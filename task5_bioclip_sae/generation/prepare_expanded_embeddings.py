"""Frozen BioCLIP2 space-format text extraction with bounded GPU batches.

Run only after GPU ownership has been granted. Stop on every exception; no
automatic smaller-batch retry, CPU fallback, or model training is implemented.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import time
import traceback

import numpy as np
import open_clip
import torch

from prepare_expanded_species import ROOT, OUT, sha, write_json


def hardware(label):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    value = {"label": label, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "nvidia_smi": result.stdout.strip()}
    with (OUT / "embedding_hardware.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value) + "\n")
    for line in result.stdout.strip().splitlines():
        fields = [p.strip() for p in line.split(",")]
        if float(fields[4]) >= 88:
            raise RuntimeError("Hardware stop: GPU temperature >=88 C")
        if float(fields[2]) - float(fields[3]) < 2048:
            raise RuntimeError("Hardware stop: GPU free memory <2 GiB")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true", help="Explicit resume after diagnosing a prior interruption; never automatic")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.batch_size != 128:
        raise ValueError("Prespecified batch size is 128")
    species_path = OUT / "data/species.json"
    rows = json.loads(species_path.read_text(encoding="utf-8"))
    n = len(rows)
    output_path = OUT / "data/embeddings_raw.npy"
    token_path = OUT / "data/token_ids.npy"
    progress_path = OUT / "embedding_progress.json"
    model_files = {p.name: sha(p) for p in sorted(args.model_path.glob("*")) if p.is_file()}
    inputs = {"species_json_sha256": sha(species_path), "model_files_sha256": model_files,
              "bioclip_revision": "2957b322090f9cb17ae72c71981c7218a28d81e0", "batch_size": args.batch_size}
    start = 0
    if output_path.exists():
        if not args.resume:
            raise RuntimeError("Output exists; refusing overwrite. Use explicit --resume only after diagnosis.")
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        assert previous["inputs"] == inputs, "Resume input mismatch"
        start = previous["rows_completed"]
        x = np.load(output_path, mmap_mode="r+")
        tokens_out = np.load(token_path, mmap_mode="r+")
        assert x.shape == (n, 768) and tokens_out.shape == (n, 77)
    else:
        if shutil.disk_usage(OUT).free < 4 * 1024**3:
            raise RuntimeError("Hardware stop: <4 GiB free disk")
        x = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(n, 768))
        tokens_out = np.lib.format.open_memmap(token_path, mode="w+", dtype="int32", shape=(n, 77))
    write_json(progress_path, {"status": "running", "rows_completed": start, "rows_total": n, "inputs": inputs})
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Hardware stop: CUDA unavailable")
    hardware("before_model_load")
    torch.cuda.reset_peak_memory_stats()
    name = "local-dir:" + str(args.model_path.resolve())
    model = open_clip.create_model(name, device="cuda", precision="fp32", require_pretrained=True).eval()
    model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(name)
    hardware("after_model_load")
    validation_rows = []
    with torch.inference_mode():
        for begin in range(start, n, args.batch_size):
            end = min(n, begin + args.batch_size)
            tax = [r["taxonomy"] for r in rows[begin:end]]
            tokens = tokenizer(tax)
            if not bool((tokens[:, -1] == 0).all()):
                bad = (tokens[:, -1] != 0).nonzero().flatten().tolist()
                write_json(OUT / "possible_truncation.json", [{"row_index": begin + j, "taxonomy": tax[j]} for j in bad])
                raise RuntimeError("Possible token truncation; inspect saved rows before continuation")
            result = model.encode_text(tokens.cuda(), normalize=False).cpu().numpy()
            if result.shape != (end - begin, 768) or not np.isfinite(result).all():
                raise RuntimeError("Nonfinite or wrong-shaped encoder output")
            x[begin:end] = result
            tokens_out[begin:end] = tokens.numpy().astype(np.int32)
            if begin == start or end == n or end % 4096 == 0:
                x.flush()
                tokens_out.flush()
                hardware(f"completed_{end}")
                write_json(progress_path, {"status": "running", "rows_completed": end, "rows_total": n, "inputs": inputs,
                                           "elapsed_seconds": time.perf_counter() - started})
                print(f"BioCLIP2 rows {end}/{n}, elapsed {time.perf_counter()-started:.1f}s", flush=True)
        # Check raw-vector equivalence against every unchanged pilot row.
        old = np.load(ROOT / "cache/aligned/data/embeddings_raw.npy")
        delta = np.abs(np.asarray(x[:len(old)]) - old)
        cos = (x[:len(old)] * old).sum(1) / (np.linalg.norm(x[:len(old)], axis=1) * np.linalg.norm(old, axis=1))
        if not np.allclose(x[:len(old)], old, atol=1e-5, rtol=1e-4):
            raise RuntimeError("Unchanged pilot embeddings failed reproduction")
        for index in [0, n // 2, n - 1]:
            single = model.encode_text(tokenizer([rows[index]["taxonomy"]]).cuda(), normalize=False).cpu().numpy()[0]
            maximum = float(np.abs(single - x[index]).max())
            assert np.allclose(single, x[index], atol=1e-5, rtol=1e-4)
            validation_rows.append({"row_index": index, "batch_vs_single_max_absolute_difference": maximum})
    hardware("after_extraction")
    # CPU checks are streamed; do not materialize a full N x N comparison.
    norms_min, norms_max = float("inf"), 0.0
    token_hash_to_row = {}
    collisions = []
    for begin in range(0, n, 4096):
        values = x[begin:begin + 4096]
        assert np.isfinite(values).all()
        norms = np.linalg.norm(values, axis=1)
        norms_min = min(norms_min, float(norms.min()))
        norms_max = max(norms_max, float(norms.max()))
        assert bool((norms > 0).all())
        for j, tok in enumerate(tokens_out[begin:begin + 4096]):
            key = hashlib.sha256(tok.tobytes()).digest()
            if key in token_hash_to_row:
                collisions.append([token_hash_to_row[key], begin + j])
            else:
                token_hash_to_row[key] = begin + j
    write_json(OUT / "token_collisions.json", collisions)
    result = {"status": "complete", "training_performed": False, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "inputs": inputs, "shape": [n, 768], "dtype": "float32", "space_format": True,
              "representation": "BioCLIP2 pooled projected raw text output, normalize=False; eval, requires_grad=False, torch.inference_mode, precision fp32",
              "model_frozen": True, "embedding_sha256": sha(output_path), "token_ids_sha256": sha(token_path),
              "script_sha256": sha(__file__), "elapsed_seconds": time.perf_counter()-started,
              "all_rows_finite": True, "token_last_position_zero_all_rows": True,
              "token_collision_pairs": len(collisions), "nonzero_norm_range": [norms_min, norms_max],
              "pilot_reproduction": {"rows": len(old), "max_absolute_difference": float(delta.max()), "mean_cosine": float(cos.mean()), "passed": True},
              "single_row_checks": validation_rows, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(), "torch_version": torch.__version__,
              "numpy_version": np.__version__, "python_version": platform.python_version(), "hardware_errors": []}
    write_json(OUT / "embedding_provenance.json", result)
    write_json(progress_path, {"status": "complete", "rows_completed": n, "rows_total": n, "inputs": inputs})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(OUT / "embedding_failure.json", {"status": "stopped", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
