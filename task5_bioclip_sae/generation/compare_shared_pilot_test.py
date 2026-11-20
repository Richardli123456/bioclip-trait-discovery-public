"""Post hoc descriptive CPU comparison on the identical original test species.

Neither checkpoint nor features are selected from this comparison. Both variance
explained values use the original pilot TRAINING mean as the common reference.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj, {name: value.numpy() for name, value in obj["state_dict"].items()}


def reconstruct(raw, state, k):
    pieces = []
    for start in range(0, len(raw), 256):
        x = raw[start:start + 256]
        positive = np.maximum(((x - state["mean"]) / state["scale"]) @ state["encoder.weight"].T + state["encoder.bias"], 0)
        indices = np.argpartition(positive, -k, axis=1)[:, -k:]
        code = np.zeros_like(positive)
        np.put_along_axis(code, indices, np.take_along_axis(positive, indices, axis=1), axis=1)
        pieces.append((code @ state["decoder.weight"].T + state["output_bias"]) * state["scale"] + state["mean"])
    return np.concatenate(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/expanded_stability_w2048_k32_v2")
    args = parser.parse_args()
    out = args.out_dir
    assert (out / "summary.json").exists() and (out / "independent_cpu_audit.json").exists(), "Run only after main audit"
    assert read(out / "independent_cpu_audit.json")["status"] == "passed"
    destination = out / "common_pilot_test_comparison.json"
    assert not destination.exists(), "Refusing output overwrite"
    plan = read(out / "protocol.json")
    assert sha(out / "protocol.json") == (out / "protocol.sha256").read_text().strip()
    original = ROOT / "cache/aligned"
    original_rows = read(original / "data/species.json")
    expanded_rows = read(Path(plan["input_dir"]) / "data/species.json")
    test_indices = np.array([i for i, row in enumerate(original_rows) if row["split"] == "test"])
    test_groups = {original_rows[i]["kingdom"] + "|" + original_rows[i]["genus"] for i in test_indices}
    expanded_training_groups = {row["split_group"] for row in expanded_rows if row["split"] == "train"}
    assert not test_groups & expanded_training_groups
    expanded_by_name = {row["scientific_name"]: row for row in expanded_rows}
    assert all(expanded_by_name[original_rows[i]["scientific_name"]]["split"] == "test" for i in test_indices)
    raw = np.load(original / "data/embeddings_raw.npy", mmap_mode="r")
    x = np.asarray(raw[test_indices])
    old_path = original / "results/sae.pt"
    new_path = out / f"seed_{plan['anchor_seed']}/sae.pt"
    old_obj, old_state = load(old_path)
    new_obj, new_state = load(new_path)
    assert sha(new_path) == read(new_path.parent / "metrics.json")["checkpoint_sha256"]
    train_ids = np.array([i for i, row in enumerate(original_rows) if row["split"] == "train"])
    old_mean_reference_error = float(np.max(np.abs(np.asarray(raw[train_ids], dtype=np.float64).mean(0) - old_state["mean"])))
    assert old_mean_reference_error < 1e-5
    common_ss = float(((x.astype(np.float64) - old_state["mean"].astype(np.float64)) ** 2).sum())
    results = {}
    for label, obj, state in [("original_5388_anchor", old_obj, old_state), ("expanded_clean_anchor", new_obj, new_state)]:
        y = reconstruct(x, state, obj["config"]["k"]).astype(np.float64)
        xx = x.astype(np.float64)
        cosine = (xx * y).sum(1) / (np.linalg.norm(xx, axis=1) * np.linalg.norm(y, axis=1))
        sse = float(((xx - y) ** 2).sum())
        results[label] = {"cosine_mean": float(cosine.mean()), "cosine_p05": float(np.quantile(cosine, .05)),
                          "mse": sse / x.size, "variance_explained_against_common_original_training_mean": 1 - sse / common_ss,
                          "normalized_mse_against_common_original_training_mean": sse / common_ss,
                          "best_epoch": obj["best_epoch"], "seed": obj["config"]["seed"]}
    report = {"status": "completed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "analysis_type": "Additional post hoc descriptive diagnostic; not a prespecified primary outcome",
              "selection_policy": "No model, checkpoint, feature, or configuration selected using this result",
              "execution": "CPU NumPy forward reconstruction from saved checkpoints; no training or GPU",
              "shared_test_species_count": len(test_indices), "shared_test_genera_count": len(test_groups),
              "expanded_training_overlap_with_original_test_genus_groups": 0,
              "all_original_test_species_remain_test": True, "test_row_indices": test_indices.tolist(),
              "common_reference": "Original aligned pilot training mean for BOTH explained-variance denominators",
              "common_reference_mean_max_error_vs_original_training_float64": old_mean_reference_error,
              "common_test_sum_squared_distance_to_original_training_mean": common_ss,
              "results": results,
              "expanded_minus_original": {"cosine_mean": results["expanded_clean_anchor"]["cosine_mean"] - results["original_5388_anchor"]["cosine_mean"],
                                         "variance_explained_common_reference": results["expanded_clean_anchor"]["variance_explained_against_common_original_training_mean"] - results["original_5388_anchor"]["variance_explained_against_common_original_training_mean"]},
              "sha256": {"script": sha(Path(__file__)), "original_checkpoint": sha(old_path), "expanded_checkpoint": sha(new_path),
                         "original_species": sha(original / "data/species.json"), "original_embeddings": sha(original / "data/embeddings_raw.npy")},
              "limitations": "Single fixed anchor seed per dataset; corpus size and composition differ jointly. This controls the evaluation species and reference mean, but does not isolate dataset size causally or validate biological traits."}
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": "completed", "test_species": len(test_indices), "results": results,
                      "expanded_minus_original": report["expanded_minus_original"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
