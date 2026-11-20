"""Bounded expanded-data SAE stability study with streamed code statistics.

Create the locked protocol with --plan-only, then consume it with --execute.
No fallback, automatic retry, output overwrite, or post-hoc gate adjustment.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_sae import TopKSAE
from experiment_common import sha, write_json

SPLITS = ("train", "validation", "test")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def table(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Guard:
    def __init__(self, out):
        self.out = out

    def __call__(self, stage):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        fields = next(csv.reader(result.stdout.splitlines()))
        record = {"stage": stage, "utc": utc(), "gpu": fields[0].strip(),
                  "total_mib": int(fields[1]), "free_mib": int(fields[2]), "temperature_c": int(fields[3])}
        with (self.out / "hardware.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["temperature_c"] >= 88 or record["free_mib"] < 2048:
            raise RuntimeError("Hardware stop: " + json.dumps(record))
        return record


def genus_key(row):
    if "genus_key" in row:
        value = row["genus_key"]
        return json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    return "|".join(str(row.get(k, "")) for k in ["kingdom", "phylum", "class_name", "order", "family", "genus"])


@torch.inference_mode()
def evaluate(model, raw, ids, chunk, guard=None, label="evaluation"):
    sums = torch.zeros(2, dtype=torch.float64, device=raw.device)
    cosines = []
    active = torch.zeros(model.width, dtype=torch.bool, device=raw.device)
    active_count = 0
    for number, batch in enumerate(ids.split(chunk)):
        if guard:
            guard(f"{label}_chunk_{number}")
        x = raw[batch]
        reconstructed, z = model(x)
        if not bool(torch.isfinite(reconstructed).all() and torch.isfinite(z).all()):
            raise RuntimeError("Nonfinite reconstruction/code during evaluation")
        # Match original metrics: subtract in float32, accumulate squares in float64.
        sums[0] += (x - reconstructed).double().square().sum()
        sums[1] += (x - model.mean).double().square().sum()
        cosines.append(nn.functional.cosine_similarity(x, reconstructed, dim=-1).cpu().numpy())
        active_count += int((z > 1e-7).sum())
        active |= (z > 1e-7).any(0)
    sse, denominator = sums.tolist()
    values = np.concatenate(cosines)
    nmse = sse / denominator
    return {"mse": sse / (len(ids) * 768), "normalized_mse": nmse,
            "variance_explained_vs_training_mean": 1 - nmse, "cosine_mean": float(values.mean()),
            "cosine_p05": float(np.quantile(values, .05)), "mean_active_features": active_count / len(ids),
            "active_dictionary_features": int(active.sum()), "dead_dictionary_features": int((~active).sum())}


def train(seed, raw, ids, mean, scale, width, k, chunk, guard, run_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TopKSAE(mean, scale, width=width, k=k).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # Preserve the original CUDA permutation and batch-indexing semantics.
    training = raw[ids["train"]]
    best, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, 201):
        guard(f"seed_{seed}_epoch_{epoch}")
        model.train()
        order = torch.randperm(len(training), device="cuda")
        total = 0.
        for batch in order.split(256):
            x = training[batch]
            prediction, _ = model(x)
            loss = ((prediction - x) / scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at seed {seed}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.normalize_decoder()
            total += float(loss.detach()) * len(x)
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            validation = evaluate(model, raw, ids["validation"], chunk, guard, f"seed_{seed}_val_epoch_{epoch}")
            value = validation["normalized_mse"]
            history.append({"epoch": epoch, "train_normalized_mse": total / len(training),
                            "validation_normalized_mse": value})
            table(run_dir / "training_history.csv", history)
            if value < best - 1e-5:
                best, best_epoch, no_improve = value, epoch, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                no_improve += 1
            print(json.dumps({"seed": seed, "epoch": epoch, "validation_nmse": value,
                              "best_epoch": best_epoch, "unimproved_checks": no_improve}), flush=True)
            if no_improve >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, epoch


@torch.inference_mode()
def save_codes(model, raw, rows, ids, run_dir, chunk, guard):
    n, width, k = len(raw), model.width, model.k
    indices = np.lib.format.open_memmap(run_dir / "code_indices.npy", mode="w+", dtype=np.int32, shape=(n, k))
    values = np.lib.format.open_memmap(run_dir / "code_values.npy", mode="w+", dtype=np.float32, shape=(n, k))
    total_support = torch.zeros(width, dtype=torch.long, device=raw.device)
    train_support = torch.zeros_like(total_support)
    top_values = torch.full((8, width), -float("inf"), device=raw.device)
    top_rows = torch.full((8, width), -1, dtype=torch.long, device=raw.device)
    for split in SPLITS:
        for number, batch in enumerate(ids[split].split(chunk)):
            guard(f"save_codes_{run_dir.name}_{split}_{number}")
            code = model.encode(raw[batch])
            if not bool(torch.isfinite(code).all()) or int((code > 0).sum(1).max()) > k:
                raise RuntimeError("Invalid TopK code")
            v, idx = code.topk(k, dim=1)
            global_rows = batch.cpu().numpy()
            indices[global_rows] = idx.cpu().numpy().astype(np.int32)
            values[global_rows] = v.cpu().numpy()
            support = (code > 1e-7).sum(0)
            total_support += support
            if split == "train":
                train_support += support
                best_v, local = code.topk(min(8, len(code)), dim=0)
                best_rows = batch[local]
                merged_values = torch.cat([top_values, best_v])
                merged_rows = torch.cat([top_rows, best_rows])
                top_values, selected = merged_values.topk(8, dim=0)
                top_rows = merged_rows.gather(0, selected)
    indices.flush()
    values.flush()
    tn, vn = top_rows.cpu().numpy(), top_values.cpu().numpy()
    train_counts, total_counts = train_support.cpu().numpy(), total_support.cpu().numpy()
    features = []
    for feature in range(width):
        # Exact floating-point ties at the eighth cutoff may be chosen by Torch topk.
        examples = [{**rows[int(i)], "row_index": int(i), "activation": float(v)}
                    for i, v in zip(tn[:, feature], vn[:, feature]) if i >= 0 and v > 1e-7]
        examples.sort(key=lambda row: (-row["activation"], row["row_index"]))
        assert all(row["split"] == "train" for row in examples)
        features.append({"feature": feature, "train_support": int(train_counts[feature]),
                         "total_support": int(total_counts[feature]), "top_training_species": examples,
                         "top_genera": len({genus_key(row) for row in examples}),
                         "top_families": len({row.get("family") for row in examples})})
    write_json(run_dir / "features.json", features)
    return train_counts


def dense_chunk(index_array, value_array, selected_rows, width, device):
    indices = torch.from_numpy(np.asarray(index_array[selected_rows]).astype(np.int64)).to(device)
    values = torch.from_numpy(np.array(value_array[selected_rows], copy=True)).to(device)
    return torch.zeros((len(selected_rows), width), dtype=torch.float32, device=device).scatter_(1, indices, values)


def correlation_from_moments(n, sx, sy, qx, qy, cross):
    numerator = cross - sx[:, None] * sy[None, :] / n
    vx = (qx - sx.square() / n).clamp_min(0)
    vy = (qy - sy.square() / n).clamp_min(0)
    den = torch.sqrt(vx[:, None] * vy[None, :])
    return torch.where(den > 1e-12, numerator / den.clamp_min(1e-12), torch.zeros_like(den)).clamp(-1, 1)


@torch.inference_mode()
def stream_correlation(a_indices, a_values, b_indices, b_values, rows, width, chunk, device, guard=None, label="correlation"):
    sx = torch.zeros(width, dtype=torch.float64, device=device)
    sy, qx, qy = sx.clone(), sx.clone(), sx.clone()
    cross = torch.zeros((width, width), dtype=torch.float64, device=device)
    for number, start in enumerate(range(0, len(rows), chunk)):
        if guard:
            guard(f"{label}_{number}")
        selected = rows[start:start + chunk]
        a = dense_chunk(a_indices, a_values, selected, width, device)
        b = dense_chunk(b_indices, b_values, selected, width, device)
        sx += a.sum(0, dtype=torch.float64)
        sy += b.sum(0, dtype=torch.float64)
        qx += a.double().square().sum(0)
        qy += b.double().square().sum(0)
        # Float32 GEMM with highest precision; float64 accumulated moments.
        cross += (a.T @ b).double()
    return correlation_from_moments(len(rows), sx, sy, qx, qy, cross)


@torch.inference_mode()
def stream_matched_validation(ai, av, bi, bv, rows, match, width, chunk, device, guard=None):
    sums = torch.zeros((5, width), dtype=torch.float64, device=device)
    for number, start in enumerate(range(0, len(rows), chunk)):
        if guard:
            guard(f"matched_validation_{number}")
        selected = rows[start:start + chunk]
        a = dense_chunk(ai, av, selected, width, device).double()
        b = dense_chunk(bi, bv, selected, width, device)[:, match].double()
        sums[0] += a.sum(0)
        sums[1] += b.sum(0)
        sums[2] += a.square().sum(0)
        sums[3] += b.square().sum(0)
        sums[4] += (a * b).sum(0)
    sx, sy, qx, qy, cross = sums
    den = torch.sqrt((qx - sx.square() / len(rows)).clamp_min(0) * (qy - sy.square() / len(rows)).clamp_min(0))
    return torch.where(den > 1e-12, (cross - sx * sy / len(rows)) / den.clamp_min(1e-12), torch.zeros_like(den)).clamp(-1, 1)


def self_test():
    torch.manual_seed(20261004)
    rng = np.random.default_rng(51)
    n, width, k = 47, 11, 3
    a = np.zeros((n, width), dtype=np.float32)
    b = np.zeros_like(a)
    for array in (a, b):
        for row in array:
            selected = rng.choice(width - 1, k, replace=False)
            row[selected] = rng.uniform(.1, 3., k)
    ai, bi = np.argsort(-a, axis=1)[:, :k], np.argsort(-b, axis=1)[:, :k]
    av, bv = np.take_along_axis(a, ai, 1), np.take_along_axis(b, bi, 1)
    actual = stream_correlation(ai, av, bi, bv, np.arange(n), width, 7, "cpu").numpy()
    x, y = a.astype(np.float64), b.astype(np.float64)
    x -= x.mean(0)
    y -= y.mean(0)
    den = np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(y, axis=0)[None, :]
    expected = np.divide(x.T @ y, den, out=np.zeros_like(den), where=den > 1e-12)
    assert np.max(np.abs(actual - expected)) < 1e-6
    matches = torch.tensor(actual.argmax(1))
    confirm = stream_matched_validation(ai, av, bi, bv, np.arange(n), matches, width, 9, "cpu").numpy()
    assert np.max(np.abs(confirm - expected[np.arange(width), matches.numpy()])) < 1e-6
    model = TopKSAE(torch.zeros(768), torch.ones(()), width=width, k=k).eval()
    raw = torch.randn(19, 768)
    result = evaluate(model, raw, torch.arange(19), 5)
    with torch.inference_mode():
        reconstructed, z = model(raw)
        expected_nmse = float((raw - reconstructed).double().square().sum() / (raw - model.mean).double().square().sum())
    # Different GEMM batch shapes can round the float32 forward pass differently.
    assert abs(result["normalized_mse"] - expected_nmse) < 1e-6
    print(json.dumps({"self_test": "passed", "streamed_correlation_max_error": float(np.max(np.abs(actual - expected))),
                      "matched_validation_max_error": float(np.max(np.abs(confirm - expected[np.arange(width), matches.numpy()]))),
                      "streamed_nmse_error": abs(result["normalized_mse"] - expected_nmse)}), flush=True)


def make_plan(args):
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sources = [args.input_dir / "data/species.json", args.input_dir / "data/embeddings_raw.npy"]
    rows = json.loads(sources[0].read_text(encoding="utf-8"))
    raw = np.load(sources[1], mmap_mode="r")
    assert raw.shape == (len(rows), 768) and raw.dtype == np.float32
    split_rows = {split: [i for i, row in enumerate(rows) if row["split"] == split] for split in SPLITS}
    assert sum(map(len, split_rows.values())) == len(rows) and all(split_rows.values())
    groups = {split: {rows[i].get("split_group", genus_key(rows[i])) for i in selected} for split, selected in split_rows.items()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    plan = {"created_utc": utc(), "status": "prespecified_before_training",
            "authorization": "User authorized all listed research steps; pause on hardware problems",
            "input_dir": str(args.input_dir.resolve()), "input_sha256": {str(path.resolve()): sha(path) for path in sources},
            "species_count": len(rows), "split_counts": {s: len(v) for s, v in split_rows.items()},
            "width": args.width, "k": args.k, "anchor_seed": 20260929, "replicate_seeds": [1, 2, 3, 4, 5],
            "evaluation_chunk": args.chunk, "train": {"max_epochs": 200, "batch_size": 256,
            "optimizer": "Adam", "learning_rate": .001, "clip_gradient_norm": 1.,
            "decoder_column_normalization": "after every optimizer step", "preprocessing": "training-only mean/global RMS",
            "selection": "Lowest validation NMSE at epoch 1/every fifth epoch; >1e-5 improvement; stop at 8 unimproved checks"},
            "matching": "Per expanded-anchor feature, best signed training Pearson correlation; same-pair signed decoder cosine; no sign flips; collisions reported",
            "gate": {"minimum_training_activation_correlation_in_every_replicate": .7,
                     "minimum_same_match_decoder_cosine_in_every_replicate": .7,
                     "validation": "diagnostic only; not used to select matches or alter the gate"},
            "test_policy": "Test metrics descriptive only; never for checkpoint or configuration selection",
            "split_validation": "Disjoint split_group when supplied (kingdom|genus); otherwise composite full-lineage genus_key",
            "conditional_sweep": "If expanded 2048/32 baseline has no passing features, only alternatives (1024,16), (1024,32), (2048,16), each same anchor and five seeds",
            "code_storage": "Two N x k .npy arrays: int32 feature indices and float32 values; no N x width full arrays",
            "anchor_identity": "New expanded-data feature IDs; earlier 729/1969/527 IDs are not transferred",
            "feature_exemplars": "Top eight training species per feature only; exact tied cutoff chosen by Torch topk",
            "hardware": {"stop_temperature_c": 88, "minimum_free_mib": 2048, "on_error": "Stop; no automatic fallback/retry"},
            "script_sha256": sha(Path(__file__)), "trainer_sha256": sha(ROOT.parent / "train_sae.py")}
    write_json(args.out_dir / "protocol.json", plan)
    (args.out_dir / "protocol.sha256").write_text(sha(args.out_dir / "protocol.json") + "\n", encoding="ascii")
    print(json.dumps({"plan_saved": str(args.out_dir / "protocol.json"), "species": len(rows), "training_started": False}), flush=True)


def execute(out):
    started = time.perf_counter()
    protocol_path = out / "protocol.json"
    plan = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert {path.name for path in out.iterdir()} == {"protocol.json", "protocol.sha256"}, "Output already contains execution artifacts; no automatic retry/overwrite"
    protocol_hash = (out / "protocol.sha256").read_text(encoding="ascii").strip()
    assert sha(protocol_path) == protocol_hash, "Protocol changed after lock"
    assert sha(Path(__file__)) == plan["script_sha256"] and sha(ROOT.parent / "train_sae.py") == plan["trainer_sha256"]
    hashes = {path: sha(path) for path in plan["input_sha256"]}
    assert hashes == plan["input_sha256"], "Input changed after protocol lock"
    guard = Guard(out)
    try:
        guard("before_load")
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available():
            raise RuntimeError("Hardware stop: CUDA unavailable")
        source = Path(plan["input_dir"])
        rows = json.loads((source / "data/species.json").read_text(encoding="utf-8"))
        raw = torch.from_numpy(np.load(source / "data/embeddings_raw.npy")).cuda()
        assert bool(torch.isfinite(raw).all())
        ids_np = {s: np.array([i for i, row in enumerate(rows) if row["split"] == s], dtype=np.int64) for s in SPLITS}
        ids = {s: torch.from_numpy(v).cuda() for s, v in ids_np.items()}
        training = raw[ids["train"]]
        mean = training.mean(0)
        scale = (training - mean).square().mean().sqrt()
        assert torch.isfinite(scale) and scale > 0
        del training
        width, k, chunk = plan["width"], plan["k"], plan["evaluation_chunk"]
        anchor = plan["anchor_seed"]
        seeds = [anchor] + plan["replicate_seeds"]
        run_metrics, all_matches = [], []
        activation_values, decoder_values, validation_values, match_ids = [], [], [], []
        anchor_indices = anchor_values = anchor_decoder = anchor_support = None
        for seed in seeds:
            run_dir = out / f"seed_{seed}"
            run_dir.mkdir()
            guard(f"before_seed_{seed}")
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model, best_epoch, epochs = train(seed, raw, ids, mean, scale, width, k, chunk, guard, run_dir)
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            checkpoint = run_dir / "sae.pt"
            torch.save({"state_dict": state, "config": {"input_dim": 768, "width": width, "k": k, "seed": seed},
                        "best_epoch": best_epoch, "source_input_sha256": hashes}, checkpoint)
            metrics = {s: evaluate(model, raw, ids[s], chunk, guard, f"final_seed_{seed}_{s}") for s in SPLITS}
            support = save_codes(model, raw, rows, ids, run_dir, chunk, guard)
            run = {"seed": seed, "best_epoch": best_epoch, "epochs_run": epochs, "metrics": metrics,
                   "checkpoint_sha256": sha(checkpoint), "indices_sha256": sha(run_dir / "code_indices.npy"),
                   "values_sha256": sha(run_dir / "code_values.npy"), "elapsed_seconds": time.perf_counter() - tick,
                   "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved()}
            indices = np.load(run_dir / "code_indices.npy", mmap_mode="r")
            values = np.load(run_dir / "code_values.npy", mmap_mode="r")
            decoder = nn.functional.normalize(model.decoder.weight.detach(), dim=0)
            if seed == anchor:
                anchor_indices, anchor_values = indices, values
                anchor_decoder, anchor_support = decoder.clone(), support.copy()
            else:
                corr = stream_correlation(anchor_indices, anchor_values, indices, values, ids_np["train"], width, chunk, "cuda", guard, f"matching_seed_{seed}")
                a, matches = corr.max(1)
                decoder_matrix = anchor_decoder.T @ decoder
                d = decoder_matrix[torch.arange(width, device="cuda"), matches]
                v = stream_matched_validation(anchor_indices, anchor_values, indices, values, ids_np["validation"], matches, width, chunk, "cuda", guard)
                m, an, dn, vn = [value.cpu().numpy() for value in [matches, a, d, v]]
                counts = np.bincount(m, minlength=width)
                best_decoder = decoder_matrix.max(1).values.cpu().numpy()
                match_ids.append(m)
                activation_values.append(an)
                decoder_values.append(dn)
                validation_values.append(vn)
                current_matches = [{"seed": seed, "anchor_feature": j, "matched_feature": int(m[j]),
                         "training_activation_correlation": float(an[j]), "same_match_decoder_cosine": float(dn[j]),
                         "validation_activation_correlation": float(vn[j]), "best_decoder_cosine_any_feature": float(best_decoder[j]),
                         "anchor_train_support": int(anchor_support[j]), "matched_train_support": int(support[m[j]]),
                         "anchor_features_sharing_match": int(counts[m[j]])} for j in range(width)]
                table(run_dir / "matches.csv", current_matches)
                all_matches.extend(current_matches)
                run.update({"unique_matched_features": int(np.count_nonzero(counts)),
                            "anchor_features_in_collisions": int((counts[m] > 1).sum()),
                            "maximum_anchor_features_sharing_match": int(counts.max()),
                            "median_activation_match": float(np.median(an)), "median_same_match_decoder_cosine": float(np.median(dn))})
                del corr, decoder_matrix
            write_json(run_dir / "metrics.json", run)
            run_metrics.append(run)
            guard(f"after_seed_{seed}")
            print(json.dumps({"finished_seed": seed, "seconds": time.perf_counter() - tick, "best_epoch": best_epoch}), flush=True)
            del model, decoder, state
        a, d, v = np.stack(activation_values), np.stack(decoder_values), np.stack(validation_values)
        stable = (a >= .7).all(0) & (d >= .7).all(0)
        stability = [{"anchor_feature": j, "min_training_activation_correlation": float(a[:, j].min()),
                      "min_same_match_decoder_cosine": float(d[:, j].min()), "min_validation_correlation": float(v[:, j].min()),
                      "activation_only_recurrent": bool((a[:, j] >= .7).all()), "passes_prespecified_gate": bool(stable[j])} for j in range(width)]
        table(out / "feature_stability.csv", stability)
        table(out / "all_feature_matches.csv", all_matches)
        for name, array in [("matches", match_ids), ("training_correlations", a), ("decoder_cosines", d), ("validation_correlations", v)]:
            np.save(out / (name + ".npy"), np.asarray(array))
        assert {path: sha(path) for path in hashes} == hashes, "Inputs changed during training"
        assert sha(protocol_path) == protocol_hash, "Protocol changed during training"
        summary = {"status": "completed", "created_utc": utc(), "species_count": len(rows),
                   "width": width, "k": k, "anchor_seed": anchor, "replicate_seeds": plan["replicate_seeds"],
                   "stable_feature_count": int(stable.sum()), "stable_features": np.flatnonzero(stable).tolist(),
                   "activation_only_recurrent_count": int((a >= .7).all(0).sum()), "input_hashes_unchanged": True,
                   "elapsed_seconds": time.perf_counter() - started, "per_seed": run_metrics,
                   "protocol_sha256": protocol_hash, "versions": {"python": sys.version, "numpy": np.__version__,
                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
                   "interpretation": "Stable features are candidates only; no biological trait has been validated by this test"}
        write_json(out / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "per_seed"}), flush=True)
    except BaseException as exc:
        write_json(out / "STOPPED.json", {"status": "stopped", "utc": utc(), "error": repr(exc),
                   "traceback": traceback.format_exc(), "automatic_retry": False, "elapsed_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "cache/expanded_tol10m_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/expanded_stability_w2048_k32_v1")
    parser.add_argument("--width", type=int, choices=[1024, 2048], default=2048)
    parser.add_argument("--k", type=int, choices=[16, 32], default=32)
    parser.add_argument("--chunk", type=int, default=4096)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.plan_only:
        assert args.chunk > 0
        make_plan(args)
    else:
        execute(args.out_dir)


if __name__ == "__main__":
    main()
