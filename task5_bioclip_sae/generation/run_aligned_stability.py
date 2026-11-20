"""Train authorized aligned SAE seed replicates; never overwrite prior artifacts.

Stop on every exception, including CUDA OOM; there is no device fallback/retry.
The stability gate is saved before training and is not fitted to the outcomes.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_sae import TopKSAE, metrics

INPUT = ROOT / "cache/aligned"
OUT = ROOT / "experiments/aligned_stability_v1"
SEEDS = [1, 2, 3, 4, 5]
ORIGINAL_SEED = 20260929
SELECTED = {729: "ladybird candidate", 1969: "owl candidate", 527: "squirrel candidate"}
THRESHOLD = 0.7


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def snapshot(label):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    snap = {"label": label, "utc": utc(), "nvidia_smi": result.stdout.strip()}
    with (OUT / "hardware.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")
    # A hardware guard, not an automatic change to experimental settings.
    for line in result.stdout.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if float(fields[4]) >= 88:
            raise RuntimeError("Hardware stop: GPU temperature >=88 C")
        if float(fields[2]) - float(fields[3]) < 2048:
            raise RuntimeError("Hardware stop: less than 2 GiB free GPU memory")
    return snap


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def norm_columns(z):
    centered = z - z.mean(0)
    return centered / centered.norm(dim=0).clamp_min(1e-12)


def train(seed, data, mean, scale):
    # Keep model, RNG call order, optimizer, objective and stopping rule identical
    # to train_sae.main. The original seed run is a reproduction control only.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TopKSAE(mean, scale).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, 201):
        model.train()
        order = torch.randperm(len(data["train"]), device="cuda")
        total = 0.0
        for batch in order.split(256):
            x = data["train"][batch]
            pred, z = model(x)
            loss = ((pred - x) / scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss, seed {seed}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_decoder()
            total += float(loss.detach()) * len(x)
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.inference_mode():
                pred, z = model(data["validation"])
                val = metrics(data["validation"], pred, z, mean)
            history.append({"epoch": epoch, "train_normalized_mse": total / len(data["train"]), "validation_normalized_mse": val["normalized_mse"]})
            if val["normalized_mse"] < best - 1e-5:
                best = val["normalized_mse"]
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, epoch, history


def independent_correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else 0.0


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    paths = ["data/embeddings_raw.npy", "data/species.json", "results/sparse_vectors.npy", "results/reconstructed_embeddings.npy", "results/sae.pt", "results/evaluation.json"]
    hashes = {p: sha(INPUT / p) for p in paths}
    protocol = {
        "created_utc": utc(), "status": "prespecified_before_training",
        "authorization": "User: proceed, finish all necessary, pause on hardware problems such as OOM",
        "input_root": str(INPUT.resolve()), "input_sha256": hashes,
        "script_sha256": sha(Path(__file__)), "trainer_sha256": sha(ROOT.parent / "train_sae.py"),
        "seeds": SEEDS, "reproduction_seed_only": ORIGINAL_SEED,
        "architecture": "768 -> 2048 nonnegative TopK k=32 -> 768; linear decoder",
        "training": {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256, "max_epochs": 200, "gradient_clip_norm": 1.0, "decoder_unit_columns_after_each_step": True, "selection": "validation NMSE at epoch 1 and every 5 epochs; minimum improvement 1e-5; stop after 8 unimproved checks", "preprocessing": "training-only mean and one global RMS; reuse fixed whole-genus splits"},
        "matching": "For each original feature, maximize signed Pearson correlation of activations across training species. Report same-pair signed decoder cosine and validation Pearson correlation. No sign flips: activations are nonnegative.",
        "matching_reuse": "Best matches can reuse new features; explicitly report collisions. SciPy unavailable; no one-to-one assignment used.",
        "candidate_gate": {"minimum_training_activation_correlation_all_five": THRESHOLD, "minimum_same_match_decoder_cosine_all_five": THRESHOLD, "validation": "confirmation reported separately; no post hoc gate changes"},
        "selected_features": SELECTED, "nulls": "One independent species-row permutation of each trained seed plus a fixed random unit-column dictionary; diagnostic references, not significance tests",
        "hardware": "CUDA required; snapshot before/after each run; stop on any exception/OOM, >=88 C, or <2 GiB free VRAM; no fallback/retry",
        "limitations": ["Matching is anchored to the original SAE, not an all-pairs equivalence proof.", "A stable feature is not necessarily a biological trait.", "5,388 animal species subset is not the complete TreeOfLife-1M species set."]
    }
    write_json(OUT / "protocol.json", protocol)
    try:
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available():
            raise RuntimeError("Hardware stop: CUDA unavailable")
        rows = json.loads((INPUT / "data/species.json").read_text(encoding="utf-8"))
        raw = torch.from_numpy(np.load(INPUT / "data/embeddings_raw.npy")).cuda()
        assert raw.shape == (5388, 768)
        ids_np = {s: np.array([i for i, row in enumerate(rows) if row["split"] == s]) for s in ["train", "validation", "test"]}
        ids = {s: torch.tensor(idx, device="cuda") for s, idx in ids_np.items()}
        data = {s: raw[idx] for s, idx in ids.items()}
        genera = {s: {rows[i]["genus"] for i in idx} for s, idx in ids_np.items()}
        assert not (genera["train"] & genera["validation"] or genera["train"] & genera["test"] or genera["validation"] & genera["test"])
        mean = data["train"].mean(0)
        scale = (data["train"] - mean).square().mean().sqrt()
        original = torch.load(INPUT / "results/sae.pt", map_location="cuda", weights_only=True)
        w0 = nn.functional.normalize(original["state_dict"]["decoder.weight"], dim=0)
        z0n = np.load(INPUT / "results/sparse_vectors.npy")
        z0 = torch.from_numpy(z0n).cuda()
        s0 = norm_columns(z0[ids["train"]])
        v0 = norm_columns(z0[ids["validation"]])
        nullgen = torch.Generator(device="cuda").manual_seed(20261004)
        randw = nn.functional.normalize(torch.randn(768, 2048, generator=nullgen, device="cuda"), dim=0)
        null_decoder = (w0.T @ randw).max(1).values.cpu().numpy()
        np.save(OUT / "random_dictionary_best_decoder_cosine.npy", null_decoder)
        feature_rows, selected_rows, runs = [], [], []
        array_matches, array_act, array_dec, array_val, array_null = [], [], [], [], []
        reproduction = None
        for seed in [ORIGINAL_SEED] + SEEDS:
            snapshot(f"before_seed_{seed}")
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            model, best_epoch, epochs, history = train(seed, data, mean, scale)
            with torch.inference_mode():
                rec, z = model(raw)
                assert bool(torch.isfinite(rec).all() and torch.isfinite(z).all())
                assert int((z > 0).sum(1).max()) <= 32
                scores = {s: metrics(raw[idx], rec[idx], z[idx], mean) for s, idx in ids.items()}
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            checkpoint_path = OUT / f"sae_seed{seed}.pt"
            torch.save({"state_dict": state, "config": {"input_dim": 768, "width": 2048, "k": 32, "seed": seed}, "best_epoch": best_epoch, "source_input_sha256": hashes}, checkpoint_path)
            zn = z.cpu().numpy()
            np.save(OUT / f"sparse_vectors_seed{seed}.npy", zn)
            table(OUT / f"training_history_seed{seed}.csv", history)
            run = {"seed": seed, "best_epoch": best_epoch, "epochs_run": epochs, "metrics": scores, "checkpoint_sha256": sha(checkpoint_path), "sparse_vectors_sha256": sha(OUT / f"sparse_vectors_seed{seed}.npy"), "elapsed_training_and_saving_seconds": time.perf_counter() - t, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved()}
            if seed == ORIGINAL_SEED:
                reproduction = {"maximum_absolute_state_difference": max(float((model.state_dict()[k] - v).abs().max()) for k, v in original["state_dict"].items()), "maximum_absolute_activation_difference": float(np.abs(zn - z0n).max()), "best_epoch_matches": best_epoch == original["best_epoch"]}
                reproduction["passed"] = reproduction["maximum_absolute_state_difference"] < 1e-6 and reproduction["maximum_absolute_activation_difference"] < 1e-5 and reproduction["best_epoch_matches"]
                write_json(OUT / "reproduction.json", reproduction)
                if not reproduction["passed"]:
                    raise RuntimeError("Original-seed reproduction failed; investigate before new-seed training")
            else:
                with torch.inference_mode():
                    corr = s0.T @ norm_columns(z[ids["train"]])
                    a, match = corr.max(1)
                    dec_matrix = w0.T @ nn.functional.normalize(model.decoder.weight.detach(), dim=0)
                    d = dec_matrix[torch.arange(2048, device="cuda"), match]
                    # Avoid computing a full held-out matrix: only confirm training-selected pairs.
                    vz = norm_columns(z[ids["validation"]])
                    valid = (v0 * vz[:, match]).sum(0)
                    perm = torch.randperm(len(ids["train"]), device="cuda", generator=nullgen)
                    null = (s0.T @ norm_columns(z[ids["train"]][perm])).max(1).values
                matchn, an, dn, vn, nulln = [x.cpu().numpy() for x in [match, a, d, valid, null]]
                counts = np.bincount(matchn, minlength=2048)
                best_dec = dec_matrix.max(1).values.cpu().numpy()
                run.update({"unique_matched_features": int(np.count_nonzero(counts)), "original_features_in_collisions": int((counts[matchn] > 1).sum()), "maximum_original_features_sharing_match": int(counts.max()), "activation_match_median": float(np.median(an)), "same_match_decoder_cosine_median": float(np.median(dn)), "validation_same_match_correlation_median": float(np.median(vn))})
                array_matches.append(matchn)
                array_act.append(an)
                array_dec.append(dn)
                array_val.append(vn)
                array_null.append(nulln)
                for j in range(2048):
                    k = int(matchn[j])
                    entry = {"seed": seed, "original_feature": j, "matched_feature": k, "training_activation_correlation": float(an[j]), "same_match_decoder_cosine": float(dn[j]), "validation_activation_correlation": float(vn[j]), "best_decoder_cosine_any_feature": float(best_dec[j]), "original_train_support": int((z0n[ids_np["train"], j] > 1e-7).sum()), "matched_train_support": int((zn[ids_np["train"], k] > 1e-7).sum()), "original_features_sharing_match": int(counts[k])}
                    feature_rows.append(entry)
                    if j in SELECTED:
                        # Separate NumPy float64 calculation checks the selected Torch match values.
                        ac = independent_correlation(z0n[ids_np["train"], j], zn[ids_np["train"], k])
                        vc = independent_correlation(z0n[ids_np["validation"], j], zn[ids_np["validation"], k])
                        dc = float(np.dot(w0[:, j].cpu().numpy().astype(np.float64), state["decoder.weight"][:, k].numpy().astype(np.float64)) / np.linalg.norm(state["decoder.weight"][:, k].numpy().astype(np.float64)))
                        assert abs(ac - entry["training_activation_correlation"]) < 2e-5
                        assert abs(vc - entry["validation_activation_correlation"]) < 2e-5
                        assert abs(dc - entry["same_match_decoder_cosine"]) < 2e-5
                        train_ids = ids_np["train"]
                        top = train_ids[np.argsort(-zn[train_ids, k], kind="stable")[:8]]
                        selected_rows.append({**entry, "candidate": SELECTED[j], "independent_train_correlation": ac, "independent_validation_correlation": vc, "independent_decoder_cosine": dc, "top8_training_species": "; ".join(rows[i]["scientific_name"] for i in top if zn[i, k] > 1e-7)})
                del corr, dec_matrix, vz
            runs.append(run)
            write_json(OUT / f"metrics_seed{seed}.json", run)
            torch.cuda.synchronize()
            snapshot(f"after_seed_{seed}")
            print(json.dumps({"seed": seed, "best_epoch": best_epoch, "test_cosine": scores["test"]["cosine_mean"], "seconds": round(time.perf_counter() - t, 2)}), flush=True)
            del model, rec, z
        a, d, v = np.stack(array_act), np.stack(array_dec), np.stack(array_val)
        stable = (a >= THRESHOLD).all(0) & (d >= THRESHOLD).all(0)
        activation_only = (a >= THRESHOLD).all(0)
        for name, value in [("matched_features", array_matches), ("training_activation_correlations", a), ("same_match_decoder_cosines", d), ("validation_activation_correlations", v), ("shuffled_species_null_best_correlations", array_null)]:
            np.save(OUT / f"{name}.npy", np.asarray(value))
        table(OUT / "all_feature_matches.csv", feature_rows)
        table(OUT / "selected_feature_matches.csv", selected_rows)
        per_feature = [{"original_feature": j, "min_training_activation_correlation": float(a[:, j].min()), "mean_training_activation_correlation": float(a[:, j].mean()), "min_same_match_decoder_cosine": float(d[:, j].min()), "min_validation_activation_correlation": float(v[:, j].min()), "activation_only_recurrent": bool(activation_only[j]), "passes_prespecified_gate": bool(stable[j])} for j in range(2048)]
        table(OUT / "feature_stability.csv", per_feature)
        assert hashes == {p: sha(INPUT / p) for p in paths}, "Input files changed"
        summary = {"status": "completed", "completed_utc": utc(), "seeds": SEEDS, "species_count": len(rows), "split_counts": {s: len(i) for s, i in ids_np.items()}, "prespecified_gate": protocol["candidate_gate"], "stable_feature_count": int(stable.sum()), "stable_features": np.flatnonzero(stable).tolist(), "activation_only_recurrent_feature_count": int(activation_only.sum()), "activation_only_recurrent_features": np.flatnonzero(activation_only).tolist(), "selected_features": [per_feature[j] for j in SELECTED], "reproduction": reproduction, "input_hashes_unchanged": True, "null_reference": {"random_dictionary_best_signed_decoder_cosine_median": float(np.median(null_decoder)), "random_dictionary_best_signed_decoder_cosine_p99": float(np.quantile(null_decoder, .99)), "shuffled_species_best_activation_correlation_median": float(np.median(array_null)), "shuffled_species_best_activation_correlation_p99": float(np.quantile(array_null, .99))}, "per_seed": runs, "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(0)}, "elapsed_seconds": time.perf_counter() - started}
        write_json(OUT / "summary.json", summary)
        report = ["# Aligned SAE seed-stability experiment", "", f"Completed {summary['completed_utc']}. **{int(stable.sum())}/2,048 original features pass the prespecified stability gate across all five new seeds.** {int(activation_only.sum())} features recur under the activation-only criterion.", "", "## Protocol", "", "Five new seeds (1–5) used the same 5,388 generator-aligned, space-separated BioCLIP 2 taxonomy embeddings, fixed whole-genus splits, 768→2,048 TopK-32 architecture, optimizer and validation selection as the original SAE. The original seed 20260929 was retrained solely as a reproduction control. Original artifacts were not modified. BioCLIP and TaxaAdapter were not trained.", "", "Before training, `protocol.json` fixed the candidate gate: training activation correlation ≥0.7 **and** cosine of the same matched decoder direction ≥0.7 in every new seed. Validation activation correlation is reported as confirmation, not used to choose matches. Matching maximizes signed training Pearson correlation independently for each original feature. Sign flips are not allowed because codes are nonnegative. SciPy was unavailable; best matches can reuse features, with collisions reported for every pair in `all_feature_matches.csv`. This is recurrence relative to the original SAE, not an all-pairs proof or a test of biological trait identity.", "", "## Selected candidates", "", "| Original feature | Candidate | Min training correlation | Min same-pair decoder cosine | Min validation correlation | Passes gate |", "|---:|---|---:|---:|---:|---|"]
        for j, label in SELECTED.items():
            f = per_feature[j]
            report.append(f"| {j} | {label} | {f['min_training_activation_correlation']:.4f} | {f['min_same_match_decoder_cosine']:.4f} | {f['min_validation_activation_correlation']:.4f} | {f['passes_prespecified_gate']} |")
        report += ["", "## Reconstruction and validation", "", "| Seed | Best epoch | Test explained variance | Test cosine | Original features in match collisions |", "|---:|---:|---:|---:|---:|"]
        for run in runs:
            report.append(f"| {run['seed']} | {run['best_epoch']} | {run['metrics']['test']['variance_explained_vs_training_mean']:.4f} | {run['metrics']['test']['cosine_mean']:.4f} | {run.get('original_features_in_collisions', 'reproduction only')} |")
        report += ["", f"Original-seed reproduction passed: {reproduction['passed']}; maximum state difference {reproduction['maximum_absolute_state_difference']:.3g}, maximum activation difference {reproduction['maximum_absolute_activation_difference']:.3g}. All source hashes remained unchanged. Selected-feature correlations and decoder cosines were independently recomputed in NumPy float64 and agree within 2e-5. Whole-genus split separation, finite outputs and TopK bounds were checked.", "", "Null references use a random decoder dictionary and one independent shuffled-species permutation for each trained seed. These describe chance best-match behavior and are not calibrated p-values; sparse activation patterns and searching 2,048 possible matches can produce high null maxima.", "", "## Interpretation and next step", "", ("No feature meets the predeclared combined recurrence gate. Do not promote the existing ladybird/owl/squirrel dimensions as stable biological traits, and do not relax thresholds after seeing the result. Existing image edits remain exploratory responses to one fitted coordinate system. Any further generation of these leads should be explicitly exploratory; a confirmatory trait experiment must first establish a reproducible representation or reproducible feature group." if not stable.any() else "Only features that pass the gate are candidates for subsequent prespecified trait measurements and paired image-generation controls. Stability alone does not establish that a feature encodes one isolated biological trait."), "", "The 5,388 species pilot is not the full TreeOfLife-1M species set. Taxonomy-associated activations may encode names and relatedness rather than separable morphology. Hardware snapshots were collected before/after every seed in `hardware.jsonl`; no fallback or retry was used.", "", "## Files", "", "- `protocol.json`: protocol saved before training, source and script hashes.", "- `summary.json`, `metrics_seed*.json`: complete reconstruction and stability metrics.", "- `all_feature_matches.csv`, `selected_feature_matches.csv`, `feature_stability.csv`: inspectable matching results, collision counts, conservative gate.", "- `*_correlations.npy`, `same_match_decoder_cosines.npy`, `matched_features.npy`: all-feature matching arrays, in seed order 1–5.", "- `sae_seed*.pt`, `sparse_vectors_seed*.npy`, `training_history_seed*.csv`: saved trained outputs; input artifacts preserved.", "- `hardware.jsonl`, `reproduction.json`: hardware monitoring and original-seed control.", ""]
        (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ["stable_feature_count", "stable_features", "activation_only_recurrent_feature_count", "selected_features", "reproduction", "elapsed_seconds"]}), flush=True)
    except Exception as exc:
        write_json(OUT / "STOPPED.json", {"utc": utc(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "hardware_or_oom_must_pause": isinstance(exc, torch.cuda.OutOfMemoryError) or any(term in str(exc).lower() for term in ["hardware", "cuda", "out of memory"]), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
