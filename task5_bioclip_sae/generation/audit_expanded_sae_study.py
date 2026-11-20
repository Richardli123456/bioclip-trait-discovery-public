"""CPU-only independent audit of completed expanded SAE outputs.

Does not import the training runner, model class, or its matching/evaluation
functions. Partial mode prints read-only completed-seed checks without writing.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import traceback

import numpy as np
import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj, {name: value.numpy() for name, value in obj["state_dict"].items()}


def read_codes(folder):
    return (np.load(folder / "code_indices.npy", mmap_mode="r"),
            np.load(folder / "code_values.npy", mmap_mode="r"))


def select_columns(indices, values, columns):
    # Independent sparse lookup; handles repeated requested feature IDs.
    return np.stack([np.where(indices == j, values, 0).sum(1, dtype=np.float64) for j in columns], axis=1)


def sparse_pearson_all_targets(ai, av, bi, bv, selected_rows, features, width):
    """Float64 sufficient statistics using sparse bincount, no dense width matrix."""
    sx, qx = np.zeros(len(features)), np.zeros(len(features))
    sy, qy = np.zeros(width), np.zeros(width)
    cross = np.zeros((len(features), width))
    for start in range(0, len(selected_rows), 4096):
        rows = selected_rows[start:start + 4096]
        a = select_columns(np.asarray(ai[rows]), np.asarray(av[rows]), features)
        indices = np.asarray(bi[rows]).ravel()
        values = np.asarray(bv[rows], dtype=np.float64)
        sy += np.bincount(indices, weights=values.ravel(), minlength=width)
        qy += np.bincount(indices, weights=(values * values).ravel(), minlength=width)
        sx += a.sum(0)
        qx += (a * a).sum(0)
        for column in range(len(features)):
            cross[column] += np.bincount(indices, weights=(a[:, column, None] * values).ravel(), minlength=width)
    n = len(selected_rows)
    den = np.sqrt(np.maximum(qx - sx * sx / n, 0)[:, None] * np.maximum(qy - sy * sy / n, 0)[None, :])
    return np.clip(np.divide(cross - sx[:, None] * sy[None, :] / n, den,
                            out=np.zeros_like(cross), where=den > 1e-12), -1, 1)


def audit_seed(folder, plan, rows, raw, selected_rows, anchor_state):
    metadata = read_json(folder / "metrics.json")
    seed, width, k = metadata["seed"], plan["width"], plan["k"]
    assert digest(folder / "sae.pt") == metadata["checkpoint_sha256"]
    assert digest(folder / "code_indices.npy") == metadata["indices_sha256"]
    assert digest(folder / "code_values.npy") == metadata["values_sha256"]
    obj, state = checkpoint(folder / "sae.pt")
    assert obj["config"] == {"input_dim": 768, "width": width, "k": k, "seed": seed}
    assert obj["source_input_sha256"] == plan["input_sha256"]
    assert np.array_equal(state["mean"], anchor_state["mean"])
    assert np.array_equal(state["scale"], anchor_state["scale"])
    assert all(np.isfinite(value).all() for value in state.values())
    assert np.max(np.abs(np.linalg.norm(state["decoder.weight"], axis=0) - 1)) < 1e-5
    history = read_csv(folder / "training_history.csv")
    expected_epochs = [1] + list(range(5, metadata["epochs_run"] + 1, 5))
    assert [int(row["epoch"]) for row in history] == expected_epochs
    best, best_epoch, patience = float("inf"), None, 0
    for record in history:
        value = float(record["validation_normalized_mse"])
        assert np.isfinite(value) and np.isfinite(float(record["train_normalized_mse"]))
        if value < best - 1e-5:
            best, best_epoch, patience = value, int(record["epoch"]), 0
        else:
            patience += 1
        assert patience < 8 or int(record["epoch"]) == metadata["epochs_run"]
    assert best_epoch == metadata["best_epoch"] == obj["best_epoch"]
    assert metadata["epochs_run"] == 200 or patience == 8
    assert abs(best - metadata["metrics"]["validation"]["normalized_mse"]) < 1e-10
    indices, values = read_codes(folder)
    assert indices.shape == values.shape == (len(rows), k)
    assert indices.dtype == np.int32 and values.dtype == np.float32
    split_array = np.array([row["split"] for row in rows])
    supports = {split: np.zeros(width, dtype=np.int64) for split in ["train", "validation", "test"]}
    active_sums = {split: 0 for split in supports}
    for start in range(0, len(rows), 8192):
        ii, vv = np.asarray(indices[start:start + 8192]), np.asarray(values[start:start + 8192])
        assert np.isfinite(vv).all() and (vv >= 0).all() and (ii >= 0).all() and (ii < width).all()
        assert (np.diff(np.sort(ii, axis=1), axis=1) > 0).all(), "Repeated feature index within code"
        for split in supports:
            mask = split_array[start:start + len(ii)] == split
            valid = vv[mask] > 1e-7
            supports[split] += np.bincount(ii[mask][valid], minlength=width)
            active_sums[split] += int(valid.sum())
    for split, support in supports.items():
        assert int((support > 0).sum()) == metadata["metrics"][split]["active_dictionary_features"]
        assert width - int((support > 0).sum()) == metadata["metrics"][split]["dead_dictionary_features"]
        assert abs(active_sums[split] / plan["split_counts"][split] - metadata["metrics"][split]["mean_active_features"]) < 1e-10
    features = read_json(folder / "features.json")
    for j, feature in enumerate(features):
        assert feature["feature"] == j
        assert feature["train_support"] == int(supports["train"][j])
        assert feature["total_support"] == sum(int(value[j]) for value in supports.values())
        assert len(feature["top_training_species"]) <= 8
        for example in feature["top_training_species"]:
            row = example["row_index"]
            assert rows[row]["split"] == example["split"] == "train"
            assert rows[row]["scientific_name"] == example["scientific_name"]
            activation = float(values[row][indices[row] == j].sum())
            assert abs(activation - example["activation"]) < 1e-7
    x = np.asarray(raw[selected_rows])
    pre = np.maximum(((x - state["mean"]) / state["scale"]) @ state["encoder.weight"].T + state["encoder.bias"], 0)
    chosen = np.argpartition(pre, -k, axis=1)[:, -k:]
    independently_encoded = np.zeros_like(pre)
    np.put_along_axis(independently_encoded, chosen, np.take_along_axis(pre, chosen, axis=1), axis=1)
    saved = np.zeros_like(pre)
    np.put_along_axis(saved, np.asarray(indices[selected_rows]), np.asarray(values[selected_rows]), axis=1)
    error = float(np.max(np.abs(independently_encoded - saved)))
    support_disagreement = int(np.count_nonzero((independently_encoded > 1e-7) != (saved > 1e-7)))
    assert error < 2e-4 and support_disagreement == 0
    decoded_independent = (independently_encoded @ state["decoder.weight"].T + state["output_bias"]) * state["scale"] + state["mean"]
    decoded_saved = (saved @ state["decoder.weight"].T + state["output_bias"]) * state["scale"] + state["mean"]
    decode_error = float(np.max(np.abs(decoded_independent - decoded_saved)))
    assert decode_error < 1e-4
    return {"seed": seed, "all_artifact_hashes_passed": True, "best_epoch_replayed": best_epoch,
            "checkpoint_selection_and_stopping_passed": True, "all_compressed_code_rows_valid": len(rows),
            "support_counts_and_training_only_exemplars_passed": True,
            "sampled_encode_max_absolute_error": error, "sampled_support_disagreements": support_disagreement,
            "sampled_decode_max_absolute_error": decode_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/expanded_stability_w2048_k32_v1")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    out, started = args.out_dir, time.perf_counter()
    if not args.partial and not (out / "summary.json").exists():
        raise RuntimeError("Training summary is not present; full audit waits for completed outputs")
    report = {"status": "running", "execution": "CPU only; independent NumPy and sparse bincount calculations; no fitting",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "auditor_sha256": digest(Path(__file__))}
    try:
        plan = read_json(out / "protocol.json")
        plan_hash = digest(out / "protocol.json")
        assert plan_hash == (out / "protocol.sha256").read_text().strip()
        assert digest(ROOT / "run_expanded_sae_study.py") == plan["script_sha256"]
        assert digest(ROOT.parent / "train_sae.py") == plan["trainer_sha256"]
        assert {path: digest(path) for path in plan["input_sha256"]} == plan["input_sha256"]
        source = Path(plan["input_dir"])
        rows = read_json(source / "data/species.json")
        raw = np.load(source / "data/embeddings_raw.npy", mmap_mode="r")
        # The v1 source audit found cross-kingdom copies of identical binomials.
        # Final datasets must remove ambiguous copies, not merely separate genera.
        canonical_names = [" ".join(row["scientific_name"].split()).casefold() for row in rows]
        assert len(set(canonical_names)) == len(rows), "Repeated global scientific name in final dataset"
        split_ids = {split: np.array([i for i, row in enumerate(rows) if row["split"] == split]) for split in ["train", "validation", "test"]}
        groups = {split: {rows[i]["split_group"] for i in ids} for split, ids in split_ids.items()}
        assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
        assert {split: len(ids) for split, ids in split_ids.items()} == plan["split_counts"]
        rng = np.random.default_rng(2026100416)
        sample = np.sort(np.concatenate([rng.choice(split_ids[split], size=size, replace=False)
                                        for split, size in [("train", 6), ("validation", 5), ("test", 5)]]))
        anchor_folder = out / f"seed_{plan['anchor_seed']}"
        anchor_obj, anchor_state = checkpoint(anchor_folder / "sae.pt")
        prior_path = source / "anchor_cpu_verification.json"
        prior = read_json(prior_path) if prior_path.exists() else None
        if prior is not None and prior["checkpoint_sha256"] == digest(anchor_folder / "sae.pt"):
            assert prior["status"] == "passed"
            assert prior["train_mean_max_abs_difference_float64_reference"] < 1e-5
            assert prior["train_global_rms_abs_difference_float64_reference"] < 1e-6
            report["training_preprocessing"] = {"independent_prior_audit_reused": str(prior_path),
                "prior_audit_sha256": digest(prior_path), "mean_max_error": prior["train_mean_max_abs_difference_float64_reference"],
                "rms_error": prior["train_global_rms_abs_difference_float64_reference"],
                "every_seed_mean_and_scale_required_identical_to_anchor": True}
        else:
            sx, squares, n = np.zeros(768), 0., 0
            for start in range(0, len(split_ids["train"]), 8192):
                x = np.asarray(raw[split_ids["train"][start:start + 8192]], dtype=np.float64)
                sx += x.sum(0)
                squares += float((x * x).sum())
                n += len(x)
            mean = sx / n
            scale = np.sqrt(squares / (n * 768) - np.mean(mean * mean))
            mean_error, scale_error = float(np.max(np.abs(mean - anchor_state["mean"]))), abs(float(scale) - float(anchor_state["scale"]))
            assert mean_error < 1e-5 and scale_error < 1e-6
            report["training_preprocessing"] = {"independent_streamed_float64": True, "mean_max_error": mean_error, "rms_error": scale_error}
        report["sampled_rows"] = sample.tolist()
        report["seed_checks"] = []
        for seed in [plan["anchor_seed"]] + plan["replicate_seeds"]:
            folder = out / f"seed_{seed}"
            if not (folder / "metrics.json").exists() and args.partial:
                continue
            result = audit_seed(folder, plan, rows, raw, sample, anchor_state)
            report["seed_checks"].append(result)
            print(json.dumps(result), flush=True)
        if args.partial:
            print(json.dumps({"partial_audit": "passed", "completed_seeds": [r["seed"] for r in report["seed_checks"]], "elapsed_seconds": time.perf_counter() - started}), flush=True)
            return
        summary = read_json(out / "summary.json")
        assert summary["status"] == "completed" and summary["protocol_sha256"] == plan_hash
        matches = read_csv(out / "all_feature_matches.csv")
        width, seeds = plan["width"], plan["replicate_seeds"]
        assert len(matches) == width * len(seeds)
        lookup = {(int(r["seed"]), int(r["anchor_feature"])): r for r in matches}
        assert len(lookup) == len(matches)
        activation = np.array([[float(lookup[s, j]["training_activation_correlation"]) for j in range(width)] for s in seeds])
        decoder = np.array([[float(lookup[s, j]["same_match_decoder_cosine"]) for j in range(width)] for s in seeds])
        validation = np.array([[float(lookup[s, j]["validation_activation_correlation"]) for j in range(width)] for s in seeds])
        match_ids = np.array([[int(lookup[s, j]["matched_feature"]) for j in range(width)] for s in seeds])
        stable = (activation >= .7).all(0) & (decoder >= .7).all(0)
        assert summary["stable_feature_count"] == int(stable.sum())
        assert summary["stable_features"] == np.flatnonzero(stable).tolist()
        assert summary["activation_only_recurrent_count"] == int((activation >= .7).all(0).sum())
        for filename, array in [("training_correlations.npy", activation), ("decoder_cosines.npy", decoder),
                                ("validation_correlations.npy", validation), ("matches.npy", match_ids)]:
            assert np.array_equal(np.load(out / filename), array)
        stability_rows = read_csv(out / "feature_stability.csv")
        assert len(stability_rows) == width
        for j, record in enumerate(stability_rows):
            assert int(record["anchor_feature"]) == j
            assert (record["passes_prespecified_gate"] == "True") == bool(stable[j])
            assert (record["activation_only_recurrent"] == "True") == bool((activation[:, j] >= .7).all())
            assert float(record["min_training_activation_correlation"]) == float(activation[:, j].min())
            assert float(record["min_same_match_decoder_cosine"]) == float(decoder[:, j].min())
            assert float(record["min_validation_correlation"]) == float(validation[:, j].min())
        # Eight features closest to the joint gate plus eight reproducibly random others.
        distance = np.abs(np.minimum(activation.min(0), decoder.min(0)) - .7)
        chosen = np.argsort(distance, kind="stable")[:8].tolist()
        others = np.setdiff1d(np.arange(width), chosen)
        chosen += rng.choice(others, 8, replace=False).tolist()
        ai, av = read_codes(anchor_folder)
        aweight = anchor_state["decoder.weight"].astype(np.float64)
        aweight /= np.linalg.norm(aweight, axis=0)
        independent_checks = []
        for sindex, seed in enumerate(seeds):
            folder = out / f"seed_{seed}"
            _, state = checkpoint(folder / "sae.pt")
            weight = state["decoder.weight"].astype(np.float64)
            weight /= np.linalg.norm(weight, axis=0)
            independent_decoder = (aweight * weight[:, match_ids[sindex]]).sum(0)
            decoder_error = float(np.max(np.abs(independent_decoder - decoder[sindex])))
            assert decoder_error < 2e-6
            counts = np.bincount(match_ids[sindex], minlength=width)
            for j in range(width):
                assert int(lookup[seed, j]["anchor_features_sharing_match"]) == int(counts[match_ids[sindex, j]])
            bi, bv = read_codes(folder)
            corr = sparse_pearson_all_targets(ai, av, bi, bv, split_ids["train"], chosen, width)
            vcorr = sparse_pearson_all_targets(ai, av, bi, bv, split_ids["validation"], chosen, width)
            saved_matches = match_ids[sindex, chosen]
            independent_best = corr.argmax(1)
            saved_pair_values = corr[np.arange(len(chosen)), saved_matches]
            activation_error = float(np.max(np.abs(saved_pair_values - activation[sindex, chosen])))
            validation_error = float(np.max(np.abs(vcorr[np.arange(len(chosen)), saved_matches] - validation[sindex, chosen])))
            match_deficit = corr.max(1) - saved_pair_values
            assert activation_error < 2e-6 and validation_error < 2e-6 and float(match_deficit.max()) < 2e-6
            result = {"seed": seed, "features": chosen, "full_dictionary_argmax_queries": len(chosen),
                      "argmax_indices_identical_count": int((independent_best == saved_matches).sum()),
                      "maximum_reported_match_deficit": float(match_deficit.max()),
                      "training_pair_correlation_max_error": activation_error,
                      "validation_pair_correlation_max_error": validation_error,
                      "all_feature_decoder_cosine_max_error": decoder_error}
            independent_checks.append(result)
            print(json.dumps(result), flush=True)
        assert digest(out / "protocol.json") == plan_hash
        report.update({"status": "passed", "all_stability_decisions_recomputed": width,
                       "stable_feature_count": int(stable.sum()), "matches_and_gate_tables_consistent": True,
                       "independent_matches": independent_checks, "protocol_sha256": plan_hash,
                       "elapsed_seconds": time.perf_counter() - started,
                       "limitations": "No optimization rerun; encode/decode checks cover 16 stratified rows per seed. Full code validity/support and all gate/decoder decisions checked; training/validation Pearson and full-dictionary argmax independently recomputed for 16 features per replicate."})
        output = out / "independent_cpu_audit.json"
        assert not output.exists(), "Audit output already exists; refusing overwrite"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"audit": "passed", "report": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    except BaseException as exc:
        if not args.partial and (out / "summary.json").exists():
            failure = out / "independent_cpu_audit_failed.json"
            if not failure.exists():
                report.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
                failure.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
