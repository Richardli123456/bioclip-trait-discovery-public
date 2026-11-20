"""CPU-only candidate selection using stable features and train/validation labels."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def centered(values, families):
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    for family in set(families):
        mask = families == family
        result[mask] -= values[mask].mean(0)
    return result


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator < 1e-14 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def family_correlation(a, b, families):
    return correlation(centered(a, families), centered(b, families))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Completed trait evaluation directory with matching manifest and joined row indices")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "summary.json").exists(), "Refusing to overwrite prior candidate confirmation"
    protocol_path = ROOT / "candidate_protocol.json"
    assert sha(protocol_path) == (ROOT / "candidate_protocol.sha256").read_text(encoding="ascii").strip(), "Candidate protocol changed after lock"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stability_summary = json.loads((args.study / "summary.json").read_text(encoding="utf-8"))
    assert stability_summary["status"] == "completed"
    study_protocol = json.loads((args.study / "protocol.json").read_text(encoding="utf-8"))
    probe = json.loads((args.benchmark / "results.json").read_text(encoding="utf-8"))
    manifest = Path(study_protocol["input_dir"]) / "data/species.json"
    assert sha(manifest) == probe["input_sha256"]["manifest"], "Probe and SAE manifests differ"
    with (args.study / "feature_stability.csv").open(encoding="utf-8") as stream:
        stability = list(csv.DictReader(stream))
    stable = sorted(int(r["anchor_feature"]) for r in stability if r["passes_prespecified_gate"].lower() == "true")
    assert stable == sorted(stability_summary["stable_features"])
    result = {"status": "completed", "study": str(args.study.resolve()), "benchmark": str(args.benchmark.resolve()), "protocol": protocol,
              "stable_feature_count": len(stable), "selected_feature": None, "candidate_confirmed": False,
              "test_trait_values_used": False, "gpu_used": False,
              "raw_probe_positive_signal_rule_passed": probe["positive_signal_rule_passed"],
              "hashes": {"candidate_protocol": sha(protocol_path), "analysis_script": sha(Path(__file__)),
                         "feature_stability": sha(args.study / "feature_stability.csv"),
                         "joined_labels": sha(args.benchmark / "joined_labels.csv"), "manifest": sha(manifest)}}
    if not stable:
        result["reason"] = "No feature passes the existing recurrence gate; no trait association was examined"
        save_json(args.output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return
    with (args.benchmark / "joined_labels.csv").open(encoding="utf-8") as stream:
        # Test rows are discarded without reading their trait-value field.
        rows = [r for r in csv.DictReader(stream) if r["split"] in {"train", "validation"}]
    split = np.asarray([r["split"] for r in rows])
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    families = np.asarray([r["family"] for r in rows])
    genera = np.asarray([r["genus"] for r in rows])
    assert not set(genera[train]) & set(genera[validation])
    ytrain = np.asarray([float(rows[i]["trait_value"]) for i in train])
    code_root = args.study / f"seed_{study_protocol['anchor_seed']}"
    indices = np.load(code_root / "code_indices.npy", mmap_mode="r")
    values = np.load(code_root / "code_values.npy", mmap_mode="r")
    assert indices.shape == values.shape == (study_protocol["species_count"], study_protocol["k"])
    selected_rows = np.asarray([int(r["embedding_row"]) for r in rows])
    selected_indices = np.asarray(indices[selected_rows])
    selected_values = np.asarray(values[selected_rows])
    result["hashes"].update({"code_indices": sha(code_root / "code_indices.npy"), "code_values": sha(code_root / "code_values.npy")})
    assert np.isfinite(selected_values).all() and (selected_values >= 0).all()
    assert (selected_indices >= 0).all() and (selected_indices < study_protocol["width"]).all()
    assert np.all(np.diff(np.sort(selected_indices, axis=1), axis=1) != 0), "Duplicate feature index in one sparse code"
    z = np.zeros((len(rows), study_protocol["width"]), dtype=np.float32)
    np.put_along_axis(z, selected_indices, selected_values, axis=1)
    ranked = []
    for feature in stable:
        activation = z[train, feature]
        active = activation > 0
        species_count = int(active.sum())
        genus_count = len(set(genera[train][active]))
        family_count = len(set(families[train][active]))
        r = family_correlation(activation, ytrain, families[train])
        eligible = species_count >= 30 and genus_count >= 10 and family_count >= 3 and r is not None
        ranked.append({"feature": feature, "active_training_species": species_count,
                       "active_training_genera": genus_count, "active_training_families": family_count,
                       "ordinary_training_r": correlation(activation, ytrain), "within_family_training_r": r,
                       "eligible": eligible})
    save_csv(args.output / "training_feature_associations.csv", ranked)
    eligible = [r for r in ranked if r["eligible"]]
    result["eligible_feature_count"] = len(eligible)
    if not eligible:
        result["reason"] = "No stable feature satisfies prespecified labelled-training support and defined-correlation criteria"
        save_json(args.output / "summary.json", result)
        print(json.dumps(result, indent=2))
        return
    selected = min(eligible, key=lambda r: (-abs(r["within_family_training_r"]), r["feature"]))
    feature = selected["feature"]
    # Save a sealed selection record before accessing any validation trait values.
    save_json(args.output / "training_selection.json", selected)
    result["selected_feature"] = feature
    result["training_selection"] = selected
    yval = np.asarray([float(rows[i]["trait_value"]) for i in validation])
    aval = z[validation, feature]
    fval, gval = families[validation], genera[validation]
    val_r = family_correlation(aval, yval, fval)
    unique_genera = sorted(set(gval))
    members = [np.flatnonzero(gval == genus) for genus in unique_genera]
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    correlations = []
    for _ in range(protocol["bootstrap"]["resamples"]):
        selected_genera = rng.integers(len(members), size=len(members))
        sample = np.concatenate([members[j] for j in selected_genera])
        r = family_correlation(aval[sample], yval[sample], fval[sample])
        if r is not None:
            correlations.append(r)
    interval = np.quantile(correlations, [0.025, 0.975]).tolist() if correlations else None
    direction = np.sign(selected["within_family_training_r"])
    confirmed = len(correlations) >= protocol["bootstrap"]["required_defined_resamples"] and val_r is not None and interval is not None and direction * val_r > 0 and (interval[0] > 0 if direction > 0 else interval[1] < 0)
    result.update({"candidate_confirmed": bool(confirmed), "validation_within_family_r": val_r,
                   "validation_ordinary_r": correlation(aval, yval), "validation_species": len(validation),
                   "validation_genera": len(unique_genera), "bootstrap_defined_resamples": len(correlations),
                   "bootstrap_undefined_resamples": protocol["bootstrap"]["resamples"] - len(correlations),
                   "validation_within_family_r_95_percent_interval": interval,
                   "reason": "Prespecified validation association confirmed" if confirmed else "Prespecified validation association did not confirm"})
    top = sorted(train, key=lambda i: (-float(z[i, feature]), rows[i]["scientific_name"]))[:20]
    save_csv(args.output / "selected_feature_training_examples.csv", [{"scientific_name": rows[i]["scientific_name"],
                     "family": rows[i]["family"], "genus": rows[i]["genus"], "activation": float(z[i, feature]),
                     "trait_value": float(rows[i]["trait_value"])} for i in top])
    save_csv(args.output / "selected_feature_validation_values.csv", [{"scientific_name": rows[i]["scientific_name"],
                     "family": rows[i]["family"], "genus": rows[i]["genus"], "activation": float(z[i, feature]),
                     "trait_value": float(rows[i]["trait_value"])} for i in validation])
    result["hashes"].update({"code_indices": sha(code_root / "code_indices.npy"), "code_values": sha(code_root / "code_values.npy")})
    save_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
