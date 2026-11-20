"""Locked, CPU-only follow-up of one stable SAE coordinate and tail/wing labels."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
GEN = EXP.parent
STUDY = EXP / "expanded_stability_w2048_k32_v2"
SOURCE = EXP / "trait_benchmark_v1"
OUT = ROOT / "tail_candidate_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_ids(keys):
    return np.unique(np.asarray(keys), return_inverse=True)[1]


def demean(x, groups):
    x = np.asarray(x, dtype=np.float64)
    n = np.bincount(groups)
    sums = np.bincount(groups, weights=x, minlength=len(n))
    return x - (sums / n)[groups]


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return None if d < 1e-14 else float(np.clip(np.dot(x, y) / d, -1, 1))


def family_corr(a, y, families):
    groups = group_ids(families)
    return corr(demean(a, groups), demean(y, groups))


def taxonomy_fit(rows, y, fp, op):
    global_mean = float(np.mean(y))
    orders, families = {}, {}
    for key in sorted({r["order"] for r in rows}):
        values = [float(v) for r, v in zip(rows, y) if r["order"] == key]
        orders[key] = (sum(values) + op * global_mean) / (len(values) + op)
    for key in sorted({r["family_key"] for r in rows}):
        selected = [(r, float(v)) for r, v in zip(rows, y) if r["family_key"] == key]
        families[key] = (sum(v for _, v in selected) + fp * orders[selected[0][0]["order"]]) / (len(selected) + fp)
    return {"global_mean": global_mean, "orders": orders, "families": families}


def predict(model, rows):
    return np.asarray([model["families"].get(r["family_key"], model["orders"].get(r["order"], model["global_mean"])) for r in rows])


def support(rows, a, threshold):
    active = [r for r, v in zip(rows, a) if v > threshold]
    return {"species": len(active), "genera": len({r["genus"] for r in active}),
            "families": len({r["family_key"] for r in active})}


def metric(y, pred):
    err = y - pred
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def endpoint(rows, raw):
    result = []
    for row in rows:
        source = raw[row["scientific_name"]]
        tail, wing = float(source["Tail.Length"]), float(source["Wing.Length"])
        assert math.isfinite(tail) and math.isfinite(wing) and tail > 0 and wing > 0
        result.append(math.log(tail / wing))
    return np.asarray(result)


def output_values(path, rows, y, a, raw, baseline=None, augmented=None):
    values = []
    for i, (r, v, z) in enumerate(zip(rows, y, a)):
        source = raw[r["scientific_name"]]
        item = {**r, "tail_length_mm": float(source["Tail.Length"]),
                "wing_length_mm": float(source["Wing.Length"]), "trait_value": float(v), "activation": float(z)}
        if baseline is not None:
            item.update(baseline_prediction=float(baseline[i]), augmented_prediction=float(augmented[i]))
        values.append(item)
    csv_write(path, values)


def confirm(split, rows, a, raw, models, beta, direction, protocol):
    c = protocol["confirmation"]
    result = {"split": split, "species": len(rows), "genera": len({r["genus"] for r in rows}),
              "active_support": support(rows, a, protocol["selection"]["active_threshold"]),
              "endpoint_values_used": False, "passed": False}
    result["support_passed"] = all(result["active_support"][key] >= c["minimum_active_" + key] for key in ["species", "genera", "families"])
    if not result["support_passed"]:
        result["reason"] = "insufficient_active_support; no endpoint values computed for this split"
        write(OUT / f"{split}_result.json", result)
        return result
    y = endpoint(rows, raw)
    result["endpoint_values_used"] = True
    f = np.asarray([r["family_key"] for r in rows])
    g = np.asarray([r["genus"] for r in rows])
    baseline = predict(models["endpoint"], rows)
    augmented = baseline + beta * (a - predict(models["activation"], rows))
    point = family_corr(a, y, f)
    difference = float(np.sqrt(np.mean((y - baseline) ** 2)) - np.sqrt(np.mean((y - augmented) ** 2)))
    members = [np.flatnonzero(g == genus) for genus in sorted(set(g))]
    rng = np.random.default_rng(protocol["bootstrap"][split + "_seed"])
    correlations, improvements = [], []
    for _ in range(protocol["bootstrap"]["resamples"]):
        sampled = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        correlation = family_corr(a[sampled], y[sampled], f[sampled])
        if correlation is not None:
            correlations.append(correlation)
        improvements.append(float(np.sqrt(np.mean((y[sampled] - baseline[sampled]) ** 2)) - np.sqrt(np.mean((y[sampled] - augmented[sampled]) ** 2))))
    ci_r = np.quantile(correlations, protocol["bootstrap"]["interval"]).tolist() if correlations else None
    ci_d = np.quantile(improvements, protocol["bootstrap"]["interval"]).tolist()
    association_passed = (len(correlations) >= protocol["bootstrap"]["required_defined_correlations"]
                          and point is not None and direction * point > 0 and ci_r is not None
                          and (ci_r[0] > 0 if direction > 0 else ci_r[1] < 0))
    prediction_passed = difference > 0 and ci_d[0] > 0
    result.update(family_centered_r=point, ordinary_r=corr(a, y), correlation_95ci=ci_r,
                  defined_correlation_resamples=len(correlations), undefined_correlation_resamples=protocol["bootstrap"]["resamples"] - len(correlations),
                  taxonomy_baseline=metric(y, baseline), taxonomy_plus_activation=metric(y, augmented),
                  rmse_improvement=difference, improvement_95ci=ci_d,
                  association_passed=bool(association_passed), prediction_passed=bool(prediction_passed),
                  passed=bool(association_passed and prediction_passed))
    result["reason"] = "all_prespecified_gates_passed" if result["passed"] else "association_or_predictive_improvement_did_not_confirm"
    output_values(OUT / f"{split}_values.csv", rows, y, a, raw, baseline, augmented)
    np.savez_compressed(OUT / f"{split}_bootstrap.npz", correlations=correlations, rmse_improvements=improvements)
    write(OUT / f"{split}_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.plan_only != args.execute, "Select exactly one execution mode"
    study = read(STUDY / "summary.json")
    study_plan = read(STUDY / "protocol.json")
    manifest = Path(study_plan["input_dir"]) / "data/species.json"
    checkpoint = STUDY / f"seed_{study_plan['anchor_seed']}"
    workbook = SOURCE / "AVONET_Supplementary_dataset_1.xlsx"
    cohort = SOURCE / "evaluation_expanded_v2/joined_labels.csv"
    inputs = [Path(__file__), ROOT / "protocol.json", STUDY / "summary.json", STUDY / "protocol.json",
              STUDY / "feature_stability.csv", STUDY / "independent_cpu_audit.json", manifest,
              workbook, cohort, SOURCE / "prepare_labels.py", checkpoint / "code_indices.npy", checkpoint / "code_values.npy"]
    assert hashlib.md5(workbook.read_bytes()).hexdigest() == "1445afdcfb6df784010c2ca034544bc8"
    hashes = {str(p.resolve()): sha(p) for p in inputs}
    assert hashes[str(manifest.resolve())] == study_plan["input_sha256"][str(manifest)]
    anchor = next(r for r in study["per_seed"] if r["seed"] == study_plan["anchor_seed"])
    assert sha(checkpoint / "code_indices.npy") == anchor["indices_sha256"]
    assert sha(checkpoint / "code_values.npy") == anchor["values_sha256"]
    assert study["stable_feature_count"] == 732 and read(STUDY / "independent_cpu_audit.json")["status"] == "passed"
    if args.plan_only:
        assert not (ROOT / "execution_lock.json").exists(), "Refusing to replace existing lock"
        write(ROOT / "execution_lock.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes,
              "scope": "Source identities, script and protocol locked before any new tail endpoint association"})
        print("Execution lock saved; no associations or model fitting performed.", flush=True)
        return
    assert read(ROOT / "execution_lock.json")["input_sha256"] == hashes, "Input/script/protocol changed after lock"
    assert not OUT.exists(), "Refusing to overwrite a prior follow-up execution"
    OUT.mkdir()
    started = time.monotonic()
    protocol = read(ROOT / "protocol.json")
    spec = importlib.util.spec_from_file_location("tail_avonet_reader", SOURCE / "prepare_labels.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    raw_rows = reader.read_sheet(workbook, 3)
    raw = {}
    for _, row in raw_rows:
        name = row["Species2"]
        assert name not in raw, "Duplicate AVONET binomial"
        raw[name] = row
    species = read(manifest)
    with cohort.open(encoding="utf-8") as f:
        rows = [{k: r[k] for k in ["scientific_name", "genus", "split", "family", "order", "embedding_row"]} for r in csv.DictReader(f)]
    assert len(rows) == 8187 and len({r["scientific_name"] for r in rows}) == len(rows)
    for row in rows:
        row["embedding_row"] = int(row["embedding_row"])
        original = species[row["embedding_row"]]
        assert all(row[k] == original[k] for k in ["scientific_name", "genus", "split", "family", "order"])
        assert original["kingdom"] == "Animalia" and original["class_name"] == "Aves"
        source = raw[row["scientific_name"]]
        assert source["Inference"] == "NO" and float(source["Complete.measures"]) >= 4
        row["family_key"] = row["order"] + "|" + row["family"]
    groups = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ["train", "validation", "test"]}
    genera = {s: {rows[i]["genus"] for i in ids} for s, ids in groups.items()}
    assert all(not genera[a] & genera[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
    csv_write(OUT / "cohort_metadata.csv", rows)
    selected_rows = np.asarray([r["embedding_row"] for r in rows])
    indices = np.load(checkpoint / "code_indices.npy", mmap_mode="r")[selected_rows]
    values = np.load(checkpoint / "code_values.npy", mmap_mode="r")[selected_rows]
    assert indices.shape == values.shape == (len(rows), 32)
    assert np.isfinite(values).all() and (values >= 0).all()
    z = np.zeros((len(rows), study["width"]), dtype=np.float64)
    np.put_along_axis(z, indices, values, axis=1)
    train = np.asarray(groups["train"])
    train_rows = [rows[i] for i in train]
    y = endpoint(train_rows, raw)
    families = np.asarray([r["family_key"] for r in train_rows])
    ranked = []
    for feature in sorted(study["stable_features"]):
        activation = z[train, feature]
        active = support(train_rows, activation, protocol["selection"]["active_threshold"])
        r = family_corr(activation, y, families)
        eligible = all(active[key] >= protocol["selection"]["minimum_active_training_" + key] for key in ["species", "genera", "families"]) and r is not None
        ranked.append({"feature": feature, **{"active_training_" + k: v for k, v in active.items()},
                       "within_family_r": r, "ordinary_r": corr(activation, y), "eligible": bool(eligible)})
    csv_write(OUT / "training_feature_associations.csv", ranked)
    eligible = [r for r in ranked if r["eligible"]]
    result = {"status": "completed", "endpoint": protocol["endpoint"], "gpu_used": False,
              "SAE_retrained": False, "stable_feature_count": 732, "eligible_training_features": len(eligible),
              "cohort_counts": {s: len(ids) for s, ids in groups.items()}, "selected_feature": None,
              "candidate_confirmed": False, "test_endpoint_used": False, "image_branch_eligible": False,
              "execution_lock_sha256": sha(ROOT / "execution_lock.json")}
    if eligible:
        chosen = min(eligible, key=lambda r: (-abs(r["within_family_r"]), r["feature"]))
        write(OUT / "training_selection.json", chosen)
        feature = chosen["feature"]
        result.update(selected_feature=feature, training_selection=chosen,
                      selection_sha256=sha(OUT / "training_selection.json"))
        a = z[train, feature]
        ids = group_ids(families)
        ac, yc = demean(a, ids), demean(y, ids)
        beta = float(np.dot(ac, yc) / np.dot(ac, ac))
        fp, op = (protocol["taxonomy_control"][k] for k in ["family_pseudocount", "order_pseudocount"])
        models = {"endpoint": taxonomy_fit(train_rows, y, fp, op), "activation": taxonomy_fit(train_rows, a, fp, op)}
        write(OUT / "training_predictors.json", {"beta": beta, "models": models, "fit_partition": "train", "family_pseudocount": fp, "order_pseudocount": op})
        output_values(OUT / "training_values.csv", train_rows, y, a, raw)
        exemplar_order = sorted(range(len(train_rows)), key=lambda i: (-a[i], train_rows[i]["scientific_name"]))[:20]
        output_values(OUT / "training_examples.csv", [train_rows[i] for i in exemplar_order], y[exemplar_order], a[exemplar_order], raw)
        validation = np.asarray(groups["validation"])
        print(f"Sealed training candidate: feature {feature}; evaluating only that candidate.", flush=True)
        result["validation"] = confirm("validation", [rows[i] for i in validation], z[validation, feature], raw, models, beta, np.sign(chosen["within_family_r"]), protocol)
        if result["validation"]["passed"]:
            test = np.asarray(groups["test"])
            result["test"] = confirm("test", [rows[i] for i in test], z[test, feature], raw, models, beta, np.sign(chosen["within_family_r"]), protocol)
            result["test_endpoint_used"] = result["test"]["endpoint_values_used"]
            result["candidate_confirmed"] = result["image_branch_eligible"] = result["test"]["passed"]
        result["reason"] = "candidate_confirmed; image_measurement_feasibility_required" if result["candidate_confirmed"] else "candidate_stopped_at_prespecified_confirmation_gate; no replacement"
    else:
        result["reason"] = "no_training_candidate_with_required_support"
    result["elapsed_seconds"] = time.monotonic() - started
    result["all_input_hashes_unchanged"] = all(sha(Path(p)) == digest for p, digest in hashes.items())
    assert result["all_input_hashes_unchanged"]
    result["limitations"] = ["Sequential new-endpoint follow-up after a negative beak experiment", "Reused cohort and SAE validation embeddings, not a pristine project-level holdout", "Species-mean morphology is not a guaranteed individual generated-image phenotype", "Association does not establish image control or biological causation"]
    write(OUT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (MemoryError, OSError) as exc:
        if OUT.exists():
            write(OUT / "STOPPED.json", {"exception": type(exc).__name__, "message": str(exc), "automatic_retry": False})
        raise
