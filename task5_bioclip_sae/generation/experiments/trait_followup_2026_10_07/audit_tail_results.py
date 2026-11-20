"""Independent source/code verification; does not compute any test endpoint."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tail_candidate_v1"
EXP = ROOT.parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def measured_rows(path, names):
    # Independent XML parser. Only retrieve morphology for requested train/val names.
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = ["".join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("x:si", ns)]
        rows = ET.fromstring(z.read("xl/worksheets/sheet3.xml")).findall("x:sheetData/x:row", ns)
        def value(cell):
            v = cell.find("x:v", ns)
            if v is None:
                return ""
            return strings[int(v.text)] if cell.get("t") == "s" else v.text
        headers = {re.sub(r"\d", "", c.get("r")): value(c) for c in rows[0]}
        fields = {"Tail.Length", "Wing.Length", "Inference", "Complete.measures"}
        result = {}
        for row in rows[1:]:
            name_cell = next((c for c in row if re.sub(r"\d", "", c.get("r")) == "A"), None)
            if name_cell is None:
                continue
            name = value(name_cell)
            if name not in names:
                continue
            result[name] = {headers[re.sub(r"\d", "", c.get("r"))]: value(c) for c in row
                            if headers.get(re.sub(r"\d", "", c.get("r"))) in fields}
    assert set(result) == names
    return result


def center_by_group(matrix, keys):
    out = np.array(matrix, dtype=np.float64, copy=True)
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for ids in groups.values():
        out[ids] -= np.mean(out[ids], axis=0)
    return out


def pearson(a, b):
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    den = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return None if den < 1e-14 else float(np.sum(a*b) / den)


def counts(rows, active):
    positive = [r for r, yes in zip(rows, active) if yes]
    return [len(positive), len({r["genus"] for r in positive}), len({r["family_key"] for r in positive})]


def independent_taxonomy(rows, target, fp, op):
    global_mean = float(np.mean(target))
    order = {}
    family = {}
    for key in {r["order"] for r in rows}:
        idx = np.asarray([i for i, r in enumerate(rows) if r["order"] == key])
        order[key] = float((target[idx].sum() + op * global_mean) / (len(idx) + op))
    for key in {r["family_key"] for r in rows}:
        idx = np.asarray([i for i, r in enumerate(rows) if r["family_key"] == key])
        family[key] = float((target[idx].sum() + fp * order[rows[idx[0]]["order"]]) / (len(idx) + fp))
    return {"global_mean": global_mean, "orders": order, "families": family}


def infer(model, rows):
    return np.asarray([model["families"][r["family_key"]] if r["family_key"] in model["families"]
                       else model["orders"].get(r["order"], model["global_mean"]) for r in rows])


def compare(a, b, tolerance=1e-11):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error < tolerance, (error, tolerance)
    return error


def main():
    started = time.monotonic()
    lock = load(ROOT / "execution_lock.json")
    hashes = {name: digest(name) for name in lock["input_sha256"]}
    assert hashes == lock["input_sha256"]
    protocol, summary = load(ROOT / "protocol.json"), load(OUT / "summary.json")
    chosen = load(OUT / "training_selection.json")
    saved_models = load(OUT / "training_predictors.json")
    study_dir = EXP / "expanded_stability_w2048_k32_v2"
    study, study_protocol = load(study_dir / "summary.json"), load(study_dir / "protocol.json")
    stable = study["stable_features"]
    assert len(stable) == len(set(stable)) == 732
    manifest = load(Path(study_protocol["input_dir"]) / "data/species.json")
    cohort = table(OUT / "cohort_metadata.csv")
    old_join = table(EXP / "trait_benchmark_v1/evaluation_expanded_v2/joined_labels.csv")
    assert len(cohort) == 8187 and len({r["scientific_name"] for r in cohort}) == 8187
    assert [(r["scientific_name"], r["embedding_row"], r["split"]) for r in cohort] == [(r["scientific_name"], r["embedding_row"], r["split"]) for r in old_join]
    for row in cohort:
        source = manifest[int(row["embedding_row"])]
        assert all(row[k] == source[k] for k in ["scientific_name", "genus", "split", "family", "order"])
        assert source["kingdom"] == "Animalia" and source["class_name"] == "Aves"
        assert row["family_key"] == row["order"] + "|" + row["family"]
    split_genera = {s: {r["genus"] for r in cohort if r["split"] == s} for s in ["train", "validation", "test"]}
    assert not split_genera["train"] & split_genera["validation"]
    assert not split_genera["train"] & split_genera["test"]
    assert not split_genera["validation"] & split_genera["test"]
    rows = {s: [r for r in cohort if r["split"] == s] for s in ["train", "validation"]}
    names = {r["scientific_name"] for values in rows.values() for r in values}
    raw = measured_rows(EXP / "trait_benchmark_v1/AVONET_Supplementary_dataset_1.xlsx", names)
    labels = {}
    activations = {}
    source_idx = np.load(study_dir / "seed_20260929/code_indices.npy", mmap_mode="r")
    source_val = np.load(study_dir / "seed_20260929/code_values.npy", mmap_mode="r")
    errors = {}
    for split in rows:
        labels[split] = []
        for row in rows[split]:
            src = raw[row["scientific_name"]]
            assert src["Inference"] == "NO" and float(src["Complete.measures"]) >= 4
            tail, wing = float(src["Tail.Length"]), float(src["Wing.Length"])
            assert tail > 0 and wing > 0 and math.isfinite(tail) and math.isfinite(wing)
            labels[split].append(math.log(tail) - math.log(wing))
        labels[split] = np.asarray(labels[split])
        selected = np.asarray([int(row["embedding_row"]) for row in rows[split]])
        indices, values = source_idx[selected], source_val[selected]
        assert np.all(np.diff(np.sort(indices, axis=1), axis=1) != 0)
        assert np.isfinite(values).all() and (values >= 0).all()
        activations[split] = np.column_stack([np.sum(np.where(indices == feature, values, 0), axis=1) for feature in stable]).astype(float)
    tr = rows["train"]
    fam = [r["family_key"] for r in tr]
    centered = center_by_group(activations["train"], fam)
    centered_y = center_by_group(labels["train"], fam)
    ranking = table(OUT / "training_feature_associations.csv")
    assert [int(r["feature"]) for r in ranking] == stable
    eligible = []
    max_corr_error = 0.
    for j, feature in enumerate(stable):
        a = activations["train"][:, j]
        observed = counts(tr, a > protocol["selection"]["active_threshold"])
        saved = ranking[j]
        assert observed == [int(saved["active_training_" + k]) for k in ["species", "genera", "families"]]
        r = pearson(centered[:, j], centered_y)
        ordinary = pearson(a, labels["train"])
        ok = (r is not None and all(observed[i] >= protocol["selection"]["minimum_active_training_" + key]
                                    for i, key in enumerate(["species", "genera", "families"])))
        assert str(ok) == saved["eligible"]
        assert (r is None) == (saved["within_family_r"] == "")
        assert (ordinary is None) == (saved["ordinary_r"] == "")
        if r is not None:
            max_corr_error = max(max_corr_error, compare(r, float(saved["within_family_r"])))
        if ordinary is not None:
            compare(ordinary, float(saved["ordinary_r"]))
        if ok:
            eligible.append((feature, r))
    assert len(eligible) == summary["eligible_training_features"] == 188
    best = min(eligible, key=lambda item: (-abs(item[1]), item[0]))
    assert best[0] == chosen["feature"] == summary["selected_feature"] == 814
    col = stable.index(best[0])
    a = activations["train"][:, col]
    # Verify the saved fixed-effect slope by an independent least-squares solver.
    beta = float(np.linalg.lstsq(centered[:, col, None], centered_y, rcond=None)[0][0])
    errors["slope_absolute_error"] = compare(beta, saved_models["beta"])
    fp, op = protocol["taxonomy_control"]["family_pseudocount"], protocol["taxonomy_control"]["order_pseudocount"]
    models = {"endpoint": independent_taxonomy(tr, labels["train"], fp, op),
              "activation": independent_taxonomy(tr, a, fp, op)}
    for kind, model in models.items():
        errors[kind + "_global_mean_error"] = compare(model["global_mean"], saved_models["models"][kind]["global_mean"])
        for group in ["orders", "families"]:
            assert set(model[group]) == set(saved_models["models"][kind][group])
            errors[kind + "_" + group + "_max_error"] = max(compare(value, saved_models["models"][kind][group][key]) for key, value in model[group].items())
    for split in ["train", "validation"]:
        saved = table(OUT / ("training_values.csv" if split == "train" else "validation_values.csv"))
        assert [r["scientific_name"] for r in saved] == [r["scientific_name"] for r in rows[split]]
        errors[split + "_label_max_error"] = compare(labels[split], [float(r["trait_value"]) for r in saved])
        errors[split + "_activation_max_error"] = compare(activations[split][:, col], [float(r["activation"]) for r in saved])
        for r in saved:
            assert float(r["tail_length_mm"]) == float(raw[r["scientific_name"]]["Tail.Length"])
            assert float(r["wing_length_mm"]) == float(raw[r["scientific_name"]]["Wing.Length"])
    val = rows["validation"]
    y, a = labels["validation"], activations["validation"][:, col]
    f = [r["family_key"] for r in val]
    rpoint = pearson(center_by_group(a, f), center_by_group(y, f))
    saved_result = load(OUT / "validation_result.json")
    active_support = counts(val, a > protocol["selection"]["active_threshold"])
    assert active_support == [saved_result["active_support"][k] for k in ["species", "genera", "families"]]
    b = infer(models["endpoint"], val)
    aug = b + beta * (a - infer(models["activation"], val))
    saved_vals = table(OUT / "validation_values.csv")
    errors["baseline_prediction_max_error"] = compare(b, [float(r["baseline_prediction"]) for r in saved_vals])
    errors["augmented_prediction_max_error"] = compare(aug, [float(r["augmented_prediction"]) for r in saved_vals])
    esq, asq = (y-b)**2, (y-aug)**2
    for tag, pred in [("taxonomy_baseline", b), ("taxonomy_plus_activation", aug)]:
        values = {"rmse": np.sqrt(np.mean((y-pred)**2)), "mae": np.mean(np.abs(y-pred)),
                  "r2": 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)}
        for key, value in values.items():
            compare(value, saved_result[tag][key])
    compare(rpoint, saved_result["family_centered_r"])
    compare(pearson(a, y), saved_result["ordinary_r"])
    # Independent weighted-cluster formulation, not concatenated bootstrap rows.
    _, gi = np.unique([r["genus"] for r in val], return_inverse=True)
    _, fi = np.unique(f, return_inverse=True)
    ng = int(gi.max()) + 1
    rng = np.random.default_rng(protocol["bootstrap"]["validation_seed"])
    correlations, improvements = [], []
    for _ in range(protocol["bootstrap"]["resamples"]):
        multiplicity = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        w = multiplicity[gi].astype(float)
        n = np.bincount(fi, weights=w)
        am = np.divide(np.bincount(fi, weights=w*a), n, out=np.zeros_like(n), where=n>0)
        ym = np.divide(np.bincount(fi, weights=w*y), n, out=np.zeros_like(n), where=n>0)
        ar, yr = a-am[fi], y-ym[fi]
        den = np.sqrt(np.sum(w*ar*ar) * np.sum(w*yr*yr))
        if den >= 1e-14:
            correlations.append(float(np.sum(w*ar*yr) / den))
        improvements.append(float(np.sqrt(np.dot(w, esq)/w.sum()) - np.sqrt(np.dot(w, asq)/w.sum())))
    saved_boot = np.load(OUT / "validation_bootstrap.npz")
    errors["bootstrap_correlation_max_error"] = compare(correlations, saved_boot["correlations"])
    errors["bootstrap_improvement_max_error"] = compare(improvements, saved_boot["rmse_improvements"])
    rci = np.quantile(correlations, [0.025, .975])
    dci = np.quantile(improvements, [0.025, .975])
    compare(rci, saved_result["correlation_95ci"])
    compare(dci, saved_result["improvement_95ci"])
    improvement = float(np.sqrt(esq.mean()) - np.sqrt(asq.mean()))
    compare(improvement, saved_result["rmse_improvement"])
    support_pass = all(active_support[i] >= protocol["confirmation"]["minimum_active_" + key] for i, key in enumerate(["species", "genera", "families"]))
    association_pass = len(correlations) >= 1900 and np.sign(best[1])*rpoint > 0 and (rci[0] > 0 if best[1] > 0 else rci[1] < 0)
    prediction_pass = improvement > 0 and dci[0] > 0
    assert support_pass and not association_pass and not prediction_pass
    assert saved_result["support_passed"] and not saved_result["association_passed"] and not saved_result["prediction_passed"]
    assert not summary["candidate_confirmed"] and not summary["image_branch_eligible"] and not summary["test_endpoint_used"]
    assert not (OUT / "test_values.csv").exists() and not (OUT / "test_result.json").exists() and not (OUT / "test_bootstrap.npz").exists()
    assert summary["execution_lock_sha256"] == digest(ROOT / "execution_lock.json")
    assert summary["selection_sha256"] == digest(OUT / "training_selection.json")
    artifacts = ["summary.json", "training_selection.json", "training_predictors.json", "cohort_metadata.csv", "training_feature_associations.csv", "training_values.csv", "validation_values.csv", "validation_result.json", "validation_bootstrap.npz"]
    verification = {
        "status": "passed", "scope": "Independent source XML parsing, sparse code reconstruction, all stable-feature training ranking, fixed training predictor verification, and weighted-genus validation bootstrap; no runner helpers imported.",
        "cohort_counts": dict(Counter(r["split"] for r in cohort)), "stable_features_checked": len(stable),
        "eligible_training_features": len(eligible), "selected_feature": best[0], "training_family_centered_r": best[1],
        "training_active_support": counts(tr, activations["train"][:, col] > 1e-7),
        "validation_active_support": active_support, "validation_family_centered_r": rpoint,
        "validation_correlation_95ci": rci.tolist(), "validation_rmse_improvement": improvement,
        "validation_improvement_95ci": dci.tolist(), "bootstrap_defined_correlations": len(correlations),
        "both_confirmation_gates_failed": True, "test_endpoint_computed": False,
        "training_correlation_max_error": max_corr_error, "numerical_errors": errors,
        "locked_inputs_hashes_verified": hashes, "result_artifact_sha256": {name: digest(OUT/name) for name in artifacts},
        "audit_script_sha256": digest(__file__), "cpu_only": True, "elapsed_seconds": time.monotonic()-started,
    }
    (OUT / "independent_verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: verification[key] for key in ["status", "selected_feature", "eligible_training_features", "validation_family_centered_r", "validation_correlation_95ci", "validation_rmse_improvement", "both_confirmation_gates_failed", "test_endpoint_computed", "numerical_errors", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
