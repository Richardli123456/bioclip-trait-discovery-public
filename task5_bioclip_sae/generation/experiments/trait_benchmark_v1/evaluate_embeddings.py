"""CPU-only prespecified trait probe with genus holdout and taxonomy controls."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ridge_path(x, y, alphas):
    """One covariance eigensystem gives exact solutions for the fixed grid."""
    mean = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-12] = 1.0
    z = (x - mean) / sd
    ymean = float(y.mean())
    gram = z.T @ z / len(y)
    eigvals, eigvecs = np.linalg.eigh(gram)
    projection = eigvecs.T @ (z.T @ (y - ymean) / len(y))
    models = []
    for alpha in alphas:
        coef = eigvecs @ (projection / (np.maximum(eigvals, 0) + alpha))
        raw_coef = coef / sd
        intercept = ymean - float(mean @ raw_coef)
        models.append({"alpha": alpha, "coef": raw_coef, "intercept": intercept})
    return models


def predict_ridge(model, x):
    return x @ model["coef"] + model["intercept"]


def taxonomy_fit(rows, y, family_pseudo, order_pseudo):
    family_vals, order_vals = defaultdict(list), defaultdict(list)
    for row, value in zip(rows, y):
        family_vals[(row["order"], row["family"])].append(float(value))
        order_vals[row["order"]].append(float(value))
    global_mean = float(y.mean())
    orders = {key: (sum(values) + order_pseudo * global_mean) / (len(values) + order_pseudo)
              for key, values in order_vals.items()}
    families = {key: (sum(values) + family_pseudo * orders[key[0]]) / (len(values) + family_pseudo)
                for key, values in family_vals.items()}
    return {"global": global_mean, "orders": orders, "families": families,
            "family_pseudocount": family_pseudo, "order_pseudocount": order_pseudo}


def taxonomy_predict(model, rows):
    return np.asarray([model["families"].get((r["order"], r["family"]),
                       model["orders"].get(r["order"], model["global"])) for r in rows])


def metrics(y, pred, genera):
    err = y - pred
    groups = sorted(set(genera))
    genus_mse = [np.mean(err[genera == genus] ** 2) for genus in groups]
    return {"n": len(y), "genera": len(groups), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "mse": float(np.mean(err ** 2)),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "genus_macro_mse": float(np.mean(genus_mse))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_expanded")
    parser.add_argument("--prepare-only", action="store_true", help="Check and join labels without fitting")
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise RuntimeError("Refusing to overwrite an evaluated test result; use a new explicitly named output")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    with (ROOT / "labels.csv").open(encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    names = [r["scientific_name"] for r in rows]
    index = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("kingdom") == "Animalia" and row.get("class_name") == "Aves":
            index[row["scientific_name"]].append(i)
    all_names = set(names)
    raw = np.load(args.embeddings, mmap_mode="r")
    assert raw.shape == (len(rows), 768), "Manifest and embedding shape mismatch"
    joined, unmatched, lineage_mismatches = [], [], []
    for label in labels:
        name = label["scientific_name"]
        if not index[name]:
            reason = "no_exact_binomial_match" if name not in all_names else "manifest_lineage_not_Animalia_Aves"
            unmatched.append({**label, "join_exclusion_reason": reason})
            if reason == "manifest_lineage_not_Animalia_Aves":
                lineage_mismatches.append({"scientific_name": name, "manifest_taxonomy": " | ".join(r["taxonomy"] for r in rows if r["scientific_name"] == name),
                                           "avonet_family": label["avonet_family"], "avonet_order": label["avonet_order"]})
            continue
        if len(index[name]) > 1:
            unmatched.append({**label, "join_exclusion_reason": "ambiguous_exact_name_within_Aves"})
            continue
        i = index[name][0]
        row = rows[i]
        assert row["genus"] == label["genus"], "Joined binomial genus disagreement"
        if row.get("class_name") != "Aves" or row.get("kingdom") != "Animalia":
            unmatched.append({**label, "join_exclusion_reason": "manifest_lineage_not_Animalia_Aves"})
            lineage_mismatches.append({"scientific_name": name, "manifest_taxonomy": row["taxonomy"],
                                       "avonet_family": label["avonet_family"], "avonet_order": label["avonet_order"]})
            continue
        joined.append({**label, "embedding_row": i, "split": row["split"],
                       "family": row["family"], "order": row["order"]})
    write_csv(args.output / "joined_labels.csv", joined)
    write_csv(args.output / "unmatched_labels.csv", unmatched)
    write_csv(args.output / "excluded_lineage_mismatches.csv", lineage_mismatches)
    write_csv(args.output / "higher_taxonomy_disagreements.csv", [r for r in joined if r["family"] != r["avonet_family"] or r["order"] != r["avonet_order"]])
    assert len(joined) >= 100, "Insufficient matched species for this prespecified benchmark"
    split = np.asarray([r["split"] for r in joined])
    genera = np.asarray([r["genus"] for r in joined])
    train, val, test = (np.flatnonzero(split == key) for key in ["train", "validation", "test"])
    for indices in [train, val, test]:
        assert len(indices) >= 20 and len(set(genera[indices])) >= 5, "Insufficient split coverage"
    assert set(train) | set(val) | set(test) == set(range(len(joined)))
    assert all(not (set(genera[a]) & set(genera[b])) for a, b in [(train, val), (train, test), (val, test)])
    audit = {
        "eligible_avonet_labels": len(labels), "matched_species": len(joined), "unmatched_species": len(unmatched),
        "exact_binomial_join": True, "no_cross_split_genera": True,
        "manifest_duplicate_name_count_all_lineages": sum(count > 1 for count in Counter(names).values()),
        "ambiguous_label_names_within_Aves": sum(len(index[r["scientific_name"]]) > 1 for r in labels),
        "excluded_non_bird_lineages": len(lineage_mismatches),
        "split_species": dict(Counter(split)),
        "split_genera": {key: len(set(genera[split == key])) for key in ["train", "validation", "test"]},
        "family_disagreements_between_sources": sum(r["family"] != r["avonet_family"] for r in joined),
        "order_disagreements_between_sources": sum(r["order"] != r["avonet_order"] for r in joined),
        "baseline_taxonomy": "Embedding manifest taxonomy, matching the information supplied to BioCLIP",
    }
    write_json(args.output / "join_audit.json", audit)
    if args.prepare_only:
        print(json.dumps(audit, indent=2))
        return
    x = np.asarray(raw[[int(r["embedding_row"]) for r in joined]], dtype=np.float64)
    y = np.asarray([float(r["trait_value"]) for r in joined])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    validations = []
    ridge_models = ridge_path(x[train], y[train], protocol["ridge"]["alpha_grid"])
    for model in ridge_models:
        model["validation_mse"] = float(np.mean((predict_ridge(model, x[val]) - y[val]) ** 2))
        validations.append({"model": "embedding_ridge", "alpha": model["alpha"],
                            "family_pseudocount": "", "order_pseudocount": "", "validation_mse": model["validation_mse"]})
    chosen_ridge = min(ridge_models, key=lambda m: (m["validation_mse"], -m["alpha"]))
    taxonomy_models = []
    train_rows, val_rows = [joined[i] for i in train], [joined[i] for i in val]
    for family in protocol["taxonomy_baseline"]["family_pseudocount_grid"]:
        for order in protocol["taxonomy_baseline"]["order_pseudocount_grid"]:
            model = taxonomy_fit(train_rows, y[train], family, order)
            model["validation_mse"] = float(np.mean((taxonomy_predict(model, val_rows) - y[val]) ** 2))
            taxonomy_models.append(model)
            validations.append({"model": "taxonomy_hierarchy", "alpha": "", "family_pseudocount": family,
                                "order_pseudocount": order, "validation_mse": model["validation_mse"]})
    chosen_tax = min(taxonomy_models, key=lambda m: (m["validation_mse"], -m["family_pseudocount"] - m["order_pseudocount"]))
    write_csv(args.output / "validation_grid.csv", validations)
    # Test labels have not been used in preprocessing, model selection or fitting.
    fit = np.concatenate([train, val])
    final_ridge = ridge_path(x[fit], y[fit], [chosen_ridge["alpha"]])[0]
    final_tax = taxonomy_fit([joined[i] for i in fit], y[fit], chosen_tax["family_pseudocount"], chosen_tax["order_pseudocount"])
    test_rows = [joined[i] for i in test]
    pred_ridge = predict_ridge(final_ridge, x[test])
    pred_tax = taxonomy_predict(final_tax, test_rows)
    pred_mean = np.full(len(test), y[fit].mean())
    result = {"join": audit, "protocol": protocol, "hardware": "CPU only", "sae_training": False,
              "fitted_models": ["linear ridge probe", "hierarchical taxonomy means"],
              "selected_ridge_alpha": chosen_ridge["alpha"],
              "selected_family_pseudocount": chosen_tax["family_pseudocount"],
              "selected_order_pseudocount": chosen_tax["order_pseudocount"],
              "test": {"embedding_ridge": metrics(y[test], pred_ridge, genera[test]),
                       "taxonomy_baseline": metrics(y[test], pred_tax, genera[test]),
                       "training_mean": metrics(y[test], pred_mean, genera[test])}}
    unique_genera = sorted(set(genera[test]))
    differences = (y[test] - pred_tax) ** 2 - (y[test] - pred_ridge) ** 2
    ridge_sums = np.asarray([((y[test] - pred_ridge)[genera[test] == genus] ** 2).sum() for genus in unique_genera])
    taxonomy_sums = np.asarray([((y[test] - pred_tax)[genera[test] == genus] ** 2).sum() for genus in unique_genera])
    counts = np.asarray([(genera[test] == genus).sum() for genus in unique_genera])
    rng = np.random.default_rng(protocol["uncertainty"]["seed"])
    resamples = rng.integers(len(unique_genera), size=(protocol["uncertainty"]["resamples"], len(unique_genera)))
    improvements = np.sqrt(taxonomy_sums[resamples].sum(1) / counts[resamples].sum(1)) - np.sqrt(ridge_sums[resamples].sum(1) / counts[resamples].sum(1))
    interval = np.quantile(improvements, [0.025, 0.975]).tolist()
    result["embedding_rmse_improvement_over_taxonomy"] = {"estimate": result["test"]["taxonomy_baseline"]["rmse"] - result["test"]["embedding_ridge"]["rmse"], "genus_bootstrap_95_percent_interval": interval}
    result["embedding_mse_improvement_over_taxonomy"] = float(differences.mean())
    result["positive_signal_rule_passed"] = bool(result["test"]["embedding_ridge"]["r2"] > 0 and interval[0] > 0)
    result["validation_seconds"] = time.time() - started
    result["input_sha256"] = {"manifest": sha(args.manifest), "embeddings": sha(args.embeddings),
                               "labels": sha(ROOT / "labels.csv"), "protocol": sha(ROOT / "protocol.json"),
                               "evaluation_script": sha(Path(__file__))}
    np.savez(args.output / "ridge_probe.npz", coef=final_ridge["coef"], intercept=final_ridge["intercept"], alpha=final_ridge["alpha"])
    write_json(args.output / "taxonomy_model.json", {**final_tax, "families": [dict(order=key[0], family=key[1], estimate=value) for key, value in final_tax["families"].items()]})
    write_csv(args.output / "test_predictions.csv", [{"scientific_name": r["scientific_name"], "genus": r["genus"], "family": r["family"],
                "order": r["order"], "trait_observed": float(y[index]), "predicted_embedding": float(a),
                "predicted_taxonomy": float(b), "predicted_mean": float(c)}
                for r, index, a, b, c in zip(test_rows, test, pred_ridge, pred_tax, pred_mean)])
    write_json(args.output / "results.json", result)
    print(json.dumps({key: result[key] for key in ["join", "test", "embedding_rmse_improvement_over_taxonomy", "positive_signal_rule_passed"]}, indent=2))


if __name__ == "__main__":
    main()
