"""CPU-only figure and descriptive follow-up for the fixed stability experiment."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/aligned_stability_v1"
ALIGNED = ROOT / "cache/aligned"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    act = np.load(OUT / "training_activation_correlations.npy")
    dec = np.load(OUT / "same_match_decoder_cosines.npy")
    assert act.shape == dec.shape == (5, 2048)
    amin, dmin = act.min(0), dec.min(0)
    assert int(((amin >= .7) & (dmin >= .7)).sum()) == summary["stable_feature_count"]
    recurring = np.asarray(summary["activation_only_recurrent_features"])
    selected = {729: ("Ladybird 729", (0.05, 0.83)), 1969: ("Owl 1969", (0.19, 0.52)), 527: ("Squirrel 527", (0.72, 0.44))}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": "#273142", "text.color": "#273142", "axes.edgecolor": "#a3aab5", "xtick.color": "#465163", "ytick.color": "#465163", "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.6, 6.7), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.24, top=.78, wspace=.30)
    fig.suptitle("Aligned SAE: reconstruction consistency and feature recurrence", x=.075, ha="left", y=.96, fontsize=17, fontweight="bold")
    fig.text(.075, .90, "5,388 species  |  768 → 2,048 TopK-32 SAE  |  five new training seeds; original model as reference", fontsize=11, color="#606a78")
    left.scatter(amin, dmin, s=12, color="#34618b", alpha=.34, linewidths=0, zorder=2)
    left.scatter(amin[recurring], dmin[recurring], s=42, edgecolor="#273142", facecolor="none", linewidth=1.1, zorder=3)
    left.add_patch(Rectangle((.7, .7), .3, .3, facecolor="#f1f3f5", zorder=0))
    left.axvline(.7, color="#777f8b", lw=1.2, ls="--", zorder=1)
    left.axhline(.7, color="#777f8b", lw=1.2, ls="--", zorder=1)
    left.text(.85, .84, "0 features\npass this gate", ha="center", va="center", fontsize=11, fontweight="bold")
    for j, (label, pos) in selected.items():
        left.scatter(amin[j], dmin[j], s=52, color="#c47b22", edgecolor="#5b3a16", linewidth=.7, zorder=4)
        left.annotate(label, (amin[j], dmin[j]), xytext=pos, fontsize=10, arrowprops={"arrowstyle": "-", "color": "#766858", "lw": .9}, color="#6a461b", va="center")
    left.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Minimum training activation correlation", ylabel="Minimum cosine of the same matched decoder")
    left.set_title("A   Feature recurrence across five seeds", loc="left", fontsize=12, fontweight="bold", pad=16)
    left.grid(color="#e8eaed", linewidth=.6, zorder=-1)
    runs = summary["per_seed"]
    values = np.array([r["metrics"]["test"]["cosine_mean"] for r in runs])
    x = np.arange(len(runs))
    right.scatter(x[1:], values[1:], color="#34618b", s=55, zorder=3)
    right.scatter(x[0], values[0], color="#273142", marker="D", s=48, zorder=3)
    right.axhline(values[0], color="#939aa5", linewidth=1, ls="--", zorder=1)
    for xi, y in zip(x, values):
        right.text(xi, y + .00022, f"{y:.4f}", ha="center", fontsize=10)
    right.set(xlim=(-.5, 5.5), ylim=(.868, .875), ylabel="Mean held-out reconstruction cosine", xlabel="SAE training initialization")
    right.set_xticks(x, ["Original", "1", "2", "3", "4", "5"])
    right.set_title("B   Reconstruction on 593 held-out species", loc="left", fontsize=12, fontweight="bold", pad=16)
    right.grid(axis="y", color="#e8eaed", linewidth=.7)
    fig.text(.075, .130, "Each left-panel point is one original feature (n = 2,048), matched separately by training activation correlation in each new seed.", fontsize=9.5)
    fig.text(.075, .095, "Dashed 0.7 thresholds define this experiment’s prespecified gate, not a universal standard. Open circles: seven activation-only recurring features.", fontsize=9.5)
    fig.text(.075, .060, "Decoder directions are signed; matches can be reused. Stable reconstruction does not establish stable coordinates or isolated biological traits.", fontsize=9.5)
    fig.text(.075, .025, "Right panel uses a focused vertical scale (0.868–0.875); points are separate runs. No confidence intervals are implied.", fontsize=9.5, color="#606a78")
    fig.savefig(OUT / "stability_summary.png", dpi=190)
    fig.savefig(OUT / "stability_summary.pdf")
    plt.close(fig)

    rows = json.loads((ALIGNED / "data/species.json").read_text(encoding="utf-8"))
    z = np.load(ALIGNED / "results/sparse_vectors.npy")
    train = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    leads = [i for i, row in enumerate(rows) if row["scientific_name"] == "Coccinella septempunctata" or row["genus"] in ["Nyctibius", "Phyllaemulor"]]
    descriptions = []
    for j in recurring:
        top = train[np.argsort(-z[train, j], kind="stable")[:8]]
        top = [int(i) for i in top if z[i, j] > 1e-7]
        descriptions.append({"original_feature": int(j), "min_training_correlation": float(amin[j]), "min_same_match_decoder_cosine": float(dmin[j]), "passes_combined_gate": False, "top8_family_counts": dict(Counter(rows[i]["family"] for i in top)), "top8_training_species": [{"species": rows[i]["scientific_name"], "family": rows[i]["family"], "activation": float(z[i, j])} for i in top], "lead_species_activations": [{"species": rows[i]["scientific_name"], "activation": float(z[i, j]), "split": rows[i]["split"]} for i in leads]})
    allmatches = list(csv.DictReader((OUT / "all_feature_matches.csv").open(encoding="utf-8")))
    best_dec = np.array([float(r["best_decoder_cosine_any_feature"]) for r in allmatches]).reshape(5, 2048)
    diagnostic = {"purpose": "Post-result descriptive diagnostic only; does not relax or replace the prespecified gate", "features_with_best_decoder_cosine_ge_0_7_in_all_seeds": int((best_dec >= .7).all(0).sum()), "highest_minimum_best_decoder_cosine": float(best_dec.min(0).max()), "activation_only_feature_descriptions": descriptions}
    (OUT / "activation_only_diagnostic.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8")
    provenance = {"source_summary_sha256": sha(OUT / "summary.json"), "source_all_matches_sha256": sha(OUT / "all_feature_matches.csv"), "source_raw_codes_sha256": sha(ALIGNED / "results/sparse_vectors.npy"), "script_sha256": sha(__file__), "png_sha256": sha(OUT / "stability_summary.png"), "pdf_sha256": sha(OUT / "stability_summary.pdf"), "point_count": 2048, "training_seed_count": 5, "test_species_count": summary["split_counts"]["test"], "cpu_only": True}
    (OUT / "figure_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"diagnostic": {k: v for k, v in diagnostic.items() if k != "activation_only_feature_descriptions"}, "features": [{"feature": f["original_feature"], "families": f["top8_family_counts"], "active_leads": [r for r in f["lead_species_activations"] if r["activation"] > 1e-7]} for f in descriptions]}, indent=2))


if __name__ == "__main__":
    main()
