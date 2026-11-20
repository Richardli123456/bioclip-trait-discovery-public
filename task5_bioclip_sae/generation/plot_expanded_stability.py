"""Source-checked static PNG/PDF figures for completed expanded SAE studies.

CPU only. Refuses unfinished summaries and never uses partial results as final.
"""
from pathlib import Path
import argparse
import csv
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiment_common import ROOT, sha, write_json

BLUE, GOLD, GRAY, TEXT = "#31688e", "#b88020", "#8c959e", "#273142"


def save_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    studies, points, reconstruction, hashes = [], [], [], {}
    locked_input_hashes = None
    for directory in args.studies:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        assert summary["status"] == "completed" and summary["input_hashes_unchanged"]
        assert summary["replicate_seeds"] == [1, 2, 3, 4, 5]
        if locked_input_hashes is None:
            locked_input_hashes = protocol["input_sha256"]
        assert protocol["input_sha256"] == locked_input_hashes, "Do not compare different corpora as a fixed-data configuration sweep"
        activation = np.load(directory / "training_correlations.npy")
        decoder = np.load(directory / "decoder_cosines.npy")
        validation = np.load(directory / "validation_correlations.npy")
        assert activation.shape == decoder.shape == validation.shape == (5, summary["width"])
        amin, dmin, vmin = activation.min(0), decoder.min(0), validation.min(0)
        passed = (amin >= .7) & (dmin >= .7)
        assert int(passed.sum()) == summary["stable_feature_count"]
        assert np.array_equal(np.flatnonzero(passed), summary["stable_features"])
        label = f"{summary['width']:,} features, TopK {summary['k']}"
        studies.append({"label": label, "summary": summary, "protocol": protocol, "directory": directory,
                        "amin": amin, "dmin": dmin, "vmin": vmin, "passed": passed})
        for feature in range(summary["width"]):
            points.append({"configuration": label, "feature": feature,
                           "min_training_activation_correlation": float(amin[feature]),
                           "min_same_match_decoder_cosine": float(dmin[feature]),
                           "min_validation_activation_correlation": float(vmin[feature]),
                           "passes_prespecified_gate": bool(passed[feature])})
        for run in summary["per_seed"]:
            for split in ["validation", "test"]:
                reconstruction.append({"configuration": label, "seed": run["seed"], "split": split,
                                       "species_count": protocol["split_counts"][split], "best_epoch": run["best_epoch"],
                                       **{k: run["metrics"][split][k] for k in ["normalized_mse", "variance_explained_vs_training_mean", "cosine_mean", "cosine_p05"]}})
        for name in ["summary.json", "protocol.json", "training_correlations.npy", "decoder_cosines.npy", "validation_correlations.npy"]:
            hashes[str((directory / name).resolve())] = sha(directory / name)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
                         "axes.labelcolor": TEXT, "axes.edgecolor": "#9ea6af", "xtick.color": TEXT,
                         "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    count = len(studies)
    cols, panel_rows = min(2, count), (count+1)//2
    fig, axes = plt.subplots(panel_rows, cols, figsize=(8.3 if cols == 1 else 6.3*cols, 6.5*panel_rows), squeeze=False)
    fig.subplots_adjust(left=.10 if cols == 1 else .075, right=.97, bottom=.21/panel_rows, top=1-.20/panel_rows,
                        hspace=.36, wspace=.28)
    lower_x = min(-.03, min(float(s["amin"].min()) for s in studies)-.025)
    lower_y = min(-.03, min(float(s["dmin"].min()) for s in studies)-.025)
    species_count = studies[0]["summary"]["species_count"]
    fig.suptitle("SAE feature stability across five training seeds", x=.075 if cols > 1 else .10, ha="left", y=.985, fontsize=16, fontweight="bold")
    subtitle = f"{species_count:,} taxonomy records | Minima over five replicate seeds; a separate anchor per configuration"
    fig.text(.075 if cols > 1 else .10, 1-.085/panel_rows, textwrap.fill(subtitle, width=98 if cols == 1 else 150), fontsize=10, color="#606a78")
    for number, study in enumerate(studies):
        ax = axes.flat[number]
        ax.add_patch(Rectangle((.7, .7), .3, .3, facecolor="#f0f3f5", edgecolor="none", zorder=0))
        ax.axvline(.7, color=TEXT, lw=1, ls="--", zorder=1)
        ax.axhline(.7, color=TEXT, lw=1, ls="--", zorder=1)
        rejected = ~study["passed"]
        ax.scatter(study["amin"][rejected], study["dmin"][rejected], s=10, color=GRAY, alpha=.4, linewidths=0, zorder=2)
        ax.scatter(study["amin"][study["passed"]], study["dmin"][study["passed"]], s=26, color=BLUE, marker="D", edgecolor=TEXT, linewidth=.35, zorder=3)
        ax.text(.98, .02, f"{int(study['passed'].sum()):,} / {study['summary']['width']:,} pass", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", color=BLUE)
        ax.set(xlim=(lower_x, 1.02), ylim=(lower_y, 1.02), xlabel="Minimum training activation correlation",
               ylabel="Minimum same-match decoder cosine")
        ax.set_title(study["label"], loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.grid(color="#e8eaed", linewidth=.6, zorder=-1)
    for number in range(count, panel_rows*cols):
        axes.flat[number].set_visible(False)
    footnote = "Each point is one anchor feature. Both minima must be ≥0.7. Matches maximize signed training activation correlation; decoder cosine uses that same match. Reused matches are allowed. Passing does not establish a biological trait."
    fig.text(.075 if cols > 1 else .10, .040/panel_rows, textwrap.fill(footnote, width=108 if cols == 1 else 175), fontsize=9)
    fig.savefig(args.out / "feature_stability.png", dpi=200)
    fig.savefig(args.out / "feature_stability.pdf")
    plt.close(fig)

    reconstruction_height = 4.0*count+1.6
    fig, axes = plt.subplots(count, 2, figsize=(12.6, reconstruction_height), squeeze=False)
    fig.subplots_adjust(left=.075, right=.97, bottom=.55/reconstruction_height, top=1-1.15/reconstruction_height, hspace=.45, wspace=.28)
    fig.suptitle("Validation and test reconstruction across initializations", x=.075, ha="left", y=1-.18/reconstruction_height, fontsize=16, fontweight="bold")
    fig.text(.075, 1-.65/reconstruction_height, "Validation selects checkpoints; test results are descriptive. Each marker is a separate training run, not a confidence interval.", fontsize=10, color="#606a78")
    for number, study in enumerate(studies):
        runs = study["summary"]["per_seed"]
        tick = np.arange(len(runs))
        for split, marker, color, shift in [("validation", "o", BLUE, -.08), ("test", "^", GOLD, .08)]:
            for column, metric in enumerate(["normalized_mse", "cosine_mean"]):
                values = np.array([r["metrics"][split][metric] for r in runs])
                axes[number, column].scatter(tick+shift, values, marker=marker, color=color, s=42, label=split.capitalize(), zorder=3)
        for column, title in enumerate(["Normalized reconstruction error", "Mean reconstruction cosine"]):
            ax = axes[number, column]
            ax.set(xlim=(-.5, len(runs)-.5), ylim=(0, 1), ylabel="NMSE (lower is better)" if column == 0 else "Cosine (higher is better)", xlabel="SAE training seed")
            ax.set_title(study["label"] + "\n" + title, loc="left", fontsize=12, fontweight="bold")
            ax.set_xticks(tick, ["Anchor" if r["seed"] == study["summary"]["anchor_seed"] else str(r["seed"]) for r in runs])
            ax.grid(axis="y", color="#e8eaed", linewidth=.6, zorder=-1)
            ax.legend(loc="center left", frameon=False, fontsize=9)
    fig.savefig(args.out / "reconstruction.png", dpi=200)
    fig.savefig(args.out / "reconstruction.pdf")
    plt.close(fig)
    save_table(args.out / "feature_stability_points.csv", points)
    save_table(args.out / "reconstruction_metrics.csv", reconstruction)
    provenance = {"status": "generated_pending_visual_inspection", "cpu_only": True, "source_hashes": hashes,
                  "script_sha256": sha(__file__), "configurations": [s["label"] for s in studies],
                  "feature_points": len(points), "reconstruction_rows": len(reconstruction),
                  "gate_recomputed_matches_summary": True, "figure_hashes": {p.name: sha(p) for p in args.out.glob("*.png")},
                  "limitations": "These are separate training runs, not confidence intervals; test metrics did not select configurations. Stable features are candidates, not validated traits."}
    write_json(args.out / "figure_provenance.json", provenance)
    lines = ["# Expanded SAE stability figures", "", f"The figures use {species_count:,} taxonomy records and the unchanged two-part 0.7 gate across all five replicate seeds.", "",
             "| Configuration | Passing features |", "|---|---:|"]
    lines.extend(f"| {s['label']} | {int(s['passed'].sum()):,} / {s['summary']['width']:,} |" for s in studies)
    lines.extend(["", "![Feature stability](feature_stability.png)", "", "![Reconstruction](reconstruction.png)", "",
                  "[Feature point data](feature_stability_points.csv) · [Reconstruction values](reconstruction_metrics.csv) · [Provenance](figure_provenance.json)", "",
                  "Every point in the feature-stability figure is a matched anchor coordinate. Passing requires both the training activation correlation and the decoder cosine for that same match to remain at least 0.7 in every replicate. Validation correlations are retained in the point table as confirmation diagnostics. Reused matches are allowed.", "",
                  "Reconstruction panels use a fixed 0–1 scale. Individual seed markers do not represent statistical uncertainty. No biological trait or image-generation effect is established by these figures.", ""])
    (args.out / "STABILITY_FIGURES.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "configuration_count": count, "feature_points": len(points), "gate_verified": True}), flush=True)


if __name__ == "__main__":
    main()
