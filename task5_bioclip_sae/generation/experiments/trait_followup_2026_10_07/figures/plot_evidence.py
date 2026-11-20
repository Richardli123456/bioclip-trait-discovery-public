"""Make standalone scientific figures from saved, verified results; no fitting.

Outputs stay in the two assigned figures folders. Source tables retain units,
denominators, uncertainty definitions, and hashes of the governing artifacts.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent
HUMAN = EXPERIMENTS / "human_review_2026_10_07"
TAIL = HERE.parent
HUMAN_OUT = HUMAN / "figures"
INK = "#183B4E"
BLUE = "#357A96"
MUTED = "#52616B"
GRAY = "#C9CFD3"
GRID = "#E1E6E9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.labelsize": 11, "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": MUTED, "axes.linewidth": .8,
                     "savefig.facecolor": "white", "pdf.fonttype": 42})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(fig, out, stem):
    outputs = []
    for extension in ["png", "pdf"]:
        path = out / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, metadata={"Creator": "Reproducible Matplotlib scientific figure"})
        outputs.append(path)
    plt.close(fig)
    return outputs


def human_figure():
    HUMAN_OUT.mkdir(exist_ok=True)
    source = HUMAN / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    h = summary["human"]
    with (HUMAN / "human_scores_corrected.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    hist = Counter(int(r["spot_count"]) for r in records if r["spot_count"] != "NA")
    missing = sum(r["spot_count"] == "NA" for r in records)
    assert {str(k): v for k, v in sorted(hist.items())} == h["exact_count_distribution"]
    assert len(records) == h["images"] == 32 and sum(hist.values()) == 27 and missing == 5
    source_rows = [{"visible_spot_count": count, "measurement_status": "count_scorable", "images": hist[count],
                    "denominator_all_images": 32, "denominator_scorable_images": 27,
                    "reviewer_type": "single human", "summary_sha256": sha(source)} for count in range(11)]
    source_rows.append({"visible_spot_count": "NA", "measurement_status": "unscorable_or_uncertain", "images": missing,
                        "denominator_all_images": 32, "denominator_scorable_images": "not applicable",
                        "reviewer_type": "single human", "summary_sha256": sha(source)})
    assert sum(r["images"] for r in source_rows) == 32
    csv_write(HUMAN_OUT / "human_count_distribution_source.csv", source_rows)
    rate_rows = []
    for name in ["count_scorable_rate", "visible_spots_among_eligible", "exactly_seven_among_scorable", "observed_scorable_seven_yield_all_images"]:
        value = h[name]
        rate_rows.append({"measure": name, "numerator": value["numerator"], "denominator": value["denominator"],
                          "proportion": value["proportion"], "ci95_lower": value["wilson_95_interval"][0],
                          "ci95_upper": value["wilson_95_interval"][1], "interval_method": "two-sided Wilson score",
                          "summary_sha256": sha(source)})
    csv_write(HUMAN_OUT / "human_rate_summary_source.csv", rate_rows)
    fig = plt.figure(figsize=(11.8, 6.8))
    fig.text(.075, .935, "Visible spots in 32 unedited ladybird images", fontsize=20, weight="bold")
    fig.text(.075, .89, "Single human review | 27 countable images; 5 uncertain / unscorable", fontsize=12, color=MUTED)
    ax = fig.add_axes([.075, .30, .68, .48])
    bars = ax.bar(range(11), [hist[i] for i in range(11)], width=.78, color=BLUE,
                  edgecolor=INK, linewidth=.7)
    bars[7].set_facecolor(INK)
    bars[7].set_hatch("///")
    for i, bar in enumerate(bars):
        ax.text(i, hist[i] + .12, str(hist[i]), ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 8)
    ax.set_xticks(range(11))
    ax.set_xlabel("Visible dark wing-cover spots")
    ax.set_ylabel("Images")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_title("Distribution among 27 countable images", loc="left", pad=13)
    ax2 = fig.add_axes([.835, .30, .095, .48], sharey=ax)
    ax2.bar([0], [missing], width=.65, color=GRAY, edgecolor=MUTED, hatch="//", linewidth=.7)
    ax2.text(0, missing + .12, str(missing), ha="center", va="bottom", fontsize=11)
    ax2.set_xticks([0], ["NA"])
    ax2.set_xlim(-.6, .6)
    ax2.tick_params(axis="y", labelleft=False, left=False)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("Unscorable", pad=13)
    ax2.yaxis.grid(True, color=GRID, linewidth=.7)
    ax2.set_axisbelow(True)
    fig.text(.075, .205, "Exactly seven: 3/27 countable images (11.1%; 95% Wilson CI 3.9%-28.1%).", fontsize=11.5, weight="bold")
    fig.text(.075, .157, "NA remains missing, never zero. Counts describe visible patches under this rater's interpretation.", fontsize=10, color=MUTED)
    fig.text(.075, .118, "The prior AI disagreed on view eligibility for all three human exact-seven images; no adjudication is implied.", fontsize=10, color=MUTED)
    fig.text(.075, .069, "Source: corrected human CSV and verified summary, 2026-10-07. No SAE edit was used for these images.", fontsize=9, color=MUTED)
    outputs = export(fig, HUMAN_OUT, "human_spot_count_distribution")
    return outputs, [source, HUMAN / "human_scores_corrected.csv"], len(source_rows)


def tail_figure():
    source = TAIL / "tail_candidate_v1/summary.json"
    s = json.loads(source.read_text(encoding="utf-8"))
    assert s["status"] == "completed" and s["selected_feature"] == 814
    assert not s["candidate_confirmed"] and not s["test_endpoint_used"] and not s["image_branch_eligible"]
    v, t = s["validation"], s["training_selection"]
    association = [
        {"split": "training", "feature": 814, "within_family_pearson_r": t["within_family_r"], "ci95_lower": "NA", "ci95_upper": "NA",
         "interval_method": "not estimated; selection statistic", "species": s["cohort_counts"]["train"],
         "active_species": t["active_training_species"], "active_genera": t["active_training_genera"], "summary_sha256": sha(source)},
        {"split": "validation", "feature": 814, "within_family_pearson_r": v["family_centered_r"], "ci95_lower": v["correlation_95ci"][0], "ci95_upper": v["correlation_95ci"][1],
         "interval_method": "2000 genus-cluster bootstrap resamples; family recentered per resample", "species": v["species"],
         "active_species": v["active_support"]["species"], "active_genera": v["active_support"]["genera"], "summary_sha256": sha(source)},
    ]
    prediction = [{"split": "validation", "model": name, "rmse": v[key]["rmse"], "units": "natural-log tail/wing ratio",
                   "species": v["species"], "genera": v["genera"], "summary_sha256": sha(source)}
                  for name, key in [("taxonomy baseline", "taxonomy_baseline"), ("taxonomy plus feature 814", "taxonomy_plus_activation")]]
    improvement = [{"quantity": "taxonomy RMSE minus taxonomy-plus-activation RMSE", "estimate": v["rmse_improvement"],
                    "ci95_lower": v["improvement_95ci"][0], "ci95_upper": v["improvement_95ci"][1],
                    "units": "natural-log tail/wing ratio", "favorable_direction": "positive",
                    "interval_method": "2000 paired genus-cluster bootstrap resamples", "summary_sha256": sha(source)}]
    csv_write(HERE / "tail_association_source.csv", association)
    csv_write(HERE / "tail_prediction_source.csv", prediction)
    csv_write(HERE / "tail_rmse_improvement_source.csv", improvement)
    fig = plt.figure(figsize=(12.3, 7.5))
    fig.text(.075, .935, "Tail-to-wing association did not confirm", fontsize=20, weight="bold")
    fig.text(.075, .887, "SAE feature 814 | Selected on training labels; evaluated on whole-genus validation holdout", fontsize=11.5, color=MUTED)
    ax = fig.add_axes([.145, .385, .335, .365])
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.scatter([t["within_family_r"]], [1], s=72, marker="s", color=INK, zorder=3)
    vr = v["family_centered_r"]
    lo, hi = v["correlation_95ci"]
    ax.errorbar([vr], [0], xerr=[[vr-lo], [hi-vr]], fmt="o", markersize=8, color=BLUE, capsize=5, linewidth=1.7, zorder=3)
    ax.set_yticks([1, 0], ["Training\n6,538 species", "Validation\n875 species"])
    ax.set_ylim(-.6, 1.75)
    ax.set_xlim(-.25, .25)
    ax.set_xticks([-.2, -.1, 0, .1, .2])
    ax.set_xlabel("Within-family Pearson correlation (r)")
    ax.set_title("A. Direction reversed; interval crosses zero", loc="left", fontsize=12, pad=18)
    ax.text(t["within_family_r"], 1.26, f"{t['within_family_r']:+.3f}", ha="center", fontsize=11, weight="bold")
    ax.text(vr, .26, f"{vr:+.3f} [{lo:+.3f}, {hi:+.3f}]", ha="center", fontsize=10.5)
    ax.xaxis.grid(True, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax2 = fig.add_axes([.61, .385, .32, .365])
    vals = [r["rmse"] for r in prediction]
    ax2.bar([0, 1], vals, width=.52, color=[GRAY, BLUE], edgecolor=[MUTED, INK], linewidth=.8)
    for i, value in enumerate(vals):
        ax2.text(i, value+.008, f"{value:.5f}", ha="center", va="bottom", fontsize=11, weight="bold")
    ax2.set_ylim(0, .29)
    ax2.set_xlim(-.65, 1.65)
    ax2.set_xticks([0, 1], ["Taxonomy", "Taxonomy\n+ feature 814"])
    ax2.set_ylabel("RMSE (natural-log ratio units)")
    ax2.set_title("B. No improvement over taxonomy", loc="left", fontsize=12, pad=18)
    ax2.yaxis.grid(True, color=GRID, linewidth=.7)
    ax2.set_axisbelow(True)
    fig.text(.075, .27, "Association gate: failed     |     Predictive-improvement gate: failed", fontsize=12, weight="bold")
    low, high = v["improvement_95ci"]
    fig.text(.075, .221, f"RMSE improvement (taxonomy minus augmented): {v['rmse_improvement']:+.5f}; 95% CI [{low:+.5f}, {high:+.5f}]. Positive favors the feature.", fontsize=10.5)
    fig.text(.075, .167, "Intervals: 2,000 genus-cluster bootstrap resamples. Validation: 875 species / 214 genera; 29 active species / 12 genera.", fontsize=9.8, color=MUTED)
    fig.text(.075, .127, "Training point is a selection statistic, not independent evidence; no training interval was estimated. Reused cohort, new endpoint.", fontsize=9.8, color=MUTED)
    fig.text(.075, .087, "Test endpoint not evaluated; no tail-image experiment launched. Stable SAE coordinates are not established biological traits.", fontsize=9.8, color=MUTED)
    fig.text(.075, .045, "Source: tail_candidate_v1/summary.json, 2026-10-07. Endpoint: ln(AVONET species-mean tail length / wing length).", fontsize=9, color=MUTED)
    outputs = export(fig, HERE, "tail_validation_evidence")
    return outputs, [source, TAIL / "protocol.json"], len(association) + len(prediction) + len(improvement)


def main():
    for make, out in [(human_figure, HUMAN_OUT), (tail_figure, HERE)]:
        outputs, inputs, source_rows = make()
        metadata = {
            "source_rows": source_rows, "source_sha256": {str(p.resolve()): sha(p) for p in inputs},
            "plot_script_sha256": sha(Path(__file__)), "plotting_library": f"Matplotlib {matplotlib.__version__}",
            "outputs_sha256": {p.name: sha(p) for p in outputs},
            "new_fits": False, "new_image_generation": False,
            "notes": "Figures use saved results. Missing observations remain separate, interval units are explicit, and negative confirmation outcomes are preserved.",
        }
        (out / "figure_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"figures": [str(p) for p in outputs], "source_rows": source_rows}))


if __name__ == "__main__":
    main()
