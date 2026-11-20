"""Summarize the completed human baseline review without changing its raw export.

Standard library only; no model fitting, image generation, or image rescoring.
The three visibility corrections below were explicitly supplied by the user.
Run from any directory with the project's existing Python environment.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "unedited_dorsal_v1"
RAW = ROOT / "human_scores_original.csv"
RAW_SHA256 = "ed129aa5afb8b07f8131a311976ec1bdaa0d95c356d1fb16fc11dbf3e00105fe"
FIELDS = ["image_id", "view_eligible", "spots_visible", "spot_count", "notes"]
CLARIFIED_IDS = ("D002", "D008", "D027")
EXPECTED_IDS = [f"D{i:03d}" for i in range(1, 33)]
Z95 = statistics.NormalDist().inv_cdf(0.975)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    assert len(rows) == 32
    assert sorted(r["image_id"] for r in rows) == EXPECTED_IDS
    return rows


def save_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def wilson(k, n):
    """Two-sided 95% Wilson score interval for a declared binomial denominator."""
    assert 0 <= k <= n and n > 0
    p = k / n
    denominator = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return {"numerator": k, "denominator": n, "proportion": p,
            "wilson_95_interval": [max(0.0, center - half), min(1.0, center + half)]}


def summarize(rows):
    eligible = [r for r in rows if r["view_eligible"] == "yes"]
    counts = [int(r["spot_count"]) for r in rows if r["spot_count"] != "NA"]
    seven_ids = [r["image_id"] for r in rows if r["spot_count"] == "7"]
    spot_yes = sum(r["spots_visible"] == "yes" for r in eligible)
    return {
        "images": len(rows),
        "eligibility_counts": dict(Counter(r["view_eligible"] for r in rows)),
        "view_eligible_rate": wilson(len(eligible), len(rows)),
        "count_scorable_rate": wilson(len(counts), len(rows)),
        "visible_spots_among_eligible": wilson(spot_yes, len(eligible)),
        "exact_count_distribution": {str(k): v for k, v in sorted(Counter(counts).items())},
        "count_mean_among_scorable": statistics.mean(counts),
        "count_median_among_scorable": statistics.median(counts),
        "count_sample_sd_among_scorable": statistics.stdev(counts) if len(counts) > 1 else None,
        "count_min_max_among_scorable": [min(counts), max(counts)],
        "exactly_seven_ids": seven_ids,
        "exactly_seven_among_scorable": wilson(len(seven_ids), len(counts)),
        "observed_scorable_seven_yield_all_images": wilson(len(seven_ids), len(rows)),
        "unscorable_ids": [r["image_id"] for r in rows if r["spot_count"] == "NA"],
        "notes_present": sum(bool(r["notes"].strip()) for r in rows),
    }


def validate(rows):
    for row in rows:
        assert row["view_eligible"] in {"yes", "no", "NA"}
        assert row["spots_visible"] in {"yes", "no", "NA"}
        assert row["spot_count"] == "NA" or row["spot_count"].isdigit()
        if row["view_eligible"] != "yes":
            assert row["spots_visible"] == row["spot_count"] == "NA"
        if row["spot_count"] != "NA":
            assert row["view_eligible"] == "yes"
            assert row["spots_visible"] == ("yes" if int(row["spot_count"]) > 0 else "no")


def percent(entry):
    low, high = entry["wilson_95_interval"]
    return (f"{entry['numerator']}/{entry['denominator']} "
            f"({100 * entry['proportion']:.1f}%; 95% Wilson interval {100 * low:.1f}%–{100 * high:.1f}%)")


def main():
    assert sha(RAW) == RAW_SHA256, "Raw human export changed"
    raw = read_csv(RAW)
    corrected = [dict(r) for r in raw]
    changes = []
    for row in corrected:
        if row["image_id"] in CLARIFIED_IDS:
            assert row["view_eligible"] == "yes"
            assert row["spots_visible"] == "NA" and row["spot_count"] == "5"
            changes.append({"image_id": row["image_id"], "field": "spots_visible", "before": "NA", "after": "yes",
                            "unchanged_confirmed_spot_count": 5})
            row["spots_visible"] = "yes"
    assert len(changes) == 3
    validate(corrected)
    save_csv(ROOT / "human_scores_corrected.csv", corrected, FIELDS)
    clarification = {
        "date_local": "2026-10-07", "timezone": "America/New_York",
        "source": "User's direct clarification in the current research chat, relayed by the supervising agent",
        "normalized_clarification_not_verbatim_quote": "D002, D008, and D027 have visible spots; the spot count is 5 for each.",
        "changes": changes, "counts_changed": False, "raw_export_preserved": True,
        "raw_export_sha256": RAW_SHA256,
        "corrected_csv_sha256": sha(ROOT / "human_scores_corrected.csv"),
        "authorization_scope": "Only the three explicitly clarified visibility fields; no AI-derived correction",
    }
    save_json(ROOT / "clarification_provenance.json", clarification)
    ai_path = BASELINE / "ai_review_provisional.csv"
    ai = read_csv(ai_path)
    validate(ai)
    ai_by_id = {r["image_id"]: r for r in ai}
    paired = []
    for h in corrected:
        a = ai_by_id[h["image_id"]]
        numeric_pair = h["spot_count"] != "NA" and a["spot_count"] != "NA"
        paired.append({
            "image_id": h["image_id"], "human_view_eligible": h["view_eligible"],
            "ai_view_eligible": a["view_eligible"],
            "view_descriptor_agrees": h["view_eligible"] == a["view_eligible"],
            "human_spots_visible": h["spots_visible"], "ai_spots_visible": a["spots_visible"],
            "spots_descriptor_agrees": h["spots_visible"] == a["spots_visible"],
            "human_spot_count": h["spot_count"], "ai_spot_count": a["spot_count"],
            "count_scorable_status_agrees": (h["spot_count"] != "NA") == (a["spot_count"] != "NA"),
            "both_numeric_counts": numeric_pair,
            "numeric_counts_agree_if_both_scorable": h["spot_count"] == a["spot_count"] if numeric_pair else "NA",
            "human_notes": h["notes"], "ai_notes": a["notes"],
        })
    disagreements = [r for r in paired if not (r["view_descriptor_agrees"] and r["spots_descriptor_agrees"]
                      and r["count_scorable_status_agrees"] and r["numeric_counts_agree_if_both_scorable"] in {True, "NA"})]
    save_csv(ROOT / "human_ai_paired_descriptors.csv", paired)
    save_csv(ROOT / "human_ai_disagreements.csv", disagreements)
    matrix = {h: {a: sum(r["human_view_eligible"] == h and r["ai_view_eligible"] == a for r in paired)
                  for a in ["yes", "no", "NA"]} for h in ["yes", "no", "NA"]}
    human_summary = summarize(corrected)
    ai_summary = summarize(ai)
    summary = {
        "status": "completed", "human_reviewers": 1,
        "human_expertise": "Not established; no biological-expert claim",
        "human_review_blinding": "Interface used randomized opaque IDs and did not show AI scores; actual compliance with seed-key avoidance was not independently verified",
        "human": human_summary, "ai_provisional_separate": ai_summary,
        "paired_comparison": {
            "eligibility_descriptor_confusion_human_rows_ai_columns": matrix,
            "eligibility_three_category_agreement": sum(r["view_descriptor_agrees"] for r in paired),
            "spots_visibility_three_category_agreement": sum(r["spots_descriptor_agrees"] for r in paired),
            "exact_count_scorable_status_agreement": sum(r["count_scorable_status_agrees"] for r in paired),
            "both_exact_count_scorable": sum(r["both_numeric_counts"] for r in paired),
            "same_numeric_count_when_both_scorable": sum(r["numeric_counts_agree_if_both_scorable"] is True for r in paired),
            "disagreement_rows": len(disagreements),
            "disagreement_definition": "Any view descriptor, spot descriptor, count-scorability, or jointly available count differs; NA-versus-number is missingness disagreement, not count error",
            "interpretation": "Descriptive comparison of one human review with one earlier provisional AI pass; no independent-human reliability or adjudication",
        },
        "feasibility_conclusion": "The human found exact visible counts usable for 27 of these 32 baseline images. This supports a small exploratory follow-up under this rater and rule, but substantial eligibility disagreement with the provisional AI calls for explicit annotation calibration. It is not SAE-trait confirmation, canonical-seven fidelity, independent-rater reliability, or evidence of an intervention effect.",
        "predeclared_feasibility_threshold": None,
        "feasibility_gate_note": "No numerical pass threshold was fixed for this 32-image human calibration; do not retroactively call a confirmatory gate passed.",
        "uncertainty_method": "Two-sided 95% Wilson score intervals with z=NormalDist.inv_cdf(0.975); descriptive uncertainty over these generation seeds with a fixed rater and generator. Intervals omit rater error, systematic model error, and species generalization.",
        "missingness": "Five NA counts stay NA, not zero. All-image seven yield counts a jointly observable event (scorable and counted seven); it does not classify unscorable images as non-seven.",
        "training_performed": False, "generation_performed": False, "images_rescored_by_ai": False,
        "input_sha256": {"raw_human_csv": sha(RAW), "ai_csv": sha(ai_path),
                         "scoring_rules": sha(BASELINE / "human_review/SCORING_RULES.md"),
                         "generation_protocol": sha(BASELINE / "protocol.json"),
                         "analysis_script": sha(Path(__file__))},
        "corrected_human_csv_sha256": sha(ROOT / "human_scores_corrected.csv"),
    }
    save_json(ROOT / "summary.json", summary)
    dist = human_summary["exact_count_distribution"]
    md = [
        "# Completed human review of the 32 unedited ladybird images", "",
        "The single human reviewer judged **27/32 images eligible and exactly countable**. Five remain uncertain/unscorable. This materially improves the measurement-feasibility evidence over the earlier provisional AI pass, while the two reviewers disagree substantially about which views qualify. The human annotations are retained as supplied; AI scores do not override them.", "",
        "## Human results", "",
        f"- Eligible and count-scorable: {percent(human_summary['count_scorable_rate'])}.",
        f"- Visible spots among eligible images: {percent(human_summary['visible_spots_among_eligible'])}.",
        f"- Exactly seven visible spots among scorable images: {percent(human_summary['exactly_seven_among_scorable'])}.",
        f"- Observed scorable-seven yield over all 32 images: {percent(human_summary['observed_scorable_seven_yield_all_images'])}.",
        f"- Exact-seven IDs: {', '.join(human_summary['exactly_seven_ids'])}.",
        f"- Unscorable IDs: {', '.join(human_summary['unscorable_ids'])}. These are NA, never zero.",
        f"- Mean count among the 27 scorable images: {human_summary['count_mean_among_scorable']:.2f}; median {human_summary['count_median_among_scorable']:g}; range {min(map(int, dist))}–{max(map(int, dist))}.", "",
        "| Visible count | Images |", "|---:|---:|",
        *[f"| {count} | {number} |" for count, number in dist.items()], "",
        "The all-image yield is the joint event 'scorable and counted seven.' It does not treat the five unscorable images as biologically non-seven. These are visible patch counts, not inferred hidden spots or confirmed anatomical species patterns. No image received a zero count. Notes were blank on all 32 exported records, so uncertainty reasons cannot be reconstructed from human notes.", "",
        "## Explicit clarification and provenance", "",
        "The raw export had spot_count=5 and view_eligible=yes but spots_visible=NA for D002, D008, and D027. The user explicitly clarified that visible spots are yes and the count is five for all three. Only those three visibility fields were changed in the derived CSV; counts, eligibility, notes, row order, and the original export are unchanged. This is a human clarification, not an AI correction.", "",
        "The page displayed randomized opaque image IDs without AI scores. We cannot independently verify whether the reviewer consulted other materials or the seed key. This is one human review with expertise unspecified, not expert adjudication or independent-human agreement.", "",
        "## Separate comparison with the earlier provisional AI review", "",
        "| Human eligibility \\ AI eligibility | yes | no | NA |", "|---|---:|---:|---:|",
        *[f"| {h} | {matrix[h]['yes']} | {matrix[h]['no']} | {matrix[h]['NA']} |" for h in ["yes", "no", "NA"]], "",
        f"Exact agreement on the three eligibility descriptors is {summary['paired_comparison']['eligibility_three_category_agreement']}/32. The earlier AI pass marked 12 eligible, 13 ineligible, and seven uncertain. All three human exact-seven images (D005, D010, D026) were marked view-ineligible by that AI pass. This is an unresolved eligibility disagreement, not grounds to invalidate or silently change the human response.", "",
        f"Human exact counts are available for 27 images, versus one for the AI. Only D011 has numeric counts from both, and both give eight. This single agreement cannot establish count reliability. Count-scorability status agrees on {summary['paired_comparison']['exact_count_scorable_status_agreement']}/32; numeric-versus-NA cases are reported as differences in measurement availability, not invented numeric errors.", "",
        "## Measurement decision and limits", "",
        "Human scoring supports a small exploratory count-measurement follow-up for this fixed model, overhead prompt, species, and rater. No numerical acceptance threshold was locked for this 32-image review, so the result is descriptive rather than a retroactively passed confirmatory test. Exact seven occurred in only 3/27 scorable images; these outputs do not reliably reproduce the canonical seven-patch appearance under this scoring.", "",
        "Before scaling a count intervention, freeze the candidate and counting rules, use explicit pose/occlusion examples to calibrate annotation, and retain paired missingness as an outcome. The visible counts do not themselves provide independent species-level labels for selecting an SAE coordinate, and they cannot rescue the failed beak feature. No new training or generation was performed for this analysis.", "",
        "The Wilson intervals describe sampling uncertainty over image seeds for fixed settings and a fixed rater. They do not include annotation errors, systematic generator errors, or cross-species uncertainty. The broad human–AI eligibility disagreement remains unresolved and should be transparent in any downstream claim.", "",
        "## Reproducible files", "",
        "- [Unchanged original human export](human_scores_original.csv)",
        "- [Corrected human CSV](human_scores_corrected.csv) and [explicit clarification provenance](clarification_provenance.json)",
        "- [Numerical summary and source hashes](summary.json)",
        "- [All paired descriptors](human_ai_paired_descriptors.csv) and [disagreements](human_ai_disagreements.csv)",
        "- [Independent verification](verification.json) and [verification script](verify_human_review.py)",
        "- [Reproducible standard-library analysis](analyze_human_review.py)",
        "- [Original human scoring rules](../unedited_dorsal_v1/human_review/SCORING_RULES.md)", "",
        "Run `python analyze_human_review.py`, then `python verify_human_review.py` from this folder. The analysis checks the fixed raw-export hash before writing derived outputs and refuses inconsistent scoring records. The separate verifier independently counts CSV records, inverts the binomial score inequality numerically to check Wilson endpoints, and checks all 32 blind images against their original generation hashes.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    assert sha(RAW) == RAW_SHA256, "Analysis changed raw export"
    print(json.dumps({"human": human_summary, "comparison": summary["paired_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
