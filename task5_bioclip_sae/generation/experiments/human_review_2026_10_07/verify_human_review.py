"""Independent arithmetic, provenance, image-hash, and link checks; no scoring."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "unedited_dorsal_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def score_interval(k, n):
    """Numerical inversion; deliberately does not reuse Wilson closed form."""
    q = NormalDist().inv_cdf(0.975) ** 2
    p_hat = k / n
    def f(p):
        return (k - n * p) ** 2 - q * n * p * (1 - p)
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, p_hat
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.0
    else:
        lo, hi = p_hat, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        upper = (lo + hi) / 2
    return lower, upper


def main():
    raw_path = ROOT / "human_scores_original.csv"
    assert digest(raw_path) == "ed129aa5afb8b07f8131a311976ec1bdaa0d95c356d1fb16fc11dbf3e00105fe"
    raw, corrected = rows(raw_path), rows(ROOT / "human_scores_corrected.csv")
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    ai = rows(BASELINE / "ai_review_provisional.csv")
    assert len(raw) == len(corrected) == len(ai) == 32
    diffs = [(a["image_id"], key, a[key], b[key]) for a, b in zip(raw, corrected)
             for key in a if a[key] != b[key]]
    assert diffs == [(id_, "spots_visible", "NA", "yes") for id_ in ["D002", "D008", "D027"]]
    yes = [r for r in corrected if r["view_eligible"] == "yes"]
    counted = [r for r in corrected if re.fullmatch(r"[0-9]+", r["spot_count"])]
    uncounted = [r for r in corrected if r["spot_count"] == "NA"]
    assert len(yes) == len(counted) == 27 and len(uncounted) == 5
    assert all(r["spots_visible"] == "yes" for r in yes)
    assert [r["image_id"] for r in uncounted] == ["D009", "D014", "D016", "D024", "D028"]
    counts = [int(r["spot_count"]) for r in counted]
    expected_histogram = {1: 1, 3: 1, 4: 4, 5: 7, 6: 2, 7: 3, 8: 5, 9: 2, 10: 2}
    actual_histogram = {i: counts.count(i) for i in sorted(set(counts))}
    assert actual_histogram == expected_histogram
    assert {int(k): v for k, v in summary["human"]["exact_count_distribution"].items()} == actual_histogram
    assert sum(counts) == 166 and sorted(counts)[13] == 6
    assert abs(summary["human"]["count_mean_among_scorable"] - 166 / 27) < 1e-14
    assert [r["image_id"] for r in counted if r["spot_count"] == "7"] == ["D005", "D010", "D026"]
    numeric_both = [(h["image_id"], int(h["spot_count"]), int(a["spot_count"]))
                    for h, a in zip(corrected, ai) if h["spot_count"] != "NA" and a["spot_count"] != "NA"]
    assert numeric_both == [("D011", 8, 8)]
    assert [r["image_id"] for r in ai] == [r["image_id"] for r in corrected]
    assert sum(h["view_eligible"] == a["view_eligible"] for h, a in zip(corrected, ai)) == 12
    assert sum(h["spots_visible"] == a["spots_visible"] for h, a in zip(corrected, ai)) == 15
    assert sum((h["spot_count"] != "NA") == (a["spot_count"] != "NA") for h, a in zip(corrected, ai)) == 6
    assert len(rows(ROOT / "human_ai_disagreements.csv")) == 30
    errors = []
    rates_checked = 0
    for section in ["human", "ai_provisional_separate"]:
        for value in summary[section].values():
            if isinstance(value, dict) and "wilson_95_interval" in value:
                expected = score_interval(value["numerator"], value["denominator"])
                errors += [abs(a - b) for a, b in zip(expected, value["wilson_95_interval"])]
                assert abs(value["proportion"] - value["numerator"] / value["denominator"]) < 1e-15
                rates_checked += 1
    assert max(errors) < 1e-12
    source_paths = {
        "raw_human_csv": raw_path, "ai_csv": BASELINE / "ai_review_provisional.csv",
        "scoring_rules": BASELINE / "human_review/SCORING_RULES.md",
        "generation_protocol": BASELINE / "protocol.json", "analysis_script": ROOT / "analyze_human_review.py",
    }
    assert all(digest(path) == summary["input_sha256"][key] for key, path in source_paths.items())
    assert digest(ROOT / "human_scores_corrected.csv") == summary["corrected_human_csv_sha256"]
    key = rows(BASELINE / "blind_key.csv")
    generated = {r["image"]: r for r in json.loads((BASELINE / "image_records.json").read_text(encoding="utf-8"))}
    assert len(key) == 32 and {r["image_id"] for r in key} == {r["image_id"] for r in corrected}
    for item in key:
        expected = generated[item["original_image"]]
        assert int(item["seed"]) == expected["seed"]
        assert digest(BASELINE / item["blind_image"]) == digest(BASELINE / item["original_image"]) == item["sha256"] == expected["sha256"]
    links = re.findall(r"\]\(([^)]+)\)", (ROOT / "REPORT.md").read_text(encoding="utf-8"))
    assert all((ROOT / target).is_file() or target == "verification.json" for target in links)
    result = {
        "status": "passed", "human_records": 32, "raw_export_hash_preserved": True,
        "only_authorized_three_visibility_changes": True, "human_eligible": 27,
        "human_exact_count_scorable": 27, "human_exactly_seven": 3, "human_unscorable": 5,
        "exact_histogram_independently_checked": True, "both_raters_numeric_counts": numeric_both,
        "three_category_view_agreement": 12, "disagreement_rows": 30,
        "wilson_intervals_checked_by_numerical_score_inversion": rates_checked,
        "maximum_interval_error": max(errors), "original_and_blind_image_hashes_verified": 32,
        "relative_report_links_checked": len(links), "source_and_analysis_hashes_verified": True,
        "verification_script_sha256": digest(Path(__file__)),
        "summary_sha256": digest(ROOT / "summary.json"),
        "image_scoring_performed": False,
        "interpretation": "Independent arithmetic/code-path verification; not a second human or AI image review",
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
