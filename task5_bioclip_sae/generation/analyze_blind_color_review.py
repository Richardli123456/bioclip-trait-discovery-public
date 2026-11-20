"""Unblind returned colour scores and tabulate them by feature and condition.

Usage: analyze_blind_color_review.py scores_A.csv [scores_B.csv ...]
Reads the package key, joins on image_id, and writes results.json / RESULTS.md beside the key.
An image counts as showing the target colour when the scorer ticked that colour; images marked
"no / unclear animal" or left unanswered are reported separately and excluded from the rate.
"""
from pathlib import Path
import argparse
import csv
import json
from collections import defaultdict

from experiment_common import ROOT, write_json

ORDER = {"add": ["add_original", "add_high", "add_random"], "remove": ["remove_original", "removed", "remove_random"]}
LABELS = {"add_original": "original", "add_high": "feature added (strongest)", "add_random": "random nudge",
          "remove_original": "original", "removed": "feature removed", "remove_random": "random nudge"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="+")
    parser.add_argument("--package", type=Path, default=ROOT / "experiments/blind_colour_review_v1")
    args = parser.parse_args()
    key_dir = args.package / "KEY_do_not_send"
    with (key_dir / "key.csv").open(encoding="utf-8") as stream:
        key = {r["image_id"]: r for r in csv.DictReader(stream)}
    colours = sorted({r["target_colour"] for r in key.values() if r["target_colour"]})
    raters = {}
    for path in args.scores:
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert {r["image_id"] for r in rows} <= set(key), f"{path}: unknown image IDs"
        name = next((r["rater"] for r in rows if r["rater"]), path.stem)
        assert name not in raters, f"Duplicate rater name {name}"
        raters[name] = {r["image_id"]: r for r in rows}

    results = {}
    for rater, answers in raters.items():
        table = {}
        for colour in colours:
            for part, conditions in ORDER.items():
                for condition in conditions:
                    # Shared add originals have no feature; they are the baseline for every colour.
                    ids = [i for i, r in key.items() if r["condition_type"] == condition
                           and (r["target_colour"] == colour or condition == "add_original")]
                    scored = [i for i in ids if answers.get(i, {}).get("clear_animal") == "yes"]
                    unclear = [i for i in ids if answers.get(i, {}).get("clear_animal") == "no"]
                    missing = len(ids) - len(scored) - len(unclear)
                    hits = sum(answers[i][colour] == "1" for i in scored)
                    table[f"{colour}:{condition}"] = {"images": len(ids), "scored": len(scored), "unclear": len(unclear),
                                                      "unanswered": missing, "target_colour": hits,
                                                      "rate": hits / len(scored) if scored else None}
        results[rater] = table

    agreement = None
    if len(raters) >= 2:
        names = list(raters)
        pairs = [(i, c) for i, r in key.items() for c in colours if r["target_colour"] in (c, "")]
        both = [(i, c) for i, c in pairs if all(raters[n].get(i, {}).get("clear_animal") == "yes" for n in names[:2])]
        same = sum(raters[names[0]][i][c] == raters[names[1]][i][c] for i, c in both)
        agreement = {"raters": names[:2], "compared_judgements": len(both), "agreement": same / len(both) if both else None}

    write_json(key_dir / "results.json", {"raters": results, "agreement": agreement,
                                          "scores": [str(p) for p in args.scores]})
    lines = ["# Blind colour scoring results", "",
             "Share of clearly visible animals the scorer marked with the target colour. Unclear and unanswered images are excluded.", ""]
    for rater, table in results.items():
        lines += [f"## Scorer: {rater}", "", "| Colour | Edit | Original | Edited | Random nudge |", "|---|---|---:|---:|---:|"]
        for colour in colours:
            for part, conditions in ORDER.items():
                cells = []
                for condition in conditions:
                    v = table[f"{colour}:{condition}"]
                    cells.append(f"{v['target_colour']}/{v['scored']}" + (f" ({v['rate']:.0%})" if v["scored"] else ""))
                lines.append(f"| {colour} | {'added' if part == 'add' else 'removed'} | " + " | ".join(cells) + " |")
        lines.append("")
    if agreement:
        lines.append(f"Agreement between {agreement['raters'][0]} and {agreement['raters'][1]} on target-colour ticks: "
                     f"{agreement['agreement']:.0%} of {agreement['compared_judgements']} judgements.")
    (key_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
