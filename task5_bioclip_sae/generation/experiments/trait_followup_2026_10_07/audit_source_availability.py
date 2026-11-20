"""Count availability for candidate AVONET endpoints; never load model outputs."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "trait_benchmark_v1"
spec = importlib.util.spec_from_file_location("avonet_source_reader", SOURCE / "prepare_labels.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
workbook = SOURCE / "AVONET_Supplementary_dataset_1.xlsx"
assert hashlib.md5(workbook.read_bytes()).hexdigest() == "1445afdcfb6df784010c2ca034544bc8"
raw = reader.read_sheet(workbook, 3)
with (SOURCE / "evaluation_expanded_v2" / "joined_labels.csv").open(encoding="utf-8") as stream:
    # Read only identity/split/taxonomy columns from the earlier exact-name join.
    joined = {r["scientific_name"]: {k: r[k] for k in ["genus", "split", "family", "order"]}
              for r in csv.DictReader(stream)}
audit = {
    "scope": "Measurement availability only; no embedding, SAE association, prediction, or new endpoint outcome analyzed.",
    "source_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
    "source_url": "https://figshare.com/articles/dataset/16586228/7",
    "source_sheet": "AVONET2_eBird",
    "source_rows": len(raw),
    "candidate_endpoints": {},
}
for endpoint, fields in {
    "ln_tail_length_over_wing_length": ["Tail.Length", "Wing.Length"],
    "ln_tarsus_length_over_wing_length": ["Tarsus.Length", "Wing.Length"],
    "hand_wing_index": ["Hand-Wing.Index"],
}.items():
    eligible, exclusions = [], Counter()
    for _, row in raw:
        try:
            measurements = [float(row[field]) for field in fields]
            complete = float(row["Complete.measures"])
        except ValueError:
            exclusions["nonnumeric_measurement"] += 1
            continue
        if not all(math.isfinite(v) and v > 0 for v in measurements) or not math.isfinite(complete):
            exclusions["nonpositive_or_nonfinite"] += 1
        elif row["Inference"] != "NO":
            exclusions["some_morphology_inferred"] += 1
        elif complete < 4:
            exclusions["fewer_than_four_complete_specimens"] += 1
        elif len(row["Species2"].split()) != 2:
            exclusions["not_binomial"] += 1
        else:
            eligible.append(row)
    matched = [joined[row["Species2"]] for row in eligible if row["Species2"] in joined]
    audit["candidate_endpoints"][endpoint] = {
        "source_fields": fields,
        "source_eligible_species": len(eligible),
        "source_genera": len({row["Species2"].split()[0] for row in eligible}),
        "source_families": len({row["Family2"] for row in eligible}),
        "exclusions": dict(exclusions),
        "matched_using_existing_exact_bird_join": len(matched),
        "matched_species_by_split": dict(Counter(row["split"] for row in matched)),
        "matched_genera_by_split": {s: len({row["genus"] for row in matched if row["split"] == s})
                                     for s in ["train", "validation", "test"]},
        "existing_join_species_missing_new_measurements": len(joined) - len(matched),
    }
    assert len(eligible) + sum(exclusions.values()) == len(raw)
    assert len(matched) == len(joined) == 8187
audit["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT / "source_availability_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
