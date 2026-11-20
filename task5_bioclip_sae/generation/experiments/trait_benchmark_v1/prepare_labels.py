"""Extract independently measured AVONET traits; no model fitting or GPU use."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://ndownloader.figshare.com/files/34480856"
PAGE = "https://figshare.com/articles/dataset/16586228/7"
MD5 = "1445afdcfb6df784010c2ca034544bc8"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_sheet(path, sheet):
    with zipfile.ZipFile(path) as archive:
        ss = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(t.itertext()) for t in ss.findall("x:si", NS)]
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
        parsed = []
        for row in root.findall("x:sheetData/x:row", NS):
            values = {}
            for cell in row:
                value = cell.find("x:v", NS)
                col = re.sub(r"\d", "", cell.attrib["r"])
                if value is not None:
                    values[col] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            parsed.append((int(row.attrib["r"]), values))
        header = parsed[0][1]
        return [(rownum, {name: values.get(col, "") for col, name in header.items()})
                for rownum, values in parsed[1:]]


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    path = ROOT / "AVONET_Supplementary_dataset_1.xlsx"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    assert hashlib.md5(path.read_bytes()).hexdigest() == MD5, "Source checksum mismatch"
    raw = read_sheet(path, 3)  # Fixed eBird taxonomy, selected before fitting.
    labels, exclusions = [], []
    for rownum, row in raw:
        name = row["Species2"]
        reason = None
        try:
            length = float(row["Beak.Length_Nares"])
            depth = float(row["Beak.Depth"])
            complete = float(row["Complete.measures"])
        except ValueError:
            reason = "missing_or_non_numeric_measurement"
        if reason is None:
            if not all(math.isfinite(v) for v in [length, depth, complete]) or min(length, depth) <= 0:
                reason = "invalid_measurement"
            elif row["Inference"] != "NO":
                reason = "some_morphology_inferred_conservative_exclusion"
            elif complete < 4:
                reason = "fewer_than_four_complete_specimens"
            elif len(name.split()) != 2:
                reason = "not_binomial"
        if reason:
            exclusions.append({"scientific_name": name, "source_row": rownum, "reason": reason})
            continue
        labels.append({
            "scientific_name": name, "genus": name.split()[0], "avonet_family": row["Family2"],
            "avonet_order": row["Order2"], "beak_length_nares_mm": length,
            "beak_depth_mm": depth, "trait_value": math.log(length / depth),
            "trait_id": "ln_beak_nares_length_over_depth", "complete_specimens": int(complete),
            "total_individuals": row["Total.individuals"], "inference": row["Inference"],
            "source_sheet": "AVONET2_eBird", "source_row": rownum, "source_url": PAGE,
        })
    assert len({r["scientific_name"] for r in labels}) == len(labels)
    write_csv(ROOT / "labels.csv", labels)
    write_csv(ROOT / "label_exclusions.csv", exclusions)
    metadata = {
        "dataset": "AVONET Supplementary dataset 1", "article_version": 7,
        "source_url": PAGE, "download_url": URL, "license": "CC BY 4.0",
        "authors": "Tobias et al. (2022)", "paper_doi": "10.1111/ele.13898",
        "source_md5": MD5, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sheet": "AVONET2_eBird", "raw_rows": len(raw), "eligible_rows": len(labels),
        "genera": len({r["genus"] for r in labels}),
        "families": len({r["avonet_family"] for r in labels}),
        "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
        "trait": "Natural logarithm of species mean beak length from nares divided by species mean beak depth",
        "measurement_geometry": "Length is nostril anterior edge to tip; depth is at nostril anterior edge; both millimetres",
        "derivation": "Ratio of published species means; not mean individual ratio; larger values mean more elongated beaks",
        "inclusion_rule": "Exact binomial; finite positive measurements; Inference=NO; Complete.measures >= 4",
        "selection_timing": "Trait, source taxonomy and inclusion rules selected before any embedding probe fitting",
        "limits": ["Independent labels are not proof of training-data independence from BioCLIP.",
                   "Species means are not image-specific measurements or sex/age-specific labels.",
                   "Exact-name joins can retain taxonomic concept differences; no automatic synonyms are introduced.",
                   "This benchmark does not validate ladybird spot counts or an SAE intervention."]
    }
    (ROOT / "label_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
