"""Build a conservative species-name manifest from pinned ToL10M metadata.

This is a broader ToL10M expansion, not the unreleased exact ToL1M subset.
No training, image downloads, name resolution, or GPU operations occur here.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cache/expanded_tol10m_v1"
REVISION = "91debffb7146c32c89d76feb1eb575b555e2ecc7"
SOURCE = "embeddings/txt_emb_species.json"
SOURCE_SHA = "844e6fabc06cac072214d566b78f40825b154efa9479eb11285030ca038b2ece"
FIELDS = ["kingdom", "phylum", "class_name", "order", "family", "genus", "epithet"]
KNOWN_SOURCE_ERRORS = {
    "Animalia Arthropoda Insecta Orthoptera Acrididae Celebesia abbotti": "https://eol.org/pages/66999756",
    "Chromista Phaeophyta Phaeophyceae Laminariales Lessoniaceae Lessonia rufa": "https://sib.gob.ar/especies/lessonia-rufa",
    "Protozoa Protozoa Spirotrichea Sporadotrichida Oxytrichidae Himantopus mexicanus": "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=227231",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def download(relative, dest):
    url = f"https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/{REVISION}/{relative}"
    if not dest.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "bioclip-trait-discovery"})
        partial = dest.with_suffix(dest.suffix + ".partial")
        with urllib.request.urlopen(request, timeout=120) as src, partial.open("wb") as f:
            for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
                f.write(chunk)
        partial.rename(dest)
    return {"url": url, "path": str(dest.resolve()), "bytes": dest.stat().st_size, "sha256": sha(dest)}


def main():
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    source_path = OUT / "sources/txt_emb_species.json"
    source = download(SOURCE, source_path)
    assert source["sha256"] == SOURCE_SHA, "Official LFS digest mismatch"
    readme = download("README.md", OUT / "sources/README.md")
    pilot_path = ROOT / "cache/aligned/data/species.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    upstream = json.loads(source_path.read_text(encoding="utf-8"))
    pilot_by_name = {(r["kingdom"], r["scientific_name"]): tuple(r[k] for k in FIELDS) for r in pilot}
    pilot_groups = {}
    for row in pilot:
        group = (row["kingdom"], row["genus"])
        assert group not in pilot_groups or pilot_groups[group] == row["split"]
        pilot_groups[group] = row["split"]
    eligible = defaultdict(dict)
    rejected = []
    exclusions = Counter()
    common_conflicts = 0
    for i, value in enumerate(upstream):
        tax, common = value
        reason = None
        if len(tax) != 7 or not all(isinstance(t, str) for t in tax):
            reason = "invalid_seven_rank_record"
        elif not all(tax):
            reason = "missing_rank"
        elif any(t != t.strip() for t in tax):
            reason = "rank_has_leading_or_trailing_whitespace"
        elif not re.fullmatch(r"[A-Z][a-zA-Z-]*", tax[5]):
            reason = "genus_not_unambiguous_ascii_name"
        elif not re.fullmatch(r"[a-z][a-z-]*", tax[6]):
            reason = "epithet_not_single_ascii_name"
        elif tax[6] in {"sp", "spp", "cf", "aff", "indet", "unidentified", "unknown", "spec", "species", "x"}:
            reason = "unresolved_species_marker"
        elif any(t.lower() in {"unknown", "unclassified", "unidentified", "incertae sedis"} for t in tax):
            reason = "unresolved_rank_marker"
        elif " ".join(tax) in KNOWN_SOURCE_ERRORS:
            reason = "externally_confirmed_lineage_error"
        if reason:
            exclusions[reason] += 1
            rejected.append({"source_index": i, "reason": reason, "taxonomy": tax})
            continue
        tax = tuple(tax)
        key = (tax[0], f"{tax[5]} {tax[6]}")
        if tax in eligible[key]:
            exclusions["exact_taxonomy_duplicate"] += 1
            common_conflicts += eligible[key][tax][1] != common
        else:
            eligible[key][tax] = (i, common)
    conflicts = []
    chosen = {}
    for key, candidates in eligible.items():
        if len(candidates) > 1:
            legacy = pilot_by_name.get(key)
            selected = legacy if legacy in candidates else None
            conflicts.append({"kingdom": key[0], "scientific_name": key[1],
                              "resolution": "original_pilot_lineage_precedence" if selected else "excluded_all_conflicting_lineages",
                              "lineages": [list(t) for t in candidates]})
            for tax, (index, common) in candidates.items():
                if tax != selected:
                    exclusions["conflicting_lineage"] += 1
                    rejected.append({"source_index": index, "reason": "conflicting_lineage", "taxonomy": tax})
            if selected:
                chosen[selected] = candidates[selected]
        else:
            tax, info = next(iter(candidates.items()))
            # A new alternative must never coexist with a different original
            # pilot lineage for the same kingdom and scientific name.
            if key in pilot_by_name and tax != pilot_by_name[key]:
                exclusions["original_pilot_lineage_precedence"] += 1
                rejected.append({"source_index": info[0], "reason": "original_pilot_lineage_precedence", "taxonomy": tax})
            else:
                chosen[tax] = info
    rows = []
    missing_pilot = []
    for i, row in enumerate(pilot):
        row = dict(row)
        tax = tuple(row[k] for k in FIELDS)
        info = chosen.pop(tax, None)
        if info is None:
            missing_pilot.append({"pilot_index": i, "scientific_name": row["scientific_name"]})
        row.update(original_pilot_index=i, source="original_inat21_pilot+tol10m" if info else "original_inat21_pilot",
                   source_index=info[0] if info else None)
        rows.append(row)
    for tax, (index, common) in sorted(chosen.items(), key=lambda item: item[0]):
        row = dict(zip(FIELDS, tax))
        group = (row["kingdom"], row["genus"])
        if group in pilot_groups:
            split = pilot_groups[group]
        else:
            value = int.from_bytes(hashlib.sha256(("expanded_tol10m_v1|" + "|".join(group)).encode()).digest()[:8], "big") / 2**64
            split = "train" if value < .8 else "validation" if value < .9 else "test"
        row.update(scientific_name=f"{tax[5]} {tax[6]}", common_name=common, taxonomy=" ".join(tax),
                   split=split, original_pilot_index=None, source="tol10m_txt_emb_species", source_index=index,
                   source_id=f"tol10m_species_index:{index}")
        rows.append(row)
    split_groups = defaultdict(set)
    genus_kingdoms = defaultdict(set)
    for i, row in enumerate(rows):
        row["row_index"] = i
        # Full lineage identifies records; kingdom+genus is a deliberately
        # coarser split group so disagreements in family cannot cause leakage.
        row["genus_key"] = "|".join(row[k] for k in FIELDS[:-1])
        row["split_group"] = f"{row['kingdom']}|{row['genus']}"
        split_groups[row["split_group"]].add(row["split"])
        genus_kingdoms[row["genus"]].add(row["kingdom"])
    assert all(len(s) == 1 for s in split_groups.values())
    assert len({r["taxonomy"] for r in rows}) == len(rows)
    assert len({(r["kingdom"], r["scientific_name"]) for r in rows}) == len(rows)
    assert all(all(rows[i][k] == p[k] for k in FIELDS + ["taxonomy", "split"]) for i, p in enumerate(pilot))
    out_json = OUT / "data/species.json"
    write_json(out_json, rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "data/species.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / "excluded_records.json", rejected)
    write_json(OUT / "lineage_conflicts.json", conflicts)
    summary = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "dataset_revision": REVISION,
               "scope": "Conservative binomial species-name subset from all released ToL10M fully labeled taxonomy entries, plus all 5388 original iNat21 Animalia pilot rows. Not exact ToL1M training membership or a taxonomically synonym-resolved species census.",
               "source": source, "readme": readme, "script_sha256": sha(__file__), "upstream_records": len(upstream),
               "final_rows": len(rows), "source_exclusions": dict(exclusions), "lineage_conflict_groups": len(conflicts),
               "original_pilot_rows": len(pilot), "original_pilot_sha256": sha(pilot_path),
               "pilot_not_found_in_filtered_source": missing_pilot, "duplicate_common_name_conflicts": common_conflicts,
               "kingdom_counts": dict(Counter(r["kingdom"] for r in rows)), "split_counts": dict(Counter(r["split"] for r in rows)),
               "split_group_counts": dict(Counter(next(iter(v)) for v in split_groups.values())),
               "cross_kingdom_genus_spellings": sum(len(v) > 1 for v in genus_kingdoms.values()),
               "species_json_sha256": sha(out_json), "species_csv_sha256": sha(OUT / "data/species.csv"),
               "known_source_error_evidence": KNOWN_SOURCE_ERRORS,
               "split_rule": "Preserve every original kingdom+genus split; novel kingdom+genus SHA256 deterministic 80/10/10. Full-lineage genus identifiers retained, but grouping is coarser to prevent reclassification leakage.",
               "validation": {"original_5388_rows_and_splits_preserved": True, "unique_taxonomy_strings": True,
                              "unique_kingdom_binomials": True, "split_groups_disjoint": True},
               "caveats": ["Synonyms with different binomials are not merged without accepted-name identifiers; this can leak related taxa across genus holdouts.",
                           "Source warns some hemihomonym lineages are incorrect; we detect explicit conflicts, not all source mislabels.",
                           "Strict ASCII binomial filtering excludes some legitimate hybrids, infraspecies, diacritics and author-bearing labels rather than guessing a canonical name.",
                           "Broader all-kingdom composition changes along with dataset size; improved stability cannot be attributed to count alone.",
                           "All 5388 previous pilot rows retain their original taxonomy even where upstream conflicts are present."]}
    write_json(OUT / "manifest_provenance.json", summary)
    print(json.dumps({k: summary[k] for k in ["final_rows", "upstream_records", "source_exclusions", "kingdom_counts", "split_counts", "split_group_counts", "lineage_conflict_groups"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
