"""Conservatively remove cross-kingdom binomial conflicts without re-encoding.

v1 is immutable and remains the extraction provenance. Exact scientific-name
duplicates across kingdoms are withheld unless the original pilot fixes a
known lineage; in that case only the pilot row is retained. This is an
exclusion policy, not an automated taxonomy correction or synonym resolution.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
import numpy as np

from prepare_expanded_species import ROOT, sha, write_json

SOURCE = ROOT / "cache/expanded_tol10m_v1"
OUT = ROOT / "cache/expanded_tol10m_v2"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "data").mkdir()
    source_species = SOURCE / "data/species.json"
    source_provenance = json.loads((SOURCE / "manifest_provenance.json").read_text(encoding="utf-8"))
    original = json.loads(source_species.read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for row in original:
        by_name[row["scientific_name"]].append(row)
    remove, conflicts = set(), []
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        known = [r for r in group if r["original_pilot_index"] is not None]
        assert len(known) <= 1
        retained = known[0]["row_index"] if known else None
        dropped = [r["row_index"] for r in group if r["row_index"] != retained]
        remove.update(dropped)
        conflicts.append({"scientific_name": name, "source_rows": group,
                          "retained_v1_row_index": retained, "excluded_v1_row_indices": dropped,
                          "policy": "Retain original pilot lineage" if known else "Withhold all ambiguous cross-kingdom versions",
                          "cross_split_before_exclusion": len({r["split"] for r in group}) > 1})
    keep = np.array([i for i in range(len(original)) if i not in remove], dtype=np.int64)
    assert np.array_equal(keep[:5388], np.arange(5388))
    rows = []
    for new_index, old_index in enumerate(keep):
        row = dict(original[int(old_index)])
        row["row_index"] = new_index
        rows.append(row)
    assert len({r["scientific_name"] for r in rows}) == len(rows)
    assert len({r["taxonomy"] for r in rows}) == len(rows)
    assert all(all(rows[i][k] == original[i][k] for k in original[i]) for i in range(5388))
    split_groups = defaultdict(set)
    for row in rows:
        split_groups[row["split_group"]].add(row["split"])
    assert all(len(s) == 1 for s in split_groups.values())
    write_json(OUT / "data/species.json", rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "data/species.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    np.save(OUT / "data/rows_from_v1.npy", keep)
    old_x = np.load(SOURCE / "data/embeddings_raw.npy", mmap_mode="r")
    old_tokens = np.load(SOURCE / "data/token_ids.npy", mmap_mode="r")
    x = np.lib.format.open_memmap(OUT / "data/embeddings_raw.npy", mode="w+", dtype=np.float32, shape=(len(rows), 768))
    tokens = np.lib.format.open_memmap(OUT / "data/token_ids.npy", mode="w+", dtype=np.int32, shape=(len(rows), 77))
    norm_min, norm_max = float("inf"), 0.
    for start in range(0, len(rows), 4096):
        ids = keep[start:start+4096]
        x[start:start+len(ids)] = old_x[ids]
        tokens[start:start+len(ids)] = old_tokens[ids]
        assert np.array_equal(x[start:start+len(ids)], old_x[ids])
        assert np.array_equal(tokens[start:start+len(ids)], old_tokens[ids])
        assert np.isfinite(x[start:start+len(ids)]).all()
        norms = np.linalg.norm(x[start:start+len(ids)], axis=1)
        norm_min, norm_max = min(norm_min, float(norms.min())), max(norm_max, float(norms.max()))
    x.flush()
    tokens.flush()
    # A genus spelling can validly be shared across kingdoms. Report these
    # records and shared vernacular names for future manual review, but do not
    # invent accepted-name mappings or move existing genus partitions.
    by_genus, by_common = defaultdict(list), defaultdict(list)
    for row in rows:
        by_genus[row["genus"]].append(row)
        common = row["common_name"].strip().casefold()
        if common:
            by_common[common].append(row)
    genus_ambiguities = [{"genus_spelling": name, "kingdoms": sorted({r["kingdom"] for r in group}),
                          "species_count": len(group), "row_indices": [r["row_index"] for r in group]}
                         for name, group in by_genus.items() if len({r["kingdom"] for r in group}) > 1]
    common_ambiguities = [{"common_name_casefolded": name,
                           "rows": [{k:r[k] for k in ["row_index", "scientific_name", "taxonomy", "split"]} for r in group]}
                          for name, group in by_common.items() if len({r["kingdom"] for r in group}) > 1]
    write_json(OUT / "cross_kingdom_binomial_exclusions.json", conflicts)
    write_json(OUT / "remaining_cross_kingdom_genus_names.json", genus_ambiguities)
    write_json(OUT / "remaining_cross_kingdom_common_names.json", common_ambiguities)
    exclusions = dict(source_provenance["source_exclusions"])
    exclusions["cross_kingdom_binomial_conflict"] = len(remove)
    provenance = {**source_provenance, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "version": "expanded_tol10m_v2", "supersedes_for_scientific_analysis": str(SOURCE.resolve()),
                  "previous_manifest_sha256": sha(source_species), "script_sha256": sha(__file__),
                  "final_rows": len(rows), "source_exclusions": exclusions,
                  "kingdom_counts": dict(Counter(r["kingdom"] for r in rows)),
                  "split_counts": dict(Counter(r["split"] for r in rows)),
                  "split_group_counts": dict(Counter(next(iter(v)) for v in split_groups.values())),
                  "cross_kingdom_genus_spellings": len(genus_ambiguities),
                  "cross_kingdom_binomial_ambiguous_groups_removed": len(conflicts),
                  "cross_kingdom_binomial_rows_removed": len(remove),
                  "original_pilot_lineages_retained_in_conflict_groups": sum(c["retained_v1_row_index"] is not None for c in conflicts),
                  "remaining_cross_kingdom_common_name_groups": len(common_ambiguities),
                  "species_json_sha256": sha(OUT / "data/species.json"), "species_csv_sha256": sha(OUT / "data/species.csv"),
                  "validation": {"original_5388_rows_and_splits_preserved": True, "globally_unique_binomials": True,
                                 "unique_taxonomy_strings": True, "split_groups_disjoint": True},
                  "additional_policy": "Global binomial ambiguity is excluded, except known pilot lineage precedence. Remaining genus spelling homonyms and shared common names are reported for manual review; no automatic relabeling or original split changes.",
                  "caveats": source_provenance["caveats"] + ["Unique scientific-name strings do not prove all biological species identities or higher lineages are correct. Shared vernacular names alone cannot distinguish true homonyms from source errors."]}
    write_json(OUT / "manifest_provenance.json", provenance)
    old_provenance = json.loads((SOURCE / "embedding_provenance.json").read_text(encoding="utf-8"))
    new_indices = {int(old): i for i, old in enumerate(keep)}
    checks = []
    for check in old_provenance["single_row_checks"]:
        new_check = dict(check)
        new_check["v1_row_index"] = check["row_index"]
        new_check["row_index"] = new_indices.get(check["row_index"])
        checks.append(new_check)
    embedding_provenance = {**old_provenance, "shape": [len(rows), 768], "version": "expanded_tol10m_v2",
                            "inputs": {**old_provenance["inputs"], "species_json_sha256": provenance["species_json_sha256"]},
                            "embedding_sha256": sha(OUT / "data/embeddings_raw.npy"),
                            "token_ids_sha256": sha(OUT / "data/token_ids.npy"), "single_row_checks": checks,
                            "nonzero_norm_range": [norm_min, norm_max],
                            "v2_quality_filter": {"method": "CPU exact row selection; zero GPU/encoder/training operations",
                                                  "all_kept_values_bitwise_identical_to_v1": True,
                                                  "row_map_sha256": sha(OUT / "data/rows_from_v1.npy"),
                                                  "source_embedding_provenance": str((SOURCE / "embedding_provenance.json").resolve()),
                                                  "source_embedding_provenance_sha256": sha(SOURCE / "embedding_provenance.json"),
                                                  "source_embedding_sha256": old_provenance["embedding_sha256"],
                                                  "script_sha256": sha(__file__)}}
    write_json(OUT / "embedding_provenance.json", embedding_provenance)
    write_json(OUT / "token_collisions.json", [])
    write_json(OUT / "embedding_progress.json", {"status": "complete", "rows_total": len(rows), "rows_completed": len(rows), "inputs": embedding_provenance["inputs"]})
    assert sha(source_species) == provenance["previous_manifest_sha256"]
    print(json.dumps({"final_rows": len(rows), "removed_rows": len(remove), "conflict_groups": len(conflicts),
                      "split_counts": provenance["split_counts"], "kingdom_counts": provenance["kingdom_counts"],
                      "remaining_cross_kingdom_genus_names": len(genus_ambiguities),
                      "remaining_cross_kingdom_common_names": len(common_ambiguities),
                      "species_sha256": provenance["species_json_sha256"], "embedding_sha256": embedding_provenance["embedding_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
