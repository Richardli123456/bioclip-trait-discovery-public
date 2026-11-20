"""One-time transparent CPU filtering of seven metadata QA failures.

The initial extraction is preserved intact. A fresh run of the current
prepare_expanded_species.py excludes these records before extraction, so this
historical migration script is unnecessary for a from-scratch reproduction.
"""
from pathlib import Path
import json
import numpy as np
from prepare_expanded_species import OUT, sha, write_json, main as prepare_manifest


def main():
    initial = json.loads((OUT / "embedding_provenance.json").read_text(encoding="utf-8"))
    assert initial["status"] == "complete" and initial["shape"] == [363199, 768]
    initial_rows = json.loads((OUT / "data/species.json").read_text(encoding="utf-8"))
    saved = OUT / "before_seven_record_quality_filter"
    saved.mkdir(exist_ok=False)
    (saved / "data").mkdir()
    paths = ["data/species.json", "data/species.csv", "data/embeddings_raw.npy", "data/token_ids.npy",
             "manifest_provenance.json", "embedding_provenance.json", "embedding_progress.json",
             "excluded_records.json", "lineage_conflicts.json", "token_collisions.json"]
    for relative in paths:
        (OUT / relative).rename(saved / relative)
    prepare_manifest()
    rows = json.loads((OUT / "data/species.json").read_text(encoding="utf-8"))
    lookup = {r["taxonomy"]: i for i, r in enumerate(initial_rows)}
    order = np.array([lookup[r["taxonomy"]] for r in rows], dtype=np.int64)
    assert len(rows) == len(initial_rows) - 7
    assert np.array_equal(order[:5388], np.arange(5388))
    np.save(OUT / "data/rows_from_initial_extraction.npy", order)
    old_x = np.load(saved / "data/embeddings_raw.npy", mmap_mode="r")
    old_tokens = np.load(saved / "data/token_ids.npy", mmap_mode="r")
    x = np.lib.format.open_memmap(OUT / "data/embeddings_raw.npy", mode="w+", dtype=np.float32, shape=(len(rows), 768))
    tokens = np.lib.format.open_memmap(OUT / "data/token_ids.npy", mode="w+", dtype=np.int32, shape=(len(rows), 77))
    for begin in range(0, len(rows), 4096):
        idx = order[begin:begin + 4096]
        x[begin:begin + len(idx)] = old_x[idx]
        tokens[begin:begin + len(idx)] = old_tokens[idx]
        assert np.array_equal(x[begin:begin + len(idx)], old_x[idx])
        assert np.array_equal(tokens[begin:begin + len(idx)], old_tokens[idx])
        assert np.isfinite(x[begin:begin + len(idx)]).all()
    x.flush()
    tokens.flush()
    kept = set(order.tolist())
    removed = [r for i, r in enumerate(initial_rows) if i not in kept]
    result = dict(initial)
    result["inputs"] = dict(initial["inputs"], species_json_sha256=sha(OUT / "data/species.json"))
    result["shape"] = [len(rows), 768]
    result["embedding_sha256"] = sha(OUT / "data/embeddings_raw.npy")
    result["token_ids_sha256"] = sha(OUT / "data/token_ids.npy")
    result["post_extraction_quality_filter"] = {
        "initial_extraction_provenance": str((saved / "embedding_provenance.json").resolve()),
        "initial_extraction_provenance_sha256": sha(saved / "embedding_provenance.json"),
        "rows_removed": removed, "method": "Exact taxonomy row selection; no numeric recomputation or model operations",
        "row_mapping_sha256": sha(OUT / "data/rows_from_initial_extraction.npy"),
        "all_rows_bitwise_identical_to_selected_initial_rows": True,
        "filter_script_sha256": sha(__file__),
        "lineage_error_evidence": "generation/experiments/trait_benchmark_v1/confirmed_nonbird_lineage_errors.json and known_source_error_evidence in manifest_provenance.json"}
    previous_collisions = json.loads((saved / "token_collisions.json").read_text(encoding="utf-8"))
    new_index = {int(old): i for i, old in enumerate(order)}
    result["single_row_checks"] = [dict(check, initial_row_index=check["row_index"],
                                        row_index=new_index.get(check["row_index"]))
                                   for check in initial["single_row_checks"]]
    collisions = [[new_index[a], new_index[b]] for a, b in previous_collisions if a in new_index and b in new_index]
    result["token_collision_pairs"] = len(collisions)
    write_json(OUT / "token_collisions.json", collisions)
    write_json(OUT / "embedding_provenance.json", result)
    write_json(OUT / "embedding_progress.json", {"status": "complete", "rows_completed": len(rows), "rows_total": len(rows), "inputs": result["inputs"]})
    write_json(OUT / "quality_filter.json", result["post_extraction_quality_filter"])
    print(json.dumps({"final_shape": result["shape"], "removed": [r["taxonomy"] for r in removed], "embedding_sha256": result["embedding_sha256"], "tokens_collisions": len(collisions)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
