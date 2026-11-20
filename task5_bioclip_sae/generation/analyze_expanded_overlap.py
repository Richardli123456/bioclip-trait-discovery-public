"""Exact binary-support neighbors and residual-preserving bridge plans.

CPU self-test/plan preparation are separate from explicitly authorized GPU
execution. No training or image generation is performed by this script.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import time
import traceback

import numpy as np
import torch

from experiment_common import ROOT, sha, write_json
from run_expanded_sae_study import Guard


def support_matrix(indices, values, width, device):
    idx = torch.from_numpy(np.array(indices, dtype=np.int64)).to(device)
    binary = torch.from_numpy(np.array(values > 1e-7, dtype=np.float16)).to(device)
    return torch.zeros((len(indices), width), dtype=torch.float16, device=device).scatter_(1, idx, binary)


@torch.inference_mode()
def exact_batch(binary, start, end, count):
    n = len(binary)
    intersection = binary[start:end] @ binary.T
    # Every dot product is the sum of <=k binary 1s and is exactly representable
    # in float16 for k<=32. Use integer ranking to impose exact index tie breaks.
    shared = intersection.to(torch.int32)
    assert bool((intersection == shared).all())
    rank = shared.to(torch.int64) * (n + 1) + (n - torch.arange(n, device=binary.device))[None, :]
    rank[torch.arange(end-start, device=binary.device), torch.arange(start, end, device=binary.device)] = -1
    neighbors = rank.topk(count, dim=1).indices
    overlaps = shared.gather(1, neighbors)
    return neighbors.cpu().numpy().astype(np.int32), overlaps.cpu().numpy().astype(np.uint8)


def cpu_expected(indices, values, query, count, width):
    active = [set(map(int, i[v > 1e-7])) for i, v in zip(indices, values)]
    score = np.array([len(active[query] & a) for a in active], dtype=np.int32)
    score[query] = -1
    order = np.lexsort((np.arange(len(score)), -score))[:count]
    return order, score[order]


def self_test():
    rng = np.random.default_rng(63)
    n, width, k, count = 35, 13, 4, 10
    indices = np.array([rng.choice(width, k, replace=False) for _ in range(n)], dtype=np.int32)
    values = rng.uniform(.1, 3, (n, k)).astype(np.float32)
    values[0] = 0
    indices[2], values[2] = indices[1], values[1]
    values[3, :2] = 0
    binary = support_matrix(indices, values, width, "cpu")
    for start in range(0, n, 7):
        end = min(start+7, n)
        got_i, got_v = exact_batch(binary, start, end, count)
        for j in range(start, end):
            expected_i, expected_v = cpu_expected(indices, values, j, count, width)
            assert np.array_equal(got_i[j-start], expected_i)
            assert np.array_equal(got_v[j-start], expected_v)
    print(json.dumps({"self_test": "passed", "queries_exhaustively_checked": n,
                      "cases": ["ties", "identical_codes", "all_zero_code", "partly_zero_code", "self_exclusion"]}), flush=True)


def plan(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rows_path = args.input_dir / "data/species.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    paths = [rows_path, args.input_dir / "data/embeddings_raw.npy", args.anchor_dir / "code_indices.npy",
             args.anchor_dir / "code_values.npy", args.anchor_dir / "sae.pt"]
    checkpoint = torch.load(paths[-1], map_location="cpu", weights_only=True)
    indices = np.load(paths[2], mmap_mode="r")
    values = np.load(paths[3], mmap_mode="r")
    width, k = checkpoint["config"]["width"], checkpoint["config"]["k"]
    assert indices.shape == values.shape == (len(rows), k)
    assert 0 < args.batch <= 512 and 0 < args.count < len(rows) and k <= 32
    plan_value = {"status": "prespecified_before_neighbor_calculation", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "input_dir": str(args.input_dir.resolve()), "anchor_dir": str(args.anchor_dir.resolve()),
                  "input_hashes": {str(p.resolve()): sha(p) for p in paths}, "species_count": len(rows),
                  "width": width, "k": k, "count": args.count, "query_batch": args.batch,
                  "export_full_csv": args.full_csv,
                  "binary_threshold": "activation >1e-7", "ranking": "Descending count of shared active coordinates; self excluded; ascending row_index breaks ties",
                  "jaccard": "Reported separately; intersection/union, defined zero for an empty union. Ranking by shared count is Jaccard-equivalent only when active counts are equal.",
                  "exactness": "float16 binary GEMM sums <=32 integers exactly; integer ranks impose deterministic ties; 128 queries independently verified with CPU postings",
                  "bridge_sources": args.bridge_sources, "bridge_kind": "residual_preserving_source",
                  "bridge_order": "Descending abs(target_code-source_code) times decoder column norm times global scale; feature index breaks ties; replace one coordinate per step; no re-TopK",
                  "bridge_endpoint": "source_raw + decoder(target_code)-decoder(source_code); source residual retained. Raw and reconstructed target controls are separately saved.",
                  "interpretation": "Sparse support overlap and bridge geometry are exploratory mathematical quantities, not evidence of biological or perceptual similarity.",
                  "script_sha256": sha(__file__), "hardware": "Stop on every exception/OOM, >=88C or <2GiB free; no automatic retries"}
    write_json(args.out / "protocol.json", plan_value)
    print(json.dumps({"plan": str(args.out / "protocol.json"), "queries": len(rows), "gpu_started": False}), flush=True)


def audit_postings(indices, values, neighbors, overlaps, width):
    n = len(indices)
    # Inverted postings provide an independent sparse CPU implementation.
    mask = np.asarray(values > 1e-7).reshape(-1)
    flat_features = np.asarray(indices).reshape(-1)[mask]
    flat_rows = np.repeat(np.arange(n, dtype=np.int32), indices.shape[1])[mask]
    sort_order = np.argsort(flat_features, kind="stable")
    sorted_rows = flat_rows[sort_order]
    boundaries = np.concatenate([[0], np.cumsum(np.bincount(flat_features, minlength=width))])
    postings = [sorted_rows[boundaries[j]:boundaries[j+1]] for j in range(width)]
    del mask, flat_features, flat_rows, sort_order
    selected = sorted(set([0, 5387, n-1] + np.random.default_rng(20261004).choice(n, size=min(125, n), replace=False).tolist()))
    for query in selected:
        features = indices[query][values[query] > 1e-7]
        score = np.zeros(n, dtype=np.int32)
        for feature in features:
            score[postings[int(feature)]] += 1
        score[query] = -1
        expected = np.lexsort((np.arange(n), -score))[:neighbors.shape[1]]
        assert np.array_equal(neighbors[query], expected), f"Ranking mismatch at query {query}"
        assert np.array_equal(overlaps[query], score[expected])
    return {"status": "passed", "independently_verified_queries": len(selected), "query_indices": selected,
            "method": "CPU inverted postings, integer intersection counts, lexsort by negative count then row index"}


def bridge_plan(out, source_name, rows, embeddings, indices, values, neighbors, checkpoint):
    matches = [i for i, row in enumerate(rows) if row["kingdom"] == "Animalia" and row["scientific_name"] == source_name]
    assert len(matches) == 1, f"Ambiguous/missing bridge source {source_name}"
    source = matches[0]
    target = int(neighbors[source, 0])
    width = checkpoint["config"]["width"]
    state = checkpoint["state_dict"]
    decoder = state["decoder.weight"].numpy()
    mean = state["mean"].numpy()
    scale = float(state["scale"])
    bias = state["output_bias"].numpy()
    zs, zt = np.zeros(width, dtype=np.float32), np.zeros(width, dtype=np.float32)
    zs[indices[source]] = values[source]
    zt[indices[target]] = values[target]
    differences = np.flatnonzero(zt != zs)
    magnitude = np.abs(zt-zs) * np.linalg.norm(decoder, axis=0) * scale
    order = sorted(map(int, differences), key=lambda f: (-float(magnitude[f]), f))
    code_states = np.empty((len(order)+1, width), dtype=np.float32)
    code_states[0] = zs
    for step, feature in enumerate(order, 1):
        code_states[step] = code_states[step-1]
        code_states[step, feature] = zt[feature]
        assert np.count_nonzero(code_states[step] != code_states[step-1]) == 1
    assert np.array_equal(code_states[-1], zt)
    raw_source, raw_target = np.array(embeddings[source]), np.array(embeddings[target])
    source_recon = (decoder @ zs + bias) * scale + mean
    target_recon = (decoder @ zt + bias) * scale + mean
    # Difference form avoids numerical round-off in the unedited source endpoint.
    edited = raw_source[None, :] + (code_states-zs) @ decoder.T * scale
    assert np.array_equal(edited[0], raw_source)
    assert np.allclose(edited[-1], raw_source + target_recon-source_recon, atol=1e-5, rtol=1e-5)
    destination = out / "bridges" / source_name.replace(" ", "_")
    destination.mkdir(parents=True, exist_ok=False)
    controls = np.stack([raw_source, raw_target, source_recon, target_recon])
    np.save(destination / "code_states.npy", code_states)
    np.save(destination / "vectors.npy", edited)
    np.save(destination / "endpoint_controls.npy", controls)
    steps = [{"step": step, "changed_feature": None if step == 0 else order[step-1],
              "active_count_above_1e-7": int(np.count_nonzero(code > 1e-7)),
              "remaining_code_l1": float(np.abs(code-zt).sum())} for step, code in enumerate(code_states)]
    result = {"source": rows[source], "target": rows[target], "steps": steps, "coordinates_changed": len(order),
              "one_coordinate_per_step_verified": True, "exact_target_code_reached": True,
              "raw_source_start_verified": True, "source_residual_preserved": True,
              "endpoint_controls_order": ["source_original", "target_original", "source_reconstruction", "target_reconstruction"],
              "target_embedding_distance_from_bridge_endpoint": float(np.linalg.norm(edited[-1]-raw_target)),
              "limitation": "Exact target SAE code with the source residual is generally not the raw target embedding. No biological or generated-image result is established by this plan.",
              "sha256": {p.name: sha(p) for p in destination.glob("*.npy")}}
    write_json(destination / "protocol.json", result)
    return {"source": source_name, "target": rows[target]["scientific_name"], "coordinates_changed": len(order), "path": str(destination.resolve())}


def execute(out):
    started = time.perf_counter()
    protocol_path = out / "protocol.json"
    assert list(out.iterdir()) == [protocol_path], "Do not overwrite or automatically resume"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert sha(__file__) == protocol["script_sha256"]
    assert {p: sha(p) for p in protocol["input_hashes"]} == protocol["input_hashes"]
    guard = Guard(out)
    try:
        guard("before_neighbor_load")
        if not torch.cuda.is_available():
            raise RuntimeError("Hardware stop: CUDA unavailable")
        torch.set_num_threads(8)
        input_dir, anchor = Path(protocol["input_dir"]), Path(protocol["anchor_dir"])
        rows = json.loads((input_dir / "data/species.json").read_text(encoding="utf-8"))
        indices = np.load(anchor / "code_indices.npy", mmap_mode="r")
        values = np.load(anchor / "code_values.npy", mmap_mode="r")
        n, width = len(rows), protocol["width"]
        binary = torch.zeros((n, width), dtype=torch.float16, device="cuda")
        active_count = np.empty(n, dtype=np.int16)
        for begin in range(0, n, 4096):
            end = min(begin+4096, n)
            idx, val = indices[begin:end], values[begin:end]
            assert np.isfinite(val).all() and (val >= 0).all() and (idx >= 0).all() and (idx < width).all()
            assert all(len(set(map(int, row))) == len(row) for row in idx), "Duplicate stored feature IDs"
            binary[begin:end] = support_matrix(idx, val, width, "cuda")
            active_count[begin:end] = (val > 1e-7).sum(1)
        neighbors = np.lib.format.open_memmap(out / "neighbor_indices.npy", mode="w+", dtype=np.int32, shape=(n, protocol["count"]))
        overlaps = np.lib.format.open_memmap(out / "shared_active.npy", mode="w+", dtype=np.uint8, shape=neighbors.shape)
        for begin in range(0, n, protocol["query_batch"]):
            end = min(n, begin+protocol["query_batch"])
            if begin % (protocol["query_batch"] * 16) == 0:
                guard(f"neighbor_queries_{begin}")
            neighbors[begin:end], overlaps[begin:end] = exact_batch(binary, begin, end, protocol["count"])
            if begin % (protocol["query_batch"] * 64) == 0:
                print(json.dumps({"completed_queries": end, "total": n, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        neighbors.flush()
        overlaps.flush()
        guard("after_gpu_neighbors")
        del binary
        torch.cuda.empty_cache()
        audit = audit_postings(indices, values, neighbors, overlaps, width)
        write_json(out / "independent_audit.json", audit)
        np.save(out / "active_counts.npy", active_count)
        selected_sources = [i for i, row in enumerate(rows) if row["kingdom"] == "Animalia" and row["scientific_name"] in protocol["bridge_sources"]]
        export_neighbors(out / "selected_neighbors.csv", selected_sources, rows, neighbors, overlaps, active_count)
        if protocol["export_full_csv"]:
            export_neighbors(out / "neighbors.csv.gz", range(n), rows, neighbors, overlaps, active_count)
        checkpoint = torch.load(anchor / "sae.pt", map_location="cpu", weights_only=True)
        embeddings = np.load(input_dir / "data/embeddings_raw.npy", mmap_mode="r")
        bridges = [bridge_plan(out, name, rows, embeddings, indices, values, neighbors, checkpoint) for name in protocol["bridge_sources"]]
        bridge_vectors, records = [], []
        for bridge in bridges:
            location = Path(bridge["path"])
            details = json.loads((location / "protocol.json").read_text(encoding="utf-8"))
            for condition, vector in zip(details["endpoint_controls_order"], np.load(location / "endpoint_controls.npy")):
                records.append({"vector_index": len(bridge_vectors), "source": bridge["source"], "target": bridge["target"],
                                "condition": condition, "step": None, "edited_feature": None})
                bridge_vectors.append(vector)
            for step, vector in zip(details["steps"], np.load(location / "vectors.npy")):
                records.append({"vector_index": len(bridge_vectors), "source": bridge["source"], "target": bridge["target"],
                                "condition": "residual_preserving_bridge", "step": step["step"], "edited_feature": step["changed_feature"]})
                bridge_vectors.append(vector)
            bridge["residual_policy"] = "Retain source_raw - decode(source_code) for every step; target_original and target_reconstruction are separate controls"
            bridge["detailed_protocol_sha256"] = sha(location / "protocol.json")
        np.save(out / "bridge_vectors.npy", np.stack(bridge_vectors))
        write_json(out / "bridge_plan.json", {"status": "ready_for_exploratory_image_generation", "records": records, "bridges": bridges,
                   "input_hashes": protocol["input_hashes"], "bridge_vectors_sha256": sha(out / "bridge_vectors.npy"),
                   "interpretation": protocol["interpretation"], "feature_gate": "Bridge geometry is exploratory and does not validate any individual feature as a trait"})
        summary = {"status": "completed", "species_count": n, "neighbors_per_species": protocol["count"],
                   "edges_in_arrays": n*protocol["count"], "rows_in_full_csv": n*protocol["count"] if protocol["export_full_csv"] else 0,
                   "closest_shared_active_mean": float(overlaps[:, 0].mean()),
                   "closest_same_genus_and_kingdom_fraction": float(np.mean([rows[i]["split_group"] == rows[int(j)]["split_group"] for i, j in enumerate(neighbors[:, 0])])),
                   "independent_cpu_audit": audit, "bridge_plans": bridges, "elapsed_seconds": time.perf_counter()-started,
                   "inputs_unchanged": {p: sha(p) for p in protocol["input_hashes"]} == protocol["input_hashes"],
                   "hardware_errors": [], "interpretation": protocol["interpretation"],
                   "output_sha256": {p.name: sha(p) for p in out.glob("*.npy")}}
        assert summary["inputs_unchanged"]
        write_json(out / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "independent_cpu_audit"}, indent=2), flush=True)
    except BaseException as exc:
        write_json(out / "STOPPED.json", {"status": "stopped", "error": repr(exc), "traceback": traceback.format_exc(), "automatic_retry": False})
        raise


def export_neighbors(path, selected_sources, rows, neighbors, overlaps, active_count):
    opener = gzip.open if str(path).endswith(".gz") else open
    options = {"compresslevel": 3} if opener is gzip.open else {}
    columns = ["source_index", "source_scientific_name", "source_kingdom", "rank", "target_index", "target_scientific_name", "target_kingdom", "shared_active", "source_active", "target_active", "jaccard"]
    with opener(path, "wt", encoding="utf-8", newline="", **options) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for source in selected_sources:
            for rank, (target, shared) in enumerate(zip(neighbors[source], overlaps[source]), 1):
                target, shared = int(target), int(shared)
                union = int(active_count[source]+active_count[target]-shared)
                writer.writerow([source, rows[source]["scientific_name"], rows[source]["kingdom"], rank, target,
                                 rows[target]["scientific_name"], rows[target]["kingdom"], shared,
                                 int(active_count[source]), int(active_count[target]), shared/union if union else 0.])


def query_neighbors(out, name):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    rows = json.loads((Path(protocol["input_dir"]) / "data/species.json").read_text(encoding="utf-8"))
    neighbors = np.load(out / "neighbor_indices.npy", mmap_mode="r")
    overlaps = np.load(out / "shared_active.npy", mmap_mode="r")
    results = []
    for source, row in enumerate(rows):
        if row["scientific_name"] == name:
            results.append({"source_index": source, "source": row,
                            "neighbors": [{"rank": rank, "target_index": int(target), "target": rows[int(target)], "shared_active": int(shared)}
                                          for rank, (target, shared) in enumerate(zip(neighbors[source], overlaps[source]), 1)]})
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "cache/expanded_tol10m_v2")
    parser.add_argument("--anchor-dir", type=Path, default=ROOT / "experiments/expanded_stability_w2048_k32_v2/seed_20260929")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/expanded_overlap_v2")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--bridge-sources", nargs="+", default=["Coccinella septempunctata", "Nyctibius griseus", "Asio otus"])
    parser.add_argument("--full-csv", action="store_true", help="Optional compressed 3.6-million-edge table; compact arrays and selected table are always saved")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--query", metavar="SCIENTIFIC_NAME")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.plan_only:
        plan(args)
    elif args.query:
        query_neighbors(args.out, args.query)
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
