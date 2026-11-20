"""Independently check support retrieval, interventions, bridges and saved images."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np

from experiment_common import ROOT, EPS, FrozenArtifacts, sha, write_json


def verify_overlap(data, overlap):
    summary = json.loads((overlap / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"] == data.provenance()
    supports = [set(np.flatnonzero(row > EPS).tolist()) for row in data.z]
    by_source = {}
    with (overlap / "neighbors.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            i, j = data.by_name[row["source"]], data.by_name[row["neighbor"]]
            assert i != j
            shared = len(supports[i] & supports[j])
            union = len(supports[i] | supports[j])
            assert int(row["shared_active"]) == shared
            assert abs(float(row["jaccard"]) - shared / union) < 1e-12
            assert (row["same_genus"] == "True") == (data.rows[i]["genus"] == data.rows[j]["genus"])
            assert (row["same_family"] == "True") == (data.rows[i]["family"] == data.rows[j]["family"])
            by_source.setdefault(i, []).append(row)
    assert len(by_source) == len(data.rows)
    assert sum(map(len, by_source.values())) == summary["row_count"]
    rng = np.random.default_rng(20261002)
    sampled = rng.choice(len(data.rows), size=min(128, len(data.rows)), replace=False)
    # Independent Python set intersections against every candidate for sampled queries.
    for i in sampled:
        ranked = sorted((j for j in range(len(data.rows)) if j != i),
                        key=lambda j: (-len(supports[i] & supports[j]), data.names[j]))
        actual = by_source[i]
        assert [data.by_name[r["neighbor"]] for r in actual] == ranked[:summary["top_k"]]
        max_shared = len(supports[i] & supports[ranked[0]])
        ties = sum(len(supports[i] & supports[j]) == max_shared for j in ranked)
        assert int(actual[0]["top_overlap_tie_count"]) == ties
    top = [by_source[i][0] for i in range(len(data.rows))]
    assert abs(np.mean([int(r["shared_active"]) for r in top]) - summary["mean_top1_shared"]) < 1e-12
    assert abs(np.mean([r["same_family"] == "True" for r in top]) - summary["top1_same_family_fraction"]) < 1e-12
    return {"all_neighbor_intersections_checked": summary["row_count"],
            "exhaustive_independent_query_rankings_checked": len(sampled)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/existing_sae_v1")
    parser.add_argument("--overlap-dir", type=Path, default=ROOT / "overlap")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    out = args.out_dir
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["training_performed"] is False and protocol["model_updates"] is False
    data = FrozenArtifacts(protocol["provenance"]["aligned_root"])
    assert data.provenance() == protocol["provenance"]
    vectors = np.load(out / "vectors.npy").astype(np.float64)
    assert vectors.shape == (len(protocol["vectors"]), 768) and np.isfinite(vectors).all()
    assert sha(out / "vectors.npy") == protocol["vector_sha256"]
    # Saved encoder/decoder round-trip without initializing or fitting an SAE.
    assert np.allclose(data.encode(data.x[:16]), data.z[:16], atol=2e-5, rtol=1e-4)
    assert np.allclose(data.decode(data.z[:16]), data.y[:16], atol=2e-5, rtol=1e-4)
    lookup = {(r["kind"], r["source"], r["condition"]): vectors[r["vector_index"]] for r in protocol["vectors"]}
    assert len(lookup) == len(protocol["vectors"])
    for selection in protocol["feature_selections"]:
        name, j, control = selection["species"], selection["feature"], selection["control_feature"]
        i = data.by_name[name]
        x, z = data.x[i].astype(np.float64), data.z[i].astype(np.float64)
        direction = z[j] * data.decoder[:, j] * data.scale
        original = lookup["intervention", name, "original"]
        assert np.array_equal(original, data.x[i])
        assert np.allclose(lookup["intervention", name, "reconstruction"], data.decode(z), atol=2e-5, rtol=1e-5)
        for condition, factor in [("remove_residual", 0), ("half_residual", 0.5), ("double_residual", 2)]:
            expected = x + (factor - 1) * direction
            assert np.allclose(lookup["intervention", name, condition], expected, atol=2e-5, rtol=1e-5)
        expected_control = x - z[control] * data.decoder[:, control] * data.scale
        assert np.allclose(lookup["intervention", name, "control_remove_residual"], expected_control, atol=2e-5, rtol=1e-5)
        random = lookup["intervention", name, "random_direction_control"] - x
        assert abs(np.linalg.norm(random) / np.linalg.norm(direction) - 1) < 1e-5
        assert abs(np.dot(random, direction) / np.dot(direction, direction)) < 1e-5
        assert selection["control_training_family_fraction"] <= 0.25
        top = selection["candidate_training_examples"]
        assert sum(r["family"] == data.rows[i]["family"] for r in top) >= 7
        assert len({r["genus"] for r in top}) >= 2
    bridge_checks = []
    support = [set(np.flatnonzero(row > EPS).tolist()) for row in data.z]
    for bridge in protocol["bridges"]:
        source, target = bridge["source"], bridge["target"]
        i, j = data.by_name[source], data.by_name[target]
        ranked = sorted((k for k in range(len(data.rows)) if k != i),
                        key=lambda k: (-len(support[i] & support[k]), data.names[k]))
        assert j == ranked[0]
        expected = data.z[i].astype(np.float64).copy()
        end = data.z[j].astype(np.float64)
        different = set(np.flatnonzero(expected != end).tolist())
        assert {r["feature"] for r in bridge["steps"]} == different
        assert len(bridge["steps"]) == len(different) == bridge["changed_dimensions"]
        assert np.allclose(lookup["bridge", source, "step_000"], data.decode(expected), atol=2e-5, rtol=1e-5)
        distance = np.linalg.norm(expected - end)
        max_active = int((expected > EPS).sum())
        for step in bridge["steps"]:
            previous = expected.copy()
            assert float(expected[step["feature"]]) == step["from"]
            expected[step["feature"]] = step["to"]
            assert np.count_nonzero(previous != expected) == 1
            assert np.linalg.norm(expected - end) < distance
            distance = np.linalg.norm(expected - end)
            max_active = max(max_active, int((expected > EPS).sum()))
            vector = lookup["bridge", source, f"step_{step['step']:03d}"]
            assert np.allclose(vector, data.decode(expected), atol=2e-5, rtol=1e-5)
        assert np.array_equal(expected, end) and distance == 0
        assert max_active == bridge["max_active_along_path"]
        bridge_checks.append({"source": source, "target": target, "single_coordinate_steps": len(different),
                              "endpoint_codes_exact": True, "max_active_features": max_active})
    overlap_result = verify_overlap(data, args.overlap_dir)
    result = {"status": "plan_verified" if args.plan_only else "passed", "training_performed": False,
              "input_artifacts_unchanged": True, "saved_encoder_decoder_roundtrip": True,
              "residual_preserving_feature_math": True, "random_control_norm_and_orthogonality": True,
              "bridge_checks": bridge_checks, "overlap_checks": overlap_result,
              "vector_sha256": sha(out / "vectors.npy"),
              "not_established": ["Specific biological trait control", "Robustness across SAE training seeds", "Full TreeOfLife-1M coverage"]}
    if not args.plan_only:
        from PIL import Image
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed" and manifest["training_performed"] is False
        assert manifest["input_checkpoints_unchanged"] and manifest["frozen_parameter_assertions_passed"]
        assert manifest["precomputed_projection_matches_upstream_exactly"]
        assert manifest["image_count"] == protocol["planned_image_count"] == len(manifest["images"])
        expected_keys = {(r["kind"], r["source"], r["condition"], seed) for r in protocol["vectors"]
                         for seed in (protocol["intervention_seeds"] if r["kind"] == "intervention" else protocol["bridge_seeds"])}
        actual_keys = {(r["kind"], r["source"], r["condition"], r["seed"]) for r in manifest["images"]}
        assert actual_keys == expected_keys
        assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
        assert sha(ROOT / "cache/adapter" / assets["weight_filename"]) == manifest["adapter_checkpoint_sha256"]
        for filename, digest in manifest["script_sha256"].items():
            assert sha(ROOT / filename) == digest
        for record in manifest["images"]:
            path = out / record["image"]
            assert sha(path) == record["image_sha256"]
            with Image.open(path) as image:
                assert image.size == (512, 512)
                image.verify()
            with Image.open(path) as image:
                assert np.asarray(image).std() > 1
        # Identical raw source vectors at fixed seed should reproduce across the two experiment branches.
        identical_references = []
        for bridge in protocol["bridges"]:
            for seed in protocol["bridge_seeds"]:
                base = next((r for r in manifest["images"] if r["kind"] == "intervention" and r["source"] == bridge["source"]
                             and r["condition"] == "original" and r["seed"] == seed), None)
                if base is None:
                    continue
                repeated = next(r for r in manifest["images"] if r["kind"] == "bridge" and r["source"] == bridge["source"]
                                and r["condition"] == "source_original" and r["seed"] == seed)
                assert base["image_sha256"] == repeated["image_sha256"]
                identical_references.append({"species": bridge["source"], "seed": seed, "identical_images": True})
        result.update({"verified_images": len(manifest["images"]),
                       "identical_raw_source_references": identical_references,
                       "manifest_sha256": sha(out / "run_manifest.json")})
    write_json(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
