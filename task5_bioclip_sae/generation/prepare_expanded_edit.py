"""Prepare one-feature dose and control vectors from a frozen expanded SAE.

CPU only. No training or image generation. New output directory required.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import torch

from experiment_common import ROOT, sha, write_json
sys.path.insert(0, str(ROOT.parent))
from train_sae import TopKSAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260929, help="SAE seed, not image seed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exploratory", action="store_true", help="Allow an unvalidated feature, explicitly marked exploratory")
    args = parser.parse_args()
    torch.set_num_threads(2)
    study = json.loads((args.study / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((args.study / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert args.seed in [study["anchor_seed"]] + study["replicate_seeds"]
    assert {p: sha(p) for p in study["input_sha256"]} == study["input_sha256"], "Study inputs have changed"
    source = Path(study["input_dir"])
    rows = json.loads((source / "data/species.json").read_text(encoding="utf-8"))
    matched = [i for i, r in enumerate(rows) if r["scientific_name"] == args.species]
    assert len(matched) == 1, "Species must resolve to exactly one saved taxonomy record"
    row_index = matched[0]
    checkpoint_path = args.study / f"seed_{args.seed}" / "sae.pt"
    saved_metrics = json.loads((checkpoint_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert sha(checkpoint_path) == saved_metrics["checkpoint_sha256"]
    assert sha(checkpoint_path.parent / "code_indices.npy") == saved_metrics["indices_sha256"]
    assert sha(checkpoint_path.parent / "code_values.npy") == saved_metrics["values_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    width, k = checkpoint["config"]["width"], checkpoint["config"]["k"]
    assert 0 <= args.feature < width
    with (args.study / "feature_stability.csv").open(encoding="utf-8") as stream:
        stability = {int(r["anchor_feature"]): r for r in csv.DictReader(stream)}
    feature = args.feature
    if args.seed != study["anchor_seed"]:
        with (args.study / f"seed_{args.seed}" / "matches.csv").open(encoding="utf-8") as stream:
            match = next(r for r in csv.DictReader(stream) if int(r["anchor_feature"]) == args.feature)
        feature = int(match["matched_feature"])
    stable = stability[args.feature]["passes_prespecified_gate"].lower() == "true"
    assert stable or args.exploratory, "Feature failed recurrence; use explicit --exploratory for a mathematical demonstration"
    model = TopKSAE(state["mean"], state["scale"], width=width, k=k).eval().requires_grad_(False)
    model.load_state_dict(state)
    raw_path = source / "data/embeddings_raw.npy"
    x = np.array(np.load(raw_path, mmap_mode="r")[row_index], dtype=np.float32)
    with torch.inference_mode():
        checked_code = model.encode(torch.from_numpy(x[None]))[0].numpy()
    code_root = checkpoint_path.parent
    stored_indices = np.load(code_root / "code_indices.npy", mmap_mode="r")[row_index]
    stored_values = np.load(code_root / "code_values.npy", mmap_mode="r")[row_index]
    z = np.zeros(width, dtype=np.float64)
    z[stored_indices] = stored_values
    assert np.allclose(z, checked_code, atol=2e-4, rtol=1e-4), "Saved and CPU-encoded sparse vectors disagree"
    assert z[feature] > 1e-7, "Selected coordinate is inactive for this species; multiplication would be a no-op"
    decoder = state["decoder.weight"].numpy().astype(np.float64)
    mean, bias, scale = state["mean"].numpy().astype(np.float64), state["output_bias"].numpy().astype(np.float64), float(state["scale"])
    def decode(code):
        return (decoder @ code + bias) * scale + mean
    reconstruction = decode(z)
    direction = z[feature] * decoder[:, feature] * scale
    controls = [int(i) for i in np.flatnonzero(z > 1e-7) if i != feature]
    control = min(controls, key=lambda f: (abs(np.linalg.norm(z[f] * decoder[:, f] * scale) - np.linalg.norm(direction)), f))
    rng = np.random.default_rng(20261004 + args.feature)
    random_direction = rng.normal(size=768)
    random_direction -= direction * (random_direction @ direction) / (direction @ direction)
    random_direction *= np.linalg.norm(direction) / np.linalg.norm(random_direction)
    assert abs(random_direction @ direction) < 1e-8
    assert np.isclose(np.linalg.norm(random_direction), np.linalg.norm(direction), rtol=1e-12)
    vectors, records = [], []
    def add(condition, vector, edited_feature=None, **details):
        records.append({"vector_index": len(vectors), "source": args.species, "target": args.species,
                        "condition": condition, "step": None, "edited_feature": edited_feature, **details})
        vectors.append(np.asarray(vector, dtype=np.float32))
    add("original", x)
    add("reconstruction", reconstruction)
    for factor, label in [(0., "remove_residual"), (.5, "half_residual"), (2., "double_residual")]:
        edited = z.copy()
        edited[feature] *= factor
        vector = x + decode(edited) - reconstruction
        assert np.allclose(vector - x, (factor - 1) * direction, atol=1e-10)
        assert np.allclose(vector - decode(edited), x - reconstruction, atol=1e-10)
        add(label, vector, feature, factor=factor, residual_preserved=True)
    edited = z.copy()
    edited[control] = 0
    add("other_active_feature_removed", x + decode(edited) - reconstruction, control, residual_preserved=True)
    add("matched_norm_random_control", x - random_direction, random_seed=20261004 + args.feature)
    matrix = np.stack(vectors)
    assert matrix.shape == (7, 768) and np.isfinite(matrix).all()
    assert np.array_equal(matrix[0], x)
    for record, vector in zip(records, matrix):
        record["edit_norm"] = float(np.linalg.norm(vector - x))
        record["cosine_to_original"] = float(np.dot(vector, x) / (np.linalg.norm(vector) * np.linalg.norm(x)))
    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / "intervention_vectors.npy", matrix)
    np.save(args.out / "source_code.npy", z.astype(np.float32))
    provenance_paths = [source / "data/species.json", raw_path, checkpoint_path, code_root / "code_indices.npy", code_root / "code_values.npy", args.study / "feature_stability.csv", args.study / "protocol.json", args.study / "summary.json", code_root / "metrics.json"]
    if args.seed != study["anchor_seed"]:
        provenance_paths.append(code_root / "matches.csv")
    write_json(args.out / "intervention_plan.json", {
        "species": rows[row_index], "embedding_row": row_index, "anchor_feature": args.feature,
        "sae_seed": args.seed, "actual_feature": feature, "activation": float(z[feature]),
        "stable_candidate": stable, "biological_trait_validated": False,
        "other_active_control_feature": int(control),
        "control_rule": "Different positive coordinate with closest full-removal norm; lower feature index breaks ties",
        "random_rule": "Fixed random Gaussian direction orthogonal to candidate, matched to full-removal norm",
        "equation": "edited_raw = raw + decode(edited_code) - decode(original_code); no TopK after editing",
        "records": records, "training_performed": False, "gpu_used": False,
        "source_sha256": {str(p.resolve()): sha(p) for p in provenance_paths},
        "script_sha256": sha(Path(__file__)), "vectors_sha256": sha(args.out / "intervention_vectors.npy"),
        "limitation": "Stable coordinates can encode taxonomy or other associations; image and external-label tests remain necessary. The other-active control is not established to be unrelated to the trait."})
    print(json.dumps({"plan": str(args.out), "species": args.species, "anchor_feature": args.feature,
                      "actual_feature": feature, "stable": stable, "vectors": len(records)}), flush=True)


if __name__ == "__main__":
    main()
