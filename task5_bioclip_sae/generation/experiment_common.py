"""Read-only use of the saved aligned SAE; no fitting or optimizer code."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "cache/aligned"
EPS = 1e-7


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class FrozenArtifacts:
    def __init__(self, root=ALIGNED):
        import torch
        torch.set_num_threads(8)
        self.root = Path(root)
        self.rows = json.loads((self.root / "data/species.json").read_text(encoding="utf-8"))
        self.x = np.load(self.root / "data/embeddings_raw.npy")
        self.z = np.load(self.root / "results/sparse_vectors.npy")
        self.y = np.load(self.root / "results/reconstructed_embeddings.npy")
        self.features = json.loads((self.root / "results/features.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(self.root / "results/sae.pt", map_location="cpu", weights_only=True)
        self.config = checkpoint["config"]
        self.state = {k: v.detach().cpu().numpy().astype(np.float64)
                      for k, v in checkpoint["state_dict"].items()}
        self.decoder = self.state["decoder.weight"]
        self.mean = self.state["mean"]
        self.scale = float(self.state["scale"])
        self.bias = self.state["output_bias"]
        self.names = [r["scientific_name"] for r in self.rows]
        self.by_name = {name: i for i, name in enumerate(self.names)}
        assert len(self.by_name) == len(self.rows), "Duplicate scientific names"
        assert self.x.shape == self.y.shape == (len(self.rows), 768)
        assert self.z.shape == (len(self.rows), self.config["width"])
        assert self.decoder.shape == (768, self.z.shape[1])
        assert all(np.isfinite(a).all() for a in [self.x, self.y, self.z, self.decoder])
        assert (self.z >= 0).all()
        assert ((self.z > EPS).sum(1) == self.config["k"]).all()

    def decode(self, codes):
        return (np.asarray(codes, dtype=np.float64) @ self.decoder.T + self.bias) * self.scale + self.mean

    def encode(self, raw):
        positive = np.maximum((np.asarray(raw, dtype=np.float64) - self.mean) / self.scale
                              @ self.state["encoder.weight"].T + self.state["encoder.bias"], 0)
        indices = np.argpartition(positive, -self.config["k"], axis=-1)[..., -self.config["k"]:]
        result = np.zeros_like(positive)
        np.put_along_axis(result, indices, np.take_along_axis(positive, indices, axis=-1), axis=-1)
        return result

    def provenance(self):
        relative = ["data/species.json", "data/embeddings_raw.npy", "results/sparse_vectors.npy",
                    "results/reconstructed_embeddings.npy", "results/sae.pt", "results/features.json"]
        return {"aligned_root": str(self.root.resolve()), "species_count": len(self.rows),
                "input_dim": 768, "sae_width": self.z.shape[1], "k": self.config["k"],
                "sha256": {name: sha(self.root / name) for name in relative}}

    def family_candidate(self, index):
        row = self.rows[index]
        candidates = [f for f in self.features if len(f["top_training_species"]) >= 8
                      and sum(r["family"] == row["family"] for r in f["top_training_species"]) >= 7
                      and f["top_genera"] >= 2 and self.z[index, f["feature"]] > EPS]
        if not candidates:
            raise ValueError(f"No eligible training-family-associated feature for {row['scientific_name']}")
        return max(candidates, key=lambda f: (self.z[index, f["feature"]], -f["feature"]))


def support_counts(support, index):
    return np.count_nonzero(support & support[index], axis=1)


def rank_neighbors(counts, names, index, keep=10):
    # Exact support overlap, with scientific-name alphabetical ties. No phenotype-based selection.
    eligible = np.arange(len(names)) != index
    indices = np.flatnonzero(eligible)
    order = np.lexsort((np.asarray(names)[indices], -counts[indices]))
    return indices[order[:keep]]


def bridge_codes(source, target, decoder, scale):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    dimensions = np.flatnonzero(source != target)
    contribution = np.abs(target - source) * np.linalg.norm(decoder, axis=0) * scale
    dimensions = dimensions[np.lexsort((dimensions, -contribution[dimensions]))]
    codes, steps = [source.copy()], []
    for feature in dimensions:
        current = codes[-1].copy()
        before = float(current[feature])
        current[feature] = target[feature]
        codes.append(current)
        steps.append({"step": len(steps) + 1, "feature": int(feature), "from": before,
                      "to": float(target[feature]), "decoder_change_norm": float(contribution[feature]),
                      "active_features": int((current > EPS).sum())})
    assert np.array_equal(codes[-1], target)
    return np.stack(codes), steps
