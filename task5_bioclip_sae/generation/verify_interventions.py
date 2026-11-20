"""Independently verify the aligned SAE artifacts and saved image comparisons."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_sae import TopKSAE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / "cache/aligned"
    rows = json.loads((run / "data/species.json").read_text(encoding="utf-8"))
    x = np.load(run / "data/embeddings_raw.npy")
    z = np.load(run / "results/sparse_vectors.npy")
    y = np.load(run / "results/reconstructed_embeddings.npy")
    assert x.shape == y.shape == (5388, 768) and z.shape == (5388, 2048)
    assert all(np.isfinite(a).all() for a in [x, z, y])
    assert (z >= 0).all() and ((z > 1e-7).sum(1) == 32).all()
    groups = {s: {r["genus"] for r in rows if r["split"] == s}
              for s in ["train", "validation", "test"]}
    assert not (groups["train"] & groups["test"] or groups["train"] & groups["validation"]
                or groups["test"] & groups["validation"])
    evals = json.loads((ROOT / "aligned_evaluation.json").read_text())
    mean = x[[r["split"] == "train" for r in rows]].astype("float64").mean(0)
    for split in groups:
        mask = [r["split"] == split for r in rows]
        a, b = x[mask].astype("float64"), y[mask].astype("float64")
        nmse = np.square(a-b).sum() / np.square(a-mean).sum()
        cosine = (a*b).sum(1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        assert abs(nmse-evals["metrics"][split]["normalized_mse"]) < 1e-6
        assert abs(cosine.mean()-evals["metrics"][split]["cosine_mean"]) < 1e-6
    state = torch.load(run / "results/sae.pt", map_location="cpu", weights_only=True)["state_dict"]
    model = TopKSAE(state["mean"], state["scale"]).eval()
    model.load_state_dict(state)
    with torch.inference_mode():
        reconstruction, sparse = model(torch.from_numpy(x[:16]))
    assert np.allclose(reconstruction.numpy(), y[:16], atol=2e-5, rtol=1e-4)
    assert np.allclose(sparse.numpy(), z[:16], atol=2e-5, rtol=1e-4)
    fmt = json.loads((ROOT / "format_comparison.json").read_text())
    provenance = json.loads((ROOT / "aligned_provenance.json").read_text())
    assert sha(run / "data/embeddings_raw.npy") == fmt["aligned_embedding_sha256"]
    for key, rel in [
        ("sae_sha256", "results/sae.pt"),
        ("sparse_vectors_sha256", "results/sparse_vectors.npy"),
        ("reconstructed_embeddings_sha256", "results/reconstructed_embeddings.npy"),
    ]:
        assert sha(run / rel) == provenance[key]
    manifest = json.loads((ROOT / "results/run_manifest.json").read_text())
    images = manifest["images"]
    assert len(images) == 8
    assert {(r["seed"], r["condition"]) for r in images} == {
        (s, c) for s in [42, 43] for c in ["original", "reconstruction", "remove", "increase"]
    }
    assert manifest["precomputed_projection_matches_upstream_exactly"]
    assert manifest["decoder_edit_direction_verified"]
    assets = json.loads((ROOT / "assets.json").read_text())
    assert sha(ROOT / "cache/adapter" / assets["weight_filename"]) == assets["weight_sha256"]
    for rel, digest in assets["code_sha256"].items():
        assert sha(ROOT / "cache/upstream" / rel) == digest
    for record in images:
        path = ROOT / "results" / record["image"]
        assert sha(path) == record["image_sha256"]
        with Image.open(path) as image:
            assert image.size == (512, 512)
            image.verify()
    result = {
        "status": "passed",
        "checks": [
            "Dimensions, finite values, nonnegative codes and 32 active features",
            "Whole-genus split isolation",
            "All reconstruction metrics independently recomputed in NumPy float64",
            "Aligned checkpoint reload reproduces saved codes and reconstructions on CPU",
            "Pinned official code and checkpoint SHA256 hashes match",
            "Eight 512x512 generated images match recorded checksums and condition/seed coverage",
            "Runtime recorded exact equivalence of the original projection and precomputed-vector path",
            "Runtime verified the expected decoder direction for the selected-feature edits",
        ],
        "not_established": ["Causal specificity of a biological trait", "Cross-species or multi-seed stability beyond this pilot"],
        "comparison_sha256": sha(ROOT / "results/comparison.png"),
    }
    (ROOT / "results/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
