"""Re-extract and fit the pilot using the released generator's taxonomy format.

The original comma-separated experiment is preserved. Large aligned artifacts
are generated under cache/ and never overwrite the earlier experiment.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioclip-model-path", type=Path,
                        default=TASK.parent / "task4_bioclip_pilot/cache/model")
    args = parser.parse_args()
    run = ROOT / "cache/aligned"
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "results").mkdir(exist_ok=True)
    rows = json.loads((TASK / "data/species.json").read_text(encoding="utf-8"))
    for row in rows:
        row["taxonomy"] = " ".join(row[k] for k in [
            "kingdom", "phylum", "class_name", "order", "family", "genus", "epithet",
        ])
    (run / "data/species.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    name = "local-dir:" + str(args.bioclip_model_path.resolve())
    model = open_clip.create_model(name, device="cuda", precision="fp32", require_pretrained=True).eval()
    tokenizer = open_clip.get_tokenizer(name)
    tokens = tokenizer([r["taxonomy"] for r in rows])
    assert bool((tokens[:, -1] == 0).all()), "Possible taxonomy truncation"
    output = []
    with torch.inference_mode():
        for i in range(0, len(rows), 128):
            output.append(model.encode_text(tokens[i:i + 128].cuda(), normalize=False).cpu().numpy())
        single = model.encode_text(tokens[:1].cuda(), normalize=False).cpu().numpy()
    x = np.concatenate(output)
    assert x.shape == (5388, 768) and np.isfinite(x).all()
    assert np.allclose(x[:1], single, atol=1e-5, rtol=1e-4)
    np.save(run / "data/embeddings_raw.npy", x)
    np.save(run / "data/token_ids.npy", tokens.numpy())
    old = np.load(TASK / "data/embeddings_raw.npy")
    similarity = (old * x).sum(1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(x, axis=1))
    comparison = {
        "original_format": "taxonomy levels separated by comma and space",
        "released_generator_format": "taxonomy levels separated by one space",
        "species_count": len(rows),
        "old_vs_aligned_cosine_mean": float(similarity.mean()),
        "old_vs_aligned_cosine_p05": float(np.quantile(similarity, .05)),
        "original_embedding_sha256": sha(TASK / "data/embeddings_raw.npy"),
        "aligned_embedding_sha256": sha(run / "data/embeddings_raw.npy"),
        "species_source_sha256": sha(TASK / "data/species.json"),
        "bioclip_model_revision": json.loads((TASK.parent / "task4_bioclip_pilot/data/model_revision.json").read_text()),
        "representation": "pooled projected raw text vectors, normalize=False, float32",
        "scope": "Same 5,388 iNat21 Animalia and genus splits as the pilot; not the full TreeOfLife species list",
    }
    (ROOT / "format_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    spec = importlib.util.spec_from_file_location("pilot_sae", TASK / "train_sae.py")
    sae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sae_module)
    sae_module.ROOT = run
    sae_module.main()
    evaluation = json.loads((run / "results/evaluation.json").read_text())
    (ROOT / "aligned_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    provenance = {
        "training_source_sha256": sha(TASK / "train_sae.py"),
        "sae_sha256": sha(run / "results/sae.pt"),
        "sparse_vectors_sha256": sha(run / "results/sparse_vectors.npy"),
        "reconstructed_embeddings_sha256": sha(run / "results/reconstructed_embeddings.npy"),
    }
    (ROOT / "aligned_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
