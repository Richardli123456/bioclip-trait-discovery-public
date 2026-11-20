"""Independent CPU checks of one saved expanded SAE anchor and compressed codes."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

from experiment_common import ROOT, sha, write_json
sys.path.insert(0, str(ROOT.parent))
from train_sae import TopKSAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "cache/expanded_tol10m_v1")
    parser.add_argument("--anchor-dir", type=Path, default=ROOT / "experiments/expanded_stability_w2048_k32_v1/seed_20260929")
    parser.add_argument("--out", type=Path, default=ROOT / "cache/expanded_tol10m_v1/anchor_cpu_verification.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = json.loads((args.input_dir / "data/species.json").read_text(encoding="utf-8"))
    raw = np.load(args.input_dir / "data/embeddings_raw.npy", mmap_mode="r")
    ck = torch.load(args.anchor_dir / "sae.pt", map_location="cpu", weights_only=True)
    ids = np.flatnonzero([r["split"] == "train" for r in rows])
    total = np.zeros(768, dtype=np.float64)
    for begin in range(0, len(ids), 4096):
        total += raw[ids[begin:begin+4096]].sum(0, dtype=np.float64)
    mean = total / len(ids)
    squares = 0.
    for begin in range(0, len(ids), 4096):
        delta = raw[ids[begin:begin+4096]].astype(np.float64) - mean
        squares += float(np.square(delta).sum())
    scale = np.sqrt(squares / (len(ids)*768))
    mean_error = float(np.abs(ck["state_dict"]["mean"].numpy()-mean).max())
    scale_error = abs(float(ck["state_dict"]["scale"])-scale)
    assert mean_error < 1e-5 and scale_error < 1e-5
    model = TopKSAE(ck["state_dict"]["mean"], ck["state_dict"]["scale"], width=ck["config"]["width"], k=ck["config"]["k"]).eval()
    model.load_state_dict(ck["state_dict"])
    indices = np.load(args.anchor_dir / "code_indices.npy", mmap_mode="r")
    values = np.load(args.anchor_dir / "code_values.npy", mmap_mode="r")
    chosen = np.unique(np.concatenate([np.array([0, 5387, len(rows)-1]), np.random.default_rng(761).choice(len(rows), 253, replace=False)]))
    stored = np.zeros((len(chosen), ck["config"]["width"]), dtype=np.float32)
    stored[np.arange(len(chosen))[:, None], indices[chosen]] = values[chosen]
    with torch.inference_mode():
        code = model.encode(torch.from_numpy(np.array(raw[chosen]))).numpy()
    maximum = float(np.abs(code-stored).max())
    support_disagreement = int(np.count_nonzero((code > 1e-7) != (stored > 1e-7)))
    assert np.allclose(code, stored, atol=1e-4, rtol=1e-4), "CPU/GPU code mismatch requires investigation"
    result = {"status": "passed", "execution": "CPU only; no fitting or GPU work", "checkpoint_sha256": sha(args.anchor_dir / "sae.pt"),
              "train_mean_max_abs_difference_float64_reference": mean_error,
              "train_global_rms_abs_difference_float64_reference": scale_error,
              "sampled_rows": chosen.tolist(), "sample_count": len(chosen), "cpu_encode_vs_stored_code_max_abs_difference": maximum,
              "sampled_binary_support_disagreements": support_disagreement,
              "limits": "Checks preprocessing across all training rows and compressed codes on 256 samples; does not reproduce optimization or all cross-run matches.",
              "script_sha256": sha(__file__)}
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "sampled_rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
