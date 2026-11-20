"""Fixed-seed raw/reconstructed/feature-edited TaxaAdapter comparisons.

This subclasses the released adapter only to accept precomputed raw BioCLIP
vectors. The original taxonomy attention, projection, decoder and sampler are
used. Candidate feature selection uses training species only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent
sys.path.insert(0, str(ROOT / "cache/upstream"))
sys.path.insert(0, str(TASK))
from taxa_adapter import IPAdapter
from train_sae import TopKSAE


class EmbeddingTaxaAdapter(IPAdapter):
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if clip_image_embeds is None:
            return super().get_image_embeds(pil_image=pil_image)
        if self.model_type != "bioclip":
            raise ValueError("Precomputed input is implemented for BioCLIP only")
        if clip_image_embeds.ndim != 2 or clip_image_embeds.shape[1] != 768:
            raise ValueError("Expected a batch of raw 768-dimensional vectors")
        dtype = self.image_proj_model.proj.weight.dtype
        vector = clip_image_embeds.to(self.device, dtype=dtype)
        return (self.image_proj_model(vector),
                self.image_proj_model(torch.zeros_like(vector)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioclip-model-path", type=Path,
                        default=TASK.parent / "task4_bioclip_pilot/cache/model")
    parser.add_argument("--species", default="Asio otus")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--prompt", default="best quality, high quality")
    args = parser.parse_args()
    started = time.perf_counter()
    run = ROOT / "cache/aligned"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = json.loads((run / "data/species.json").read_text(encoding="utf-8"))
    i = next(i for i, r in enumerate(rows) if r["scientific_name"] == args.species)
    row = rows[i]
    x = torch.from_numpy(np.load(run / "data/embeddings_raw.npy")[i:i+1]).cuda()
    saved_z = np.load(run / "results/sparse_vectors.npy")
    checkpoint = torch.load(run / "results/sae.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    sae = TopKSAE(state["mean"], state["scale"]).eval().cuda()
    sae.load_state_dict(state)
    with torch.inference_mode():
        reconstruction, z = sae(x)
    assert np.allclose(z.cpu().numpy(), saved_z[i:i+1], atol=2e-5, rtol=1e-4)
    features = json.loads((run / "results/features.json").read_text(encoding="utf-8"))
    # Family-associated candidates, not presumed independent visual traits.
    candidates = [f for f in features if len(f["top_training_species"]) >= 8
                  and sum(r["family"] == row["family"] for r in f["top_training_species"]) >= 7
                  and f["top_genera"] >= 2 and float(z[0, f["feature"]]) > 0]
    if not candidates:
        raise ValueError("No active candidate with adequate training-family support")
    chosen = max(candidates, key=lambda f: float(z[0, f["feature"]]))
    j = chosen["feature"]
    removed = z.clone()
    removed[0, j] = 0
    increased = z.clone()
    increased[0, j] *= 2
    with torch.inference_mode():
        without = sae.decode(removed)
        doubled = sae.decode(increased)
        direction = z[0, j] * sae.decoder.weight[:, j] * sae.scale
        assert torch.allclose((without - reconstruction)[0], -direction, atol=1e-5, rtol=1e-4)
        assert torch.allclose((doubled - reconstruction)[0], direction, atol=1e-5, rtol=1e-4)
    selection = {
        "species": row, "feature": j, "original_activation": float(z[0, j]),
        "selection_rule": "Strongest active feature with at least 7/8 top training examples in this species' family and at least two genera",
        "candidate_label": row["family"] + "-associated feature; no specific causal trait assumed",
        "training_examples": chosen["top_training_species"],
        "train_support": chosen["train_support"],
        "feature_indices": "Indices from the space-formatted aligned SAE, distinct from the original comma-formatted pilot",
    }
    (ROOT / "feature_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("Candidate feature", j, "activation", round(float(z[0, j]), 4), flush=True)
    name = "local-dir:" + str(args.bioclip_model_path.resolve())
    bio = open_clip.create_model(name, device="cuda", precision="fp32", require_pretrained=True).eval()
    tokenizer = open_clip.get_tokenizer(name)
    vae = AutoencoderKL.from_pretrained(ROOT / "cache/vae", torch_dtype=torch.float16)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, steps_offset=1,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        ROOT / "cache/sd15", torch_dtype=torch.float16, variant="fp16",
        scheduler=scheduler, vae=vae, feature_extractor=None, safety_checker=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    adapter = EmbeddingTaxaAdapter(
        pipe, image_encoder_path=None,
        ip_ckpt=str(ROOT / "cache/adapter/taxaadapter_tol-1m.bin"),
        device="cuda", num_tokens=4, model_type="bioclip",
        bioclip=bio, taxabind=torch.nn.Identity(),
    )
    with torch.inference_mode():
        tokens = tokenizer([row["taxonomy"]]).cuda()
        official_vector = adapter.image_encoder.encode_text(tokens)
        official_projection = IPAdapter.get_image_embeds(adapter, pil_image=tokens)
        injected_projection = adapter.get_image_embeds(clip_image_embeds=official_vector)
        assert all(torch.equal(a, b) for a, b in zip(official_projection, injected_projection))
        precision_cosine = float(torch.nn.functional.cosine_similarity(x, official_vector.float()).item())
    conditions = {
        "original": x, "reconstruction": reconstruction,
        "remove": without, "increase": doubled,
    }
    image_records = []
    for seed in args.seeds:
        for condition, vector in conditions.items():
            images = adapter.generate(
                clip_image_embeds=vector, num_samples=1, seed=seed,
                prompt=args.prompt, guidance_scale=args.guidance,
                scale=1.0, num_inference_steps=args.steps, height=512, width=512,
            )
            assert len(images) == 1 and images[0].size == (512, 512)
            pixels = np.asarray(images[0])
            assert np.isfinite(pixels).all() and float(pixels.std()) > 1
            filename = f"seed_{seed}_{condition}.png"
            images[0].save(out / filename)
            image_records.append({
                "seed": seed, "condition": condition, "image": filename,
                "image_sha256": sha(out / filename),
                "raw_embedding_norm": float(vector.norm()),
                "cosine_to_original": float(torch.nn.functional.cosine_similarity(x, vector).item()),
                "relative_input_change": float((vector - x).norm() / x.norm()),
            })
            print("Saved", filename, flush=True)
    manifest = {
        "species": args.species, "taxonomy": row["taxonomy"], "feature": j,
        "prompt": args.prompt, "seeds": args.seeds, "steps": args.steps,
        "guidance_scale": args.guidance, "adapter_scale": 1.0,
        "resolution": [512, 512], "scheduler": "DDIM, upstream inference settings",
        "edit": "Hold all other sparse activations fixed; set selected activation to zero or twice its original value",
        "input_format": "Raw space-separated BioCLIP 2 taxonomy vectors, float32 extraction cast to fp16 at adapter projection",
        "fp32_input_vs_official_fp16_encoder_cosine": precision_cosine,
        "precomputed_projection_matches_upstream_exactly": True,
        "decoder_edit_direction_verified": True,
        "adapter_checkpoint_sha256": sha(ROOT / "cache/adapter/taxaadapter_tol-1m.bin"),
        "aligned_sae_sha256": sha(run / "results/sae.pt"),
        "gpu": torch.cuda.get_device_name(),
        "packages": {n: importlib.metadata.version(n) for n in [
            "torch", "torchvision", "diffusers", "transformers", "open_clip_torch", "huggingface_hub",
        ]},
        "images": image_records, "elapsed_seconds": time.perf_counter() - started,
        "limits": [
            "One species and two seeds constitute a smoke test, not a trait-specificity or stability benchmark",
            "Candidate feature selected using taxonomy association, which is confounded with visual appearance",
            "Edited codes need not lie on the natural embedding distribution",
            "Unedited SAE reconstruction can itself alter generated images",
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A static scientific comparison figure, with all conditions labeled.
    cell = 384
    canvas = Image.new("RGB", (4*cell, 88 + len(args.seeds)*(cell + 52)), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"TaxaAdapter + SAE | {args.species} | aligned feature {j}", fill="#153346", font=font(27))
    draw.text((18, 48), "Generated images. Same prompt and seed within each row; only the taxonomy vector changes.", fill="#334b59", font=font(18))
    labels = ["Original vector", "SAE reconstruction", f"Feature {j} removed", f"Feature {j} doubled"]
    for row_index, seed in enumerate(args.seeds):
        for col, (condition, label) in enumerate(zip(conditions, labels)):
            left = col*cell
            top = 88 + row_index*(cell + 52)
            draw.text((left+12, top+8), f"Seed {seed}: {label}", fill="#153346", font=font(19))
            image = Image.open(out / f"seed_{seed}_{condition}.png").convert("RGB").resize((cell, cell))
            canvas.paste(image, (left, top+44))
    canvas.save(out / "comparison.png")
    print("Completed controlled generation smoke test", flush=True)


if __name__ == "__main__":
    main()
