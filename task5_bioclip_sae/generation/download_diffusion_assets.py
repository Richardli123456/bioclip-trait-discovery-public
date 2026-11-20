"""Download the pinned SD 1.5 and VAE components used by released inference code."""
from pathlib import Path
import json

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_REV = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
VAE_REV = "31f26fdeee1355a5c34592e401dd41e45d25a493"


def main():
    snapshot_download(
        BASE_REPO, revision=BASE_REV, local_dir=str(ROOT / "cache/sd15"),
        allow_patterns=[
            "model_index.json", "scheduler/*.json", "tokenizer/*",
            "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "README.md", "LICENSE*",
        ], max_workers=4,
    )
    snapshot_download(
        VAE_REPO, revision=VAE_REV, local_dir=str(ROOT / "cache/vae"),
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors", "README.md", "LICENSE*"],
    )
    (ROOT / "diffusion_assets.json").write_text(json.dumps({
        "base_model": BASE_REPO, "base_revision": BASE_REV,
        "base_alias_used_upstream": "runwayml/stable-diffusion-v1-5",
        "vae_model": VAE_REPO, "vae_revision": VAE_REV,
    }, indent=2), encoding="utf-8")
    print("Pinned diffusion components ready", flush=True)


if __name__ == "__main__":
    main()
