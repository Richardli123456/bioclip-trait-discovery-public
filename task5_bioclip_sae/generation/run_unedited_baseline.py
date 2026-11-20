"""Generate a prespecified, unedited 32-seed dorsal-view calibration packet.

Inference only. The original saved taxonomy embedding is used without SAE
encoding, decoding, selection, or intervention. Refuses output overwrites and
stops on hardware guard failures without automatic retries or fallbacks.
"""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
import traceback

import numpy as np

from experiment_common import ROOT, ALIGNED, sha, write_json
from run_frozen_experiments import load_generator

OUT = ROOT / "experiments/unedited_dorsal_v1"
SPECIES = "Coccinella septempunctata"
SEEDS = list(range(1000, 1032))
PROMPT = "best quality, high quality, macro photograph of a single adult beetle, seen directly from above, dorsal view, centered, whole body visible, sharp focus"
NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"


def hardware_sample(stage):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    fields = next(csv.reader(completed.stdout.splitlines()))
    sample = {"stage": stage, "gpu": fields[0].strip(), "total_mib": int(fields[1]),
              "free_mib": int(fields[2]), "temperature_c": int(fields[3]),
              "unix_time": time.time()}
    with (OUT / "hardware.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample) + "\n")
    if sample["temperature_c"] >= 88 or sample["free_mib"] < 2048:
        raise RuntimeError("Hardware guard stop: " + json.dumps(sample))
    return sample


def frozen_hashes(modules):
    """Hash every parameter and buffer, one CPU tensor at a time."""
    result = {}
    for index, module in enumerate(modules):
        assert not module.training
        assert all(not p.requires_grad for p in module.parameters())
        digest = hashlib.sha256()
        for name, value in module.state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        result[f"{index}_{type(module).__name__}"] = digest.hexdigest()
    return result


def make_packet(records):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
    shuffled = np.random.default_rng(2026100401).permutation(len(records))
    blinded = []
    blind_dir = OUT / "human_review/images"
    blind_dir.mkdir(parents=True)
    for number, index in enumerate(shuffled, 1):
        record = records[int(index)]
        ident = f"D{number:03d}"
        filename = f"{ident}.png"
        shutil.copyfile(OUT / record["image"], blind_dir / filename)
        blinded.append({"image_id": ident, "blind_image": "human_review/images/" + filename,
                        "seed": record["seed"], "original_image": record["image"],
                        "sha256": record["sha256"]})
    with (OUT / "blind_key.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blinded[0]))
        writer.writeheader()
        writer.writerows(blinded)
    for start in range(0, len(blinded), 6):
        # Two columns and three rows. Each image retains native 512x512 pixels.
        canvas = Image.new("RGB", (1024, 3 * 552), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, record in enumerate(blinded[start:start + 6]):
            left, top = (slot % 2) * 512, (slot // 2) * 552
            draw.text((left + 12, top + 8), record["image_id"], fill="#172c3c", font=font)
            with Image.open(OUT / record["blind_image"]) as im:
                canvas.paste(im.convert("RGB"), (left, top + 40))
        canvas.save(OUT / f"human_review/contact_sheet_{start // 6 + 1:02d}.png")
    with (OUT / "human_review/scoring.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "view_eligible", "spots_visible", "spot_count", "notes"])
        writer.writeheader()
        writer.writerows({"image_id": r["image_id"]} for r in blinded)
    (OUT / "human_review/SCORING_RULES.md").write_text(
        "# Human scoring instructions\n\n"
        "Review the 32 randomized images once, using the native-resolution images or six contact sheets. "
        "Do not consult the seed key until all scores are saved. This is an unedited-baseline calibration, "
        "so all images have the same condition; randomization conceals seed order.\n\n"
        "- view_eligible: yes only for a clear actual dorsal/top view with both wing covers visible, "
        "sufficient focus, and no substantial occlusion. Low side-angle views are no. Uncertainty is NA.\n"
        "- spots_visible: yes/no only when view_eligible=yes and the answer is clear; otherwise NA. "
        "Distinguish markings on the wing covers from head/pronotum markings, legs, glare, or shadows.\n"
        "- spot_count: a nonnegative integer only when view_eligible=yes and all visible wing-cover spots "
        "can be confidently distinguished. Zero means an eligible, visibly spotless pair of wing covers. "
        "Blurred, hidden, merged, or ambiguous markings give NA, with a note. Do not infer hidden spots.\n"
        "- notes: describe any ambiguity, low view angle, blur, occlusion, or uncertain mark boundary.\n\n"
        "Keep NA separate from zero. Report eligible/32, any-spots/eligible, count-scorable/32, and "
        "the distribution of counts among count-scorable images. Human annotations are currently blank. "
        "An AI review, if later added, must be separately identified and is not human or expert validation.\n",
        encoding="utf-8")
    return blinded


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        preflight = hardware_sample("before_load")
        rows = json.loads((ALIGNED / "data/species.json").read_text(encoding="utf-8"))
        index = next(i for i, row in enumerate(rows) if row["scientific_name"] == SPECIES)
        row = rows[index]
        vector = np.load(ALIGNED / "data/embeddings_raw.npy")[index].copy()
        assert vector.shape == (768,) and np.isfinite(vector).all()
        assert "," not in row["taxonomy"]
        np.save(OUT / "original_embedding.npy", vector)
        adapter_path = ROOT / "cache/adapter/taxaadapter_tol-1m.bin"
        input_hashes = {str(path): sha(path) for path in [ALIGNED / "data/species.json",
                       ALIGNED / "data/embeddings_raw.npy", adapter_path]}
        assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
        assert input_hashes[str(adapter_path)] == assets["weight_sha256"]
        protocol = {"species": row, "seeds": SEEDS, "prompt": PROMPT, "negative_prompt": NEGATIVE,
                    "input": "Original raw space-separated taxonomy embedding; no SAE used",
                    "training_performed": False, "model_updates": False, "planned_image_count": 32,
                    "steps": 50, "guidance_scale": 6.0, "adapter_scale": 1.0, "resolution": [512, 512],
                    "scheduler": "Pinned TaxaAdapter DDIM settings", "input_sha256": input_hashes,
                    "original_embedding_sha256": sha(OUT / "original_embedding.npy"),
                    "blind_randomization_seed": 2026100401,
                    "scoring": "Human review; both wing covers visible from above; uncertainty NA, never zero",
                    "hardware_stop": {"temperature_c_at_least": 88, "free_mib_below": 2048,
                                      "cuda_errors": "Stop; no automatic fallback or retry"},
                    "interpretation": "Generator visibility calibration; no causal trait claims or interventions"}
        write_json(OUT / "protocol.json", protocol)
        print("Saved protocol; loading frozen generator", flush=True)
        adapter, _, tokenizer, modules = load_generator(ROOT.parent.parent.parent / "task4_bioclip_pilot/cache/model")
        import torch
        torch.set_num_threads(8)
        assert not torch.is_grad_enabled()
        hardware_sample("after_load")
        hashes_before = frozen_hashes(modules)
        with torch.inference_mode():
            official = adapter.image_encoder.encode_text(tokenizer([row["taxonomy"]]).cuda()).float()
            x = torch.from_numpy(vector[None]).cuda()
            cosine = float(torch.nn.functional.cosine_similarity(x, official).item())
            assert cosine > 0.999
        generation_started = time.perf_counter()
        for seed in SEEDS:
            hardware_sample(f"before_seed_{seed}")
            tick = time.perf_counter()
            image = adapter.generate(clip_image_embeds=x, num_samples=1, seed=seed,
                                     prompt=PROMPT, negative_prompt=NEGATIVE, guidance_scale=6.0,
                                     scale=1.0, num_inference_steps=50, height=512, width=512)[0]
            assert image.size == (512, 512) and np.asarray(image).std() > 1
            name = f"original_seed_{seed}.png"
            image.save(OUT / name)
            hardware_sample(f"after_seed_{seed}")
            records.append({"seed": seed, "image": name, "sha256": sha(OUT / name),
                            "seconds": time.perf_counter() - tick, "size": [512, 512]})
            write_json(OUT / "image_records.json", records)
            print(f"Saved {len(records)}/32; seed {seed}; {records[-1]['seconds']:.2f}s", flush=True)
        generation_seconds = time.perf_counter() - generation_started
        assert frozen_hashes(modules) == hashes_before, "Model state changed during inference"
        assert {path: sha(path) for path in input_hashes} == input_hashes, "Saved inputs changed"
        blinded = make_packet(records)
        for record in records:
            assert sha(OUT / record["image"]) == record["sha256"]
        for record in blinded:
            assert sha(OUT / record["blind_image"]) == record["sha256"]
        hardware = [json.loads(line) for line in (OUT / "hardware.jsonl").read_text().splitlines()]
        manifest = {"status": "completed", "training_performed": False, "model_updates": False,
                    "model_state_sha256": hashes_before, "all_model_state_hashes_unchanged": True,
                    "all_input_hashes_unchanged": True, "frozen_parameter_assertions_passed": True,
                    "image_count": len(records), "image_sha256_checks_passed": True,
                    "human_scoring_status": "blank; requires actual human review", "ai_scoring_performed": False,
                    "generation_seconds": generation_seconds, "elapsed_seconds": time.perf_counter() - started,
                    "gpu": preflight["gpu"], "minimum_free_mib": min(r["free_mib"] for r in hardware),
                    "maximum_temperature_c": max(r["temperature_c"] for r in hardware),
                    "hardware_guard_triggered": False, "official_embedding_cosine": cosine,
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ["torch", "numpy", "diffusers", "transformers", "open_clip_torch"]},
                    "script_sha256": {name: sha(ROOT / name) for name in
                                      ["run_unedited_baseline.py", "run_frozen_experiments.py", "run_interventions.py"]}}
        write_json(OUT / "run_manifest.json", manifest)
        (OUT / "REPORT.md").write_text(
            "# Unedited dorsal-view generator calibration\n\n"
            f"Generated 32 unedited {SPECIES} images at 512 x 512 using seeds 1000-1031 and "
            "one prespecified overhead-view prompt. Neither spots nor spot count were named in the ordinary prompt. "
            "The original space-formatted BioCLIP vector was injected without any SAE operation.\n\n"
            f"Generation took {generation_seconds:.1f} seconds. All model state hashes and saved input hashes "
            f"were unchanged. GPU: {preflight['gpu']}; maximum sampled temperature "
            f"{manifest['maximum_temperature_c']} C; minimum sampled free VRAM {manifest['minimum_free_mib']} MiB. "
            "No hardware guard, OOM, or CUDA error occurred.\n\n"
            "## Review packet\n\n"
            "[Scoring rules](human_review/SCORING_RULES.md) | [Blank human scoring CSV](human_review/scoring.csv)\n\n"
            + "\n".join(f"- [Native-resolution contact sheet {n}](human_review/contact_sheet_{n:02d}.png)" for n in range(1, 7))
            + "\n\nIndividual blind images are in human_review/images/. The seed key is separate in blind_key.csv. "
            "No human or AI scores have been fabricated. The generator's spot reliability remains unmeasured "
            "until actual review is completed. This packet supplies the requested calibration images and strict "
            "rules, not biological validation.\n\n"
            "Full provenance: protocol.json, run_manifest.json, image_records.json, and hardware.jsonl.\n",
            encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
    except BaseException as exc:
        write_json(OUT / "STOPPED.json", {"status": "stopped", "error": repr(exc),
                   "traceback": traceback.format_exc(), "completed_images": len(records),
                   "elapsed_seconds": time.perf_counter() - started, "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
