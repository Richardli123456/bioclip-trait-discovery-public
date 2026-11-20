"""Residual-preserving controls and one-coordinate species bridges. Inference only.

The script never calls the preparation/training scripts and refuses to overwrite
an experiment directory. --plan-only writes the complete numerical plan without
loading the generator. All pretrained modules are frozen before image inference.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import sys
import time

from experiment_common import (ROOT, ALIGNED, EPS, FrozenArtifacts, bridge_codes,
                               cosine, rank_neighbors, sha, support_counts, write_json)
import numpy as np


def plan(data, species, bridge_sources):
    vectors, records, selections, bridges = [], [], [], []

    def add(kind, source, condition, vector, **details):
        index = len(vectors)
        vectors.append(np.asarray(vector, dtype=np.float32))
        records.append({"vector_index": index, "kind": kind, "source": source,
                        "condition": condition, **details})

    for name in species:
        i = data.by_name[name]
        x, z = data.x[i].astype(np.float64), data.z[i].astype(np.float64)
        reconstructed = data.decode(z)
        candidate = data.family_candidate(i)
        j = candidate["feature"]
        direction = z[j] * data.decoder[:, j] * data.scale
        low_family = []
        for f in data.features:
            control = f["feature"]
            top = f["top_training_species"]
            fraction = sum(r["family"] == data.rows[i]["family"] for r in top) / max(1, len(top))
            if control != j and z[control] > EPS and top and fraction <= 0.25:
                control_norm = float(np.linalg.norm(z[control] * data.decoder[:, control] * data.scale))
                low_family.append((abs(control_norm - np.linalg.norm(direction)), control, fraction))
        if not low_family:
            raise ValueError(f"No active low-family-association control for {name}")
        _, control, control_fraction = min(low_family)
        rng = np.random.default_rng(20261002 + j)
        random_direction = rng.normal(size=768)
        random_direction -= direction * np.dot(random_direction, direction) / np.dot(direction, direction)
        random_direction *= np.linalg.norm(direction) / np.linalg.norm(random_direction)
        selections.append({"species": name, "feature": j, "activation": float(z[j]),
                           "selection_rule": "Strongest active feature with >=7/8 training top examples in the family and >=2 genera",
                           "candidate_training_examples": candidate["top_training_species"],
                           "control_feature": control, "control_training_family_fraction": control_fraction,
                           "control_training_examples": data.features[control]["top_training_species"],
                           "feature_change_norm": float(np.linalg.norm(direction)),
                           "control_change_norm": float(np.linalg.norm(z[control] * data.decoder[:, control] * data.scale)),
                           "random_control_change_norm": float(np.linalg.norm(random_direction)),
                           "interpretation": "Family-associated candidate; no named trait assumed"})
        add("intervention", name, "original", x)
        add("intervention", name, "reconstruction", reconstructed)
        for factor, condition in [(0.0, "remove_residual"), (0.5, "half_residual"), (2.0, "double_residual")]:
            edited = z.copy()
            edited[j] *= factor
            # Add back the original reconstruction residual: x' = x + D(z') - D(z).
            vector = x + data.decode(edited) - reconstructed
            assert np.allclose(vector - x, (factor - 1) * direction, atol=1e-10)
            add("intervention", name, condition, vector, feature=j, factor=factor,
                residual_preserved=True, edited_active_features=int((edited > EPS).sum()))
        control_code = z.copy()
        control_code[control] = 0
        add("intervention", name, "control_remove_residual", x + data.decode(control_code) - reconstructed,
            feature=control, residual_preserved=True, edited_active_features=int((control_code > EPS).sum()))
        add("intervention", name, "random_direction_control", x - random_direction,
            random_seed=20261002 + j, norm_matched=True, orthogonal_to_candidate=True)

    support = data.z > EPS
    for name in bridge_sources:
        i = data.by_name[name]
        counts = support_counts(support, i)
        ranked = rank_neighbors(counts, data.names, i, 10)
        target_index = int(ranked[0])
        target = data.names[target_index]
        codes, steps = bridge_codes(data.z[i], data.z[target_index], data.decoder, data.scale)
        decoded = data.decode(codes)
        bridge = {"source": name, "target": target, "shared_active": int(counts[target_index]),
                  "jaccard": float(counts[target_index] / (64 - counts[target_index])),
                  "selection": "Highest binary SAE-support overlap among all non-self species; alphabetical ties",
                  "equally_top_ranked_peers": int(np.count_nonzero(counts[np.arange(len(counts)) != i] == counts[target_index])),
                  "coordinate_order": "Descending absolute activation change times decoder-direction norm; feature-index ties",
                  "path": "Decoded SAE codes: source reconstruction to target reconstruction",
                  "reconstruction_residual_policy": "No residual added on this path; original endpoints generated separately",
                  "changed_dimensions": len(steps), "steps": steps,
                  "max_active_along_path": int((codes > EPS).sum(1).max()),
                  "source_reconstruction_cosine": cosine(decoded[0], data.x[i]),
                  "target_reconstruction_cosine": cosine(decoded[-1], data.x[target_index]),
                  "source_residual_norm": float(np.linalg.norm(data.x[i] - decoded[0])),
                  "target_residual_norm": float(np.linalg.norm(data.x[target_index] - decoded[-1]))}
        bridges.append(bridge)
        add("bridge", name, "source_original", data.x[i], target=target, step=None)
        for step_index, vector in enumerate(decoded):
            add("bridge", name, f"step_{step_index:03d}", vector, target=target, step=step_index,
                edited_feature=None if step_index == 0 else steps[step_index - 1]["feature"],
                active_features=int((codes[step_index] > EPS).sum()),
                code_distance_to_target=float(np.linalg.norm(codes[step_index] - codes[-1])))
        add("bridge", name, "target_original", data.x[target_index], target=target, step=None)
    return np.stack(vectors), records, selections, bridges


def load_generator(model_path):
    # Import after explicitly choosing writable/offline caches. Existing local weights only.
    os.environ.setdefault("HF_HOME", str(ROOT / "cache/hf_runtime"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache/hf_runtime/transformers"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import open_clip
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
    sys.path.insert(0, str(ROOT))
    from run_interventions import EmbeddingTaxaAdapter
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded inference setup requires CUDA")
    torch.set_grad_enabled(False)
    name = "local-dir:" + str(model_path.resolve())
    bio, _, preprocess = open_clip.create_model_and_transforms(
        name, device="cuda", precision="fp32", require_pretrained=True)
    bio.eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(ROOT / "cache/vae", torch_dtype=torch.float16, local_files_only=True)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
    pipe = StableDiffusionPipeline.from_pretrained(
        ROOT / "cache/sd15", torch_dtype=torch.float16, variant="fp16", scheduler=scheduler,
        vae=vae, feature_extractor=None, safety_checker=None, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    adapter = EmbeddingTaxaAdapter(pipe, image_encoder_path=None,
                                  ip_ckpt=str(ROOT / "cache/adapter/taxaadapter_tol-1m.bin"),
                                  device="cuda", num_tokens=4, model_type="bioclip",
                                  bioclip=bio, taxabind=torch.nn.Identity())
    modules = [bio, pipe.unet, pipe.vae, pipe.text_encoder, adapter.image_proj_model]
    for module in modules:
        module.eval().requires_grad_(False)
    assert all(not p.requires_grad for module in modules for p in module.parameters())
    return adapter, preprocess, open_clip.get_tokenizer(name), modules


def make_panels(out, records, seeds, bridge_seeds):
    from PIL import Image, ImageDraw
    from run_interventions import font
    intervention_names = list(dict.fromkeys(r["source"] for r in records if r["kind"] == "intervention"))
    for name in intervention_names:
        relevant = [r for r in records if r["kind"] == "intervention" and r["source"] == name]
        labels = [r["condition"] for r in relevant if r["seed"] == seeds[0]]
        cell, heading, label_height = 256, 80, 55
        canvas = Image.new("RGB", (len(labels) * cell, heading + len(seeds) * (cell + label_height)), "#f4f6f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), f"Frozen SAE interventions | {name}", font=font(25), fill="#173648")
        draw.text((14, 46), "Fixed prompt/seed per row; feature edits retain the original reconstruction residual.",
                  font=font(17), fill="#334b59")
        for row, seed in enumerate(seeds):
            for column, condition in enumerate(labels):
                record = next(r for r in relevant if r["seed"] == seed and r["condition"] == condition)
                left, top = column * cell, heading + row * (cell + label_height)
                draw.text((left + 8, top + 5), f"Seed {seed}", font=font(16), fill="#173648")
                draw.text((left + 8, top + 26), condition.replace("_residual", " + residual").replace("_", " "),
                          font=font(14), fill="#173648")
                with Image.open(out / record["image"]) as image:
                    canvas.paste(image.convert("RGB").resize((cell, cell)), (left, top + label_height))
        canvas.save(out / (name.replace(" ", "_") + "_interventions.png"))
    for source in dict.fromkeys(r["source"] for r in records if r["kind"] == "bridge"):
        relevant = [r for r in records if r["kind"] == "bridge" and r["source"] == source]
        target = relevant[0]["target"]
        max_step = max(r["step"] for r in relevant if r["step"] is not None)
        selected = sorted(set(np.linspace(0, max_step, 6).round().astype(int).tolist()))
        conditions = ["source_original"] + [f"step_{step:03d}" for step in selected] + ["target_original"]
        cell, heading, label_height = 256, 85, 55
        canvas = Image.new("RGB", (len(conditions) * cell, heading + len(bridge_seeds) * (cell + label_height)), "#f4f6f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 10), f"SAE bridge | {source} -> {target}", font=font(25), fill="#173648")
        draw.text((14, 47), "Selected frames shown; every intervening one-coordinate step was generated and saved.",
                  font=font(17), fill="#334b59")
        for row, seed in enumerate(bridge_seeds):
            for column, condition in enumerate(conditions):
                record = next(r for r in relevant if r["seed"] == seed and r["condition"] == condition)
                left, top = column * cell, heading + row * (cell + label_height)
                label = condition.replace("_", " ")
                if record["step"] == 0:
                    label = "source reconstructed"
                elif record["step"] == max_step:
                    label = "target reconstructed"
                draw.text((left + 8, top + 5), f"Seed {seed}", font=font(16), fill="#173648")
                draw.text((left + 8, top + 26), label, font=font(15), fill="#173648")
                with Image.open(out / record["image"]) as image:
                    canvas.paste(image.convert("RGB").resize((cell, cell)), (left, top + label_height))
        canvas.save(out / (source.replace(" ", "_") + "_bridge.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/existing_sae_v1")
    parser.add_argument("--species", nargs="+", default=["Asio otus", "Coccinella septempunctata", "Tamiasciurus douglasii"])
    parser.add_argument("--bridge-sources", nargs="*", default=["Asio otus"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--bridge-seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--prompt", default="best quality, high quality")
    parser.add_argument("--bioclip-model-path", type=Path,
                        default=ROOT.parent.parent / "task4_bioclip_pilot/cache/model")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    for values in [args.species, args.bridge_sources, args.seeds, args.bridge_seeds]:
        if len(set(values)) != len(values):
            raise ValueError("Duplicate species or seeds would overwrite images")
    if not args.species and not args.bridge_sources:
        raise ValueError("Nothing to run")
    if args.steps < 1 or not args.seeds or (args.bridge_sources and not args.bridge_seeds):
        raise ValueError("Steps and seed lists must be nonempty")
    started = time.perf_counter()
    data = FrozenArtifacts(args.aligned_dir)
    vectors, plan_records, selections, bridges = plan(data, args.species, args.bridge_sources)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir
    np.save(out / "vectors.npy", vectors)
    protocol = {"training_performed": False, "model_updates": False,
                "intervention_species": args.species, "intervention_seeds": args.seeds,
                "bridge_seeds": args.bridge_seeds, "prompt": args.prompt,
                "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
                "steps": args.steps, "guidance_scale": args.guidance, "adapter_scale": 1.0,
                "resolution": [512, 512], "scheduler": "Pinned upstream DDIM settings",
                "intervention_equation": "x_edited = x_original + decode(z_edited) - decode(z_original)",
                "random_control": "Equal input-change norm, orthogonal to the candidate decoder direction; fixed seed",
                "active_feature_control": "Feature with <=25% same-family training examples and closest removal norm; not an exact norm match",
                "bridge_endpoint_policy": "Decoded source and target codes; raw endpoint images are separate references",
                "feature_selections": selections, "bridges": bridges, "vectors": plan_records,
                "vector_sha256": sha(out / "vectors.npy"), "provenance": data.provenance(),
                "planned_image_count": sum(len(args.seeds) if r["kind"] == "intervention" else len(args.bridge_seeds)
                                           for r in plan_records),
                "limits": ["Preselected family associations are hypotheses, not named biological traits",
                           "Edited and bridge codes can fall outside the natural text-embedding distribution",
                           "Bridges are order-dependent and may temporarily have more than 32 active features",
                           "BioCLIP image/text cosine is a model-internal species-alignment proxy, not independent trait validation",
                           "No full TreeOfLife-1M species coverage or new SAE training is performed"]}
    write_json(out / "protocol.json", protocol)
    print(f"Prepared {len(vectors)} vectors / {protocol['planned_image_count']} images; no training", flush=True)
    if args.plan_only:
        return
    adapter, preprocess, tokenizer, modules = load_generator(args.bioclip_model_path)
    import torch
    from taxa_adapter import IPAdapter
    precision_checks = {}
    checked_species = set(args.species + args.bridge_sources + [b["target"] for b in bridges])
    with torch.inference_mode():
        for name in sorted(checked_species):
            i = data.by_name[name]
            tokens = tokenizer([data.rows[i]["taxonomy"]]).cuda()
            official = adapter.image_encoder.encode_text(tokens)
            projected = IPAdapter.get_image_embeds(adapter, pil_image=tokens)
            injected = adapter.get_image_embeds(clip_image_embeds=official)
            assert all(torch.equal(a, b) for a, b in zip(projected, injected))
            precision_checks[name] = cosine(data.x[i], official.float().cpu().numpy()[0])
    assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    checkpoint_sha_before = sha(ROOT / "cache/adapter" / assets["weight_filename"])
    assert checkpoint_sha_before == assets["weight_sha256"]
    image_records = []
    generated_count = 0
    for record in plan_records:
        vector = vectors[record["vector_index"]]
        tensor = torch.from_numpy(vector[None]).cuda()
        seeds = args.seeds if record["kind"] == "intervention" else args.bridge_seeds
        source_index = data.by_name[record["source"]]
        for seed in seeds:
            image = adapter.generate(clip_image_embeds=tensor, num_samples=1, seed=seed,
                                     prompt=args.prompt, guidance_scale=args.guidance,
                                     scale=1.0, num_inference_steps=args.steps, height=512, width=512)[0]
            assert image.size == (512, 512) and np.asarray(image).std() > 1
            with torch.inference_mode():
                pixels = preprocess(image).unsqueeze(0).cuda().to(torch.float16)
                image_embedding = adapter.image_encoder.encode_image(pixels).float().cpu().numpy()[0]
            filename = f"{record['kind']}_{record['source'].replace(' ', '_')}_{record['condition']}_seed_{seed}.png"
            image.save(out / filename)
            output = {**record, "seed": seed, "image": filename, "image_sha256": sha(out / filename),
                      "embedding_norm": float(np.linalg.norm(vector)),
                      "input_cosine_to_source": cosine(vector, data.x[source_index]),
                      "input_change_norm": float(np.linalg.norm(vector - data.x[source_index])),
                      "image_bioclip_cosine_to_source": cosine(image_embedding, data.x[source_index])}
            if record["kind"] == "bridge":
                output["image_bioclip_cosine_to_target"] = cosine(image_embedding, data.x[data.by_name[record["target"]]])
            image_records.append(output)
            generated_count += 1
            if generated_count % 12 == 0 or generated_count == protocol["planned_image_count"]:
                print(f"Generated {generated_count}/{protocol['planned_image_count']} images", flush=True)
                write_json(out / "image_records.json", image_records)
    assert all(not p.requires_grad for module in modules for p in module.parameters())
    assert sha(ROOT / "cache/adapter" / assets["weight_filename"]) == checkpoint_sha_before
    assert data.provenance() == protocol["provenance"], "Input artifacts changed during inference"
    make_panels(out, image_records, args.seeds, args.bridge_seeds)
    manifest = {"status": "completed", "training_performed": False, "model_updates": False,
                "frozen_parameter_assertions_passed": True, "planned_image_count": protocol["planned_image_count"],
                "image_count": len(image_records), "adapter_checkpoint_sha256": checkpoint_sha_before,
                "input_checkpoints_unchanged": True, "precomputed_projection_matches_upstream_exactly": True,
                "fp32_vs_official_fp16_cosine_by_species": precision_checks,
                "vector_sha256": sha(out / "vectors.npy"), "gpu": torch.cuda.get_device_name(),
                "packages": {n: importlib.metadata.version(n) for n in ["torch", "numpy", "diffusers", "transformers", "open_clip_torch"]},
                "script_sha256": {name: sha(ROOT / name) for name in ["experiment_common.py", "run_frozen_experiments.py", "run_interventions.py"]},
                "assets_manifest_sha256": sha(ROOT / "assets.json"),
                "diffusion_assets_manifest_sha256": sha(ROOT / "diffusion_assets.json"),
                "images": image_records, "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "run_manifest.json", manifest)
    print(f"Completed {len(image_records)} images in {manifest['elapsed_seconds']:.1f}s; all weights frozen", flush=True)


if __name__ == "__main__":
    main()
