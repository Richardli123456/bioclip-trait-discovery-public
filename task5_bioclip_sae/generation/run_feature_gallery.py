"""Feature gallery: add or remove one stable expanded-SAE feature and render through frozen TaxaAdapter.

Add: set the feature's activation to a (from its current value c, usually 0);
edited = raw + (a - c) * scale * decoder[:, f], equal to decode(z with z_f = a) - decode(z) + raw,
so the reconstruction residual is kept. Levels at or below c are skipped.
Remove/double: feature active; edited = raw + (factor - 1) * z_f * scale * decoder[:, f].
Random control: Gaussian direction orthogonal to the feature direction, matched to the
largest edit norm for that species. Inference only; refuses an existing output directory.
Without --config this reproduces the v1 orange/striped run.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiment_common import ROOT, sha, write_json
from render_expanded_bridge import guard
from run_frozen_experiments import load_generator
from run_unedited_baseline import frozen_hashes

STUDY = ROOT / "experiments/expanded_stability_w2048_k32_v2"
CORPUS = ROOT / "cache/expanded_tol10m_v2"
V1_CONFIG = {
    "features": {"938": {"label": "orange (aurantiaca, naranja)", "add_levels": [5.0, 10.0, 15.0],
                         "remove_species": ["Hypsypops rubicundus", "Danaus plexippus"]},
                 "975": {"label": "striped (vittata, trivittata)", "add_levels": [5.0, 10.0, 15.0],
                         "remove_species": ["Apantesis vittata", "Cassida vittata"]}},
    "add_species": ["Passer domesticus", "Sciurus carolinensis", "Ardea alba", "Carabus nemoralis"]}
NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"
# Hue windows on PIL's 0-255 HSV scale (degrees / 360 * 255); saturated, not dark pixels only.
HUES = {"red": [(0, 7), (246, 255)], "orange": [(10, 30)], "yellow": [(31, 45)], "blue": [(140, 180)]}


def colour_fractions(path):
    # Crude whole-image colour measure, background included.
    hsv = np.asarray(Image.open(path).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (s > 128) & (v > 100)
    return {name: float((vivid & np.any([(h >= lo) & (h <= hi) for lo, hi in windows], axis=0)).mean())
            for name, windows in HUES.items()}


def build_plan(config, rows, x_all, codes_idx, codes_val, decoder, scale):
    by_name = {r["scientific_name"]: i for i, r in enumerate(rows)}
    features = {int(f): spec for f, spec in config["features"].items()}
    vectors, records = [], []

    def add(species, row, condition, vector, feature=None, **details):
        records.append({"vector_index": len(vectors), "species": species,
                        "common_name": rows[row].get("common_name"), "embedding_row": row,
                        "condition": condition, "feature": feature, **details})
        vectors.append(np.asarray(vector, dtype=np.float32))

    def random_control(direction, norm, salt):
        rng = np.random.default_rng(20261023 + salt)
        r = rng.normal(size=768)
        r -= direction * (r @ direction) / (direction @ direction)
        r *= norm / np.linalg.norm(r)
        assert abs(r @ direction) < 1e-8
        return r

    def code(row):
        return dict(zip(np.asarray(codes_idx[row]).tolist(), np.asarray(codes_val[row]).tolist()))

    for species in config["add_species"]:
        row = by_name[species]
        x = np.asarray(x_all[row], dtype=np.float64)
        add(species, row, "original", x)
        for f, spec in features.items():
            current = code(row).get(f, 0.0)
            unit = scale * decoder[:, f]
            levels = [a for a in spec["add_levels"] if a > current]
            assert levels, f"{species}: every add level is below its current activation of feature {f}"
            for a in levels:
                add(species, row, f"add_{a:g}", x + (a - current) * unit, f, activation=a, initial_activation=current)
            add(species, row, "random_control", x + random_control(unit, (levels[-1] - current) * np.linalg.norm(unit), f + row),
                f, matched_norm_of=f"add_{levels[-1]:g}")
    for f, spec in features.items():
        for species in spec["remove_species"]:
            row = by_name[species]
            x = np.asarray(x_all[row], dtype=np.float64)
            zf = code(row).get(f, 0.0)
            assert zf > 1e-7, f"{species} does not use feature {f}"
            direction = zf * scale * decoder[:, f]
            add(species, row, "original", x, f, activation=zf)
            add(species, row, "remove", x - direction, f, activation=0.0)
            add(species, row, "double", x + direction, f, activation=2 * zf)
            add(species, row, "random_control", x + random_control(direction, np.linalg.norm(direction), f + row),
                f, matched_norm_of="remove")
    matrix = np.stack(vectors)
    assert np.isfinite(matrix).all() and matrix.shape[1] == 768
    for record, vector in zip(records, matrix):
        x = np.asarray(x_all[record["embedding_row"]], dtype=np.float64)
        record["edit_norm"] = float(np.linalg.norm(vector - x))
        record["relative_edit"] = record["edit_norm"] / float(np.linalg.norm(x))
    return matrix, records


def contact_sheets(out, results, seeds):
    font = ImageFont.truetype("arial.ttf", 22)
    groups = {}
    for r in results:
        groups.setdefault((r["feature_group"], r["species"]), []).append(r)
    for (group, species), items in groups.items():
        conditions = list(dict.fromkeys(r["condition"] for r in items))
        tile, label = 256, 36
        sheet = Image.new("RGB", (tile * len(conditions), label + tile * len(seeds)), "white")
        draw = ImageDraw.Draw(sheet)
        for c, condition in enumerate(conditions):
            draw.text((c * tile + 8, 6), condition, fill="black", font=font)
            for s, seed in enumerate(seeds):
                item = next(r for r in items if r["condition"] == condition and r["seed"] == seed)
                sheet.paste(Image.open(out / item["image"]).resize((tile, tile)), (c * tile, label + s * tile))
        name = f"sheet_{group}_{species.replace(' ', '_')}.png"
        sheet.save(out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/feature_gallery_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3000, 3001, 3002, 3003])
    parser.add_argument("--prompt", default="best quality, high quality")
    parser.add_argument("--bioclip-model-path", type=Path, default=ROOT.parent.parent.parent / "task4_bioclip_pilot/cache/model")
    parser.add_argument("--config", type=Path, help="JSON with features (label, add_levels, remove_species) and add_species")
    parser.add_argument("--plan-only", action="store_true", help="Print the vector plan and stop; no output or GPU")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else V1_CONFIG
    features = [int(f) for f in config["features"]]

    checkpoint_path = STUDY / "seed_20260929/sae.pt"
    metrics = json.loads((checkpoint_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert sha(checkpoint_path) == metrics["checkpoint_sha256"], "SAE checkpoint changed"
    with (STUDY / "feature_stability.csv").open(encoding="utf-8") as stream:
        stability = {int(r["anchor_feature"]): r for r in csv.DictReader(stream)}
    assert all(stability[f]["passes_prespecified_gate"] == "True" for f in features)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    decoder = state["decoder.weight"].numpy().astype(np.float64)
    scale = float(state["scale"])
    rows = json.loads((CORPUS / "data/species.json").read_text(encoding="utf-8"))
    x_all = np.load(CORPUS / "data/embeddings_raw.npy", mmap_mode="r")
    codes_idx = np.load(checkpoint_path.parent / "code_indices.npy", mmap_mode="r")
    codes_val = np.load(checkpoint_path.parent / "code_values.npy", mmap_mode="r")
    matrix, records = build_plan(config, rows, x_all, codes_idx, codes_val, decoder, scale)
    if args.plan_only:
        for r in records:
            print(f"{r['species']:28s} {r['condition']:15s} f={r['feature']} act={r.get('activation', '')} "
                  f"edit={r['edit_norm']:.2f} ({r['relative_edit']:.0%})")
        print(len(records), "vectors,", len(records) * len(args.seeds), "images")
        return

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    np.save(out / "vectors.npy", matrix)
    weight_path = ROOT / "cache/adapter/taxaadapter_tol-1m.bin"
    assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    assert sha(weight_path) == assets["weight_sha256"], "Adapter differs from pinned released checkpoint"
    protocol = {"config": config, "seeds": args.seeds, "prompt": args.prompt,
                "negative_prompt": NEGATIVE, "steps": 50, "guidance_scale": 6.0, "adapter_scale": 1.0,
                "resolution": [512, 512], "image_count": len(records) * len(args.seeds),
                "sae": {"study": str(STUDY), "seed": 20260929, "checkpoint_sha256": metrics["checkpoint_sha256"]},
                "vectors_sha256": sha(out / "vectors.npy"), "records": records,
                "selection": "Features chosen from Latin-name stems of stable features; species chosen before generation",
                "training_performed": False, "model_updates": False, "script_sha256": sha(Path(__file__))}
    write_json(out / "protocol.json", protocol)

    started, results = time.perf_counter(), []
    try:
        guard(out, "before_load")
        adapter, _, _, modules = load_generator(args.bioclip_model_path)
        torch.set_num_threads(8)
        model_hashes = frozen_hashes(modules)
        calibration = json.loads((ROOT / "experiments/unedited_dorsal_v1/run_manifest.json").read_text(encoding="utf-8"))
        assert model_hashes == calibration["model_state_sha256"], "Generator state differs from the unedited calibration"
        guard(out, "after_load")
        for record in records:
            raw = torch.from_numpy(matrix[record["vector_index"]][None]).cuda()
            if record["species"] in config["add_species"]:
                group = "add_original" if record["condition"] == "original" else f"add_{record['feature']}"
            else:
                group = f"remove_{record['feature']}"
            for seed in args.seeds:
                picture = adapter.generate(clip_image_embeds=raw, num_samples=1, seed=seed,
                                           prompt=args.prompt, negative_prompt=NEGATIVE, guidance_scale=6.,
                                           scale=1., num_inference_steps=50, height=512, width=512)[0]
                assert picture.size == (512, 512) and np.asarray(picture).std() > 1
                name = f"images/v{record['vector_index']:03d}_seed_{seed}.png"
                picture.save(out / name)
                colours = colour_fractions(out / name)
                results.append({**record, "feature_group": group, "seed": seed, "image": name,
                                "sha256": sha(out / name), "orange_fraction": colours["orange"],
                                "colour_fractions": colours})
            write_json(out / "image_records.json", results)
            guard(out, f"after_vector_{record['vector_index']}")
            print(json.dumps({"saved": len(results), "planned": protocol["image_count"]}), flush=True)
        assert frozen_hashes(modules) == model_hashes
        # The add-sheets show the shared original next to each feature's edits.
        sheet_rows = []
        for r in results:
            if r["feature_group"] == "add_original":
                for f in features:
                    sheet_rows.append({**r, "feature_group": f"add_{f}"})
            else:
                sheet_rows.append(r)
        contact_sheets(out, sheet_rows, args.seeds)
        hardware = [json.loads(line) for line in (out / "hardware.jsonl").read_text().splitlines()]
        write_json(out / "run_manifest.json", {"status": "completed", "image_count": len(results),
                   "elapsed_seconds": time.perf_counter() - started, "model_state_sha256": model_hashes,
                   "all_model_hashes_unchanged": True, "maximum_temperature_c": max(r["temperature_c"] for r in hardware),
                   "minimum_free_mib": min(r["free_mib"] for r in hardware), "training_performed": False})
    except BaseException as error:
        write_json(out / "STOPPED.json", {"error": repr(error), "traceback": traceback.format_exc(),
                   "saved_images": len(results), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
