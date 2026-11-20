"""Render a saved one-coordinate expanded-SAE bridge through frozen TaxaAdapter.

Inference only; preserves all numerical plans and refuses a second execution.
Human biological interpretation is not supplied by this renderer.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import html
import json
import subprocess
import time
import traceback

import numpy as np
from experiment_common import ROOT, sha, write_json
from run_frozen_experiments import load_generator
from run_unedited_baseline import frozen_hashes


def guard(out, stage):
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    fields = next(csv.reader(result.stdout.splitlines()))
    row = {"stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "gpu": fields[0].strip(), "total_mib": int(fields[1]),
           "free_mib": int(fields[2]), "temperature_c": int(fields[3])}
    with (out / "hardware.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    if row["temperature_c"] >= 88 or row["free_mib"] < 2048:
        raise RuntimeError("Hardware stop: " + json.dumps(row))
    return row


def make_viewer(out, records, source, target, seeds, kind="bridge"):
    # Native saved pixels; controls only change which existing PNG is displayed.
    payload = json.dumps(records).replace("</", "<\\/")
    title = html.escape(f"{source} → {target}")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded SAE bridge</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1100px;padding:0 18px;color:#172c3c;background:#f5f7fa}h1{font-size:26px}button,select,input{font:inherit;margin:5px;padding:8px}#images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:8px 0}img{max-width:100%;width:512px;height:auto}figcaption{font-weight:600}#description{min-height:3em;background:white;padding:14px;border-radius:8px}input{width:75%}.note{color:#526371}</style>
<h1>__TITLE__</h1><p>Expanded SAE: one coordinate changes per step, retaining the source reconstruction residual. Saved model-generated images; this page runs no model.</p>
<p class="note">An apparent change is an exploratory observation, not evidence that a coordinate controls an independent biological trait. The last residual-preserving step can differ from the raw target.</p>
<label>Condition <select id="condition"></select></label><button id="previous">Previous</button><button id="next">Next</button><br>
<input id="slider" type="range" min="0" step="1"><span id="position"></span><p id="description"></p><div id="images"></div>
<p><a href="run_manifest.json">Generation provenance</a> · <a href="image_records.json">Every image and hash</a></p>
<script>const records=__RECORDS__,seeds=__SEEDS__;const base=records.filter(r=>r.seed===seeds[0]);const select=document.querySelector('#condition'),slider=document.querySelector('#slider');base.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.step===null?r.condition:`Step ${r.step} · feature ${r.edited_feature===null?'unchanged':r.edited_feature}`;select.append(o)});slider.max=base.length-1;
function show(i){i=Math.max(0,Math.min(base.length-1,Number(i)));select.value=slider.value=i;document.querySelector('#position').textContent=`${i+1} / ${base.length}`;const item=base[i];document.querySelector('#description').textContent=`${item.condition} · ${item.step===null?'endpoint reference':`step ${item.step}`} · ${item.edited_feature===null?'no edited coordinate at this reference':`coordinate ${item.edited_feature}`}`;const box=document.querySelector('#images');box.replaceChildren();seeds.forEach(seed=>{const r=records.find(r=>r.seed===seed&&r.vector_index===item.vector_index),fig=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=r.image;im.alt=`${r.condition}, seed ${seed}`;im.width=512;im.height=512;cap.textContent=`Seed ${seed}`;fig.append(cap,im);box.append(fig)});document.querySelector('#previous').disabled=i===0;document.querySelector('#next').disabled=i===base.length-1}
select.onchange=()=>show(select.value);slider.oninput=()=>show(slider.value);document.querySelector('#previous').onclick=()=>show(Number(select.value)-1);document.querySelector('#next').onclick=()=>show(Number(select.value)+1);show(0);</script></html>"""
    if kind != "bridge":
        document = document.replace("Expanded SAE: one coordinate changes per step, retaining the source reconstruction residual.",
                                    "Controlled expanded-SAE vector comparison. Read the locked protocol for conditions and measurement rules.")
        document = document.replace(" The last residual-preserving step can differ from the raw target.", "")
        title = html.escape(source + " — controlled vector comparison")
    document = document.replace("__TITLE__", title).replace("__RECORDS__", payload).replace("__SEEDS__", json.dumps(seeds))
    (out / "viewer.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/expanded_bridge_images_v1")
    parser.add_argument("--source", default="Coccinella septempunctata")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2000, 2001])
    parser.add_argument("--prompt", default="best quality, high quality")
    parser.add_argument("--kind", choices=["bridge", "trait_pilot", "trait_calibration"], default="bridge")
    parser.add_argument("--plan-file", default="bridge_plan.json")
    parser.add_argument("--vectors-file", default="bridge_vectors.npy")
    parser.add_argument("--bioclip-model-path", type=Path, default=ROOT.parent.parent.parent / "task4_bioclip_pilot/cache/model")
    args = parser.parse_args()
    assert args.seeds and len(set(args.seeds)) == len(args.seeds)
    paths = [args.plan_dir / args.plan_file, args.plan_dir / args.vectors_file]
    plan = json.loads(paths[0].read_text(encoding="utf-8"))
    vectors = np.load(paths[1])
    selected = [r for r in plan["records"] if r["source"] == args.source]
    assert selected and vectors.ndim == 2 and vectors.shape[1] == 768 and np.isfinite(vectors).all()
    assert len({r["vector_index"] for r in selected}) == len(selected)
    assert all(isinstance(r["vector_index"], int) and 0 <= r["vector_index"] < len(vectors) for r in selected)
    assert len({r["target"] for r in selected}) == 1
    if args.kind == "bridge":
        selected.sort(key=lambda r: (0 if r["condition"] == "source_original" else 1 if r["condition"] == "source_reconstruction" else 2 if r["step"] is not None else 3 if r["condition"] == "target_reconstruction" else 4, r["step"] if r["step"] is not None else 0))
    expected_vector_hash = plan.get("bridge_vectors_sha256", plan.get("vectors_sha256"))
    if expected_vector_hash:
        assert sha(paths[1]) == expected_vector_hash, "Saved vector plan changed"
    weight_path = ROOT / "cache/adapter/taxaadapter_tol-1m.bin"
    assets = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    assert sha(weight_path) == assets["weight_sha256"], "Adapter differs from pinned released checkpoint"
    upstream_files = [ROOT / "cache/upstream" / name for name in assets["code_sha256"] if name.startswith("taxa_adapter/") and name.endswith(".py")]
    for path in upstream_files:
        relative = path.relative_to(ROOT / "cache/upstream").as_posix()
        assert sha(path) == assets["code_sha256"][relative], "Upstream inference code changed: " + relative
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    hashes = {str(p.resolve()): sha(p) for p in paths + [weight_path, ROOT / "assets.json"] + upstream_files}
    protocol = {"training_performed": False, "model_updates": False, "source": args.source,
                "target": selected[0]["target"], "seeds": args.seeds,
                "prompt": args.prompt, "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
                "steps": 50, "guidance_scale": 6.0, "adapter_scale": 1.0, "resolution": [512, 512],
                "image_count": len(selected) * len(args.seeds), "plan_sha256": hashes,
                "record_selection": "All saved states and references for the prespecified source; no image-based selection",
                "kind": args.kind,
                "interpretation": "Exploratory representation bridge, not a biological trait intervention test" if args.kind == "bridge" else "Prespecified controlled-vector experiment; image endpoint validity and causal interpretation require separate evidence",
                "script_sha256": {p.name: sha(p) for p in [Path(__file__), ROOT / "run_frozen_experiments.py", ROOT / "run_interventions.py", ROOT / "run_unedited_baseline.py"]}}
    write_json(out / "protocol.json", protocol)
    started = time.perf_counter()
    results = []
    try:
        guard(out, "before_load")
        adapter, _, _, modules = load_generator(args.bioclip_model_path)
        import torch
        torch.set_num_threads(8)
        original_model_hashes = frozen_hashes(modules)
        calibration_manifest = ROOT / "experiments/unedited_dorsal_v1/run_manifest.json"
        calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
        assert original_model_hashes == calibration["model_state_sha256"], "Generator state differs from the fresh unedited calibration"
        guard(out, "after_load")
        for item in selected:
            raw = torch.from_numpy(vectors[item["vector_index"]][None].astype(np.float32)).cuda()
            for seed in args.seeds:
                guard(out, f"before_vector_{item['vector_index']}_seed_{seed}")
                picture = adapter.generate(clip_image_embeds=raw, num_samples=1, seed=seed,
                    prompt=protocol["prompt"], negative_prompt=protocol["negative_prompt"],
                    guidance_scale=6., scale=1., num_inference_steps=50, height=512, width=512)[0]
                assert picture.size == (512, 512) and np.asarray(picture).std() > 1
                name = f"images/vector_{item['vector_index']:04d}_seed_{seed}.png"
                picture.save(out / name)
                results.append({**item, "seed": seed, "image": name, "sha256": sha(out / name)})
                write_json(out / "image_records.json", results)
                guard(out, f"after_vector_{item['vector_index']}_seed_{seed}")
                print(json.dumps({"saved": len(results), "planned": protocol["image_count"]}), flush=True)
        assert frozen_hashes(modules) == original_model_hashes
        assert {path: sha(path) for path in hashes} == hashes
        assert all(sha(out / r["image"]) == r["sha256"] for r in results)
        hardware = [json.loads(line) for line in (out / "hardware.jsonl").read_text().splitlines()]
        write_json(out / "run_manifest.json", {"status": "completed", "image_count": len(results),
                   "elapsed_seconds": time.perf_counter() - started, "all_model_hashes_unchanged": True,
                   "model_state_sha256": original_model_hashes, "all_input_and_image_hashes_verified": True,
                   "matches_unedited_calibration_model_state": True, "calibration_manifest_sha256": sha(calibration_manifest),
                   "maximum_temperature_c": max(r["temperature_c"] for r in hardware),
                   "minimum_free_mib": min(r["free_mib"] for r in hardware), "hardware_errors": [],
                   "training_performed": False, "model_updates": False})
        make_viewer(out, results, args.source, selected[0]["target"], args.seeds, args.kind)
    except BaseException as error:
        write_json(out / "STOPPED.json", {"error": repr(error), "traceback": traceback.format_exc(),
                   "saved_images": len(results), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
