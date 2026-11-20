"""Build an offline viewer for saved images. No model execution or training."""
from pathlib import Path
import argparse
import json

from experiment_common import ROOT

TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BioCLIP SAE experiment viewer</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#183344;background:#edf2f4}*{box-sizing:border-box}
body{max-width:1260px;margin:auto;padding:24px}h1{font-size:29px;margin-bottom:8px}h2{font-size:21px}
p{line-height:1.5}.intro{color:#466271}.badge{display:inline-block;background:#dceee6;padding:6px 11px;border-radius:6px;font-size:13px}
section{background:white;border:1px solid #d8e1e7;border-radius:10px;margin-top:24px;padding:22px}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:16px;margin:18px 0}label{display:flex;flex-direction:column;gap:7px;font-size:14px}
select,button{font:inherit;padding:8px 12px;border:1px solid #b8c8d2;border-radius:5px;background:white;color:#183344}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.bridge{display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:14px;align-items:start}
figure{margin:0}figure img{display:block;width:100%;aspect-ratio:1;border-radius:6px;background:#e8eef2}
figcaption{margin:8px 0;font-size:14px;font-weight:600}.details{font-size:14px;line-height:1.65;color:#405d6e;min-height:24px}
.slider{flex:1;min-width:260px}input[type=range]{width:100%}.note{padding:12px 16px;background:#f3f6f8;border-left:3px solid #7896aa;font-size:14px}
.counter{font-variant-numeric:tabular-nums;font-weight:600}a{color:#235f87}
@media(max-width:700px){body{padding:12px}section{padding:16px}.bridge{grid-template-columns:1fr}.pair{grid-template-columns:1fr}}
</style></head><body>
<span class="badge">Existing checkpoints · inference only · 188 saved images</span>
<h1>BioCLIP SAE experiment viewer</h1>
<p class="intro">Compare fixed-seed feature interventions and inspect every coordinate in the long-eared to short-eared owl bridge. Images are saved model outputs; changing these controls only selects an image.</p>
<section><h2>Feature interventions</h2>
<p>Selected-feature edits preserve the original reconstruction residual: <strong>x′ = x + D(z′) − D(z)</strong>.</p>
<div class="controls"><label>Species<select id="species"></select></label><label>Image seed<select id="seed"></select></label><label>Condition<select id="condition"></select></label></div>
<div class="pair"><figure><img id="original" alt="Original taxonomy embedding image"><figcaption>Original taxonomy embedding</figcaption></figure><figure><img id="edited" alt="Selected intervention image"><figcaption id="editedCaption"></figcaption></figure></div>
<p id="featureInfo" class="details"></p><p id="scores" class="details"></p>
<p class="note">Candidates were selected by family association among training examples. BioCLIP cosine measures model-internal species alignment; no specific biological trait has been established. The active-feature control has a different edit magnitude; the random-direction control has an exactly matched input-change norm.</p>
</section>
<section><h2>One-coordinate species bridge</h2><p id="bridgeTitle"></p>
<div class="controls"><label>Image seed<select id="bridgeSeed"></select></label><button id="previous" type="button">Previous step</button><button id="next" type="button">Next step</button><label class="slider">SAE coordinate replacements<input id="step" type="range" min="0" step="1" value="0"></label><span id="counter" class="counter"></span></div>
<div class="bridge"><figure><img id="sourceReference" alt="Original source species reference"><figcaption>Original source embedding</figcaption></figure><figure><img id="bridgeFrame" alt="Selected bridge step"><figcaption id="bridgeCaption"></figcaption></figure><figure><img id="targetReference" alt="Original target species reference"><figcaption>Original target embedding</figcaption></figure></div>
<p id="stepInfo" class="details"></p><p id="bridgeScores" class="details"></p>
<p class="note">The path starts at the source SAE reconstruction and ends at the target SAE reconstruction. Each adjacent saved step changes exactly one coordinate. The original source and target above provide separate references. The path order follows decoder-change magnitude and can affect the visual result.</p>
</section><p class="intro">See <a href="REPORT.md">experiment report</a>, <a href="verification.json">verification</a> and <a href="summary.json">score summary</a>. No training was performed in this continuation.</p>
<script>const data=__DATA__;
const $=id=>document.getElementById(id);
const labels={original:'Original embedding',reconstruction:'Unedited SAE reconstruction',remove_residual:'Selected feature: 0× + residual',half_residual:'Selected feature: 0.5× + residual',double_residual:'Selected feature: 2× + residual',control_remove_residual:'Other active feature: removed + residual',random_direction_control:'Random direction: matched norm'};
const addOptions=(id,values,label=x=>x)=>{for(const value of values){const item=document.createElement('option');item.value=value;item.textContent=label(value);$(id).append(item)}};
addOptions('species',data.species);addOptions('seed',data.seeds);addOptions('condition',Object.keys(labels),x=>labels[x]);
$('condition').value='remove_residual';
const find=(kind,source,condition,seed)=>data.images.find(r=>r.kind===kind&&r.source===source&&r.condition===condition&&r.seed===Number(seed));
function showIntervention(){const name=$('species').value,seed=$('seed').value,condition=$('condition').value;
const raw=find('intervention',name,'original',seed),edited=find('intervention',name,condition,seed),selection=data.selections.find(r=>r.species===name);
$('original').src=raw.image;$('edited').src=edited.image;$('editedCaption').textContent=labels[condition];
$('featureInfo').textContent=`Selected feature ${selection.feature}; original activation ${selection.activation.toFixed(4)}. Active-feature control ${selection.control_feature}.`;
$('scores').textContent=`BioCLIP species alignment: original ${raw.image_bioclip_cosine_to_source.toFixed(4)}; selected image ${edited.image_bioclip_cosine_to_source.toFixed(4)}. Input-change norm ${edited.input_change_norm.toFixed(4)}.`;}
for(const id of ['species','seed','condition'])$(id).addEventListener('change',showIntervention);showIntervention();
const bridge=data.bridge;addOptions('bridgeSeed',data.bridgeSeeds);$('step').max=bridge.changed_dimensions;
$('bridgeTitle').textContent=`${bridge.source} → ${bridge.target}: ${bridge.shared_active}/32 shared active features, ${bridge.changed_dimensions} coordinate changes.`;
function showBridge(){const seed=$('bridgeSeed').value,step=Number($('step').value),condition=`step_${String(step).padStart(3,'0')}`,record=find('bridge',bridge.source,condition,seed);
$('sourceReference').src=find('bridge',bridge.source,'source_original',seed).image;$('targetReference').src=find('bridge',bridge.source,'target_original',seed).image;$('bridgeFrame').src=record.image;
$('counter').textContent=`${step} / ${bridge.changed_dimensions}`;$('previous').disabled=step===0;$('next').disabled=step===bridge.changed_dimensions;
$('bridgeCaption').textContent=step===0?'Source reconstructed':step===bridge.changed_dimensions?'Target reconstructed':`Bridge step ${step}`;
const edit=step?bridge.steps[step-1]:null;$('stepInfo').textContent=edit?`Changed feature ${edit.feature}: ${edit.from.toFixed(4)} → ${edit.to.toFixed(4)}. Active coordinates: ${record.active_features}. Remaining code distance: ${record.code_distance_to_target.toFixed(4)}.`:`Starting source code; ${record.active_features} active coordinates.`;
$('bridgeScores').textContent=`BioCLIP alignment: source ${record.image_bioclip_cosine_to_source.toFixed(4)}; target ${record.image_bioclip_cosine_to_target.toFixed(4)}. These scores are proxies, not trait annotations.`;}
$('bridgeSeed').addEventListener('change',showBridge);$('step').addEventListener('input',showBridge);
$('previous').addEventListener('click',()=>{$('step').value=Number($('step').value)-1;showBridge()});$('next').addEventListener('click',()=>{$('step').value=Number($('step').value)+1;showBridge()});showBridge();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/existing_sae_v1")
    args = parser.parse_args()
    protocol = json.loads((args.out_dir / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and len(protocol["bridges"]) == 1
    data = {"species": protocol["intervention_species"], "seeds": protocol["intervention_seeds"],
            "bridgeSeeds": protocol["bridge_seeds"], "bridge": protocol["bridges"][0],
            "selections": protocol["feature_selections"], "images": manifest["images"]}
    serialized = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    html = TEMPLATE.replace("__DATA__", serialized).replace("188 saved images", f"{manifest['image_count']} saved images")
    (args.out_dir / "viewer.html").write_text(html, encoding="utf-8")
    print(f"Saved offline viewer: {args.out_dir / 'viewer.html'}", flush=True)


if __name__ == "__main__":
    main()
