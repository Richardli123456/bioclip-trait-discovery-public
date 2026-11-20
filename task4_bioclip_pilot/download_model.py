"""Pin and download the official BioCLIP 2 configuration and safetensors weights."""
from pathlib import Path
import json, urllib.request, time
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parent
out=ROOT/'cache'/'model'
out.mkdir(parents=True,exist_ok=True)
pin=ROOT/'data'/'model_revision.json'
if pin.exists():
    info=json.loads(pin.read_text())
else:
    with urllib.request.urlopen('https://huggingface.co/api/models/imageomics/bioclip-2') as r:
        metadata=json.load(r)
    info={'repo_id':'imageomics/bioclip-2','revision':metadata['sha']}
    pin.write_text(json.dumps(info,indent=2))
for filename in ['open_clip_config.json','open_clip_model.safetensors']:
    print('Downloading',filename,'revision',info['revision'],flush=True)
    path=hf_hub_download(info['repo_id'],filename,revision=info['revision'],local_dir=str(out))
    print('Ready',path,flush=True)
