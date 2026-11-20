"""Extract BioCLIP 2 vectors for all 5,388 Animalia in official ToL-10M iNat21 metadata."""
from pathlib import Path
import csv, hashlib, json, time
from collections import Counter,defaultdict
import numpy as np
import torch,open_clip

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/'task4_bioclip_pilot'
SOURCE_REV='91debffb7146c32c89d76feb1eb575b555e2ecc7'
SOURCE_URL=f'https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/{SOURCE_REV}/metadata/naming/inat21_name_lookup.json'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    start=time.perf_counter()
    src=ROOT/'data/inat21_name_lookup.json'
    data=json.loads(src.read_text())
    rows=[]
    for key,(taxa,common,ids) in data.items():
        if taxa[0]!='Animalia':continue
        assert len(taxa)==7 and all(isinstance(t,str) and t.strip() for t in taxa)
        assert ' ' not in taxa[-1], taxa
        rows.append(dict(scientific_name=taxa[5]+' '+taxa[6],common_name=common,
            kingdom=taxa[0],phylum=taxa[1],class_name=taxa[2],order=taxa[3],family=taxa[4],genus=taxa[5],
            epithet=taxa[6],taxonomy=', '.join(taxa),source_id=key))
    rows.sort(key=lambda r:r['scientific_name'])
    assert len(rows)==5388 and len({r['scientific_name'] for r in rows})==len(rows)
    # Split whole genera, stratified by class; classes with <3 genera remain training-only.
    classes=defaultdict(set)
    for row in rows: classes[row['class_name']].add(row['genus'])
    allocation={}
    for cls,genera in sorted(classes.items()):
        ordered=sorted(genera,key=lambda g:hashlib.sha256(('20260929:'+cls+':'+g).encode()).hexdigest())
        n=max(1,round(len(ordered)*.1)) if len(ordered)>=3 else 0
        for i,genus in enumerate(ordered):
            allocation[(cls,genus)]='validation' if i<n else 'test' if i<2*n else 'train'
    for row in rows: row['split']=allocation[(row['class_name'],row['genus'])]
    for a,b in [('train','test'),('train','validation'),('validation','test')]:
        assert not ({r['genus'] for r in rows if r['split']==a}&{r['genus'] for r in rows if r['split']==b})
    (ROOT/'data/species.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (ROOT/'data/species.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    device='cuda';torch.manual_seed(0);torch.set_float32_matmul_precision('highest')
    model_name='local-dir:'+str(PILOT/'cache/model')
    model=open_clip.create_model(model_name,device=device,precision='fp32',require_pretrained=True).eval()
    tokenizer=open_clip.get_tokenizer(model_name)
    tokens=tokenizer([r['taxonomy'] for r in rows])
    assert bool((tokens[:,-1]==0).all()),'Possible truncation'
    out=[]
    with torch.inference_mode():
        for i in range(0,len(rows),128):
            out.append(model.encode_text(tokens[i:i+128].to(device),normalize=False).float().cpu().numpy())
            if i%1024==0:print('Encoded',min(i+128,len(rows)),'/',len(rows),flush=True)
        single=model.encode_text(tokens[:1].to(device),normalize=False).float().cpu().numpy()
    x=np.concatenate(out);assert x.shape==(5388,768) and np.isfinite(x).all()
    assert np.allclose(x[:1],single,rtol=1e-4,atol=1e-5)
    np.save(ROOT/'data/embeddings_raw.npy',x)
    np.save(ROOT/'data/embeddings_normalized.npy',x/np.linalg.norm(x,axis=1,keepdims=True))
    np.save(ROOT/'data/token_ids.npy',tokens.numpy())
    metadata=dict(source_url=SOURCE_URL,source_revision=SOURCE_REV,source_sha256=sha(src),
        model=json.loads((PILOT/'data/model_revision.json').read_text()),shape=list(x.shape),
        gpu=torch.cuda.get_device_name(),prompt='Kingdom, Phylum, Class, Order, Family, Genus, specific epithet',
        representation='OpenCLIP pooled projected text output, normalize=False, float32',
        embedding_sha256=sha(ROOT/'data/embeddings_raw.npy'),species_sha256=sha(ROOT/'data/species.csv'),
        splits=dict(Counter(r['split'] for r in rows)),classes=dict(Counter(r['class_name'] for r in rows)),
        families=len({r['family'] for r in rows}),genera=len({r['genus'] for r in rows}),
        splitting='whole genera, deterministic SHA256 ordering within class; 80/10/10 genus targets',
        elapsed_seconds=time.perf_counter()-start,
        limitation='Public TreeOfLife-10M iNat21 animal subset, not confirmed exact TaxaAdapter TreeOfLife-1M training list.')
    (ROOT/'data/dataset_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
