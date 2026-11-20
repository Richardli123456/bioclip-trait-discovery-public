"""Extract real BioCLIP 2 taxonomy vectors and validate a 20-species cosine pilot."""
from pathlib import Path
import csv, hashlib, importlib.metadata, json, platform, time
from datetime import datetime, timezone
import numpy as np
import torch
import open_clip

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    start=time.perf_counter()
    rows=json.loads((ROOT/'data/species.json').read_text(encoding='utf-8'))
    assert len(rows)==20 and len({r['scientific_name'] for r in rows})==20
    model_path=ROOT/'cache/model'
    model_name='local-dir:'+str(model_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    model=open_clip.create_model(model_name,device=device,precision='fp32',require_pretrained=True)
    model.eval()
    tokenizer=open_clip.get_tokenizer(model_name)
    prompts=[r['taxonomy'] for r in rows]
    # No common names, prompt ensembles, or photo embeddings enter the main matrix.
    tokens=tokenizer(prompts).to(device)
    assert bool((tokens[:,-1]==0).all()), 'Check unexpected prompt truncation.'
    with torch.inference_mode():
        raw=model.encode_text(tokens,normalize=False).float().cpu().numpy()
        # Re-encode singly to check the saved results are stable across batch size.
        single=np.concatenate([model.encode_text(tokens[i:i+1],normalize=False).float().cpu().numpy()
                               for i in range(len(rows))])
    assert raw.shape==(20,768) and np.isfinite(raw).all()
    norms=np.linalg.norm(raw,axis=1,keepdims=True)
    assert np.all(norms>0)
    normalized=raw/norms
    sim=normalized@normalized.T
    assert np.allclose(sim,sim.T,atol=1e-6)
    assert np.allclose(np.diag(sim),1,atol=1e-5)
    assert np.max(np.abs(sim))<=1.00001
    assert np.allclose(raw,single,rtol=1e-4,atol=1e-5)
    np.save(OUT/'embeddings_raw.npy',raw)
    np.save(OUT/'embeddings_normalized.npy',normalized)
    np.save(OUT/'cosine_similarity.npy',sim)
    np.save(OUT/'token_ids.npy',tokens.cpu().numpy())
    with (OUT/'embeddings_raw.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scientific_name']+[f'dim_{i:03d}' for i in range(768)])
        for row,vec in zip(rows,raw): w.writerow([row['scientific_name']]+vec.tolist())
    with (OUT/'cosine_similarity.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scientific_name']+[r['scientific_name'] for r in rows])
        for row,vec in zip(rows,sim): w.writerow([row['scientific_name']]+vec.tolist())
    nearest=[]
    for i,row in enumerate(rows):
        ranking=sorted((j for j in range(20) if j!=i),key=lambda j:(-float(sim[i,j]),j))[:3]
        entry={'scientific_name':row['scientific_name'],'common_name':row['common_name'],
               'reference_page':row['reference_page']}
        for rank,j in enumerate(ranking,1):
            entry.update({f'neighbor_{rank}':rows[j]['scientific_name'],
                          f'neighbor_{rank}_common':rows[j]['common_name'],
                          f'cosine_{rank}':float(sim[i,j]),
                          f'same_genus_{rank}':row['genus']==rows[j]['genus']})
        nearest.append(entry)
    (OUT/'nearest_neighbors.json').write_text(json.dumps(nearest,indent=2),encoding='utf-8')
    with (OUT/'nearest_neighbors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(nearest[0]));w.writeheader();w.writerows(nearest)
    pairs=[(float(sim[i,j]),rows[i]['scientific_name'],rows[j]['scientific_name'])
           for i in range(20) for j in range(i+1,20)]
    pairs.sort(reverse=True)
    pin=json.loads((ROOT/'data/model_revision.json').read_text())
    metadata={**pin,'created_utc':datetime.now(timezone.utc).isoformat(),
        'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'python':platform.python_version(), 'packages':{p:importlib.metadata.version(p) for p in
          ['torch','torchvision','numpy','open_clip_torch','huggingface_hub','safetensors']},
        'shape':list(raw.shape),'dtype':str(raw.dtype),'precision':'fp32; TF32 disabled',
        'checkpoint_sha256':digest(model_path/'open_clip_model.safetensors'),
        'taxonomy_file_sha256':digest(ROOT/'data/species.csv'),
        'embedding_sha256':digest(OUT/'embeddings_raw.npy'),
        'prompt_format':'Kingdom, Phylum, Class, Order, Family, Genus, specific epithet',
        'pooling':'OpenCLIP encode_text: pooled end-of-text representation followed by text_projection',
        'input_names':prompts,'row_order':[r['scientific_name'] for r in rows],
        'token_lengths':[int((t!=0).sum()) for t in tokens.cpu()],
        'raw_norm_range':[float(norms.min()),float(norms.max())],
        'batch_vs_single_max_absolute_difference':float(np.max(np.abs(raw-single))),
        'top1_same_genus_count':sum(r['same_genus_1'] for r in nearest),
        'top3_same_genus_count':sum(r[f'same_genus_{k}'] for r in nearest for k in [1,2,3]),
        'strongest_pairs':pairs[:10], 'elapsed_seconds':time.perf_counter()-start,
        'scope':'20 hand-selected Passerellidae species; neighbors only within this subset. '
          'No classification benchmark, SAE training, trait validation, or TaxaAdapter inference.'}
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['shape','gpu','elapsed_seconds',
          'top1_same_genus_count','top3_same_genus_count','batch_vs_single_max_absolute_difference',
          'strongest_pairs']},indent=2),flush=True)

if __name__=='__main__': main()
