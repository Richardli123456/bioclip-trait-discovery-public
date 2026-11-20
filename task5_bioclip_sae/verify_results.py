"""Meaningful artifact checks, independent metric recomputation and SAE round-trip."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from PIL import Image
from train_sae import TopKSAE
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/species.json').read_text());meta=json.loads((ROOT/'data/dataset_metadata.json').read_text())
x=np.load(ROOT/'data/embeddings_raw.npy');z=np.load(ROOT/'results/sparse_vectors.npy');y=np.load(ROOT/'results/reconstructed_embeddings.npy')
assert x.shape==(5388,768) and z.shape==(5388,2048) and y.shape==x.shape
assert all(np.isfinite(a).all() for a in [x,z,y]) and (z>=0).all()
assert ((z>1e-7).sum(1)==32).all()
assert hashlib.sha256((ROOT/'data/embeddings_raw.npy').read_bytes()).hexdigest()==meta['embedding_sha256']
assert hashlib.sha256((ROOT/'data/species.csv').read_bytes()).hexdigest()==meta['species_sha256']
splits={s:{r['genus'] for r in rows if r['split']==s} for s in ['train','validation','test']}
assert not (splits['train']&splits['test'] or splits['train']&splits['validation'] or splits['test']&splits['validation'])
checkpoint=torch.load(ROOT/'results/sae.pt',map_location='cpu',weights_only=True)
model=TopKSAE(checkpoint['state_dict']['mean'],checkpoint['state_dict']['scale']).eval()
model.load_state_dict(checkpoint['state_dict'])
with torch.inference_mode():
    loaded_y,loaded_z=model(torch.from_numpy(x[:16]))
    assert np.allclose(loaded_y.numpy(),y[:16],atol=2e-5,rtol=1e-4)
    assert np.allclose(loaded_z.numpy(),z[:16],atol=2e-5,rtol=1e-4)
    # A single-feature intervention follows exactly its decoder direction.
    j=int(loaded_z[0].argmax());changed=loaded_z[:1].clone();changed[0,j]=0
    delta=model.decode(changed)-model.decode(loaded_z[:1])
    expected=-loaded_z[0,j]*model.decoder.weight[:,j]*model.scale
    assert torch.allclose(delta[0],expected,atol=1e-5,rtol=1e-4)
scores=json.loads((ROOT/'results/evaluation.json').read_text())
trainmean=x[[i for i,r in enumerate(rows) if r['split']=='train']].astype('float64').mean(0)
for s in splits:
    ii=[i for i,r in enumerate(rows) if r['split']==s]
    a=x[ii].astype('float64');b=y[ii].astype('float64')
    nmse=np.square(a-b).sum()/np.square(a-trainmean).sum()
    cosine=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
    assert abs(nmse-scores['metrics'][s]['normalized_mse'])<1e-6
    assert abs(cosine.mean()-scores['metrics'][s]['cosine_mean'])<1e-6
features=json.loads((ROOT/'results/selected_features.json').read_text())
assert len(features)==5
for f in features:
    assert len(f['photo_examples'])==4
    for r in f['photo_examples']:
        assert r['split']=='train' and rows[r['row_index']]['scientific_name']==r['scientific_name']
        assert abs(r['activation']-float(z[r['row_index'],f['feature']]))<1e-6
        Image.open(ROOT/r['photo_local']).verify()
verification={'status':'passed','checks':['Shapes, finite arrays and nonnegative sparse codes',
    'Exactly 32 active features per species','Pinned input and embedding SHA256 hashes',
    'No genus leakage between train/validation/test','Reloaded checkpoint reproduces saved codes and reconstructions',
    'Single-feature edit equals the expected decoder-direction change',
    'All split reconstruction metrics independently recomputed in NumPy float64',
    'Five selected features have four attributed, valid reference images each',
    'All displayed example activations match saved vectors and use training species'],
    'not_tested':['TaxaAdapter image-generation behavior','Causal biological trait specificity','Stability across SAE seeds']}
(ROOT/'results/verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2))
