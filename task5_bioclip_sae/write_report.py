"""Write the measured results and evidence-bounded candidate-feature catalog."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,importlib.metadata,json
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'data/dataset_metadata.json').read_text())
evaluation=json.loads((ROOT/'results/evaluation.json').read_text())
features=json.loads((ROOT/'results/selected_features.json').read_text())
interpret=json.loads((ROOT/'data/feature_interpretations.json').read_text())
release=json.loads((ROOT/'data/taxaadapter_release_status.json').read_text())
rows=json.loads((ROOT/'data/species.json').read_text())
report=['# SAE pilot: 5,388 animal species','',
 'The expanded embedding dataset, small SAE, held-out evaluation and five candidate-feature photo panels are complete. This is an exploratory analysis of **BioCLIP 2 taxonomy embeddings**. It does not yet establish biological trait controls or include generated images.',
 '', '## What was completed','',
 '- Extracted **5,388 × 768** raw float32 taxonomy embeddings with the official pretrained BioCLIP 2 checkpoint.',
 '- Used all 5,388 Animalia entries in the official TreeOfLife-10M iNat21 naming metadata: **27 classes, 715 families and 2,988 genera**.',
 '- Trained a **768 → 2,048 → 768 TopK SAE**, with **32 nonzero features per species**.',
 '- Held out entire genera for validation and testing; no genus appears in more than one split.',
 '- Selected five exploratory features from training-only activation lists and inspected four attributed real photos per feature.',
 '', '## Evaluation','',
 '| Split | Species | Genera | Reconstructed variation | Mean reconstruction cosine |',
 '|---|---:|---:|---:|---:|']
for s in ['train','validation','test']:
    m=evaluation['metrics'][s];g=len({r['genus'] for r in rows if r['split']==s})
    report.append(f"| {s.title()} | {data['splits'][s]:,} | {g:,} | {100*m['variance_explained_vs_training_mean']:.2f}% | {m['cosine_mean']:.4f} |")
report += ['',
 '**Interpretation:** the SAE reconstructs 83.44% of training variation but only 56.31% for held-out test genera. This is a substantial generalization gap. Treat it as a working exploratory model; reconstruction fidelity needs improvement before interpreting image changes as precise trait interventions.',
 '',
 '“Reconstructed variation” is 1 − (squared reconstruction error / squared distance to the training mean). It is not classification accuracy. The test cosine fifth percentile is '+f"{evaluation['metrics']['test']['cosine_p05']:.4f}.",
 '',
 f"Training stopped after {evaluation['epochs_run']} epochs; the best validation checkpoint was epoch {evaluation['best_epoch']}. Neither test scores nor photo appearance selected that checkpoint. All 2,048 dictionary features activate somewhere in training (zero globally dead training features). Exactly 32 active coordinates is imposed by TopK, not a discovered property of biology.",
 '', '### Reconstruction references','',
 '| Representation | Test reconstructed variation | Test mean cosine |',
 '|---|---:|---:|']
for name,m in [('PCA, 32 dense components',evaluation['baselines']['pca_32']['test']),
               ('SAE, 32 active of 2,048',evaluation['metrics']['test']),
               ('PCA, 128 dense components',evaluation['baselines']['pca_128']['test'])]:
    report.append(f"| {name} | {100*m['variance_explained_vs_training_mean']:.2f}% | {m['cosine_mean']:.4f} |")
report += ['',
 'PCA was fitted only on training vectors. These references show the reconstruction/sparsity trade-off; they are not parameter-matched interpretability comparisons. The SAE modestly outperforms PCA-32 here, while PCA-128 retains substantially more information. Better reconstruction alone would not establish interpretable traits.',
 '', '## Five candidate features','',
 '**These labels are human hypotheses after photo inspection, not model-provided names or confirmed causal traits.** Many highly activating species share taxonomy words. Each feature may represent an entire animal group or multiple correlated attributes. Feature numbers are zero-based and specific to this checkpoint.',
 '', '| Feature | Candidate visual association | Active training species | Active test species |',
 '|---|---|---:|---:|']
for f in features:
    j=str(f['feature']);report.append(f"| {j} | {interpret[j]['label']} | {f['train_support']} | {f['test_support']} |")
report += ['','An activation is counted when greater than 1e-7. Support counts include low activations; they do not mean every active species exhibits the proposed trait. Test counts are descriptive, not a held-out validation of the human labels.','']
catalog=[];credits=['# Reference photo credits','',
 'Twenty real photographs used for noncommercial research inspection. Images were scaled proportionally for the panels; biological content was not edited. Original licenses remain in force. Photos did not enter the SAE or BioCLIP embedding calculation.','',
 '| Feature | Species | Attribution | License | Source |','|---|---|---|---|---|']
for f in features:
    j=str(f['feature']);t=interpret[j]
    report += [f"### Feature {j}: {t['label']}",'',f"![Feature {j} examples](feature_{j}.jpg)",'',
        '**Observed in photos:** '+t['observation'],'','**Candidate interpretation:** '+t['hypothesis'],'',
        '**Competing explanation:** '+t['alternative'],'','**Future generation test:** '+t['future_test'],'',
        'Top eight training species by activation:','',
        '| Scientific name | Source common name | Activation | Family |','|---|---|---:|---|']
    for r in f['top_training_species']:
        report.append(f"| *{r['scientific_name']}* | {r['common_name'] or '—'} | {r['activation']:.4f} | {r['family']} |")
    report += ['','The panel favors different genera among these eight; it is not necessarily the top four by score. One image per species does not cover sex, life stage, seasonal plumage or other variation.','']
    for r in f['photo_examples']:
        catalog.append({'feature':f['feature'],'candidate_label':t['label'],'scientific_name':r['scientific_name'],
            'source_common_name':r['common_name'],'genus':r['genus'],'family':r['family'],
            'activation':r['activation'],'split':r['split'],'row_index':r['row_index'],
            'photo_url':r['photo_url'],'photo_credit':r['attribution'],'photo_license':r['license'],
            'interpretation_status':'Exploratory association; no causal generation test',
            'observation':t['observation'],'alternative_explanation':t['alternative']})
        credits.append(f"| {j} | {r['scientific_name']} | {r['attribution']} | {r['license']} | [Photo]({r['photo_url']}) · [Taxon]({r['taxon_url']})"+
             (f" · [Observation]({r['observation_url']})" if r['observation_url'] else '')+' |')
report += ['## Method and limitations','',
 'Taxonomy strings use the same pilot format: `Kingdom, Phylum, Class, Order, Family, Genus, specific epithet`. Common names and photographs are excluded from embedding extraction. Names are preserved from the official training metadata and may differ from modern taxonomy or common-name conventions.',
 '',
 'The SAE subtracts the training mean and divides by one global training RMS; this transformation is inverted when reconstructing the 768-dimensional output. Encoder: linear layer, ReLU, retain largest 32 activations. Decoder: linear dictionary with unit-length columns and a learned output bias. Training: Adam, learning rate 0.001, batch 256, seed 20260929, gradient clipping 1.0. Validation error is checked every five epochs; the best checkpoint is saved.',
 '',
 'The split groups complete genera and is stratified by taxonomic class. Classes containing fewer than three genera remain training-only. Class representation is imbalanced: insects and birds dominate. All species may already have been seen during BioCLIP pretraining; “held out” here refers only to SAE training.',
 '',
 'Feature selection was manual and exploratory, based on recognizability of training top-activation lists. No feature-selection accuracy, trait annotations, null controls, multi-seed stability or causal specificity were measured. Photos illustrate candidate interpretations rather than prove them. A feature can be a taxonomy/word feature, a whole-animal identity feature, or a mixture of visual traits.',
 '', '## Remaining dependency: TaxaAdapter','',
 'This animal subset comes from the public TreeOfLife-10M iNat21 metadata. **It is not confirmed to be the exact TreeOfLife-1M TaxaAdapter species list.** Exact generator input formatting, embedding normalization and adapter compatibility remain unverified.',
 '',
 f"The public [TaxaAdapter repository](https://github.com/Imageomics/TaxaAdapter) was checked directly at {release['checked_utc']}. Its root contained only `.gitignore`, `LICENSE` and `README.md`. No runnable generator/checkpoint was present there. We did not contact Professor Chao or send messages.",
 '',
 'To run the causal phase, obtain the TreeOfLife-1M checkpoint, inference code, exact taxonomy list and extraction/normalization settings from the authors. Confirm whether the assignment means 200+ or 200K+ species. Then first compare original embeddings against unedited SAE reconstructions under fixed seeds; only after that compare feature removal/increase. A prepared request is in `../MODEL_FILES_REQUEST.md`.',
 '', '## Saved artifacts','',
 '- `../data/embeddings_raw.npy`: 5,388 × 768 float32 input embeddings.',
 '- `../data/embeddings_normalized.npy`: unit-length copies; the SAE was trained on the raw vectors with its own recorded centering/scaling.',
 '- `../data/species.csv`: exact taxonomy, source IDs and split for every row.',
 '- `sae.pt`: reloadable SAE weights, preprocessing buffers and configuration.',
 '- `sparse_vectors.npy`: 5,388 × 2,048 sparse vectors (stored as a dense NumPy array).',
 '- `reconstructed_embeddings.npy`: 5,388 × 768 reconstructed vectors.',
 '- `feature_catalog.csv`: 20 real examples across five candidate features, with observations and source links.',
 '- `features.json`: support counts and top eight training examples for all 2,048 features.',
 '- `selected_features.json`, `feature_*.jpg`, `photo_credits.md`: inspected examples and attribution.',
 '- `evaluation.json`, `training_history.csv`, `verification.json`, `run_manifest.json`: measurements, checks and provenance.',
 '', '## Validation performed','',
 'Verified dimensions, finiteness, exact TopK sparsity, source/embedding hashes and absence of genus overlap. Reloaded the checkpoint on CPU and reproduced saved GPU reconstructions/codes within floating-point tolerance. Independently recomputed all split reconstruction metrics in NumPy float64. Verified the mathematical effect of a one-feature edit on the decoder output. Opened all reference photos and visually inspected all five assembled panels. No TaxaAdapter runtime or image-generation test has occurred.',
 '', '## Sources','',
 f"- [Pinned official TreeOfLife-10M iNat21 naming metadata]({data['source_url']})",
 '- [Official BioCLIP 2 model](https://huggingface.co/imageomics/bioclip-2)',
 '- [TaxaAdapter paper](https://arxiv.org/abs/2603.26128)',
 '- [SAE interpretation and intervention reference](https://arxiv.org/abs/2502.06755)',
 '- [All photo credits and links](photo_credits.md)',
 '']
(ROOT/'results/RESULTS.md').write_text('\n'.join(report),encoding='utf-8')
(ROOT/'results/photo_credits.md').write_text('\n'.join(credits)+'\n',encoding='utf-8')
with (ROOT/'results/feature_catalog.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(catalog[0]));w.writeheader();w.writerows(catalog)
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'model_revision':data['model'],
    'dataset_revision':data['source_revision'],'gpu':data['gpu'],
    'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch','numpy','Pillow']},
    'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
       [ROOT/'results/sae.pt',ROOT/'data/embeddings_raw.npy',ROOT/'results/sparse_vectors.npy',
        ROOT/'data/species.csv',ROOT/'train_sae.py',ROOT/'prepare_embeddings.py']}}
(ROOT/'results/run_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Wrote results, feature catalog, credits and run manifest.')
