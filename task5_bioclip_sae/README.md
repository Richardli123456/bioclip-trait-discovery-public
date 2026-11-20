# BioCLIP 2 SAE experiments

## Read the training code with a saved result

Start with the small, comma-formatted pilot. In [prepare_embeddings.py](prepare_embeddings.py), follow the species rows and their genus-based split into the raw embedding array. In [train_sae.py](train_sae.py), read `TopKSAE.encode`, `decode`, then `metrics`. A batch of 256 species enters as 256 × 768, expands to 256 × 2,048 activations with at most 32 positive entries per row, and decodes back to 256 × 768. The SAE learns by comparing its reconstruction with the original input. The [technical guide](../TECHNICAL_GUIDE.md) derives the loss and explains why the decoder columns are normalized.

Check the arithmetic in [evaluation.json](results/evaluation.json). Under `metrics.test`, subtract `normalized_mse` (0.43689) from one to recover the explained fraction (0.56311). Compare this with the training fraction (0.83439). The gap shows why the held-out genera matter. The saved checkpoint came from epoch 15, selected using validation genera. This exercise describes the 5,388-animal pilot. For the larger, space-formatted corpus, follow the [expanded runner](generation/run_expanded_sae_study.py) and its [separate results](generation/experiments/expanded_followup_2026_10_04/REPORT.md).

## Study reports

Start with the [follow-up report](generation/experiments/trait_followup_2026_10_07/REPORT.md). The [completed human review](generation/experiments/human_review_2026_10_07/REPORT.md) found **27/32 unedited ladybird images scorable, exactly seven visible spots in 3/27, and five remaining NA**. This is one human review with substantial unresolved eligibility disagreement against the earlier provisional AI pass.

For the new natural-log tail/wing length ratio, training selected feature **814** from the existing 732 recurrent features. Its family-centered correlation changed from **-0.1467** in training to **+0.0803** in validation, and it added no predictive improvement over taxonomy. The prespecified validation gate failed; test endpoint values were not computed and no tail-image branch was launched. This follow-up reused the existing SAE without retraining.

The [expanded report](generation/experiments/expanded_followup_2026_10_04/REPORT.md) retains **362,460 corrected TreeOfLife-10M taxonomy records, 768-dimensional space-formatted embeddings, 732/2,048 SAE features recurring across all five new seeds**, and the **98-image exploratory bridge**. The earlier independent raw-vector beak-shape probe and SAE candidate failed validation. Biological trait control remains unvalidated, and the exact TreeOfLife-1M training manifest remains unavailable.

Read [results/RESULTS.md](results/RESULTS.md) for the historical 5,388-animal experiment and five candidate-feature photo panels. Its [aligned stability follow-up](generation/experiments/followup_2026_10_04/REPORT.md) found zero passing features on that smaller corpus. The earlier [inference-only continuation](generation/experiments/existing_sae_v1/REPORT.md) retains 188 verified images, overlap rankings and an owl bridge; human review of its older 28-image ladybird set remains deferred. The [generation guide](generation/README.md) separates these historical pilots from current expanded work.

This original pilot uses real pretrained BioCLIP 2 embeddings, a trained 2,048-feature TopK SAE, and held-out genera. Candidate visual associations remain hypotheses. A separate [TaxaAdapter generation pilot](generation/README.md) uses the released code/checkpoint and a newly fitted SAE aligned to its taxonomy format.

## Reproduce the original 5,388-animal pilot

From the repository root in PowerShell, after following [environment setup](../README.md#set-up-an-independent-environment):

```powershell
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/prepare_embeddings.py'
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/train_sae.py'
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/inspect_features.py'
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/make_feature_sheets.py'
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/verify_results.py'
& './.venv/Scripts/python.exe' 'task5_bioclip_sae/write_report.py'
```

The official checkpoint is downloaded into the sibling Task 4 folder; both tasks use the independent root `.venv`. Large arrays and `results/sae.pt` are locally available in the prepared copy but excluded from Git. A fresh clone must run extraction and training before loading or verifying the SAE.

`prepare_embeddings.py` reads the committed official metadata in `data/inat21_name_lookup.json`; its source URL, revision and hash are recorded in `data/dataset_metadata.json`. Every row in `species.csv` maps to the same row in all arrays. The source is the public TreeOfLife-10M iNat21 animal subset, not a verified exact TaxaAdapter training subset.

`data/feature_interpretations.json` contains human photo observations. Re-review those if the model, selected features or photos change. Feature indices are zero-based and can change when retraining. The shown run uses a fixed seed but small platform/library floating-point differences can alter TopK ties and training trajectories.

## Load the saved SAE

Run from `task5_bioclip_sae` with the same Python environment:

```python
import torch
from train_sae import TopKSAE

checkpoint = torch.load('results/sae.pt', map_location='cpu', weights_only=True)
state = checkpoint['state_dict']
model = TopKSAE(state['mean'], state['scale'], width=2048, k=32).eval()
model.load_state_dict(state)
# raw_vectors must be float32 [batch, 768] in the exact recorded BioCLIP format.
# with torch.inference_mode():
#     reconstructed, sparse = model(raw_vectors)
```

## Historical pilot reconstruction result

Test-genera reconstruction: 56.31% variation explained, mean cosine 0.8874. Training reconstruction: 83.44%. The generalization gap is material. The five photo associations are plausible candidates for further experiments, not demonstrated independent trait controls.

`MODEL_FILES_REQUEST.md` and the dependency discussion in the original results report are historical notes from before the generator release. The official code and TreeOfLife-1M adapter weights are now downloaded for the separate generation experiment; see [its pinned assets](generation/assets.json). The exact full target species list is still not confirmed.

## Which SAE should be used?

| Experiment | Taxonomy separators | Checkpoint location | Feature examples |
| --- | --- | --- | --- |
| Original pilot | Comma and space | `results/sae.pt` | 192, 190, 208, 598, 287 |
| Generator-aligned pilot | Space only | `generation/cache/aligned/results/sae.pt` | 1969 in the first owl intervention |
| Corrected expanded v2 | Space only | `generation/experiments/expanded_stability_w2048_k32_v2/seed_20260929/sae.pt` | 732 recurrent features; beak-shape candidate 506 and later tail/wing candidate 814 failed validation |

These runs share the primary SAE dimensions but use different input vectors or corpora. Load each checkpoint with its matching preprocessing buffers, taxonomy format and species-row manifest. Feature indices are specific to the trained model. The original extraction/training commands above assert 5,388 species; current expanded artifacts and their separate runner are documented in the [expanded report](generation/experiments/expanded_followup_2026_10_04/REPORT.md). The preliminary expanded v1 was superseded by corrected v2.
