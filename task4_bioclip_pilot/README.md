# BioCLIP 2 taxonomy embedding pilot

## Follow one bird through the pipeline

1. Find *Spizella passerina* in the [species table](data/species.csv). The encoder receives its taxonomy string. The bird photographs are visual references for the reader.
2. Open [run_embeddings.py](run_embeddings.py) and follow the text tokens into the pooled embedding. Each row has 768 values. The script keeps a raw copy and a unit-length copy, then takes dot products between normalized rows.
3. Find the bird in [nearest_neighbors.csv](results/nearest_neighbors.csv). Worthen's Sparrow scores 0.9045, just above Brewer's Sparrow at 0.9007. Both share the genus *Spizella*. Those are similarities among the 20 sampled birds. A cosine of 0.9045 is not a 90.45% probability of correct identification.

To see what happens when thousands of species are encoded, continue to the [SAE study](../task5_bioclip_sae/README.md).

Start with [results/RESULTS.md](results/RESULTS.md). This completed experiment extracts real pretrained BioCLIP 2 embeddings for 20 sparrows/towhees, computes cosine neighbors, and compares attributed real photos.

## Run Task 4

From the repository root in PowerShell, after following [environment setup](../README.md#set-up-an-independent-environment):

```powershell
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/run_embeddings.py'
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/write_results.py'
```

Source inputs are included; download the model using the root setup instructions. Running embeddings does not download images or retrain a model.

## Recreate inputs

```powershell
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/prepare_data.py' --taxonomy-only
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/download_model.py'
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/run_embeddings.py'
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/fetch_reference_photos.py'
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/make_contact_sheet.py'
& './.venv/Scripts/python.exe' 'task4_bioclip_pilot/write_results.py'
```

`data/visual_observations.json` contains manually reviewed descriptions of the downloaded reference photos, not outputs of BioCLIP. Re-review those descriptions if you replace the photos.

## Environment

Use the independent root `.venv` described in the [repository setup](../README.md#set-up-an-independent-environment). The original run used Windows, Python 3.12, CUDA 13.0 and RTX 5090. Its full dependency snapshot is `requirements-lock.txt`; exact model and embedding provenance are in `results/run_metadata.json`.

`cache/` holds approximately 1.71 GB of official model weights plus download/package caches. Both `cache/` and `.venv/` are excluded from Git. Results and metadata remain ordinary reviewable files. No SAE was trained and no TaxaAdapter images were generated.
