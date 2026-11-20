"""Assemble the requested table from measured neighbors and inspected photo notes."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/species.json').read_text())
neighbors=json.loads((ROOT/'results/nearest_neighbors.json').read_text())
photos=json.loads((ROOT/'data/photo_references.json').read_text())
notes=json.loads((ROOT/'data/visual_observations.json').read_text())
meta=json.loads((ROOT/'results/run_metadata.json').read_text())
assert [r['scientific_name'] for r in rows]==[r['scientific_name'] for r in photos]
assert set(notes)=={r['scientific_name'] for r in rows}
output=[]
for r,n,p in zip(rows,neighbors,photos):
    output.append({**n,'visual_observation_against_nearest_match':notes[r['scientific_name']],
                   'reference_photo':p['photo_url'],'photo_credit':p['attribution'],
                   'photo_license':p['license_code'],'taxonomy_source':r['taxonomy_source']})
with (ROOT/'results/species_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
lines=[
 '# BioCLIP 2: 20-species starter experiment',
 '',
 'Completed on the local RTX 5090. The official BioCLIP 2 text encoder produced a **20 × 768 float32 matrix** from verified full taxonomies for 20 Passerellidae species across five genera. No training was required.',
 '',
 '**Result:** all 20 nearest neighbors are in the same genus. The strongest pair is Dark-eyed junco / Yellow-eyed junco (cosine **0.945233**). These are similarities within this hand-selected group, not species-classification accuracy or trait-discovery results.',
 '',
 '## Main comparison table',
 '',
 'Scores are cosine similarities, not probabilities. Self-matches are excluded. Visual notes describe the selected real photos only and compare each row with its first neighbor.',
 '',
 '| Species | Closest match | Second match | Third match | Photo observations |',
 '|---|---|---|---|---|'
]
for r in output:
    matches=[f"{r[f'neighbor_{k}_common']} ({r[f'cosine_{k}']:.4f})" for k in [1,2,3]]
    lines.append(f"| **{r['common_name']}** — *{r['scientific_name']}* | "+' | '.join(matches)+
                 f" | {r['visual_observation_against_nearest_match']} [Photo]({r['reference_photo']}) |")
lines += [
 '', '## Three useful comparisons', '',
 '- **Dark-eyed vs Yellow-eyed junco — 0.9452:** strongest embedding pair. Gray plumage and overall form are similar, while eye color and back/head coloration differ in these references.',
 '- **White-crowned vs Golden-crowned sparrow — 0.9115:** similar brown wings and pale underparts, but visibly different crown markings.',
 '- **Eastern vs Spotted towhee — 0.8458:** similar dark heads, rusty sides and pale bellies; white wing/back spotting is a promising candidate trait for a later controlled edit.',
 '', '## Reference photos', '',
 '![Twenty real reference photos](reference_photos.jpg)',
 '',
 'These photos were inspected for the notes above; they were not fed to the text encoder. One photo per species does not cover sex, age, season or subspecies variation. The Harris\'s sparrow reference is distant and faces away; the Black-chinned sparrow is small in the frame. No fine-detail conclusions were drawn from those views. Full attribution and source links appear in [photo credits](photo_credits.md).',
 '', '## Method and reproducibility', '',
 '1. Hand-select 20 species from Spizella (6), Zonotrichia (5), Junco (3), Melospiza (3), and Pipilo (3). All belong to Passerellidae.',
 '2. Verify each complete taxonomy using an exact, accepted species match in the GBIF API. Save all source responses.',
 '3. Encode the literal string `Kingdom, Phylum, Class, Order, Family, Genus, specific epithet`. No common name, extra natural-language prefix or prompt averaging is used.',
 '4. Load the official `imageomics/bioclip-2` configuration and safetensors checkpoint at a pinned revision. Use OpenCLIP `encode_text(..., normalize=False)` in evaluation mode, float32, with gradients disabled.',
 '5. Save raw vectors. Separately L2-normalize them and compute the pairwise dot products to obtain cosine similarity. Sort descending and exclude self-matches.',
 '6. Obtain exact-name-matched iNaturalist reference photos and inspect their visible appearance. Store the photo sources, attribution and licenses separately from the model measurements.',
 '',
 f"**Model revision:** `{meta['revision']}`. Full package versions, SHA-256 hashes, exact inputs, GPU and validation results are in [run metadata](run_metadata.json).",
 '', '## Verification', '',
 '- Exactly 20 unique species, 20 nontruncated token sequences and 768 finite coordinates per vector.',
 '- Cosine matrix is symmetric; diagonal entries are one within numerical tolerance; values lie in the valid range.',
 f"- Batch extraction agrees with extracting each species separately; maximum absolute raw-vector difference: **{meta['batch_vs_single_max_absolute_difference']:.8g}**.",
 '- Nearest matches exclude the query species. All 20 have a same-genus first match; 49 of 60 top-three neighbor slots are same-genus.',
 '- All 20 reference image files were opened and visually inspected in the contact sheet. Dependency consistency check passed.',
 '', '## What this does and does not establish', '',
 '**Established:** pretrained BioCLIP 2 runs successfully on this PC, produces the expected vector shape, and yields coherent within-genus neighbors for this pilot.',
 '',
 '**Not established:** individual biological trait dimensions, causal control, full-dataset performance, or exact compatibility with the TreeOfLife-1M TaxaAdapter checkpoint. Related species share much of the input taxonomy string, so similarity may reflect shared taxonomic text as well as learned visual information. Photo observations do not explain which embedding coordinates cause a match. The 20 species have not been verified as members of the TaxaAdapter training subset.',
 '',
 'This uses the projected, pooled BioCLIP 2 text output. Before connecting it to TaxaAdapter or training the full SAE, verify the generator checkpoint\'s exact text formatting, pooling and normalization. Twenty vectors are a pipeline smoke test, not an adequate basis for a large trait-discovery SAE.',
 '', '## Files', '',
 '- `species_comparison.csv`: the requested 20-row table, three nearest matches, scores, photo observations and source links.',
 '- `embeddings_raw.npy` / `embeddings_raw.csv`: unnormalized 20 × 768 vectors in the order listed in `run_metadata.json`.',
 '- `embeddings_normalized.npy`: unit-length copies used for cosine similarity.',
 '- `cosine_similarity.npy` / `cosine_similarity.csv`: full 20 × 20 matrix.',
 '- `nearest_neighbors.json` / `nearest_neighbors.csv`: measured top-three matches without visual interpretation.',
 '- `reference_photos.jpg`, `photo_credits.md`: inspected photo overview and attribution.',
 '', '## Sources', '',
 '- [Official BioCLIP 2 model](https://huggingface.co/imageomics/bioclip-2)',
 '- [Official BioCLIP 2 project](https://imageomics.github.io/bioclip-2/)',
 '- GBIF taxonomy URLs are recorded for every row in `../data/species.csv`.',
 '- iNaturalist photo and taxon URLs are recorded in `../data/photo_references.json`; local source responses are in `../data/sources/`.',
 '',
 'Suggested next step: select several clear species pairs from this table and expand the taxonomy embedding collection substantially before training an SAE.'
]
(ROOT/'results/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
credits=['# Reference photo credits','','Photos are included for noncommercial research inspection. Each original retains its listed license. Photos were scaled proportionally for the contact sheet; no biological content was edited.','',
         '| Species | Photographer / attribution | License | Original photo | Taxon / identification source |',
         '|---|---|---|---|---|']
for p in photos:
    credits.append(f"| {p['common_name']} (*{p['scientific_name']}*) | {p['attribution']} | {p['license_code']} | [Photo {p['photo_id']}]({p['photo_url']}) | [Taxon]({p['taxon_url']})"+
                   (f" · [Observation]({p['observation_url']})" if p['observation_url'] else '')+' |')
(ROOT/'results/photo_credits.md').write_text('\n'.join(credits)+'\n',encoding='utf-8')
print('Wrote RESULTS.md, species_comparison.csv, photo_credits.md')
