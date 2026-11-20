# Method scope and correspondence to the papers

Audit date: 2026-10-04. **This project extends the released TaxaAdapter inference path with a separately trained TopK SAE. It is not an exact reproduction of either paper's experiments.** This audit did not run models or training.

## TaxaAdapter compatibility

The [TaxaAdapter paper, v1, sections 4 and A.1](https://arxiv.org/html/2603.26128v1) specifies a frozen BioCLIP 2 taxonomy encoder by default, full seven-level taxonomic input, and a separate taxonomy-conditioning branch alongside ordinary text conditioning. Section 6.1 describes TreeOfLife-1M as a one-million-image subset spanning more than 200,000 species. The original [PDF](https://arxiv.org/pdf/2603.26128) was too large for the web extractor; the matching arXiv HTML version was inspected.

For the executable input convention, we follow the authors' pinned release rather than punctuation in the paper's prose examples. [Released inference code](https://github.com/Imageomics/TaxaAdapter/blob/1f8f64a53b2d5ac88aaa1e98ea0ed12e35b4a506/inference.py) loads `imageomics/bioclip-2`, tokenizes the supplied taxonomy string and treats spaces as taxonomic-level separators. The released sample data use spaces. The [adapter implementation](https://github.com/Imageomics/TaxaAdapter/blob/1f8f64a53b2d5ac88aaa1e98ea0ed12e35b4a506/taxa_adapter/taxa_adapter.py) sets the BioCLIP input dimension to 768 and passes `encode_text` output directly into its projection; it does not insert L2 normalization. It projects a zero vector for the unconditional taxonomy branch.

Our extraction explicitly uses `normalize=False`. We retain one raw 768-dimensional vector per taxonomy record, with seven fields from kingdom through species epithet separated by spaces. SAE centering and RMS scaling are reversed by its decoder before generator injection. We use the author-released [TreeOfLife-1M checkpoint](https://huggingface.co/mridulk/TaxaAdapter/blob/b70f29d0d5f834e8d3f3718e00e055646874e430/taxaadapter_tol-1m.bin), pinned and hashed in [assets.json](../../assets.json). The Stable Diffusion 1.5 projection and attention path remain intact; our subclass permits precomputed vectors at the existing projection input.

The earlier [smoke-test manifest](../../results/run_manifest.json) records exact equality of the original and precomputed projection paths for its tested input. Extraction uses FP32 while released inference casts the encoder/projection to FP16; that same check reported cosine 0.9999991 between those encoder precisions. This is evidence of interface compatibility, not bitwise reproduction of all paper outputs.

Our generic or view-specific ordinary prompts, species sets, seeds and scoring differ from the paper's benchmark. The current **362,460 records with globally unique binomials** come from the broader TreeOfLife-10M release with documented filtering. The preliminary 363,192-row v1 was superseded after cross-kingdom ambiguity and source-lineage errors were identified; ambiguous names were conservatively excluded unless an original pilot lineage was known. These records are not the verified TreeOfLife-1M training manifest, not all animals, and not a synonym-resolved species census. The published adapter is reused, not retrained here.

## Differences from the SAE reference

The inspiration is [Interpretable and Testable Vision Features via Sparse Autoencoders, v2](https://arxiv.org/html/2502.06755v2), especially sections 3, 7 and 8 ([PDF](https://arxiv.org/pdf/2502.06755v2)). Its demonstrated tasks are classification and segmentation, using intermediate CLIP/DINOv2 image-patch activations. Our input is the final BioCLIP 2 taxonomy-text embedding, and our downstream task is conditional image generation.

| Component | SAE reference | This project |
|---|---|---|
| Sparsity | ReLU with L1 penalty; track L0 | ReLU followed by hard TopK; reconstruction loss only |
| Dictionary | 24,576 features | Primary 2,048; conditional comparisons include 1,024 |
| Active dimensions | Learned under penalty | At most 32 or 16, imposed by configuration |
| Input scaling | Mean subtraction and unit-norm activations | Training mean subtraction and one global training RMS |
| Training scale | 100 million patch activations; batch 16,384 | Species-text vectors; batch 256 and validation-based early stopping |
| Decoder constraint | Normalize columns and remove parallel gradient component | Normalize columns after updates; ordinary gradient clipping |
| Rate schedule | 500-step warmups | Fixed Adam learning rate 0.001 |
| Evaluation | Visual exemplars and task interventions | Genus splits, cross-seed matching, external trait labels and controlled generation |

The common intervention principle is to preserve reconstruction error: `x_edited = x + D(z_edited) - D(z)`. Our later controlled feature edits follow that equation. The initial eight-image pilot instead used decoded edited vectors directly; it should be treated as a historical smoke test. The earlier owl bridge also used reconstruction endpoints, whereas the expanded bridge protocol retains the source residual. These experiments must be identified separately.

Our five-seed recurrence threshold, AVONET probe, trait-candidate rules and support-overlap bridges are additional experimental choices. The SAE reference does not establish that this taxonomy-text setup will produce stable biological traits. A reproducible direction, a trait correlation and a visible image change are separate findings; none alone establishes isolated biological trait control.
