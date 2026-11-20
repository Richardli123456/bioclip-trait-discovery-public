# Independent trait benchmark

Updated on 2026-10-04 for the cleaned `expanded_tol10m_v2` corpus. **The embedding probe predicts beak shape but does not outperform the taxonomy baseline. The prespecified positive-signal rule failed.** The trait labels and evaluation protocol were fixed before the original fit and were unchanged for the data-quality reanalysis. This is a **bird beak-shape benchmark**, separate from the ladybird image-scoring experiment.

## Why the label panel uses birds

The bounded ladybird reference audit did not produce an adequate fixed dark-spot-count dataset for genus holdout. The [UK Ladybird Survey identification guide](https://www.coleoptera.org.uk/sites/default/files/Ladybird%20descriptions_Info%20pack_NEW_v.5.pdf) distinguishes variable color morphs, variable counts, fused markings and light-colored spots. A numeric common name is insufficient evidence for one invariant dark-spot label. Of the ten local species checked in [ladybird_reference_audit.csv](ladybird_reference_audit.csv), only the seven-spot reference supplied one fixed dark-count phenotype. This is not a claim that reliable labels cannot be assembled from other sources; it records the limitation of this bounded audit.

The seven-spot reference is useful for judging whether a generated adult resembles the typical species pattern. It is not a score for every individual image. Occlusion and view must be assessed separately, and human review of the generated images remains necessary.

## Independently measured labels

The primary source is [AVONET Supplementary dataset 1, version 7](https://figshare.com/articles/dataset/16586228/7), released under CC BY 4.0 by Tobias and colleagues. The associated [paper](https://doi.org/10.1111/ele.13898) describes the measurement dataset. The downloaded workbook was verified against the publisher's MD5; its SHA-256, source URLs and exclusions are recorded in [label_provenance.json](label_provenance.json).

The endpoint is `ln(Beak.Length_Nares / Beak.Depth)`: natural logarithm of published species-mean nostril-to-tip length divided by species-mean depth at the nostrils. Both inputs use millimetres. Higher values describe more elongated beaks. This is our derived ratio, not a pre-existing AVONET categorical label. It is a ratio of means, not the mean of individual ratios.

The fixed eBird table contains 10,661 species rows. We retained **8,677 species, 2,180 genera and 243 families** with positive finite dimensions, no inferred morphology and at least four specimens with complete measurements. We excluded 1,232 rows with fewer than four complete specimens and 752 with at least one inferred morphological value. The conservative inference exclusion applies even when the inferred field was not one of the two beak measurements.

- [labels.csv](labels.csv): eligible measured-species labels, exact source rows and URLs.
- [label_exclusions.csv](label_exclusions.csv): every excluded source row and reason.
- [prepare_labels.py](prepare_labels.py): reproducible standard-library workbook extraction.
- [confirmed_nonbird_lineage_errors.json](confirmed_nonbird_lineage_errors.json): three exact names found with non-bird higher lineages in the initial expanded TreeOfLife manifest. These were sent for explicit exclusion before SAE training; no labels or taxonomy were silently corrected.

## Evaluation fixed before fitting

[protocol.json](protocol.json) defines one endpoint and one held-out test. Exact scientific-name matching uses only Animalia/Aves manifest rows. It introduces no inferred synonyms. The fixed manifest's train, validation and test assignments are reused, with a mandatory assertion that no genus crosses partitions.

A linear ridge probe predicts the trait from the original 768-dimensional space-formatted BioCLIP 2 vectors. Feature means and scales use training data only. The penalty is chosen on validation data from a fixed grid. A taxonomy control predicts training-only family means, shrunk toward order and global means; its smoothing strengths are also chosen using validation data. The taxonomy supplied to that baseline is exactly the embedding manifest's taxonomy.

After selection, both models are refit on train plus validation. The primary test measure is species-weighted RMSE. A paired 2,000-resample genus-cluster bootstrap gives the interval for taxonomy RMSE minus embedding RMSE. The conservative positive-signal rule requires a positive test R2 and a 95% interval entirely above zero. No test result was used to select the endpoint, models or thresholds. The v2 reanalysis followed knowledge of the original results, so its test is **not historically untouched**.

The evaluator is [evaluate_embeddings.py](evaluate_embeddings.py). It uses CPU NumPy only, at most two configured BLAS threads, and never imports PyTorch or updates BioCLIP/TaxaAdapter/SAE weights. An independent normal-equation check and known/unseen taxonomy fallback checks passed; see [method_verification.json](method_verification.json).

## Current v2 results and preserved v1 history

The v2 corpus conservatively removed ambiguous cross-kingdom binomial records unless an original pilot lineage was known, retaining that pilot record only. This decision followed broader taxonomy QA; it did not use trait labels, predictions or SAE outcomes. The previous [v1 report](REPORT_v1_archived.md) and [v1 results](evaluation_expanded/results.json) remain preserved. The [reanalysis context](evaluation_expanded_v2/reanalysis_context.json) was recorded before the v2 fit and explicitly states that the v1 outcome was already known.

The v2 exact-name join contains **8,187 bird species**: 6,538 training species (1,697 genera), 875 validation species (214 genera) and 774 test species (197 genera). No genus crosses a split. All 490 unmatched eligible labels are saved. The cleaned 362,460-row manifest has globally unique binomials. The benchmark still requires Animalia/Aves membership. Its matched species, trait values and corresponding raw embeddings are exactly unchanged from v1, although row indices were recomputed for the new array. The test prediction CSV is byte-identical to v1.

The source taxonomies disagree on family for 109 matched species and on order for 374. These cases are logged, and the taxonomy baseline consistently uses the same taxonomy supplied in the embeddings. They were not silently corrected after seeing outcomes.

| Predictor | Test RMSE, log-ratio units | Test MAE | Test R2 | Genus-macro MSE |
|---|---:|---:|---:|---:|
| BioCLIP embedding ridge | 0.27179 | 0.19657 | 0.65591 | 0.08806 |
| Family/order taxonomy baseline | **0.26095** | **0.17874** | **0.68279** | **0.07530** |
| Training-plus-validation mean | 0.46396 | 0.32904 | -0.00273 | 0.22996 |

Taxonomy RMSE minus embedding RMSE is **-0.01083**, with a paired genus-bootstrap 95% interval of **[-0.05664, 0.03918]**. Positive values would favor embeddings. The observed point estimate favors taxonomy, and the interval spans both signs. This does not establish an improvement beyond the taxonomy control. It does not establish that embeddings lack trait information: the raw probe performs substantially better than a global mean.

Validation selected ridge alpha 1, family pseudocount 0 and order pseudocount 64. All grid results are retained. The original evaluation and unchanged-protocol data-quality reanalysis agree exactly. This is a replication of the same predictions on a cleaned input manifest, not a new independent confirmation. CPU fitting completed without hardware errors, and no GPU was used.

- [results.json](evaluation_expanded_v2/results.json): current metrics, protocol and input hashes.
- [test_predictions.csv](evaluation_expanded_v2/test_predictions.csv): every held-out prediction.
- [validation_grid.csv](evaluation_expanded_v2/validation_grid.csv): all prespecified tuning results.
- [joined_labels.csv](evaluation_expanded_v2/joined_labels.csv) and [join_audit.json](evaluation_expanded_v2/join_audit.json): persistent v2 row alignment and split audit.
- [higher_taxonomy_disagreements.csv](evaluation_expanded_v2/higher_taxonomy_disagreements.csv) and [unmatched_labels.csv](evaluation_expanded_v2/unmatched_labels.csv): matching limitations.
- [ridge_probe.npz](evaluation_expanded_v2/ridge_probe.npz) and [taxonomy_model.json](evaluation_expanded_v2/taxonomy_model.json): fitted CPU predictors.
- [verification.json](evaluation_expanded_v2/verification.json): independently recomputed CSV metrics, source-dimension target arithmetic, unchanged-protocol hashes, disjoint-genus checks and exact v1/v2 prediction comparison; all passed.

## Completed SAE association analysis

The CPU-only [candidate analyzer](analyze_sae_candidate.py) evaluated the completed [primary expanded SAE](../expanded_stability_w2048_k32_v2/summary.json), using `evaluation_expanded_v2`. It checked that the benchmark's manifest hash matched the SAE study before reading row-indexed codes. Its [protocol](candidate_protocol.json) was locked before those stability results and has a saved [SHA-256](candidate_protocol.sha256). It only considered features passing the existing recurrence rule, with positive support on at least 30 labelled training birds from at least 10 genera and three families. It ranked one candidate by absolute training correlation after removing family means; ordinary correlations were also retained.

Only that selected candidate receives validation analysis. A 2,000-resample genus bootstrap recomputes validation family means in each sample. Confirmation needs at least 1,900 defined resamples and an interval excluding zero in the training-selected direction. Test traits are not used. A negative branch is saved when no feature qualifies. This is an exploratory association analysis; it cannot reverse the negative raw-probe comparison or establish an isolated image-generation trait.

The analyzer passed a numerical example in which ordinary correlation is positive but the within-family correlation is exactly -1, plus a constant-feature undefined-correlation check; see [candidate_method_verification.json](candidate_method_verification.json).

Of **732 stable features, 188 met the labelled-training support criteria**. Feature **506** was selected on training data alone; it belongs to the new expanded SAE and must not be identified with an earlier pilot feature bearing the same number.

| Association statistic | Training | Validation |
|---|---:|---:|
| Active labelled species | 184 | 14 |
| Active genera | 52 | 4 |
| Active families | 18 | 3 |
| Ordinary Pearson correlation | -0.01146 | 0.14477 |
| Within-family Pearson correlation | -0.15027 | -0.03064 |

The validation bootstrap produced **1,904 defined and 96 undefined resamples**. Its within-family correlation interval was **[-0.07380, 0.02858]**, crossing zero. The minimum-defined-resamples rule passed, but the association-confirmation rule **failed**. Limited positive validation support is a diagnostic limitation, not a changed eligibility threshold. We did not select a replacement feature after viewing validation results and did not inspect test traits for this candidate analysis.

Consequently, this benchmark has **reproducible SAE features but no validated beak-shape candidate** under its fixed rule. The trait-specific generation branch is not qualified by these results. A support-neighbor or bridge demonstration remains representation exploration and must be labeled accordingly.

- [Candidate summary](sae_candidate_w2048_k32_v2/summary.json): full confirmation result, hashes and bootstrap counts.
- [Training associations](sae_candidate_w2048_k32_v2/training_feature_associations.csv): all stable features, including ineligible ones.
- [Training selection](sae_candidate_w2048_k32_v2/training_selection.json): single selected feature, saved before validation values were used.
- [Training examples](sae_candidate_w2048_k32_v2/selected_feature_training_examples.csv) and [validation values](sae_candidate_w2048_k32_v2/selected_feature_validation_values.csv): inspectable evidence, without test traits.
- [Candidate verification](sae_candidate_w2048_k32_v2/verification.json): independent ranking and within-family correlation checks, unchanged protocol hash and correct negative decision; all passed.

## Interpretation boundaries

A passing probe would establish a predictive association for one independently measured trait on genera held out from **probe fitting**. It would not prove those species were absent from BioCLIP pretraining, prove that individual SAE dimensions represent that trait, or validate a causal edit. Family and order controls reduce one obvious taxonomy explanation but cannot remove every phylogenetic or textual-name cue. Exact binomial joins can still link different taxonomic concepts across source versions; higher-rank disagreements are retained in an audit.

Species means combine specimens and may conceal sex, age and geographic variation. Beak ratios in photographs also depend on pose and projection. Thus the dataset does not supply per-image ground truth, and positive results would still require trait-specific image measurements and controlled interventions.
