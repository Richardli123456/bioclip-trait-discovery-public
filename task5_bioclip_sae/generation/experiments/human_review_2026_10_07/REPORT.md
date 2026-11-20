# Completed human review of the 32 unedited ladybird images

The single human reviewer judged **27/32 images eligible and exactly countable**. Five remain uncertain/unscorable. This materially improves the measurement-feasibility evidence over the earlier provisional AI pass, while the two reviewers disagree substantially about which views qualify. The human annotations are retained as supplied; AI scores do not override them.

## Human results

- Eligible and count-scorable: 27/32 (84.4%; 95% Wilson interval 68.2%–93.1%).
- Visible spots among eligible images: 27/27 (100.0%; 95% Wilson interval 87.5%–100.0%).
- Exactly seven visible spots among scorable images: 3/27 (11.1%; 95% Wilson interval 3.9%–28.1%).
- Observed scorable-seven yield over all 32 images: 3/32 (9.4%; 95% Wilson interval 3.2%–24.2%).
- Exact-seven IDs: D005, D010, D026.
- Unscorable IDs: D009, D014, D016, D024, D028. These are NA, never zero.
- Mean count among the 27 scorable images: 6.15; median 6; range 1–10.

| Visible count | Images |
|---:|---:|
| 1 | 1 |
| 3 | 1 |
| 4 | 4 |
| 5 | 7 |
| 6 | 2 |
| 7 | 3 |
| 8 | 5 |
| 9 | 2 |
| 10 | 2 |

The all-image yield is the joint event 'scorable and counted seven.' It does not treat the five unscorable images as biologically non-seven. These are visible patch counts, not inferred hidden spots or confirmed anatomical species patterns. No image received a zero count. Notes were blank on all 32 exported records, so uncertainty reasons cannot be reconstructed from human notes.

## Explicit clarification and provenance

The raw export had spot_count=5 and view_eligible=yes but spots_visible=NA for D002, D008, and D027. The user explicitly clarified that visible spots are yes and the count is five for all three. Only those three visibility fields were changed in the derived CSV; counts, eligibility, notes, row order, and the original export are unchanged. This is a human clarification, not an AI correction.

The page displayed randomized opaque image IDs without AI scores. We cannot independently verify whether the reviewer consulted other materials or the seed key. This is one human review with expertise unspecified, not expert adjudication or independent-human agreement.

## Separate comparison with the earlier provisional AI review

| Human eligibility \ AI eligibility | yes | no | NA |
|---|---:|---:|---:|
| yes | 11 | 10 | 6 |
| no | 0 | 0 | 0 |
| NA | 1 | 3 | 1 |

Exact agreement on the three eligibility descriptors is 12/32. The earlier AI pass marked 12 eligible, 13 ineligible, and seven uncertain. All three human exact-seven images (D005, D010, D026) were marked view-ineligible by that AI pass. This is an unresolved eligibility disagreement, not grounds to invalidate or silently change the human response.

Human exact counts are available for 27 images, versus one for the AI. Only D011 has numeric counts from both, and both give eight. This single agreement cannot establish count reliability. Count-scorability status agrees on 6/32; numeric-versus-NA cases are reported as differences in measurement availability, not invented numeric errors.

## Measurement decision and limits

Human scoring supports a small exploratory count-measurement follow-up for this fixed model, overhead prompt, species, and rater. No numerical acceptance threshold was locked for this 32-image review, so the result is descriptive rather than a retroactively passed confirmatory test. Exact seven occurred in only 3/27 scorable images; these outputs do not reliably reproduce the canonical seven-patch appearance under this scoring.

Before scaling a count intervention, freeze the candidate and counting rules, use explicit pose/occlusion examples to calibrate annotation, and retain paired missingness as an outcome. The visible counts do not themselves provide independent species-level labels for selecting an SAE coordinate, and they cannot rescue the failed beak feature. No new training or generation was performed for this analysis.

The Wilson intervals describe sampling uncertainty over image seeds for fixed settings and a fixed rater. They do not include annotation errors, systematic generator errors, or cross-species uncertainty. The broad human–AI eligibility disagreement remains unresolved and should be transparent in any downstream claim.

## Reproducible files

- [Unchanged original human export](human_scores_original.csv)
- [Corrected human CSV](human_scores_corrected.csv) and [explicit clarification provenance](clarification_provenance.json)
- [Numerical summary and source hashes](summary.json)
- [All paired descriptors](human_ai_paired_descriptors.csv) and [disagreements](human_ai_disagreements.csv)
- [Independent verification](verification.json) and [verification script](verify_human_review.py)
- [Reproducible standard-library analysis](analyze_human_review.py)
- [Original human scoring rules](../unedited_dorsal_v1/human_review/SCORING_RULES.md)

Run `python analyze_human_review.py`, then `python verify_human_review.py` from this folder. The analysis checks the fixed raw-export hash before writing derived outputs and refuses inconsistent scoring records. The separate verifier independently counts CSV records, inverts the binomial score inequality numerically to check Wilson endpoints, and checks all 32 blind images against their original generation hashes.
