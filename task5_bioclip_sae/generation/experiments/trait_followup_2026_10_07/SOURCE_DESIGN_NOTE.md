# Next independently labelled endpoint: relative tail length

Source/design assessment, 2026-10-07. This note chooses a feasible next endpoint before examining its SAE associations. It does not report a successful trait, fit a model, or replace the failed beak experiment.

**Recommend one primary endpoint: `ln(Tail.Length / Wing.Length)` from the pinned AVONET eBird species means.** Call it the log tail-to-wing length ratio. It is a ratio of published means, not the mean of specimen-level ratios, and not an allometrically adjusted or sex-specific measure of tail elongation. The rationale is existing independently measured labels, broad coverage, and a plausible visible relative-length phenotype. No new trait association, model prediction, or generated image was used for this choice.

## Source and availability

The cached [AVONET version 7 dataset](https://figshare.com/articles/dataset/16586228/7) has publisher-verified MD5 `1445afdcfb6df784010c2ca034544bc8`; this audit rechecked it. The [AVONET paper](https://doi.org/10.1111/ele.13898) describes direct measurements and species means. Workbook Metadata row 22 defines wing length from the wrist joint to the longest primary tip on the unflattened wing. Row 26 defines tail length from central tail-feather emergence at the skin to the longest tail-feather tip. Complete.measures counts specimens with all morphometric measurements; Inference=NO excludes inferred morphology under the conservative existing rule.

The count-only [availability audit](source_availability_audit.json), reproducible with [audit_source_availability.py](audit_source_availability.py), finds:

| Candidate endpoint | Eligible source species | Exact matched birds | Train / validation / test |
|---|---:|---:|---:|
| Log tail / wing length | 8,677 | 8,187 | 6,538 / 875 / 774 |
| Log tarsus / wing length | 8,677 | 8,187 | 6,538 / 875 / 774 |
| Hand-wing index | 8,677 | 8,187 | 6,538 / 875 / 774 |

All use finite positive fields, Inference=NO, at least four complete specimens, and an exact binomial. The source includes 2,180 genera and 243 families; matched split genus counts are 1,697 / 214 / 197. There are no missing measurements among the previous 8,187 exact-name matched birds. These are availability counts only: no effect sizes, target distributions, or new validation/test associations were inspected. Exact membership of the image generator's TreeOfLife-1M training set remains unknown.

## Measurement limitations and bounded choice

Tail-to-wing length is a relative measure: either a longer tail or a shorter wing can increase it. An image experiment must record both component measurements as well as the ratio, and cannot conclude isolated tail control from the ratio alone. The primary [comparative tail study](https://doi.org/10.1046/j.1420-9101.1999.00009.x) uses log-length regression residuals and explains why different body-size references can give different conclusions. Our simple log ratio is an explicit endpoint, not a reproduction of that study's allometric residual.

A side-view image may hide the tail-feather root or wrist joint, foreshorten one structure, spread or bend feathers, or contain malformed anatomy. Therefore a visible feather silhouette ratio is a **new image proxy**, not automatically an AVONET measurement. Before interpreting edits, require a fixed whole-bird lateral-view prompt and blinded assessment of full tail, wingtip and relevant landmarks; unclear cases remain NA. If anatomical landmarks are not consistently visible, stop the quantitative image branch or validate a separately named proxy against real labelled images. Do not silently reinterpret the endpoint after viewing edits.

Species means pool sexes and individuals. Ornamented tails can have pronounced sex/age/season variation. A male-specific generated image must not be presented as matching an unqualified pooled species mean. Fixed prompt specifications, actual visible phenotype, and external species information are needed before selecting an image pilot.

Hand-wing index needs secondary/primary landmarks; tarsus length needs joint landmarks that can be hidden by feathers or pose. Neither is a clearly easier first image endpoint. They are documented alternatives, **not additional tests authorized by this recommendation**. Body mass is not directly identifiable from an unscaled generated photograph. No additional dataset is necessary for this bounded follow-up, and this note makes no claim that AVONET is globally optimal.

## Statistical boundaries before analysis

1. Freeze this one endpoint, the exact manifest/split mapping, candidate support thresholds, ranking rule, taxonomy control, confidence procedure and stopping rule before reading associations. Keep the existing SAE and stable-feature set fixed.
2. The beak endpoint and its negative held-out results are already known. The follow-up is sequential exploration. Preserve that history and do not describe the reused partitions as a historically pristine project-level test. New tail associations have not been viewed in this assessment; that narrower statement is supportable.
3. Rank candidate features using training labels only. Lock one selected feature before validation. Do not replace a failed candidate with the runner-up, re-partition genera, or change labels/thresholds after a negative result. If several endpoints are actually tested, account for that declared family of tests and report every outcome.
4. Keep whole-genus partitions and use the embedding manifest's family/order taxonomy for the baseline. The old exact join has 109 family and 374 order disagreements with AVONET; retain the audit. Within-family adjustment and genus bootstrap reduce specific confounds but are not full phylogenetic control or proof of species absence from BioCLIP pretraining.
5. A feature association passing confirmation would justify a small controlled image pilot, conditional on measurement feasibility. Use paired seeds, original/reconstruction/edit/matched control conditions, and blinded scoring. Keep reconstruction quality, association, image change, and biological causal claims separate. A failed gate is a completed negative result, not a reason to force a positive replacement.

No model files or historical experiment artifacts were changed by this assessment. All calculations used CPU-only source parsing and count aggregation.
