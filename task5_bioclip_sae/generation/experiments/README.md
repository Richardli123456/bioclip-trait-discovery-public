# Which experiment answers my question?

The experiments use three kinds of evidence. SAE training runs measure how well a dictionary captures BioCLIP vectors. Generator runs show what happens to synthetic images after an edit. Bird measurements and human annotations test whether a proposed interpretation holds up against an observable outcome. Use the reports below to trace a claim back to the comparison that produced it.

| Question | Evidence to inspect | How to read it |
|---|---|---|
| Does a feature survive retraining? | [Expanded study](expanded_followup_2026_10_04/REPORT.md) and [matching protocol](expanded_stability_w2048_k32_v2/protocol.json) | Start from one anchor coordinate. Its best activation match in each replicate must also point in a similar decoder direction. |
| Can a feature change colour in a generated animal? | [Orange and stripe edits](feature_gallery_v1/REPORT.md), [remaining five candidates](feature_gallery_v2/REPORT.md) | Compare addition, removal and a displacement of matched size. In the [Yellowhammer removal sheet](feature_gallery_v2/sheet_remove_766_Emberiza_citrinella.png), track the body separately from the background. |
| Can raw vectors predict measured beak shape? | [AVONET benchmark](trait_benchmark_v1/REPORT.md) | Compare held-out predictions with the family/order baseline. Related birds can share beak shapes even without a dedicated shape feature. |
| Can a single feature predict a measured trait? | [Tail/wing follow-up](trait_followup_2026_10_07/REPORT.md) | Read the training selection before the validation result. The selected association changed sign and the study stopped before testing. |
| Can a reviewer count a generated animal's spots? | [Human scoring report](human_review_2026_10_07/REPORT.md) | Keep the 32 rendered images, the 27 scorable views and the three exact-seven observations as distinct denominators or counts. |
| What happens between two species codes? | [Potoo bridge](expanded_bridge_potoo_v1/REPORT.md) and [owl bridge](expanded_bridge_owl_v1/REPORT.md) | Compare the same step across seeds. The path is defined by decoder displacements, and the source residual remains in each edited vector. |

**Choosing the correct model.** The [first frozen-SAE continuation](existing_sae_v1/REPORT.md) and [small-corpus stability report](followup_2026_10_04/REPORT.md) used the earlier 5,388-species dataset. For results concerning 362,460 records, follow the corrected expanded v2 study. Feature IDs belong to the checkpoint named in the experiment.

**Running an experiment.** Start with the [generation environment and commands](../README.md) for the original pilot. Reading the linked reports, protocols and image sheets requires no model downloads. Rebuilding the expanded results requires assets listed in the expanded report, including large local arrays and checkpoints. The [technical guide](../../../TECHNICAL_GUIDE.md) explains the equations used across these experiments.
