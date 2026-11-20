# Human scoring instructions

Review the 32 randomized images once, using the native-resolution images or six contact sheets. Do not consult the seed key until all scores are saved. This is an unedited-baseline calibration, so all images have the same condition; randomization conceals seed order.

- view_eligible: yes only for a clear actual dorsal/top view with both wing covers visible, sufficient focus, and no substantial occlusion. Low side-angle views are no. Uncertainty is NA.
- spots_visible: yes/no only when view_eligible=yes and the answer is clear; otherwise NA. Distinguish markings on the wing covers from head/pronotum markings, legs, glare, or shadows.
- spot_count: a nonnegative integer only when view_eligible=yes and all visible wing-cover spots can be confidently distinguished. Zero means an eligible, visibly spotless pair of wing covers. Blurred, hidden, merged, or ambiguous markings give NA, with a note. Do not infer hidden spots.
- notes: describe any ambiguity, low view angle, blur, occlusion, or uncertain mark boundary.

Keep NA separate from zero. Report eligible/32, any-spots/eligible, count-scorable/32, and the distribution of counts among count-scorable images. Human annotations are currently blank. An AI review, if later added, must be separately identified and is not human or expert validation.
