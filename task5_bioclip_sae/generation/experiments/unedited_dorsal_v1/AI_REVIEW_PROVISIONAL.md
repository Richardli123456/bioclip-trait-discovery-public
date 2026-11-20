# Provisional AI review of the 32-image dorsal-view pilot

This is one AI visual review, authorized as preliminary screening while human review remains pending. It is neither an independent human review nor biological expert validation. The reviewer inspected all six blind contact sheets and individual native-resolution images for close decisions. The separate seed key was not opened before annotations were saved. All images have the same unedited condition, so this conceals seed order rather than intervention identity.

The existing strict scoring rule was applied: a clear actual dorsal view with both wing covers visible; low side views fail; uncertainty is NA. A visible mark is counted only when its wing-cover boundary can be distinguished. Glare, merged marks, and ambiguous edge regions produce NA rather than zero.

| Measure | Provisional result |
|---|---:|
| Images reviewed | 32 |
| Dorsal view eligible | 12/32 |
| View ineligible | 13/32 |
| Eligibility uncertain | 7/32 |
| Eligible images with visible wing-cover spots | 12/12 |
| Exact visible spot count scorable | 1/32 |
| Count distribution among scorable images | One image with 8 visible marks |

The sole provisional exact count is D011. It includes eight distinct visible black wing-cover marks, including partly edge-touching marks; head and pronotum markings are excluded. This decision particularly needs human verification. The remaining 31 images are not assigned a numerical count. **No image was scored as having zero spots.**

The new unedited images visibly contain wing-cover spots in multiple eligible dorsal views. This supports continuing a visibility/calibration investigation; it does not yet support a reliable exact-count intervention experiment. The prompt and fresh seeds both differ from the original four-seed pilot, so these results do not isolate a prompt effect. No confidently scored seven-spot image in this conservative review is not an estimate of a zero seven-spot generation rate: most images have missing exact counts.

## Files and outstanding human review

- [Provisional AI annotations](ai_review_provisional.csv)
- [Machine-readable totals and limitations](ai_review_provisional_summary.json)
- [Unchanged blank human scoring sheet](human_review/scoring.csv)
- [Original scoring rules](human_review/SCORING_RULES.md)

The human scoring sheet remains blank. Human review should adjudicate eligibility and spot boundaries without treating these AI labels as ground truth. There is no second-reviewer agreement claim and no bootstrap analysis at this calibration stage.
