# Correction to the ladybird scoring interpretation

Added October 4, 2026 after the user's review. Original annotation files, numerical summaries and artifact audits are preserved as records of the initial AI scoring. This correction supersedes their interpretation as evidence of measurement readiness.

The reported seven exact count agreements were **seven images from only two image seeds**. Six were highly similar seed-42 images from paired conditions, all assigned zero visible spots. They do not provide six independent replications across generation seeds. The remaining case was B002, a reconstruction-condition image at seed 45, assigned seven visible dark patches by both AI passes.

| Blind ID | Condition | Seed | Initial agreed count | Corrected treatment |
|---|---|---:|---:|---|
| B004 | original | 42 | 0 | Unscorable for the stated dorsal-view endpoint |
| B009 | other active feature removed | 42 | 0 | Unscorable for the stated dorsal-view endpoint |
| B020 | reconstruction | 42 | 0 | Unscorable for the stated dorsal-view endpoint |
| B022 | selected feature doubled | 42 | 0 | Unscorable for the stated dorsal-view endpoint |
| B023 | selected feature removed | 42 | 0 | Unscorable for the stated dorsal-view endpoint |
| B024 | selected feature halved | 42 | 0 | Unscorable for the stated dorsal-view endpoint |

On visual reassessment, those six low-angle images do not adequately establish that both wing covers are mostly visible from above, as required by the locked rule. Their visible red area may appear unspotted, but that does not justify an eligible whole-wing-cover count of zero. The initial scorer accepted them too readily. Their corrected eligible counts are missing, not zero.

The other three original-vector seeds were already judged unscorable by both AI passes. After withdrawing the seed-42 eligibility decision, **none of the four original-vector images has an accepted count under this stricter application of the original rule**. Consequently this pilot supplies no reliable estimate of original-vector spot-count fidelity. It also cannot establish that the generator usually fails to draw spots across other seeds or prompts: the original sample contains only four seeds and inadequate views.

B002 remains one AI-agreed candidate count, not an independent biological verification or a demonstrated causal SAE effect. The disputed B016/B021 decisions remain unresolved. No new complete human review of all 28 images has been performed. The two AI passes used the same model family; they are not independent human raters, and exact count agreement is not evidence of biological accuracy.

## Revised immediate experiment

Use the original, unedited space-formatted species embedding for Coccinella septempunctata. Generate 32 fresh seeds with a fixed prompt requesting a single adult beetle viewed directly from above, with the whole body visible and in focus. Do not specify a spot count or desired spot appearance in the prompt. No SAE reconstruction or intervention is needed.

Retain every output. For each image, a human reviewer should record three basic fields: acceptable dorsal view (yes/no/uncertain), visible dark wing-cover spots (yes/no/uncertain), and visible patch count when scorable. Report the dorsal-view count out of all 32, then the spotted fraction and count distribution among eligible images with the denominator stated. Unscorable images are never treated as zero spots. This is a feasibility pilot; the larger 672-image intervention plan and bootstrap analysis are unnecessary at this stage.

If clear dorsal views are rare, improve and independently retest the viewing setup. If clear views are common but spots remain rare, prioritize another observable trait or address generator fidelity before treating spot removal as an informative endpoint. A feature-addition experiment might still be possible, but would answer a different, explicitly stated question.

In parallel, obtain a verified species manifest and expand raw space-formatted taxonomy embeddings. Rerun the same SAE architecture and prespecified stability rule to isolate the dataset change. The exact TreeOfLife-1M manifest was not found in the official GitHub/model releases checked again on October 4. Broader TreeOfLife-10M taxonomy can support a separately labeled expansion; it cannot establish exact TreeOfLife-1M coverage.

Passing stability would satisfy the representation prerequisite, not prove biological trait validity. The unedited-image pilot and human assessment still determine which trait experiment is usable.
