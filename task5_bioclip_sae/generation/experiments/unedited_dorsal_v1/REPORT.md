# Unedited dorsal-view generator calibration

**October 7 review update:** [human scoring is now complete](../human_review_2026_10_07/REPORT.md): 27/32 images were judged eligible and countable, with five NA. The generation and provisional-AI records below describe their earlier stages.

Generated 32 unedited Coccinella septempunctata images at 512 x 512 using seeds 1000-1031 and one prespecified overhead-view prompt. Neither spots nor spot count were named in the ordinary prompt. The original space-formatted BioCLIP vector was injected without any SAE operation.

Generation took 68.2 seconds. All model state hashes and saved input hashes were unchanged. GPU: NVIDIA GeForce RTX 5090; maximum sampled temperature 60 C; minimum sampled free VRAM 20857 MiB. No hardware guard, OOM, or CUDA error occurred.

## Review packet

[Scoring rules](human_review/SCORING_RULES.md) | [Blank human scoring CSV](human_review/scoring.csv)

- [Native-resolution contact sheet 1](human_review/contact_sheet_01.png)
- [Native-resolution contact sheet 2](human_review/contact_sheet_02.png)
- [Native-resolution contact sheet 3](human_review/contact_sheet_03.png)
- [Native-resolution contact sheet 4](human_review/contact_sheet_04.png)
- [Native-resolution contact sheet 5](human_review/contact_sheet_05.png)
- [Native-resolution contact sheet 6](human_review/contact_sheet_06.png)

Individual blind images are in human_review/images/. The seed key is separate in blind_key.csv. A subsequent [single provisional AI review](AI_REVIEW_PROVISIONAL.md) assessed all 32 images: 12 eligible dorsal views, 13 ineligible and 7 uncertain. All 12 eligible images show spots, but only one exact count was accepted (8). All other counts are NA, never zero. Human annotations remain blank by user request. The generation manifest records the earlier pre-annotation stage; the separate AI files record the later screening. This is measurement screening, not human or biological validation.

Full provenance: protocol.json, run_manifest.json, image_records.json, and hardware.jsonl.
