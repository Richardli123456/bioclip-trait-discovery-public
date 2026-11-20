# Sources and attribution

This repository combines original experiment code and reports with third-party metadata and reference photographs. No blanket license is applied to those third-party materials; retain their attribution and respect their individual terms.

- [BioCLIP](https://imageomics.github.io/bioclip/) and [BioCLIP 2](https://imageomics.github.io/bioclip-2/) provide the scientific background.
- The official [BioCLIP 2 model](https://huggingface.co/imageomics/bioclip-2) is downloaded separately at revision `2957b322090f9cb17ae72c71981c7218a28d81e0`; model weights are not committed.
- Task 5 uses [TreeOfLife-10M naming metadata](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/91debffb7146c32c89d76feb1eb575b555e2ecc7/metadata/naming), pinned to revision `91debffb7146c32c89d76feb1eb575b555e2ecc7`. Its source URL and checksum are in `task5_bioclip_sae/data/dataset_metadata.json`. See the upstream dataset card for terms.
- Task 4 taxonomy metadata includes GBIF source records. Preserve source identifiers and provenance in its data folder.
- Reference photographs retain their individual photographer credits and licenses, including attribution, share-alike and noncommercial terms where indicated. See [Task 4 photo credits](task4_bioclip_pilot/results/photo_credits.md) and [Task 5 photo credits](task5_bioclip_sae/results/photo_credits.md). Source JSON files also preserve provider metadata.
- Research references: [TaxaAdapter](https://arxiv.org/abs/2603.26128) and [SAE interpretation/intervention reference](https://arxiv.org/abs/2502.06755).
- The generation pilot caches [official TaxaAdapter code](https://github.com/Imageomics/TaxaAdapter) and [adapter weights](https://huggingface.co/mridulk/TaxaAdapter), with their MIT license. Pinned Stable Diffusion 1.5 and VAE components retain their own model licenses; see the [generation instructions](task5_bioclip_sae/generation/README.md). Generated comparison images are explicitly labeled as model outputs.

Photo observations and selected-feature interpretations are exploratory annotations by this project, not expert-validated trait labels or claims made by the photo providers.
