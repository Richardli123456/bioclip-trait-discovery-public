# How BioCLIP trait discovery works

What happens if we take the vector for a sparrow, change one learned number, and use the edited vector to generate a picture? In one experiment, increasing feature 766 made the sparrow appear yellow in all four image seeds reviewed. Setting the same feature to zero in a Yellowhammer made the generated bird appear brown in all four. This is evidence that a direction in the model's representation can influence generated colour. The images were judged with the conditions visible, and some edits also changed the surroundings or shape. Whether the direction selectively controls plumage colour remains open. [Gallery, complete image sheets and measurements](task5_bioclip_sae/generation/experiments/feature_gallery_v2/REPORT.md)

This guide follows one species through the code, then explains how the study tested three different claims. Reconstruction asks whether a sparse code preserves the original vector. Recurrence asks whether independent fits learn similar features. Trait validation asks whether those features predict an independently measured property or consistently change a specific part of an image. Each question needs its own comparison.

## 1. From a species record to 768 numbers

The input is a row of taxonomy. The expanded pipeline cleans the species records, joins seven taxonomy levels with spaces and passes the resulting text through the frozen BioCLIP 2 text encoder. One row produces one raw vector $x$ with 768 components. BioCLIP's prior image–text training makes text vectors interesting for studying appearance. The extraction itself sees the species description as words. Two related birds can therefore have similar vectors simply because many of their taxonomy words are identical. Reference photographs illustrate species for readers and play no part in this encoding step. [Species preparation](task5_bioclip_sae/generation/prepare_expanded_species_v2.py) · [Expanded vector extraction](task5_bioclip_sae/generation/prepare_expanded_embeddings.py)

The early 20-bird pilot looked up neighbours by cosine similarity, $c(x,y)=(x^T y)/(\|x\|_2\|y\|_2)$. This compares directions while discarding length. After dividing each row by its length, a matrix of pairwise dot products gives the cosine scores. Exclude the query itself and take the highest remaining score to find its nearest neighbour. The SAE and generator work with the raw vectors. Even the separator in the input matters here. Switching from commas to the generator's spaces changed the nearest neighbour for about 53% of species in the format comparison. [Pilot extraction and ranking](task4_bioclip_pilot/run_embeddings.py) · [Format alignment](task5_bioclip_sae/generation/prepare_alignment.py)

| Dataset and representation | What it answers |
|---|---|
| 5,388 animals, comma-separated text | Historical SAE pilot. Whole genera were assigned to train, validation or test, giving 4,254 / 541 / 593 species. |
| Same animals, space-separated text | Historical generator-aligned pilot. The change of input representation required a separate SAE fit. |
| Corrected TreeOfLife-10M corpus, space-separated text | Main recurrence and follow-up experiments. Of 362,460 records, 289,829 went to training, 36,065 to validation and 36,566 to test. The corpus includes multiple kingdoms. |

Holding out genera asks whether the newly trained SAE generalizes to genera outside its training split. Earlier pretraining of BioCLIP and TaxaAdapter is a separate matter. The public TreeOfLife-10M records also do not establish the exact species membership of the generator's TreeOfLife-1M training set. [Expanded study and corpus provenance](task5_bioclip_sae/generation/experiments/expanded_followup_2026_10_04/REPORT.md)

## 2. What the sparse autoencoder learns

Imagine describing each species vector as a mixture of at most 32 entries from a dictionary of 2,048 directions. Each dictionary entry is itself a vector of length 768. The encoder chooses entries and their strengths. The decoder adds their weighted contributions together. A feature number identifies a dictionary entry in a specific checkpoint. The training objective supplies no labels such as “yellow” or “owl.”

For $N$ training rows of dimension $d=768$, the code computes a mean vector $\mu=N^{-1}\sum_i x_i$ and a single scale $s=\sqrt{\sum_i\|x_i-\mu\|_2^2/(Nd)}$. Every species uses those same training statistics. In column-vector notation the forward pass is

$$
u=(x-\mu)/s,\qquad
a=\mathrm{ReLU}(W_e u+b_e),\qquad
z=\mathrm{TopK}_{32}(a),\qquad
\hat x=\mu+s(Dz+b_o).
$$

$W_e$ has shape $2048\times768$ and $D$ has shape $768\times2048$. ReLU replaces negative preactivations with zero. TopK keeps the 32 largest values in their original coordinate positions and zeros the rest. For a toy vector $a=(0,0.3,1.7,0.8)$ with $k=2$, the sparse code would be $(0,0,1.7,0.8)$. Its decoded contribution in standardized units would be 1.7 times the third decoder column plus 0.8 times the fourth. The real architecture imposes this sparsity in advance.

Here is the central operation in readable PyTorch, using the names from the pilot model. For a batch of size $B$, the input and output have shape $B\times768$, and the intermediate sparse tensor has shape $B\times2048$.

```python
positive = F.relu(self.encoder((raw - self.mean) / self.scale))
values, indices = positive.topk(self.k, dim=-1)
z = torch.zeros_like(positive).scatter(-1, indices, values)
reconstruction = (self.decoder(z) + self.output_bias) * self.scale + self.mean
```

The pilot minimizes the mean squared error of $(\hat x-x)/s$ over species and components with Adam. Backpropagation updates the encoder and decoder. The code renormalizes each decoder column to unit length after an optimizer step, which keeps the size of a feature's contribution interpretable through its activation. Validation error chooses the saved checkpoint. Training-only PCA at 32 and 128 components provides a linear reconstruction reference. PCA uses the same fixed subspace for all rows, whereas the SAE can choose a different subset of dictionary entries for each species. [TopKSAE, training loop and PCA comparison](task5_bioclip_sae/train_sae.py) · [Expanded training runner](task5_bioclip_sae/generation/run_expanded_sae_study.py)

## 3. How well does the representation survive?

For a held-out set $H$, the reported explained fraction is

$$
1-\frac{\sum_{i\in H}\|x_i-\hat x_i\|_2^2}{\sum_{i\in H}\|x_i-\mu_{\rm train}\|_2^2}.
$$

Predicting the training mean for every row scores zero. Perfect reconstruction scores one. The expanded anchor scores **66.31%** on its test set, so its squared error is about one third of the training-mean predictor's error. The mean cosine between an original and reconstructed test vector is **0.90788**. Cosine can stay high even if vector length changes, which is one reason to report both quantities. Neither metric measures how accurately a bird's colour is represented. [Expanded evaluation](task5_bioclip_sae/generation/experiments/expanded_followup_2026_10_04/REPORT.md)

## 4. Would another training run find the same feature?

Feature 766 in a second fit could encode something entirely different. The study fits one anchor SAE and five additional SAEs with different random seeds. For anchor feature $j$, collect its activations over training species into a vector $v_j$. In each replicate, pick the feature $k$ with the largest signed Pearson correlation $r(v_j,v_k)$. Correlation measures whether the same species tend to activate both features. Then compare the **same pair's** decoder columns with cosine similarity. This asks whether activating the pair would move reconstructed vectors in the same direction. A match must score at least 0.7 on both measures in every replicate. Matching is per anchor feature, so two anchors can select the same replicate feature. [Matching protocol](task5_bioclip_sae/generation/experiments/expanded_stability_w2048_k32_v2/protocol.json)

**732 of 2,048** anchor features passed. The smaller space-aligned pilot had zero passing features under the combined rule. Corpus size and composition both changed. Among the expanded recurrent features, roughly 76% have all eight top species in one family. A feature responding reliably to a particular lineage is an informative result, but the lineage can share names, appearance and many other properties at once. [Expanded study](task5_bioclip_sae/generation/experiments/expanded_followup_2026_10_04/REPORT.md) · [Repository summary of top-species composition](README.md)

## 5. How a one-feature edit reaches the image

Suppose the original sparrow vector is $x$, its code is $z$, and we want coordinate $j$ to have strength $t$. Set $z'_j=t$ and leave every other coordinate unchanged. Simply decoding $z'$ would mix the intended edit with reconstruction error. The gallery keeps the original residual $r=x-\hat x$ and uses

$$
x_{\rm edit}=r+\mathrm{decode}(z')=x+s(t-z_j)D_{:,j}.
$$

This follows because the mean, output bias and unchanged columns cancel when we subtract the two decoded vectors. If $t=z_j$, the result is the original input exactly in the difference formula. Setting $t=0$ removes the feature contribution. Doubling sets $t=2z_j$. To add an initially weak feature, the gallery uses target levels of 0.4, 0.8 and 1.2 times its top training activation, skipping levels already reached. These are absolute target strengths. [Gallery vector construction](task5_bioclip_sae/generation/run_feature_gallery.py)

The edited 768-vector enters TaxaAdapter's learned projection. In the injection code, `get_image_embeds` sends the supplied vector through `image_proj_model` and sends a zero vector through the same projection for the unconditional branch. The downstream diffusion generator uses that conditioning along with the text prompt and image seed. Pairing original and edited renders at the same seed holds the starting randomness fixed. A same-length random edit asks whether a comparably large displacement can produce a similar visual change. The original eight-image pilot also compared the raw input against its plain reconstruction, which showed why preserving the residual mattered. [Injection code](task5_bioclip_sae/generation/run_interventions.py) · [Generation settings and pilot](task5_bioclip_sae/generation/README.md)

**Reading the yellow result.** Researchers searched the top species of recurrent features for appearance-related Latin name stems. Seven candidates went into the galleries. Orange 938, red 709 and yellow 766 showed the clearest exploratory effects. For 766, whole-image yellow pixels averaged 0.8% for originals and 1.2%, 2.8% and 4.8% across increasing targets, compared with 0.2% under the random control. Those percentages aggregate four addition species and four seeds. They count background pixels too. The Yellowhammer removal comparison asks a complementary question about an animal already high on the feature. Across the experiments, changes in framing, objects and anatomy limit conclusions about selectivity. Four image seeds from one edited checkpoint also measure a different kind of repeatability from five independently trained SAEs. [Gallery evidence](task5_bioclip_sae/generation/experiments/feature_gallery_v2/REPORT.md) · [Blind scoring package](task5_bioclip_sae/generation/experiments/blind_colour_review_v1/README.md)

## 6. Can an activation predict a measured biological trait?

AVONET provides bird measurements independent of the taxonomy text. The beak endpoint is $y=\ln(L_{\rm nares}/D_{\rm beak})$, using species-mean nostril-to-tip length and depth at the nostrils. A higher ratio means a more elongated beak. A supervised probe learns to predict this outcome from the raw embedding. Its test $R^2$ was about 0.66, compared with about 0.68 using family/order averages. A positive score alone would be easy to overinterpret here. A predictor that knows the family already captures much of the difference between species. [Endpoint, split, probe and baseline](task5_bioclip_sae/generation/experiments/trait_benchmark_v1/REPORT.md)

The tail study asks a narrower question about one SAE coordinate. Its endpoint is $\ln(\text{species-mean tail length}/\text{species-mean wing length})$. Candidate 814 was selected using training data. Family centering compares deviations within a family, reducing the influence of broad differences between families. The family-centered correlation went from −0.1467 in training to +0.0803 in validation, reversing sign. Adding the feature also failed to improve the taxonomy prediction. Under the prespecified stopping rule, test endpoint values were left uncomputed and no tail-image experiment followed. [Tail analysis and decision](task5_bioclip_sae/generation/experiments/trait_followup_2026_10_07/REPORT.md)

Image-based endpoints have a different failure mode. In 32 unedited seven-spot ladybird renders, a human reviewer could count spots in 27. Only three of those showed exactly seven visible spots. Five unscorable views remain missing. A study of spot-count edits would therefore have to distinguish a true count change from changes in viewpoint, occlusion and the generator's own inconsistent baseline. [Human scoring report](task5_bioclip_sae/generation/experiments/human_review_2026_10_07/REPORT.md)

## 7. What species neighbours and bridges show

Define a species' support $S$ as the indices of its active SAE features. The expanded search ranks neighbours by shared count $|S_a\cap S_b|$ and also reports Jaccard similarity $|S_a\cap S_b|/|S_a\cup S_b|$. For two species with 32 active features each and 24 shared, the union has 40 features and Jaccard is 0.60. Two species can share many feature IDs while using them at very different strengths. [Exact search and scoring](task5_bioclip_sae/generation/analyze_expanded_overlap.py)

A bridge makes those differences visible one at a time. Starting with species A's code, the code replaces each differing coordinate with its value in species B, largest decoded displacement first. At step $m$, the generator receives $x_A+sD(z^{(m)}-z_A)$. At the final step the code equals B's code, but the preserved reconstruction residual still belongs to A. Raw B and reconstructed B are therefore important endpoint controls. The ladybird example used 44 coordinate changes and two image seeds. Its intermediate colour patterns describe how the generator responds along this chosen path. They carry no evolutionary time scale or ancestry interpretation. [Bridge implementation](task5_bioclip_sae/generation/analyze_expanded_overlap.py) · [Bridge images and endpoint comparisons](task5_bioclip_sae/generation/experiments/expanded_followup_2026_10_04/REPORT.md)

## Reading or reproducing the repository

For a first walkthrough, open the gallery's full sheets alongside the vector-edit equation in section 5. Follow one species and one seed across original, low, medium, high and random conditions. Look separately at the animal and its background. Then read the AVONET report to see how the standard of evidence changes when independent measurements are available.

For a code walkthrough, start with `TopKSAE.encode`, `decode` and `metrics` in the pilot trainer. Follow the expanded training runner for the large-corpus experiment, then the gallery runner for edited input construction and the adapter override for injection. The repository README supplies PowerShell commands for the historical pilot and a separate generation environment. A fresh clone omits large arrays, model caches and checkpoints. The expanded study has its own preparation and artifact index. Reproducing a feature number requires its corresponding checkpoint, input format and feature dictionary. [Setup and historical commands](README.md) · [Generation environment](task5_bioclip_sae/generation/README.md) · [Expanded protocol and artifact index](task5_bioclip_sae/generation/experiments/expanded_followup_2026_10_04/REPORT.md)
