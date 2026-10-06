# OneFormer Class-Text Intervention Study

This repository contains the implementation and experimental notebooks for a controlled Representation Learning study of class-description text in OneFormer.

The project investigates whether changing the text associated with a fixed segmentation target during short OneFormer fine-tuning changes semantic, instance, or panoptic segmentation performance on Cityscapes.

## Research Question

**How do controlled changes to class-description text during OneFormer fine-tuning affect semantic, instance, and panoptic segmentation?**

The central hypothesis is that semantically equivalent wording should preserve segmentation behavior more closely than deliberately incorrect class wording.

## Experimental Design

The manipulated target is the Cityscapes **car** class. The image, segmentation mask, numeric class label, starting checkpoint, image order, optimization settings, and matched seeds are kept fixed. Only the associated class-description text is changed.

| Condition | Car-target text | Role |
|---|---|---|
| B0 | No additional fine-tuning | Pretrained starting reference |
| C0 | `a photo with a car` | Matched wording control |
| S1 | `a photo with an automobile` | Synonym substitution |
| T1 | `a photograph containing a car` | Template/paraphrase change |
| W1 | `a photo with a bicycle` | Deliberately incorrect class word |

C0 is the primary causal reference because C0, S1, T1, and W1 undergo the same fine-tuning procedure and differ only in the class-description text.

## Fine-Tuning Protocol

The controlled adaptation experiment uses:

- 50 training images
- 20 development images
- 500 Cityscapes validation images
- input resolution: `512 × 256`
- 50 optimizer updates
- batch size: 2
- learning rate: `1e-6`
- matched seeds: 42, 43, and 44
- common starting checkpoint
- identical image order
- identical trainable parameters and optimization settings

The final data preparation contained 377 intended car-text substitutions across 47 of the 50 training images.

## OneFormer Text Pathway

In this study, class-description text is a **training-time supervision signal**, not a free-form inference prompt.

The tested mechanism is:

**Class text → text embedding → query-text contrastive loss → gradient → parameter update → segmentation output**

A pre-experiment diagnostic verified that changing only the class text changes tokenization, text embeddings, contrastive loss, and gradients, while pre-update class and mask logits remain unchanged.

## Notebooks

### `OneFormer_Cityscapes_Project.ipynb`

Main experimental notebook containing the Cityscapes/OneFormer workflow, including:

- OneFormer Cityscapes setup
- official baseline reproduction
- dataset and annotation preparation
- controlled fine-tuning
- semantic segmentation evaluation
- instance segmentation evaluation
- panoptic segmentation evaluation
- matched-seed experiments
- result collection and validation

### `OneFormer_Text_Experiments.ipynb`

Additional text-intervention and diagnostic analyses, including:

- class-text intervention checks
- OneFormer text-encoder inspection
- causal-attention-mask verification
- query-text contrastive-pathway diagnostics
- fixed-reference text-embedding analysis
- semantic output-stability analysis
- predicted car-region overlap
- per-class changed-pixel analysis
- prediction-transition analysis
- representative and stress-case comparisons

## Pipeline Validation

Before interpreting the text interventions, the official OneFormer Cityscapes Swin-L evaluation pipeline was reproduced closely:

| Metric | Published | Reproduced | Difference |
|---|---:|---:|---:|
| Semantic mIoU | 83.00 | 82.74 | -0.26 |
| Instance AP | 45.60 | 45.53 | -0.07 |
| Panoptic PQ | 67.20 | 66.55 | -0.65 |

These values are used only to validate the evaluation pipeline. They are not treated as a before/after comparison with the later `512 × 256` controlled fine-tuning experiment.

## Main Results

Across the three matched seeds, the four fine-tuned conditions remained very close:

| Condition | mIoU (%) | AP (%) | PQ (%) |
|---|---:|---:|---:|
| C0 | 68.887 ± 0.025 | 21.971 ± 0.044 | 47.438 ± 0.061 |
| S1 | 68.911 ± 0.009 | 21.961 ± 0.045 | 47.494 ± 0.039 |
| T1 | 68.887 ± 0.025 | 21.968 ± 0.045 | 47.441 ± 0.063 |
| W1 | 68.899 ± 0.031 | 21.969 ± 0.034 | 47.514 ± 0.077 |

Relative to matched C0, the semantic effects were:

- S1 − C0: `+0.0242 pp`
- T1 − C0: `−0.0003 pp`
- W1 − C0: `+0.0121 pp`

All paired semantic bootstrap intervals included zero.

## Representation and Output Stability

The fixed-reference text embeddings changed substantially:

- S1 cosine to C0: `−0.0217`
- T1 cosine to C0: `0.9963`
- W1 cosine to C0: `−0.7654`

Despite these representation differences, the downstream semantic predictions remained highly stable.

Mean semantic agreement relative to C0 exceeded `99.95%` across the tested interventions, and predicted semantic car-region IoU exceeded `99.94%`.

This indicates that substantial changes in the text representation did not translate into a clear systematic aggregate segmentation shift under the tested short fine-tuning regime.

## Data Integrity and Mechanism Checks

A text-mapping issue affecting sidewalk and building labels was identified during development and corrected before the final experiments.

The corrected data were verified against the original Cityscapes annotations, including:

- 50/50 training images
- 1,166/1,166 mask rows
- 1,166/1,166 numeric class-label rows
- all C0/S1/T1/W1 text strings
- 50/50 prepared-file hashes

The OneFormer text pathway was also checked independently. After correcting the causal attention mask across all six text-transformer layers, the implementation matched the reference encoder exactly in the controlled comparison.

## Main Conclusion

Under this controlled short-adaptation regime, class-description wording clearly changes OneFormer's text representation and optimization signal, but none of the tested wording interventions produced a clear systematic dataset-level change in semantic, instance, or panoptic segmentation performance.

Localized class- and image-specific differences were still observable.

## Scope and Limitations

The conclusions are limited to:

- one OneFormer checkpoint family
- Cityscapes
- one manipulated class
- 50 training images
- 50 optimizer updates
- three matched seeds
- `512 × 256` resolution
- learning rate `1e-6`
- short fine-tuning from a pretrained model

The study does not test training from scratch or establish general language robustness across datasets or architectures.

## Data and Model Files

The raw Cityscapes dataset, large model checkpoints, intermediate predictions, and large experiment archives are not included in this repository because of size and data-distribution constraints.

The repository focuses on the experimental code and analysis notebooks needed to document and reproduce the study workflow.

## Software

- OneFormer
- PyTorch
- Hugging Face Transformers
- Cityscapes evaluation tools
- Google Colab
- Google Drive

Fine-tuning used Transformers `5.16.1`. Post-hoc text-embedding analysis used Transformers `5.17.0`.

## Author

**Maisha Fahmida**<br>  
Project Representation Learning<br>
Friedrich-Alexander-Universität Erlangen-Nürnberg
