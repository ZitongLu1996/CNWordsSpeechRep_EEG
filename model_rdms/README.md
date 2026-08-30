# Finalized model-based RDM bundle

## Purpose

These files contain the finalized model-based representational dissimilarity matrices used in the RSA analyses reported in the manuscript (Figures 3–5). They are direct exports of the saved analysis inputs, not reconstructed or recomputed alternatives.

## Stimuli

All matrices use this fixed order: 我, 你, 吃, 喝, 好, 不, 冷, 热, 左, 右 (conditions B1–B10). `stimulus_labels.csv` provides zero-based indices, condition IDs, characters, and English glosses.

## Models

### Visual form

- Internal model: `visual_alexnet_conv2`
- Source: ImageNet-pretrained AlexNet (`AlexNet_Weights.IMAGENET1K_V1`)
- Layer: `alexnet.features[3]`, the second convolutional layer
- Representation: flattened convolutional feature map from the experimental character images
- RDM: pairwise cosine distance

### Semantic

- Internal model: `semantic_tencent`
- Source: Tencent AI Lab Chinese word/phrase embeddings
- Embedding dimensionality: 200
- RDM: pairwise cosine distance

### Phonetic features

- Internal model: `phonology_distributed_feature_v3_92d_refined`
- Full description: distributed phonetic-feature model
- Representation: four 22-dimensional segmental slots (onset, glide, nucleus, coda; 88 dimensions) plus four-dimensional lexical-tone one-hot coding, for 92 dimensions total
- RDM: pairwise cosine distance
- References: https://doi.org/10.1093/cercor/bhw300 and https://doi.org/10.1016/j.jml.2009.05.001

The finalized project source explicitly contains all 92 column labels. They are exported in `refined92d_feature_labels.csv`; no labels were inferred or invented for this bundle.

## File formats

- `.npy`: raw numeric matrix in NumPy format
- `.csv`: UTF-8 human-readable matrix with Chinese-character row and column labels
- `.npz`: combined machine-readable bundle containing all three RDMs and stimulus labels
- `preview/*.png`: inspection-only heatmaps; these previews do not alter the stored values and are not manuscript figures

## RDM convention

Rows and columns use the identical fixed stimulus order. The diagonal represents self-dissimilarity and is zero; each RDM is symmetric. All three models use cosine distance. Values in NPY, CSV, and NPZ are the original finalized analysis values and have not been normalized, ranked, z-scored, or rescaled.

## Validation

| Model | Shape | Max asymmetry | Max absolute diagonal | Min | Max |
|---|---:|---:|---:|---:|---:|
| Visual form | 10 × 10 | 0 | 0 | 0 | 0.14724213933715247 |
| Semantic | 10 × 10 | 0 | 0 | 0 | 0.74867050876538166 |
| Phonetic features | 10 × 10 | 0 | 0 | 0 | 0.81666030059435779 |

## Scope

These files contain the finalized model RDMs used for the reported analyses; users do not need to reconstruct AlexNet features, Tencent embeddings, or the 92D representations in order to visualize or reuse the published RDMs. No participant data are included.
