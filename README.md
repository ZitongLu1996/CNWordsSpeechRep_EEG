# Chinese word representations show distinct temporal dynamics across speech modes

## Overview

Portable analysis code for time-resolved decoding and representational similarity analyses of Chinese word representations across Imagined, Silent, and Overt speech in 30 participants.

## Data availability

Participant data are not included. Download Zhao et al. (2025) from [Science Data Bank](https://cstr.cn/31253.11.sciencedb.24416).

## Repository structure

`data/` holds user-supplied preprocessed epochs, `scripts/decoding/` and `scripts/rsa/` hold the retained analysis chain, `model_rdms/` holds finalized small model inputs, `outputs/` receives generated results, and `docs/` records data and reproducibility details.

## Installation

```bash
conda env create -f environment.yml
conda activate chinese-word-representations
```

Alternatively, use `pip install -r requirements.txt` in a clean environment.

## Downloading the data

Download the public dataset using the link above. See [data/README.md](data/README.md) for the exact subset and structure.

## Data placement

Place the public preprocessed mode directories directly under `data/`. No source-code path edits are needed.

## Validate dataset

```bash
python check_data.py
```

For external storage: `python check_data.py --data-root /Volumes/data/ChineseSpeechDataset`. The same overrides work with `run_analysis.py`. A persistent optional override can be placed in `config/config.yaml`, based on the example file.

## Analysis pipeline

See [docs/ANALYSIS_PIPELINE.md](docs/ANALYSIS_PIPELINE.md). For example:

```bash
python run_analysis.py neural-rdms
python run_analysis.py within-mode-rsa
python run_analysis.py cross-mode-rsa
```

## Running analyses

Run `python run_analysis.py --help` for available named steps. Paths resolve from the repository location, not the current working directory. Generated files go to `outputs/`.

## Model RDMs

The finalized Visual form (`visual_alexnet_conv2`), Semantic (`semantic_tencent`), and Phonetic features (`phonology_distributed_feature_v3_92d_refined`) RDM bundle is included. The refined model contains onset, glide, nucleus, and coda blocks of 22 dimensions each plus four one-hot lexical-tone dimensions (92 total). See `model_rdms/README.md`.

## Figure reproduction

Figure-reproduction materials will be added separately.

## Citation

See `CITATION.cff`; complete the author and article metadata before release.

## License

A license has not yet been selected. See `LICENSE_TO_CHOOSE.md`.
