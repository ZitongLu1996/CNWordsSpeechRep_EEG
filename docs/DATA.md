# Data and analysis inputs

The source dataset is Zhao et al. (2025), hosted by [Science Data Bank](https://cstr.cn/31253.11.sciencedb.24416). The study uses 30 participants, three EEG modes (Imagined, Silent, Overt), and Silent/Overt sEMG.

The ten conditions are B1 我, B2 你, B3 吃, B4 喝, B5 好, B6 不, B7 冷, B8 热, B9 左, and B10 右. Import-time MNE event codes are not treated as analysis labels. Only exact annotation names `B1(1)`–`B10(10)` are mapped to integers 1–10; every unexpected or composite label is excluded and logged.

Direct inputs are preprocessed EEGLAB epochs at `data/<mode>/SXX/EEG/SXX.set` and, for Silent/Overt only, `data/<mode>/SXX/sEMG/SXX.set`, with the corresponding `.fdt` files. The repository does not need raw `.cdt` files or audio. Analysis-specific processing crops EEG/sEMG epochs to −0.1–0.9 s and resamples to 250 Hz. Outputs are written only under `outputs/`.
