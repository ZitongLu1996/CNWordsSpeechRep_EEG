# Data placement

Download the Zhao et al. (2025) dataset from [Science Data Bank](https://cstr.cn/31253.11.sciencedb.24416). This repository does not redistribute participant recordings.

The analysis directly reads the preprocessed EEGLAB epoch pairs (`.set` plus its referenced `.fdt`) for 30 participants. Copy the three speech-mode directories from the public dataset into this directory. Keep the participant, signal, and filename structure unchanged:

```text
data/
├── imagined speech/
│   ├── S01/EEG/S01.set + S01.fdt
│   └── ... S02–S30 ...
├── silent speech/
│   ├── S01/EEG/S01.set + S01.fdt
│   ├── S01/sEMG/S01.set + S01.fdt
│   └── ... S02–S30 ...
└── overt speech/
    ├── S01/EEG/S01.set + S01.fdt
    ├── S01/sEMG/S01.set + S01.fdt
    └── ... S02–S30 ...
```

EEG inputs are the files below each `EEG/` directory. sEMG inputs are below `sEMG/`. All three modes are used for EEG analyses; only Silent and Overt sEMG are used for the final control analyses. Audio, raw `.cdt` recordings, and Imagined sEMG are not direct inputs.

The scripts crop the supplied epochs to −0.1–0.9 s and resample them to 250 Hz during analysis. Do not interpret 250 Hz as the sampling rate of the original public files.

Only exact event names `B1(1)` through `B10(10)` are retained. Any other or composite event label is excluded and logged. The labels denote 我, 你, 吃, 喝, 好, 不, 冷, 热, 左, 右, respectively.

Run `python check_data.py` from anywhere to validate placement. For data stored elsewhere, use `--data-root /path/to/data` or copy `config/config.example.yaml` to `config/config.yaml` and edit it. CLI values take precedence over config values, which take precedence over repository defaults.
