"""Shared repository-relative configuration for the RSA pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DATA_PATH = Path(os.environ.get("CWR_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
RSA_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("CWR_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()
MODEL_RDM_DIR = Path(os.environ.get("CWR_MODEL_RDM_ROOT", PROJECT_ROOT / "model_rdms")).expanduser().resolve()
FIGURES_DIR = RESULTS_DIR / "figures"

sub_ids = [f"S{i:02d}" for i in range(1, 31)]
tasks = ["imagined speech", "overt speech", "silent speech"]
EEG_TASKS = tasks.copy()
SEMG_TASKS = ["overt speech", "silent speech"]

# Never replace this mapping with MNE's import-specific numeric event codes.
FIXED_EVENT_ID = {
    "B1(1)": 1,
    "B2(2)": 2,
    "B3(3)": 3,
    "B4(4)": 4,
    "B5(5)": 5,
    "B6(6)": 6,
    "B7(7)": 7,
    "B8(8)": 8,
    "B9(9)": 9,
    "B10(10)": 10,
}
LABEL_TO_WORD = {
    1: "我",
    2: "你",
    3: "吃",
    4: "喝",
    5: "好",
    6: "不",
    7: "冷",
    8: "热",
    9: "左",
    10: "右",
}
RSA_LABEL_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RSA_WORD_ORDER = ["我", "你", "吃", "喝", "好", "不", "冷", "热", "左", "右"]

tmin = -0.1
tmax = 0.9
sfreq = 250

navg = 4
time_win = 5
time_step = 5
time_opt = "average"

TENCENT_EMBEDDING_PATH = None  # Finalized model RDMs are bundled; embeddings are not required.
