#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:21:45 2026

Modifications:
    1. resample to 250 Hz 
    2. set time_win = 5 and time_step = 5 
    3. change time_opt from "features" to "average"
    03,28,2026
    4. FIXED LABELS: B1-B10 → 1-10 globally 
    04,25, 2026

"""
# server_decoding_3_bitranfer.py
# ------------------------------------------------------------------------------

# Repository root/
#   ├── server_scripts/                
#   │   └── server_decoding_3_bitranfer.py  
#   ├── 3.preprocessed data/            
#   │   ├── imagined speech/
#   │   │   ├── S01/EEG/S01.set
#   │   │   └── ... remaining participants
#   │   ├── overt speech/
#   │   │   ├── S01/EEG/S01.set
#   │   │   └── ... remaining participants
#   │   └── silent speech/
#   │       ├── S01/EEG/S01.set
#   │       └── ... remaining participants
#   └── 5.Analysis/                     
#       └── decoding_results/      
# All cross-task bidirectional decoding results (.h5 files) are saved to:
# outputs/decoding/cross_mode/
# File naming format: SXX_task1_to_task2_accs.h5     
# ------------------------------------------------------------------------------
import os
import mne
import numpy as np
import h5py
from pathlib import Path
import sys
from neurora.decoding import bidirectional_transfer_decoding
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.events import retain_valid_events
from scripts.utils.paths import resolve_paths


# ---------------------------
# Analysis parameters
# ---------------------------
sub_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07",
            "S08", "S09", "S10", "S11", "S12", "S13", "S14",
            "S15", "S16", "S17", "S18", "S19", "S20", "S21",
            "S22", "S23", "S24", "S25", "S26", "S27", "S28",
            "S29", "S30"]

task_pairs = [("imagined speech", "overt speech"),
              ("imagined speech", "silent speech"),
              ("overt speech", "silent speech")]

n_conditions = 10
navg = 4
time_win = 5
time_step = 5
iter_n = 10
pca_components = 0.95
smooth = True
time_opt = "average"

# ---------------------------
# Exact mapping of B1(1)-B10(10) event labels.
# ---------------------------
FIXED_EVENT_ID = {
    'B1(1)': 1,
    'B2(2)': 2,
    'B3(3)': 3,
    'B4(4)': 4,
    'B5(5)': 5,
    'B6(6)': 6,
    'B7(7)': 7,
    'B8(8)': 8,
    'B9(9)': 9,
    'B10(10)': 10
}

# ---------------------------
# Repository-relative paths
# ---------------------------
_paths = resolve_paths()
base_path = _paths["data_root"]
save_dir = _paths["output_root"] / "decoding" / "cross_mode"
os.makedirs(save_dir, exist_ok=True)

print("Current working directory: {}".format(os.getcwd()))
print("Decoding results save directory: {}".format(save_dir))

# ---------------------------
# Main participant loop
# ---------------------------
for sub in sub_ids:
    sub_seed = int(sub[1:])
    np.random.seed(sub_seed)
  

    print("\n===== Processing subject {} | seed = {} =====".format(sub, sub_seed))

    task_data = {}
    task_labels = {}
    unique_tasks = set([t for pair in task_pairs for t in pair])

    for task in unique_tasks:
        file_path = os.path.join(base_path, task, sub, "EEG", "{}.set".format(sub))

        if not os.path.exists(file_path):
            print(" Missing file: {}, skip {}.".format(file_path, task))
            continue

        try:
            epochs = mne.read_epochs_eeglab(file_path)
            epochs, fixed_labels = retain_valid_events(epochs)
            epochs.crop(tmin=-0.1, tmax=0.9)
            epochs.resample(250)

            # ---------------------------
            # Align the retained event labels across modes.
            # ---------------------------
            corrected_labels = np.asarray(fixed_labels)

            task_data[task] = epochs.get_data()
            task_labels[task] = corrected_labels

            print("Loaded {} data for {}: shape = {}".format(task, sub, task_data[task].shape))
            print("Labels successfully aligned: B1(1)->1 to B10(10)->10")

        except Exception as e:
            print("Error loading {}: {}, skip {} for {}".format(file_path, str(e), task, sub))
            continue

    # ---------------------------
    # Bidirectional decoding
    # ---------------------------
    for t1, t2 in task_pairs:
        if t1 not in task_data or t2 not in task_data:
            continue

        data1, labels1 = task_data[t1], task_labels[t1]
        data2, labels2 = task_data[t2], task_labels[t2]

        min_trials_per_cond = []
        for c in range(1, n_conditions+1):
            n1 = np.sum(labels1 == c)
            n2 = np.sum(labels2 == c)
            min_trials_per_cond.append(min(n1, n2))

        new_data1, new_labels1 = [], []
        new_data2, new_labels2 = [], []

        for c in range(1, n_conditions+1):
            idx1 = np.where(labels1 == c)[0]
            idx2 = np.where(labels2 == c)[0]
            n_select = min_trials_per_cond[c-1]

            if n_select == 0:
                print("No trials for condition {} in {}/{} for {}, skip.".format(c, t1, t2, sub))
                continue

            sel1 = np.random.choice(idx1, n_select, replace=False)
            sel2 = np.random.choice(idx2, n_select, replace=False)

            new_data1.append(data1[sel1])
            new_labels1.append(labels1[sel1])
            new_data2.append(data2[sel2])
            new_labels2.append(labels2[sel2])

        if len(new_data1) == 0:
            continue

        data1_bal = np.concatenate(new_data1, axis=0)[np.newaxis, ...]
        labels1_bal = np.concatenate(new_labels1, axis=0)[np.newaxis, ...]
        data2_bal = np.concatenate(new_data2, axis=0)[np.newaxis, ...]
        labels2_bal = np.concatenate(new_labels2, axis=0)[np.newaxis, ...]

        try:
            print("Running bidirectional decoding: {} ↔ {}".format(t1, t2))
            acc1to2, acc2to1 = bidirectional_transfer_decoding(
                data1_bal, labels1_bal, data2_bal, labels2_bal,
                n=n_conditions, navg=navg, time_opt=time_opt,
                time_win=time_win, time_step=time_step, iter=iter_n,
                normalization=True, pca=True, pca_components=pca_components, smooth=smooth)

            save_fname = "{}_{}_to_{}_accs.h5".format(sub, t1.replace(' ', '_'), t2.replace(' ', '_'))
            save_fpath = os.path.join(save_dir, save_fname)

            with h5py.File(save_fpath, "w") as f:
                f.create_dataset("Con1toCon2", data=acc1to2)
                f.create_dataset("Con2toCon1", data=acc2to1)

            print(" Saved: {}".format(save_fname))

        except Exception as e:
            print("Decoding error: {}".format(e))
            continue

print("\n==============================================")
print("All subjects processing completed!")
print("All results saved in directory: {}".format(save_dir))
print("==============================================")
