#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:00:44 2026

Modifications:
    1. resample to 250 Hz 
    2. set time_win = 5 and time_step = 5 
    3. change time_opt from "features" to "average"

"""

import mne
import numpy as np
import os
import h5py
from pathlib import Path
import sys
from neurora.decoding import tbyt_decoding_kfold
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.events import retain_valid_events
from scripts.utils.paths import resolve_paths

# ---------------------------
# Core settings
# ---------------------------

sub_ids = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10",
           "S11","S12","S13","S14","S15","S16","S17","S18","S19","S20",
           "S21","S22","S23","S24","S25","S26","S27","S28","S29","S30"]

tasks = ["imagined speech", "overt speech", "silent speech"]

_paths = resolve_paths()
base_path = _paths["data_root"]
save_dir = _paths["output_root"] / "decoding" / "within_mode"
os.makedirs(save_dir, exist_ok=True)

# ---------------------------
# Participant-level decoding
# ---------------------------
for task in tasks:
    print(f"\n=== Processing task: {task} ===")


    for sub in sub_ids:
        file_path = os.path.join(base_path, task, sub, "EEG", f"{sub}.set")
        if not os.path.exists(file_path):
            print(f"{sub} file missing, skip.")
            continue

        # Load epochs.
        epochs = mne.read_epochs_eeglab(file_path)
        epochs, labels = retain_valid_events(epochs)
        epochs.crop(tmin=-0.1, tmax=0.9)  # Retain the -100 to 900 ms interval.
        
        # Resample to 250 Hz.
        epochs.resample(250)

        data = epochs.get_data()           # trials x channels x times
        labels = np.asarray(labels)         # cleaned B1(1)-B10(10) analysis labels
        

        # Record the number of trials.
        n_trials = data.shape[0]

        print(f"{sub}: data shape {data.shape}, labels shape {labels.shape}")

        # NeuroRA requires a participant dimension.
        data = data[np.newaxis, ...]       # 1 x trials x channels x times
        labels = labels[np.newaxis, ...]   # 1 x trials

        # ---------------------------
        # K-fold decoding
        # ---------------------------
        accs = tbyt_decoding_kfold(
            data,
            labels,
            n=10,                # 10 Chinese characters
            navg=4,              # trial averaging
            time_opt="average",
            time_win=5,
            time_step=5,
            nfolds=5,
            nrepeats=10,
            normalization=True,
            pca=False,
            smooth=True
        )

        print(f"{sub} decoding done, ACC shape: {accs.shape}")

        # ---------------------------
        # Save one result file per participant.
        # ---------------------------
        os.makedirs(save_dir, exist_ok=True)  # Ensure that the output directory exists.
        save_path = os.path.join(save_dir, f"{sub}_{task.replace(' ','_')}_accs.h5")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("accs", data=accs)

        print(f"Saved: {save_path}")
