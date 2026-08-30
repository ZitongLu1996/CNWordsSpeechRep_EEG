#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:46:24 2026

Modifications:
    1. resample to 250 Hz 
    2. set time_win = 5 and time_step = 5 
    3. change time_opt from "features" to "average"
    03,28,2026

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Temporal Decoding (CTC) for EEG data

"""

import os
import h5py
import numpy as np
import mne
from pathlib import Path
import sys
from neurora.decoding import ct_decoding_kfold
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.events import retain_valid_events
from scripts.utils.paths import resolve_paths

# ---------------------------
# Repository-relative path configuration
# ---------------------------
_paths = resolve_paths()
base_path = _paths["data_root"]
save_dir = _paths["output_root"] / "decoding" / "temporal_generalization"


os.makedirs(save_dir, exist_ok=True)  # Ensure that the output directory exists.

# ---------------------------
# Analysis configuration
# ---------------------------
sub_ids = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10",
           "S11","S12","S13","S14","S15","S16","S17","S18","S19","S20",
           "S21","S22","S23","S24","S25","S26","S27","S28","S29","S30"]

tasks = ["imagined speech", "overt speech", "silent speech"]

# Temporal-generalization decoding parameters
navg = 4
time_win = 5
time_step = 5
nfolds = 5
nrepeats = 10
smooth = True
time_opt="average"
# ---------------------------
# run decoding
# ---------------------------
total_tasks = len(tasks) * len(sub_ids)
completed_tasks = 0

for task in tasks:
    for sub in sub_ids:
        # Construct the participant input path.
        file_path = os.path.join(base_path, task, sub, "EEG", "{}.set".format(sub))
        
        # Verify that the input file exists.
        if not os.path.exists(file_path):
            print("{} - {}: file missing, skip.".format(sub, task))
            completed_tasks += 1
            print("Progress: {}/{} ({:.1f}%)".format(completed_tasks, total_tasks, completed_tasks/total_tasks*100))
            continue

        try:
            # Load the epochs and apply analysis-specific preprocessing.
            epochs = mne.read_epochs_eeglab(file_path)
            epochs, clean_labels = retain_valid_events(epochs)
            epochs.crop(tmin=-0.1, tmax=0.9)
            
            # Resample to 250 Hz.
            epochs.resample(250)
            
            data = epochs.get_data()[np.newaxis, ...]
            labels = np.asarray(clean_labels)[np.newaxis, ...]

            print("Running CT: {} - {}, data shape {}".format(sub, task, data.shape))
            
            # Compute temporal-generalization decoding.
            acc_ct = ct_decoding_kfold(
                data, labels, n=10,
                navg=navg, time_win=time_win, time_step=time_step, time_opt=time_opt,
                nfolds=nfolds, nrepeats=nrepeats, pca = False,
                normalization=True, smooth=smooth
            )

            # Save the result, replacing spaces in the mode name with underscores.
            save_file = os.path.join(save_dir, "{}_{}_ct.h5".format(sub, task.replace(' ','_')))
            with h5py.File(save_file, "w") as f:
                f.create_dataset("accs", data=acc_ct)
            
            completed_tasks += 1
            print("Saved: {}".format(save_file))
            print("Progress: {}/{} ({:.1f}%)".format(completed_tasks, total_tasks, completed_tasks/total_tasks*100))

        except Exception as e:
            print("CTC failed for {} - {}: {}".format(sub, task, str(e)))
            completed_tasks += 1
            print("Progress: {}/{} ({:.1f}%)".format(completed_tasks, total_tasks, completed_tasks/total_tasks*100))
            continue

print("\nAll CTC decoding completed! Results saved to: {}".format(save_dir))
