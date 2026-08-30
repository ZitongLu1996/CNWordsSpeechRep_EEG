#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-by-Time Decoding for EEG Data
Server Version


Modifications:
    1. resample to 250 Hz 
    2. set time_win = 5 and time_step = 5 
    3. change time_opt from "features" to "average"
    4. null distribution based on permutation >> latency

"""


import os
import h5py
import numpy as np
import mne
from neurora.decoding import tbyt_decoding_kfold
from joblib import Parallel, delayed
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.events import retain_valid_events
from scripts.utils.paths import resolve_paths
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------------------
_paths = resolve_paths()
project_dir = _paths["project_root"]
base_path = _paths["data_root"]
save_dir = _paths["output_root"] / "decoding" / "latency"
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Participants and speech modes
# ------------------------------------------------------------------------------
sub_ids = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10",
           "S11","S12","S13","S14","S15","S16","S17","S18","S19","S20",
           "S21","S22","S23","S24","S25","S26","S27","S28","S29","S30"]

tasks = ["imagined speech", "overt speech", "silent speech"]


# ------------------------------------------------------------------------------
N_PERM = 1000                #
N_JOBS = min(6, os.cpu_count())  # 
rng = np.random.default_rng(seed=42)  # Reproducible random-number generator.

# ------------------------------------------------------------------------------
# Decode one participant and speech mode.
# ------------------------------------------------------------------------------
def run_single_subject_task(sub, task):
    try:
        file_path = os.path.join(base_path, task, sub, "EEG", "{}.set".format(sub))
        if not os.path.exists(file_path):
            print("{} - {}: file missing, skip.".format(sub, task))
            return

        epochs = mne.read_epochs_eeglab(file_path)
        epochs, labels = retain_valid_events(epochs)
        epochs.crop(tmin=-0.1, tmax=0.9)
        epochs.resample(250)
        data = epochs.get_data()
        labels = np.asarray(labels)
        

        # Decode the original labels.
        data_1sub = data[np.newaxis, ...]
        labels_1sub = labels[np.newaxis, ...]
        accs = tbyt_decoding_kfold(
            data_1sub, labels_1sub,
            n=10, navg=4, time_opt="average",
            time_win=5, time_step=5,
            nfolds=5, nrepeats=10,
            normalization=True, pca=False, smooth=True
        )
        accs = accs.squeeze()
        print("{} - {}: original decoding done, ACC shape: {}".format(sub, task, accs.shape))

        # ------------------------------------------------------------------
        # Permutation Null Distribution
        # ------------------------------------------------------------------
        perm_accs = []
        
        for b in range(N_PERM):
        
            # Permute the labels.
            perm_labels = rng.permutation(labels)
        
            perm_data_1sub = data[np.newaxis, ...]
            perm_labels_1sub = perm_labels[np.newaxis, ...]
        
            perm_curve = tbyt_decoding_kfold(
                perm_data_1sub, perm_labels_1sub,
                n=10, navg=4, time_opt="average",
                time_win=5, time_step=5,
                nfolds=5, nrepeats=1,
                normalization=True, pca=False, smooth=True
            )
            
            if (b + 1) % 200 == 0:
                print(f"  > {sub}-{task}: {b+1}/{N_PERM} perms done.")
        
            perm_accs.append(perm_curve.squeeze())
        
        perm_accs = np.array(perm_accs)
        
        print("{} - {}: permutation done, shape: {}".format(sub, task, perm_accs.shape))

        # Save the true and permutation decoding results.
        save_name = "{}_{}_accs.h5".format(sub, task.replace(' ', '_'))
        save_path = os.path.join(save_dir, save_name)
        with h5py.File(save_path, "w") as f:
            f.create_dataset("accs", data=accs)
            f.create_dataset("perm_accs", data=perm_accs)
            f.attrs['times'] = epochs.times[::5]  # Store the analysis time axis.

        print("{} - {}: saved successfully.".format(sub, task))

    except Exception as e:
        print("{} - {}: error - {}".format(sub, task, str(e)))

# ------------------------------------------------------------------------------
# Run participant-mode jobs in parallel using the loky backend.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Server decoding start...")
    print("Project dir: {}".format(project_dir))
    print("Save dir: {}".format(save_dir))
    print("N_JOBS: {}".format(N_JOBS))
    print("N_PERM: {}".format(N_PERM))
    print("="*60)

    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_single_subject_task)(sub, task)
        for task in tasks
        for sub in sub_ids
    )

    print("\nAll done!")
