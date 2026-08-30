#!/usr/bin/env python3
"""Build condition-mean, time-resolved neural RDMs for all planned datasets."""

import csv
import gc
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np

from utils.config import (
    BASE_DATA_PATH,
    EEG_TASKS,
    FIXED_EVENT_ID,
    LABEL_TO_WORD,
    RESULTS_DIR,
    RSA_LABEL_ORDER,
    RSA_WORD_ORDER,
    SEMG_TASKS,
    sfreq,
    sub_ids,
    time_step,
    time_win,
    tmax,
    tmin,
)
from utils.io_utils import load_eeglab_epochs_fixed_labels
from utils.neural_rdm_utils import (
    build_condition_mean_data,
    compute_neurora_eeg_rdm,
    make_rdm_time_axis,
    save_neural_rdm_h5,
)


EEG_OUTPUT_DIR = RESULTS_DIR / "neural_rdms" / "EEG"
SEMG_OUTPUT_DIR = RESULTS_DIR / "neural_rdms" / "sEMG"
SUMMARY_PATH = RESULTS_DIR / "neural_rdms" / "neural_rdm_all_subjects_summary.csv"
LOG_PATH = RESULTS_DIR / "logs" / "build_neural_rdms_all_subjects.txt"
SUMMARY_FIELDS = [
    "subject",
    "task",
    "signal_type",
    "status",
    "output_path",
    "n_trials",
    "n_channels",
    "n_times",
    "min_trials_per_label",
    "max_trials_per_label",
    "neurora_input_shape",
    "rdm_shape",
    "n_time_windows",
    "time_start",
    "time_end",
    "n_nan",
    "n_inf",
    "error_message",
]


def iter_datasets():
    """Yield exactly 90 EEG and 60 sEMG datasets in subject-major order."""
    for subject in sub_ids:
        for task in EEG_TASKS:
            yield subject, task, "EEG"
        for task in SEMG_TASKS:
            yield subject, task, "sEMG"


def source_path(subject, task, signal_type):
    """Return the configured EEGLAB input path."""
    return BASE_DATA_PATH / task / subject / signal_type / f"{subject}.set"


def output_path(subject, task, signal_type):
    """Return the required subject/task/signal HDF5 path."""
    output_dir = EEG_OUTPUT_DIR if signal_type == "EEG" else SEMG_OUTPUT_DIR
    task_slug = task.replace(" ", "_")
    return output_dir / f"{subject}_{task_slug}_{signal_type}_rdms.h5"


def empty_summary(subject, task, signal_type, destination):
    """Create a complete summary row populated with safe empty values."""
    return {
        "subject": subject,
        "task": task,
        "signal_type": signal_type,
        "status": "pending",
        "output_path": str(destination),
        "n_trials": "",
        "n_channels": "",
        "n_times": "",
        "min_trials_per_label": "",
        "max_trials_per_label": "",
        "neurora_input_shape": "",
        "rdm_shape": "",
        "n_time_windows": "",
        "time_start": "",
        "time_end": "",
        "n_nan": "",
        "n_inf": "",
        "error_message": "",
    }


def shape_text(shape):
    """Render a NumPy shape compactly for CSV output."""
    return "x".join(str(value) for value in shape)


def decode_h5_strings(values):
    """Decode an HDF5 string array to normal Python strings."""
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def verify_saved_h5(path, expected_rdms, expected_times):
    """Reopen and fully validate one saved neural-RDM HDF5 file."""
    required = {"rdms", "time_axis", "label_order", "word_order", "metadata_json"}
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != required:
            raise ValueError(f"HDF5 datasets mismatch: got {sorted(handle.keys())}.")
        rdms = handle["rdms"][:]
        times = handle["time_axis"][:]
        labels = handle["label_order"][:].tolist()
        words = decode_h5_strings(handle["word_order"][:])
        raw_metadata = handle["metadata_json"][()]
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(raw_metadata)
    if rdms.shape != expected_rdms.shape or not np.allclose(rdms, expected_rdms):
        raise ValueError("Reloaded HDF5 RDMs differ from computed RDMs.")
    if times.shape != expected_times.shape or not np.allclose(times, expected_times):
        raise ValueError("Reloaded HDF5 time_axis differs from computed time_axis.")
    if labels != RSA_LABEL_ORDER or words != RSA_WORD_ORDER:
        raise ValueError("Reloaded HDF5 label_order or word_order is incorrect.")
    if metadata.get("label_order") != RSA_LABEL_ORDER or metadata.get("word_order") != RSA_WORD_ORDER:
        raise ValueError("HDF5 metadata label_order or word_order is incorrect.")
    if not np.isfinite(rdms).all():
        raise ValueError("Reloaded HDF5 RDMs contain NaN or infinite values.")
    if not np.allclose(rdms, np.swapaxes(rdms, -1, -2), rtol=1e-7, atol=1e-8):
        raise ValueError("Reloaded HDF5 RDMs are not symmetric.")
    if not np.allclose(np.diagonal(rdms, axis1=-2, axis2=-1), 0.0, atol=1e-8):
        raise ValueError("Reloaded HDF5 RDM diagonals are not zero.")


def build_metadata(subject, task, signal_type, data, neurora_data, rdms, time_axis,
                   label_counts, load_info):
    """Build complete provenance metadata for one neural-RDM file."""
    return {
        "subject": subject,
        "task": task,
        "signal_type": signal_type,
        "source_file": str(source_path(subject, task, signal_type)),
        "original_data_shape": list(data.shape),
        "neurora_input_shape": list(neurora_data.shape),
        "rdm_shape": list(rdms.shape),
        "n_time_windows": int(rdms.shape[0]),
        "label_order": RSA_LABEL_ORDER,
        "word_order": RSA_WORD_ORDER,
        "label_to_word": LABEL_TO_WORD,
        "label_counts": {str(label): label_counts[label] for label in RSA_LABEL_ORDER},
        "min_trials_per_label": int(min(label_counts.values())),
        "max_trials_per_label": int(max(label_counts.values())),
        "trial_average_mode": "condition_mean_all_trials",
        "time_win": time_win,
        "time_step": time_step,
        "rdm_method": "correlation",
        "sfreq_hz": sfreq,
        "time_start": float(time_axis[0]),
        "time_end": float(time_axis[-1]),
        "n_nan": int(np.isnan(rdms).sum()),
        "n_inf": int(np.isinf(rdms).sum()),
        "removed_event_counts": load_info.get("removed_event_counts", {}),
        "n_trials_before_event_filter": load_info.get("n_trials_before"),
        "n_trials_after_event_filter": load_info.get("n_trials_after"),
        "original_event_id": load_info.get("original_event_id", {}),
        "trial_event_counts_before": load_info.get("trial_event_counts_before", {}),
    }


def process_dataset(subject, task, signal_type, log):
    """Build, atomically save, and verify one dataset without propagating failure."""
    source = source_path(subject, task, signal_type)
    destination = output_path(subject, task, signal_type)
    temporary = destination.with_suffix(".tmp.h5")
    row = empty_summary(subject, task, signal_type, destination)
    started = time.monotonic()
    log(f"\n[{subject} | {task} | {signal_type}]")
    log(f"source: {source}")
    try:
        if not source.is_file():
            raise FileNotFoundError(f"Required EEGLAB file does not exist: {source}")
        data, labels, times, load_info = load_eeglab_epochs_fixed_labels(
            source, tmin, tmax, sfreq, FIXED_EVENT_ID
        )
        unique_labels = sorted(int(value) for value in np.unique(labels))
        if unique_labels != RSA_LABEL_ORDER:
            raise ValueError(f"Expected all fixed labels 1-10; got {unique_labels}.")
        label_counts = {
            label: int(np.sum(labels == label)) for label in RSA_LABEL_ORDER
        }
        neurora_data, condition_metadata = build_condition_mean_data(
            data, labels, RSA_LABEL_ORDER
        )
        expected_input_shape = (10, 1, 1, data.shape[1], data.shape[2])
        if neurora_data.shape != expected_input_shape:
            raise ValueError(
                f"NeuroRA input shape {neurora_data.shape} != {expected_input_shape}."
            )
        rdms = compute_neurora_eeg_rdm(
            neurora_data,
            time_win=time_win,
            time_step=time_step,
            method="correlation",
        )
        time_axis = make_rdm_time_axis(times, time_win=time_win, time_step=time_step)
        expected_rdm_shape = (time_axis.size, 10, 10)
        if rdms.shape != expected_rdm_shape:
            raise ValueError(f"RDM shape {rdms.shape} != {expected_rdm_shape}.")
        if time_axis.size != rdms.shape[0]:
            raise ValueError("time_axis length does not match n_time_windows.")
        n_nan = int(np.isnan(rdms).sum())
        n_inf = int(np.isinf(rdms).sum())
        if n_nan or n_inf:
            raise ValueError(f"RDMs contain n_nan={n_nan}, n_inf={n_inf}.")
        if not np.allclose(rdms, np.swapaxes(rdms, -1, -2), rtol=1e-7, atol=1e-8):
            raise ValueError("RDMs are not symmetric.")
        if not np.allclose(np.diagonal(rdms, axis1=-2, axis2=-1), 0.0, atol=1e-8):
            raise ValueError("RDM diagonals are not zero.")
        metadata = build_metadata(
            subject, task, signal_type, data, neurora_data, rdms,
            time_axis, label_counts, load_info
        )
        metadata["condition_mean_metadata"] = condition_metadata
        destination.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        save_neural_rdm_h5(
            temporary, rdms, time_axis, RSA_LABEL_ORDER, RSA_WORD_ORDER, metadata
        )
        verify_saved_h5(temporary, rdms, time_axis)
        temporary.replace(destination)

        row.update({
            "status": "ok",
            "n_trials": int(data.shape[0]),
            "n_channels": int(data.shape[1]),
            "n_times": int(data.shape[2]),
            "min_trials_per_label": min(label_counts.values()),
            "max_trials_per_label": max(label_counts.values()),
            "neurora_input_shape": shape_text(neurora_data.shape),
            "rdm_shape": shape_text(rdms.shape),
            "n_time_windows": int(rdms.shape[0]),
            "time_start": f"{time_axis[0]:.9f}",
            "time_end": f"{time_axis[-1]:.9f}",
            "n_nan": n_nan,
            "n_inf": n_inf,
        })
        log(f"label_counts: {label_counts}")
        log(f"removed_event_counts: {load_info.get('removed_event_counts', {})}")
        log(f"data_shape: {data.shape}")
        log(f"neurora_input_shape: {neurora_data.shape}")
        log(f"rdm_shape: {rdms.shape}")
        log(f"time_axis: n={time_axis.size}, start={time_axis[0]:.9f}, end={time_axis[-1]:.9f}")
        log(f"n_nan={n_nan}, n_inf={n_inf}")
        log("symmetry_check: PASS; diagonal_check: PASS; h5_reload_check: PASS")
        log(f"output: {destination}")
        log(f"status: ok; elapsed_seconds={time.monotonic() - started:.3f}")
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        row.update({
            "status": "failed",
            "error_message": f"{type(exc).__name__}: {exc}",
        })
        log(f"status: failed; elapsed_seconds={time.monotonic() - started:.3f}")
        log(traceback.format_exc().rstrip())
    finally:
        gc.collect()
    return row


def write_summary(rows):
    """Write the current complete batch summary CSV."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run all 150 planned datasets while continuing after individual failures."""
    EEG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(LOG_PATH.parent / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(LOG_PATH.parent / ".matplotlib"))
    datasets = list(iter_datasets())
    if len(datasets) != 150:
        raise RuntimeError(f"Expected 150 planned datasets; got {len(datasets)}.")
    header = [
        "Step 6: all-subject neural RDM build",
        "=" * 80,
        f"subjects: {sub_ids[0]}-{sub_ids[-1]} (n={len(sub_ids)})",
        f"EEG tasks: {EEG_TASKS}",
        f"sEMG tasks: {SEMG_TASKS}",
        "imagined speech sEMG: excluded",
        "label path: MNE numeric code -> event name -> FIXED_EVENT_ID -> labels 1-10",
        f"label_order: {RSA_LABEL_ORDER}",
        f"word_order: {RSA_WORD_ORDER}",
        "trial_average_mode: condition_mean_all_trials",
        f"eegRDM: time_opt=1, time_win={time_win}, time_step={time_step}, method=correlation",
        f"planned datasets: {len(datasets)}",
    ]
    LOG_PATH.write_text("\n".join(header) + "\n", encoding="utf-8")

    def log(message):
        """Print and append one immediately flushed log line."""
        print(message, flush=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    rows = []
    batch_started = time.monotonic()
    for index, (subject, task, signal_type) in enumerate(datasets, start=1):
        log(f"\n--- dataset {index}/{len(datasets)} ---")
        rows.append(process_dataset(subject, task, signal_type, log))
        # Preserve a usable partial summary if a later process-level failure occurs.
        write_summary(rows)

    eeg_rows = [row for row in rows if row["signal_type"] == "EEG"]
    semg_rows = [row for row in rows if row["signal_type"] == "sEMG"]
    eeg_ok = sum(row["status"] == "ok" for row in eeg_rows)
    semg_ok = sum(row["status"] == "ok" for row in semg_rows)
    total_failed = sum(row["status"] != "ok" for row in rows)
    log("\n[batch summary]")
    log(f"EEG successful={eeg_ok}, failed={len(eeg_rows) - eeg_ok}")
    log(f"sEMG successful={semg_ok}, failed={len(semg_rows) - semg_ok}")
    log(f"total successful={len(rows) - total_failed}, failed={total_failed}")
    log(f"summary_csv: {SUMMARY_PATH}")
    log(f"elapsed_seconds={time.monotonic() - batch_started:.3f}")
    log("OVERALL: PASS" if total_failed == 0 else "OVERALL: COMPLETE_WITH_FAILURES")


if __name__ == "__main__":
    main()

