"""Adapters from fixed-label trial data to validated NeuroRA neural RDMs."""

import json
from pathlib import Path
import warnings

import h5py
import numpy as np

from utils.config import LABEL_TO_WORD


def build_condition_mean_data(data, labels, label_order):
    """Average all trials per condition and build condition-first NeuroRA input.

    Parameters
    ----------
    data : array_like, shape [n_trials, n_channels, n_times]
        Fixed-label epoched data.
    labels : array_like, shape [n_trials]
        Corrected trigger labels, never raw MNE numeric event codes.
    label_order : sequence of int
        Required condition ordering.

    Returns
    -------
    neurora_data : ndarray, shape [n_cons, 1, 1, n_channels, n_times]
        One condition mean per condition, with singleton subject/trial axes.
    metadata : dict
        Trial counts, word ordering, and input shape.
    """
    array = np.asarray(data)
    trial_labels = np.asarray(labels)
    order = [int(label) for label in label_order]

    if array.ndim != 3:
        raise ValueError(
            "data must have shape trials x channels x times; "
            f"got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("data must contain only finite numeric values.")
    if trial_labels.ndim != 1 or trial_labels.size != array.shape[0]:
        raise ValueError(
            "labels must be one-dimensional and match the trial axis; "
            f"got labels {trial_labels.shape}, data {array.shape}."
        )
    if len(order) == 0 or len(order) != len(set(order)):
        raise ValueError("label_order must be non-empty and contain unique labels.")
    unknown = sorted(set(int(label) for label in np.unique(trial_labels)) - set(order))
    if unknown:
        raise ValueError(f"labels contain values outside label_order: {unknown}.")

    counts = {label: int(np.sum(trial_labels == label)) for label in order}
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"All requested labels must be present; missing labels: {missing}.")

    condition_means = np.stack(
        [array[trial_labels == label].mean(axis=0) for label in order]
    )
    neurora_data = condition_means[:, np.newaxis, np.newaxis, :, :]
    expected_shape = (len(order), 1, 1, array.shape[1], array.shape[2])
    if neurora_data.shape != expected_shape:
        raise RuntimeError(
            f"Internal NeuroRA-shape error: got {neurora_data.shape}, expected {expected_shape}."
        )

    metadata = {
        "aggregation": "condition_mean_all_trials",
        "trial_counts_by_label": {str(label): counts[label] for label in order},
        "label_order": order,
        "word_order": [LABEL_TO_WORD[label] for label in order],
        "n_trials": int(array.shape[0]),
        "n_channels": int(array.shape[1]),
        "n_times": int(array.shape[2]),
        "neurora_input_shape": list(neurora_data.shape),
    }
    return neurora_data, metadata


def compute_neurora_eeg_rdm(neurora_data, time_win=5, time_step=5, method="correlation"):
    """Compute and validate time-resolved neural RDMs with NeuroRA eegRDM."""
    from neurora.rdm_cal import eegRDM

    array = np.asarray(neurora_data)
    if array.ndim != 5:
        raise ValueError(
            "neurora_data must have shape n_cons x n_subs x n_trials x n_channels x n_times; "
            f"got {array.shape}."
        )
    if array.shape[1:3] != (1, 1):
        raise ValueError(
            "condition_mean input must have singleton subject and trial axes; "
            f"got {array.shape[1:3]}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("neurora_data must contain only finite numeric values.")
    if not isinstance(time_win, int) or not isinstance(time_step, int):
        raise TypeError("time_win and time_step must be integers in samples.")
    if time_win <= 0 or time_step <= 0 or time_win > array.shape[-1]:
        raise ValueError("time_win/time_step must be positive and time_win must fit n_times.")
    if method not in {"correlation", "amplitude"}:
        raise ValueError("method must be 'correlation' or 'amplitude'.")

    output = np.asarray(
        eegRDM(
            array,
            chl_opt=0,
            time_opt=1,
            time_win=time_win,
            time_step=time_step,
            method=method,
        )
    )
    n_cons = array.shape[0]
    n_windows = (array.shape[-1] - time_win) // time_step + 1
    expected_shape = (1, n_windows, n_cons, n_cons)
    if output.shape != expected_shape:
        raise ValueError(f"Unexpected eegRDM output shape {output.shape}; expected {expected_shape}.")
    rdms = output[0]
    if not np.isfinite(rdms).all():
        n_bad = int(np.size(rdms) - np.isfinite(rdms).sum())
        raise ValueError(f"NeuroRA RDM output contains {n_bad} NaN/inf values.")
    if not np.allclose(rdms, np.swapaxes(rdms, -1, -2), rtol=1e-7, atol=1e-8):
        raise ValueError("NeuroRA RDM output is not symmetric.")
    if not np.allclose(np.diagonal(rdms, axis1=-2, axis2=-1), 0.0, atol=1e-8):
        warnings.warn("NeuroRA RDM diagonal is not numerically zero.", RuntimeWarning, stacklevel=2)
    return rdms


def make_rdm_time_axis(times, time_win=5, time_step=5):
    """Return the mean sample time at the center of every NeuroRA window."""
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1 or sample_times.size == 0:
        raise ValueError(f"times must be a non-empty one-dimensional vector; got {sample_times.shape}.")
    if not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) <= 0):
        raise ValueError("times must contain finite, strictly increasing values.")
    if not isinstance(time_win, int) or not isinstance(time_step, int):
        raise TypeError("time_win and time_step must be integers in samples.")
    if time_win <= 0 or time_step <= 0 or time_win > sample_times.size:
        raise ValueError("time_win/time_step must be positive and time_win must fit times.")
    starts = range(0, sample_times.size - time_win + 1, time_step)
    return np.asarray([sample_times[start:start + time_win].mean() for start in starts])


def _json_ready(value):
    """Convert nested NumPy/Path values to JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_neural_rdm_h5(save_path, rdms, time_axis, label_order, word_order, metadata):
    """Save validated neural RDMs and provenance metadata in HDF5 format."""
    path = Path(save_path)
    rdm_array = np.asarray(rdms, dtype=float)
    axis = np.asarray(time_axis, dtype=float)
    labels = np.asarray(label_order, dtype=int)
    words = [str(word) for word in word_order]

    if rdm_array.ndim != 3 or rdm_array.shape[1] != rdm_array.shape[2]:
        raise ValueError(f"rdms must have shape n_windows x n_cons x n_cons; got {rdm_array.shape}.")
    if not np.isfinite(rdm_array).all():
        raise ValueError("rdms contain NaN or infinite values.")
    if axis.shape != (rdm_array.shape[0],) or not np.isfinite(axis).all():
        raise ValueError("time_axis must be finite and match the RDM window dimension.")
    if labels.shape != (rdm_array.shape[1],) or len(words) != rdm_array.shape[1]:
        raise ValueError("label_order and word_order must match the RDM condition dimension.")

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(_json_ready(metadata), ensure_ascii=False, sort_keys=True)
    utf8 = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("rdms", data=rdm_array)
        handle.create_dataset("time_axis", data=axis)
        handle.create_dataset("label_order", data=labels)
        handle.create_dataset("word_order", data=np.asarray(words, dtype=object), dtype=utf8)
        handle.create_dataset("metadata_json", data=metadata_json, dtype=utf8)

