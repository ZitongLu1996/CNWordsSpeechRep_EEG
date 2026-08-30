"""I/O helpers that enforce stable EEGLAB trigger labels."""

from collections import Counter
from pathlib import Path

import numpy as np


def _string_counts(values):
    """Return deterministic, JSON-friendly counts."""
    return dict(sorted(Counter(str(value) for value in values).items()))


def load_eeglab_epochs_fixed_labels(file_path, tmin, tmax, sfreq, fixed_event_id):
    """Load EEGLAB epochs and map event names to fixed trigger labels.

    MNE numeric codes are used only to recover event names. Trials whose event
    names are not exact keys in ``fixed_event_id`` are removed from both data
    and labels and reported in the returned metadata.
    """
    import mne

    file_path = Path(file_path)
    epochs = mne.read_epochs_eeglab(file_path, verbose="ERROR")
    original_event_id = {str(k): int(v) for k, v in epochs.event_id.items()}
    event_names = {value: name for name, value in original_event_id.items()}

    numeric_codes = epochs.events[:, -1].copy()
    trial_names = [event_names.get(int(code), f"<unmapped_numeric_code:{code}>")
                   for code in numeric_codes]
    valid_mask = np.asarray([name in fixed_event_id for name in trial_names], dtype=bool)

    # Crop/resample once, then apply the same mask to data and labels.
    epochs.crop(tmin=tmin, tmax=tmax)
    epochs.resample(sfreq)
    all_data = epochs.get_data(copy=True)
    data = all_data[valid_mask]
    corrected_labels = np.asarray(
        [fixed_event_id[name] for name, keep in zip(trial_names, valid_mask) if keep],
        dtype=int,
    )

    removed_names = [name for name, keep in zip(trial_names, valid_mask) if not keep]
    info = {
        "original_event_id": original_event_id,
        "trial_event_counts_before": _string_counts(trial_names),
        "removed_event_counts": _string_counts(removed_names),
        "corrected_label_counts": _string_counts(corrected_labels.tolist()),
        "n_trials_before": int(len(trial_names)),
        "n_trials_after": int(len(corrected_labels)),
    }
    return data, corrected_labels, epochs.times.copy(), info

