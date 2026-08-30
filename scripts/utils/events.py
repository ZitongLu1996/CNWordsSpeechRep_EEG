"""Strict cleaning of the ten experimental event labels."""
from __future__ import annotations

import logging

VALID_EVENT_ID = {f"B{i}({i})": i for i in range(1, 11)}


def retain_valid_events(epochs):
    """Return epochs containing only exact B1(1)-B10(10) labels; log exclusions."""
    present = set(epochs.event_id)
    unexpected = sorted(present - set(VALID_EVENT_ID))
    if unexpected:
        logging.warning("Excluding unexpected/composite event labels: %s", unexpected)
    available = {name: code for name, code in VALID_EVENT_ID.items() if name in present}
    missing = sorted(set(VALID_EVENT_ID) - present)
    if missing:
        raise ValueError(f"Missing required event labels: {missing}")
    cleaned = epochs[list(available)].copy()
    # MNE import codes can differ; map exact annotation names back to analysis labels.
    code_to_label = {cleaned.event_id[name]: VALID_EVENT_ID[name] for name in available}
    labels = [code_to_label[int(code)] for code in cleaned.events[:, -1]]
    return cleaned, labels
