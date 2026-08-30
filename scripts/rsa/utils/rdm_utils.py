"""Validated RDM utilities and a transparent partial-correlation backend."""

from pathlib import Path
import warnings

import numpy as np
from scipy.stats import rankdata


_VAR_TOL = 1e-12


def validate_square_rdm(rdm, name="rdm"):
    """Validate an RDM and return it as a floating-point NumPy array."""
    array = np.asarray(rdm)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a 2D square matrix; got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values; got dtype {array.dtype}.")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.allclose(np.diag(array), 0.0, atol=1e-8):
        warnings.warn(f"{name} has a non-zero diagonal.", RuntimeWarning, stacklevel=2)
    if not np.allclose(array, array.T, rtol=1e-7, atol=1e-8):
        warnings.warn(f"{name} is not symmetric.", RuntimeWarning, stacklevel=2)
    return array


def vectorize_rdm(rdm, check=True):
    """Return the upper triangle of a square RDM, excluding its diagonal."""
    array = validate_square_rdm(rdm) if check else np.asarray(rdm, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"rdm must be square; got shape {array.shape}.")
    return array[np.triu_indices(array.shape[0], k=1)]


def stack_model_rdms(model_rdms, model_order=None):
    """Validate and stack a model-RDM dictionary in a reproducible order."""
    if not isinstance(model_rdms, dict) or not model_rdms:
        raise ValueError("model_rdms must be a non-empty dictionary.")
    order = list(model_rdms) if model_order is None else list(model_order)
    if len(order) != len(set(order)):
        raise ValueError("model_order contains duplicate names.")
    missing = [name for name in order if name not in model_rdms]
    extra = [name for name in model_rdms if name not in order]
    if missing or extra:
        raise ValueError(f"model_order mismatch; missing={missing}, extra={extra}.")
    arrays = [validate_square_rdm(model_rdms[name], name=name) for name in order]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All model RDMs must have the same shape; got {sorted(shapes)}.")
    return np.stack(arrays), order


def rank_transform_vector(x):
    """Rank-transform a finite one-dimensional numeric vector."""
    vector = _validate_vector(x, "x")
    return rankdata(vector, method="average").astype(float)


def _validate_vector(x, name):
    """Return a finite numeric one-dimensional vector."""
    array = np.asarray(x)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric; got dtype {array.dtype}.")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def residualize(y, X=None):
    """Regress controls and an intercept from a vector using least squares."""
    vector = _validate_vector(y, "y")
    if np.var(vector) <= _VAR_TOL:
        warnings.warn("y has near-zero variance.", RuntimeWarning, stacklevel=2)
    if X is None:
        return vector - vector.mean()
    controls = np.asarray(X)
    if controls.size == 0:
        return vector - vector.mean()
    if controls.ndim == 1:
        controls = controls[:, np.newaxis]
    if controls.ndim != 2 or controls.shape[0] != vector.size:
        raise ValueError(
            "X must have shape n_pairs x n_controls; "
            f"got {controls.shape} for y length {vector.size}."
        )
    if not np.issubdtype(controls.dtype, np.number):
        raise TypeError(f"X must be numeric; got dtype {controls.dtype}.")
    controls = controls.astype(float, copy=False)
    if not np.isfinite(controls).all():
        raise ValueError("X contains NaN or infinite values.")
    design = np.column_stack([np.ones(vector.size), controls])
    coefficients = np.linalg.lstsq(design, vector, rcond=None)[0]
    return vector - design @ coefficients


def pearson_corr_safe(a, b):
    """Return Pearson r, or NaN with a warning for a constant vector."""
    first = _validate_vector(a, "a")
    second = _validate_vector(b, "b")
    if first.size != second.size:
        raise ValueError(f"a and b must have equal length; got {first.size} and {second.size}.")
    if np.var(first) <= _VAR_TOL or np.var(second) <= _VAR_TOL:
        warnings.warn("Pearson correlation is undefined for near-constant input.", RuntimeWarning, stacklevel=2)
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _controls_to_matrix(controls, n_pairs):
    """Normalize controls supplied as a list or array to column orientation."""
    if controls is None:
        return None
    if isinstance(controls, (list, tuple)):
        if not controls:
            return None
        vectors = [_validate_vector(item, f"controls[{i}]") for i, item in enumerate(controls)]
        if any(vector.size != n_pairs for vector in vectors):
            raise ValueError("Every control vector must have the same length as y and x.")
        return np.column_stack(vectors)
    matrix = np.asarray(controls)
    if matrix.size == 0:
        return None
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2 or matrix.shape[0] != n_pairs:
        raise ValueError(
            "controls must have shape n_pairs x n_controls; "
            f"got {matrix.shape} for n_pairs={n_pairs}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"controls must be numeric; got dtype {matrix.dtype}.")
    matrix = matrix.astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("controls contain NaN or infinite values.")
    return matrix


def partial_corr_vector(y, x, controls=None, method="spearman"):
    """Compute transparent partial Pearson or partial Spearman correlation."""
    neural = _validate_vector(y, "y")
    target = _validate_vector(x, "x")
    if neural.size != target.size:
        raise ValueError(f"y and x must have equal length; got {neural.size} and {target.size}.")
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'.")
    control_matrix = _controls_to_matrix(controls, neural.size)
    n_controls = 0 if control_matrix is None else control_matrix.shape[1]
    if neural.size <= n_controls + 2:
        raise ValueError("Not enough observations for the requested number of controls.")

    if method == "spearman":
        neural = rank_transform_vector(neural)
        target = rank_transform_vector(target)
        if control_matrix is not None:
            control_matrix = np.column_stack(
                [rank_transform_vector(control_matrix[:, i]) for i in range(n_controls)]
            )

    neural_residual = residualize(neural, control_matrix)
    target_residual = residualize(target, control_matrix)
    r = pearson_corr_safe(neural_residual, target_residual)
    return {
        "r": r,
        "method": method,
        "n_pairs": int(neural.size),
        "n_controls": int(n_controls),
    }


def partial_corr_rdm(neural_rdm, target_rdm, control_rdms=None, method="spearman"):
    """Compute partial correlation between RDM upper-triangle vectors."""
    neural = validate_square_rdm(neural_rdm, "neural_rdm")
    target = validate_square_rdm(target_rdm, "target_rdm")
    if neural.shape != target.shape:
        raise ValueError(f"neural_rdm and target_rdm shapes differ: {neural.shape} vs {target.shape}.")
    controls = []
    for i, control in enumerate(control_rdms or []):
        array = validate_square_rdm(control, f"control_rdms[{i}]")
        if array.shape != neural.shape:
            raise ValueError(f"control_rdms[{i}] has shape {array.shape}; expected {neural.shape}.")
        controls.append(vectorize_rdm(array, check=False))
    return partial_corr_vector(
        vectorize_rdm(neural, check=False),
        vectorize_rdm(target, check=False),
        controls=controls,
        method=method,
    )


def _resolve_controls(model_order, controls_by_model):
    """Resolve and validate target-specific model-control names."""
    if controls_by_model is None:
        return {name: [other for other in model_order if other != name] for name in model_order}
    unknown_targets = [name for name in controls_by_model if name not in model_order]
    if unknown_targets:
        raise ValueError(f"controls_by_model has unknown targets: {unknown_targets}.")
    resolved = {}
    for name in model_order:
        controls = list(controls_by_model.get(name, []))
        unknown = [item for item in controls if item not in model_order]
        if unknown:
            raise ValueError(f"Controls for {name} contain unknown models: {unknown}.")
        if name in controls:
            raise ValueError(f"Target model {name} cannot control for itself.")
        if len(controls) != len(set(controls)):
            raise ValueError(f"Controls for {name} contain duplicates.")
        resolved[name] = controls
    return resolved


def compute_model_partial_rsas(neural_rdm, model_rdms, controls_by_model=None, method="spearman"):
    """Compute one partial-RSA value per model, controlling specified models."""
    neural = validate_square_rdm(neural_rdm, "neural_rdm")
    _, order = stack_model_rdms(model_rdms)
    if any(np.asarray(model_rdms[name]).shape != neural.shape for name in order):
        raise ValueError("Every model RDM must have the same shape as neural_rdm.")
    control_names = _resolve_controls(order, controls_by_model)
    results = {}
    for name in order:
        controls = control_names[name]
        result = partial_corr_rdm(
            neural,
            model_rdms[name],
            [model_rdms[item] for item in controls],
            method=method,
        )
        result["controls"] = controls
        results[name] = result
    return results


def compute_temporal_model_partial_rsas(eeg_rdms, model_rdms, controls_by_model=None, method="spearman"):
    """Compute model-wise partial RSA for a time series of neural RDMs."""
    neural = np.asarray(eeg_rdms)
    if neural.ndim != 3 or neural.shape[1] != neural.shape[2]:
        raise ValueError(
            "eeg_rdms must have shape n_times x n_cons x n_cons; "
            f"got {neural.shape}."
        )
    if not np.issubdtype(neural.dtype, np.number) or not np.isfinite(neural).all():
        raise ValueError("eeg_rdms must contain only finite numeric values.")
    _, order = stack_model_rdms(model_rdms)
    controls = _resolve_controls(order, controls_by_model)
    rsa_values = np.empty((neural.shape[0], len(order)), dtype=float)
    for time_index, neural_rdm in enumerate(neural):
        results = compute_model_partial_rsas(neural_rdm, model_rdms, controls, method)
        rsa_values[time_index] = [results[name]["r"] for name in order]
    n_cons = neural.shape[-1]
    metadata = {
        "method": method,
        "n_times": int(neural.shape[0]),
        "n_models": int(len(order)),
        "model_order": order,
        "n_cons": int(n_cons),
        "n_pairs": int(n_cons * (n_cons - 1) // 2),
        "controls_by_model": controls,
    }
    return rsa_values, order, metadata


def save_rdm_heatmap(rdm, labels, save_path, title=None):
    """Save an RDM heatmap with explicit condition labels using Matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    array = validate_square_rdm(rdm)
    labels = list(labels)
    if len(labels) != array.shape[0]:
        raise ValueError(f"Expected {array.shape[0]} labels; got {len(labels)}.")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap="viridis", interpolation="nearest")
    ticks = np.arange(array.shape[0])
    cjk_candidates = (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    )
    cjk_path = next((path for path in cjk_candidates if path.is_file()), None)
    cjk_font = FontProperties(fname=str(cjk_path)) if cjk_path else None
    axis.set_xticks(ticks, labels=labels, rotation=45, ha="right", fontproperties=cjk_font)
    axis.set_yticks(ticks, labels=labels, fontproperties=cjk_font)
    if title:
        axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Dissimilarity")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
