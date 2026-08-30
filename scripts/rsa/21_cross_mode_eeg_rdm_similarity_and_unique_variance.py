#!/usr/bin/env python3
"""Step 21: cross-mode EEG geometry and information-specific unique variance."""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata, t

from utils.config import RESULTS_DIR, MODEL_RDM_DIR, RSA_LABEL_ORDER, RSA_WORD_ORDER, sub_ids
from utils.rdm_utils import partial_corr_vector, vectorize_rdm

ROOT = Path(__file__).resolve().parent
OUT = RESULTS_DIR / "rsa_results/step21_cross_mode_eeg_rdm_similarity_and_unique_variance"
SIM_H5 = OUT / "cross_mode_similarity/step21_cross_mode_eeg_rdm_similarity_all_pairs.h5"
UNIQUE_H5 = OUT / "unique_variance/step21_information_specific_unique_variance_without_sEMG_control.h5"
SIM_CSV = OUT / "group_stats/step21_cross_mode_similarity_cluster_stats.csv"
UNIQUE_CSV = OUT / "group_stats/step21_unique_variance_cluster_stats.csv"
SUMMARY_CSV = OUT / "group_stats/step21_summary.csv"
LOG = OUT / "logs/step21_cross_mode_eeg_rdm_similarity_and_unique_variance.txt"

TASKS = ["imagined speech", "silent speech", "overt speech"]
PAIRS = [("imagined speech", "silent speech", "imagined_silent"), ("imagined speech", "overt speech", "imagined_overt"), ("silent speech", "overt speech", "silent_overt")]
INFORMATION = ["visual", "semantic", "phonetic"]
MODEL_NAMES = {"visual": "visual_alexnet_conv2", "semantic": "semantic_tencent", "phonetic": "phonology_distributed_feature_v3_92d_refined"}
MODEL_PATHS = {
    "visual": MODEL_RDM_DIR / "visual_alexnet_conv2_rdm.h5",
    "semantic": MODEL_RDM_DIR / "semantic_tencent_rdm.h5",
    "phonetic": MODEL_RDM_DIR / "phonology_distributed_feature_v3_92d_refined_rdm.h5",
}
N_PERM, SEED, ALPHA = 1000, 20260823, 0.05


def decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_rdm(array, expected_shape, name):
    value = np.asarray(array, float)
    if value.shape != expected_shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: invalid shape/finite QC: {value.shape}")
    if expected_shape == (10, 10):
        matrices = value[np.newaxis]
    else:
        matrices = value
    if not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-8):
        raise ValueError(f"{name}: RDM is not symmetric")
    if not np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0, atol=1e-8):
        raise ValueError(f"{name}: RDM diagonal is not zero")
    return value


def load_neural(path):
    with h5py.File(path, "r") as h5:
        key = "rdms" if "rdms" in h5 else "neural_rdms"
        rdms = validate_rdm(h5[key][...], (50, 10, 10), str(path))
        times = np.asarray(h5["time_axis"][...], float)
        labels = [int(value) for value in h5["label_order"][...]]
        words = decode(h5["word_order"][...])
    if times.shape != (50,) or labels != list(RSA_LABEL_ORDER) or words != list(RSA_WORD_ORDER):
        raise ValueError(f"Neural metadata/order mismatch: {path}")
    return rdms, times


def load_all_neural():
    neural = np.empty((3, 30, 50, 10, 10), float)
    paths, reference_time = [], None
    for task_index, task in enumerate(TASKS):
        for subject_index, subject in enumerate(sub_ids):
            path = RESULTS_DIR / "neural_rdms/EEG" / f"{subject}_{task.replace(' ', '_')}_EEG_rdms.h5"
            rdms, times = load_neural(path)
            neural[task_index, subject_index] = rdms
            paths.append(str(path))
            if reference_time is None:
                reference_time = times
            elif not np.array_equal(times, reference_time):
                raise ValueError(f"Time-axis mismatch: {path}")
    return neural, reference_time, paths


def load_models():
    models, paths = {}, {}
    for information in INFORMATION:
        path = MODEL_PATHS[information]
        with h5py.File(path, "r") as h5:
            key = "rdm" if "rdm" in h5 else "rdms"
            rdm = validate_rdm(h5[key][...], (10, 10), str(path))
            labels = [int(value) for value in h5["label_order"][...]]
            words = decode(h5["word_order"][...])
        if labels != list(RSA_LABEL_ORDER) or words != list(RSA_WORD_ORDER):
            raise ValueError(f"Model metadata/order mismatch: {path}")
        models[information] = vectorize_rdm(rdm)
        paths[information] = str(path)
    return models, paths


def rank_rows(vectors):
    return rankdata(np.asarray(vectors, float), axis=1, method="average")


def residualize_rows(ranked_rows, control_vectors):
    controls = np.column_stack([rankdata(vector, method="average") for vector in control_vectors])
    design = np.column_stack([np.ones(ranked_rows.shape[1]), controls])
    coefficients = np.linalg.lstsq(design, ranked_rows.T, rcond=None)[0]
    return (ranked_rows.T - design @ coefficients).T


def cross_correlation_matrix(rows_a, rows_b):
    centered_a = rows_a - rows_a.mean(axis=1, keepdims=True)
    centered_b = rows_b - rows_b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered_a, axis=1)[:, None] * np.linalg.norm(centered_b, axis=1)[None, :]
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError("Near-constant residual vector in cross-correlation matrix")
    return centered_a @ centered_b.T / denominator


def compute_maps(neural, model_vectors):
    triangle_vectors = np.empty((3, 30, 50, 45), float)
    for task_index in range(3):
        for subject_index in range(30):
            triangle_vectors[task_index, subject_index] = np.stack([vectorize_rdm(rdm) for rdm in neural[task_index, subject_index]])
    ranked = rank_rows(triangle_vectors.reshape(-1, 45)).reshape(3, 30, 50, 45)
    similarity = np.empty((3, 30, 50, 50), float)
    unique = np.empty((3, 3, 30, 50, 50), float)
    r_without = np.empty_like(unique)
    r_with = np.empty_like(unique)
    validation_errors = []
    for pair_index, (task_a, task_b, _) in enumerate(PAIRS):
        task_a_index, task_b_index = TASKS.index(task_a), TASKS.index(task_b)
        for subject_index in range(30):
            rows_a, rows_b = ranked[task_a_index, subject_index], ranked[task_b_index, subject_index]
            similarity[pair_index, subject_index] = cross_correlation_matrix(rows_a, rows_b)
            for information_index, target in enumerate(INFORMATION):
                controls_without = [model_vectors[name] for name in INFORMATION if name != target]
                controls_with = controls_without + [model_vectors[target]]
                residual_a_without = residualize_rows(rows_a, controls_without)
                residual_b_without = residualize_rows(rows_b, controls_without)
                residual_a_with = residualize_rows(rows_a, controls_with)
                residual_b_with = residualize_rows(rows_b, controls_with)
                without_map = cross_correlation_matrix(residual_a_without, residual_b_without)
                with_map = cross_correlation_matrix(residual_a_with, residual_b_with)
                r_without[information_index, pair_index, subject_index] = without_map
                r_with[information_index, pair_index, subject_index] = with_map
                unique[information_index, pair_index, subject_index] = without_map ** 2 - with_map ** 2
                if subject_index == 0 and pair_index == 0:
                    i, j = information_index, information_index + 1
                    expected_without = partial_corr_vector(triangle_vectors[task_a_index, 0, i], triangle_vectors[task_b_index, 0, j], controls_without, method="spearman")["r"]
                    expected_with = partial_corr_vector(triangle_vectors[task_a_index, 0, i], triangle_vectors[task_b_index, 0, j], controls_with, method="spearman")["r"]
                    validation_errors.extend([abs(expected_without - without_map[i, j]), abs(expected_with - with_map[i, j])])
    if not np.isfinite(similarity).all() or not np.isfinite(unique).all():
        raise ValueError("NaN/inf in Step 21 maps")
    if max(validation_errors) > 1e-10:
        raise ValueError(f"Batched partial-Spearman mismatch: {max(validation_errors)}")
    return similarity, unique, r_without, r_with, max(validation_errors)


def normalize_cluster(cluster, shape):
    if isinstance(cluster, tuple):
        mask = np.zeros(shape, bool)
        mask[cluster] = True
        return mask
    mask = np.asarray(cluster, bool)
    return mask.reshape(shape)


def cluster_test(data, tail, threshold, permutation_cluster_1samp_test):
    T_obs, clusters, p_values, H0 = permutation_cluster_1samp_test(
        data, threshold=threshold, tail=tail, n_permutations=N_PERM,
        out_type="mask", seed=SEED, verbose=False,
    )
    masks = np.stack([normalize_cluster(cluster, T_obs.shape) for cluster in clusters]) if len(clusters) else np.zeros((0,) + T_obs.shape, bool)
    significant_mask = np.zeros(T_obs.shape, np.int8)
    for mask, p_value in zip(masks, p_values):
        if p_value < ALPHA:
            significant_mask[mask] = 1 if data[:, mask].mean() > 0 else -1
    return {"T_obs": T_obs, "clusters": masks, "p_values": np.asarray(p_values), "H0": np.asarray(H0), "significant_mask": significant_mask}


CLUSTER_FIELDS = ["analysis_type", "information_type", "task_pair", "task_A", "task_B", "n_subjects", "n_times_y", "n_times_x", "statistical_test", "tail", "cluster_forming_threshold", "n_permutations", "seed", "cluster_id", "cluster_p_value", "significant_cluster", "cluster_start_y_time", "cluster_end_y_time", "cluster_start_x_time", "cluster_end_x_time", "cluster_size_n_points", "cluster_mean_value", "cluster_peak_value", "peak_t", "peak_y_time", "peak_x_time", "cluster_sign", "direction_interpretation", "notes"]
SUMMARY_FIELDS = ["analysis_type", "information_type", "task_pair", "has_significant_cluster", "n_significant_clusters", "significant_windows_summary", "peak_value", "peak_y_time", "peak_x_time", "min_cluster_p", "summary_note"]


def rows_for_result(data, result, analysis_type, information, pair_index, threshold, tail_label):
    task_a, task_b, pair_name = PAIRS[pair_index]
    cluster_rows, windows = [], []
    for cluster_id, (mask, p_value) in enumerate(zip(result["clusters"], result["p_values"])):
        y_indices, x_indices = np.where(mask)
        mean_value = float(data[:, mask].mean())
        mean_map = data.mean(axis=0)
        local = np.abs(result["T_obs"][mask])
        local_peak = int(np.argmax(local))
        peak_y, peak_x = y_indices[local_peak], x_indices[local_peak]
        significant = bool(p_value < ALPHA)
        sign = "positive" if mean_value > 0 else "negative"
        if analysis_type == "cross_mode_similarity":
            interpretation = "positive cross-mode EEG representational alignment"
        elif mean_value > 0:
            interpretation = "target information uniquely explains cross-mode shared EEG geometry"
        else:
            interpretation = "controlling target information increases residual cross-mode EEG similarity, suggesting possible suppressor-like or variance-reallocation effect"
        if significant:
            windows.append(f"y={TIME_AXIS[y_indices.min()]:.6f}–{TIME_AXIS[y_indices.max()]:.6f}s, x={TIME_AXIS[x_indices.min()]:.6f}–{TIME_AXIS[x_indices.max()]:.6f}s, {sign}, p={p_value:.6g}")
        cluster_rows.append({
            "analysis_type": analysis_type, "information_type": information, "task_pair": pair_name,
            "task_A": task_a, "task_B": task_b, "n_subjects": data.shape[0], "n_times_y": data.shape[1], "n_times_x": data.shape[2],
            "statistical_test": "mne.stats.permutation_cluster_1samp_test", "tail": tail_label,
            "cluster_forming_threshold": threshold, "n_permutations": N_PERM, "seed": SEED,
            "cluster_id": cluster_id, "cluster_p_value": float(p_value), "significant_cluster": significant,
            "cluster_start_y_time": float(TIME_AXIS[y_indices.min()]), "cluster_end_y_time": float(TIME_AXIS[y_indices.max()]),
            "cluster_start_x_time": float(TIME_AXIS[x_indices.min()]), "cluster_end_x_time": float(TIME_AXIS[x_indices.max()]),
            "cluster_size_n_points": int(mask.sum()), "cluster_mean_value": mean_value,
            "cluster_peak_value": float(mean_map[peak_y, peak_x]), "peak_t": float(result["T_obs"][peak_y, peak_x]),
            "peak_y_time": float(TIME_AXIS[peak_y]), "peak_x_time": float(TIME_AXIS[peak_x]), "cluster_sign": sign,
            "direction_interpretation": interpretation, "notes": "cluster correction within this 2D map; no global task-pair/information correction",
        })
    mean_map = data.mean(axis=0)
    peak = np.unravel_index(np.argmax(np.abs(mean_map)), mean_map.shape)
    summary = {
        "analysis_type": analysis_type, "information_type": information, "task_pair": pair_name,
        "has_significant_cluster": bool(windows), "n_significant_clusters": len(windows),
        "significant_windows_summary": "; ".join(windows), "peak_value": float(mean_map[peak]),
        "peak_y_time": float(TIME_AXIS[peak[0]]), "peak_x_time": float(TIME_AXIS[peak[1]]),
        "min_cluster_p": float(np.min(result["p_values"])) if len(result["p_values"]) else np.nan,
        "summary_note": "correction within this 2D map; no global correction across task pairs or information types",
    }
    return cluster_rows, summary


def save_h5(path, main_name, main_values, stats, metadata, information_axis=False, extras=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset(main_name, data=main_values)
        for name, value in (extras or {}).items():
            h5.create_dataset(name, data=value)
        h5.create_dataset("subject_ids", data=np.asarray(sub_ids, dtype=object), dtype=text_dtype)
        if information_axis:
            h5.create_dataset("information_types", data=np.asarray(INFORMATION, dtype=object), dtype=text_dtype)
        h5.create_dataset("task_pairs", data=np.asarray([pair[2] for pair in PAIRS], dtype=object), dtype=text_dtype)
        h5.create_dataset("task_A", data=np.asarray([pair[0] for pair in PAIRS], dtype=object), dtype=text_dtype)
        h5.create_dataset("task_B", data=np.asarray([pair[1] for pair in PAIRS], dtype=object), dtype=text_dtype)
        h5.create_dataset("time_axis", data=TIME_AXIS)
        h5.create_dataset("metadata_json", data=json.dumps(metadata, ensure_ascii=False), dtype=text_dtype)
        h5.create_dataset("T_obs", data=np.stack([result["T_obs"] for result in stats]))
        h5.create_dataset("significant_cluster_masks", data=np.stack([result["significant_mask"] for result in stats]))
        clusters_group = h5.create_group("clusters")
        p_values_group = h5.create_group("cluster_p_values")
        h0_group = h5.create_group("H0")
        for index, result in enumerate(stats):
            clusters_group.create_dataset(str(index), data=result["clusters"])
            p_values_group.create_dataset(str(index), data=result["p_values"])
            h0_group.create_dataset(str(index), data=result["H0"])
            group = h5.create_group(f"statistics/{index}")
            group.create_dataset("T_obs", data=result["T_obs"])
            group.create_dataset("clusters", data=result["clusters"])
            group.create_dataset("cluster_p_values", data=result["p_values"])
            group.create_dataset("H0", data=result["H0"])
            group.create_dataset("significant_mask", data=result["significant_mask"])


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global TIME_AXIS
    os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
    local_packages = ROOT / "external_models/python_packages"
    if local_packages.is_dir() and str(local_packages) not in sys.path:
        sys.path.insert(0, str(local_packages))
    import mne
    from mne.stats import permutation_cluster_1samp_test

    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = ["Step 21 cross-mode EEG RDM similarity and unique variance without sEMG control"]
    try:
        neural, TIME_AXIS, neural_paths = load_all_neural()
        model_vectors, model_paths = load_models()
        similarity, unique, r_without, r_with, backend_error = compute_maps(neural, model_vectors)
        positive_threshold = float(t.ppf(1 - 0.05, len(sub_ids) - 1))
        two_sided_threshold = float(t.ppf(1 - 0.05 / 2, len(sub_ids) - 1))
        similarity_stats = [cluster_test(similarity[pair_index], 1, positive_threshold, permutation_cluster_1samp_test) for pair_index in range(3)]
        unique_stats = [cluster_test(unique[info_index, pair_index], 0, two_sided_threshold, permutation_cluster_1samp_test) for info_index in range(3) for pair_index in range(3)]

        similarity_rows, unique_rows, summaries = [], [], []
        for pair_index, result in enumerate(similarity_stats):
            rows, summary = rows_for_result(similarity[pair_index], result, "cross_mode_similarity", "EEG_RDM_similarity", pair_index, positive_threshold, "positive")
            similarity_rows += rows; summaries.append(summary)
        for info_index, information in enumerate(INFORMATION):
            for pair_index in range(3):
                result = unique_stats[info_index * 3 + pair_index]
                rows, summary = rows_for_result(unique[info_index, pair_index], result, "information_specific_unique_variance", information, pair_index, two_sided_threshold, "two-sided")
                unique_rows += rows; summaries.append(summary)
        write_csv(SIM_CSV, similarity_rows, CLUSTER_FIELDS)
        write_csv(UNIQUE_CSV, unique_rows, CLUSTER_FIELDS)
        write_csv(SUMMARY_CSV, summaries, SUMMARY_FIELDS)

        common_metadata = {
            "source_step": "Step 21", "based_on": "Step 13 cross-mode EEG RDM similarity",
            "eeg_rdm_source": str(RESULTS_DIR / "neural_rdms/EEG"), "model_rdm_source": model_paths,
            "sEMG_control": False, "information_types": INFORMATION,
            "n_permutations": N_PERM, "seed": SEED, "correction_scope": "within each 2D map", "global_correction": False,
        }
        similarity_metadata = {**common_metadata, "analysis": "EEG-RDM cross-temporal Spearman similarity", "cross_mode_similarity_stat": "MNE positive-tail 2D cluster permutation", "matrix_axis_0": "task_A_time_y", "matrix_axis_1": "task_B_time_x"}
        unique_metadata = {**common_metadata, "unique_variance_definition": "delta_R2 = r_without_target_squared - r_with_target_squared", "partial_spearman_backend": "upper triangle + average rank transform + least-squares residualization with intercept + residual Pearson", "unique_variance_stat": "MNE two-sided 2D cluster permutation", "matrix_axis_0": "task_A_time_y", "matrix_axis_1": "task_B_time_x"}
        save_h5(SIM_H5, "similarity_maps", similarity, similarity_stats, similarity_metadata)
        save_h5(UNIQUE_H5, "unique_variance_maps", unique, unique_stats, unique_metadata, information_axis=True, extras={"r_without_target": r_without, "r_with_target": r_with})

        log += [
            f"EEG_RDM_paths_count={len(neural_paths)}", *[f"EEG_RDM={path}" for path in neural_paths],
            *[f"model_RDM[{name}]={path}" for name, path in model_paths.items()],
            "subject_order_check=PASS (S01-S30 across all tasks)", "time_axis_check=PASS (identical saved 50-bin axes)",
            f"task_pair_order={[pair[2] for pair in PAIRS]}", f"model_order={[MODEL_NAMES[name] for name in INFORMATION]}",
            f"cross_mode_similarity_shape={similarity.shape}", f"unique_variance_shape={unique.shape}",
            f"r_without_target_shape={r_without.shape}", f"r_with_target_shape={r_with.shape}",
            f"partial_backend_validation_max_abs_error={backend_error:.3g}", f"mne_version={mne.__version__}", f"mne_path={mne.__file__}",
            f"Figure4A threshold={positive_threshold}; tail=1; seed={SEED}; n_permutations={N_PERM}",
            f"Figure4B-D threshold={two_sided_threshold}; tail=0; seed={SEED}; n_permutations={N_PERM}",
        ]
        for pair_index, result in enumerate(similarity_stats):
            log.append(f"similarity/{PAIRS[pair_index][2]}: candidate={len(result['p_values'])}, significant={int(np.sum(result['p_values'] < ALPHA))}")
        for info_index, information in enumerate(INFORMATION):
            for pair_index in range(3):
                result = unique_stats[info_index * 3 + pair_index]
                positive = int(np.any(result["significant_mask"] > 0))
                negative = int(np.any(result["significant_mask"] < 0))
                log.append(f"unique/{information}/{PAIRS[pair_index][2]}: candidate={len(result['p_values'])}, significant={int(np.sum(result['p_values'] < ALPHA))}, positive_cluster_present={positive}, negative_cluster_present={negative}")
        log += ["sEMG control/model used=NO", "Step 13/19/20 results modified=NO", f"outputs={SIM_H5}; {UNIQUE_H5}; {SIM_CSV}; {UNIQUE_CSV}; {SUMMARY_CSV}", "OVERALL: PASS"]
    except Exception:
        log += [traceback.format_exc(), "OVERALL: FAILED"]
        LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
        raise
    LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log[-40:]))


if __name__ == "__main__":
    main()
