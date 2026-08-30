#!/usr/bin/env python3
"""Step 20E: paired task differences of saved Step 14B refined-92D curves."""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/rsa_results/step14b_EEG_partial_without_sEMG_control/modelset_v3_92d_refined/EEG_partial_rsa_without_sEMG_v3_92d_refined.h5"
OUT = ROOT / "results/rsa_results/step20e_task_state_paired_difference_without_sEMG_control_refined92d"
H5_OUT = OUT / "difference_curves/step20e_paired_difference_curves_without_sEMG_control_refined92d.h5"
STATS_CSV = OUT / "group_stats/step20e_paired_difference_cluster_stats.csv"
SUMMARY_CSV = OUT / "group_stats/step20e_paired_difference_summary.csv"
TIMEPOINT_CSV = OUT / "group_stats/step20e_paired_difference_timepoint_summary.csv"
LOG = OUT / "logs/step20e_task_state_paired_difference_without_sEMG_control_refined92d.txt"

TASKS = ["imagined speech", "silent speech", "overt speech"]
MODELS = ["semantic_tencent", "visual_alexnet_conv2", "phonology_distributed_feature_v3_92d_refined"]
COMPARISONS = [
    ("imagined speech", "silent speech", "imagined_minus_silent"),
    ("imagined speech", "overt speech", "imagined_minus_overt"),
    ("silent speech", "overt speech", "silent_minus_overt"),
]
INITIAL = {"imagined speech": "I", "silent speech": "S", "overt speech": "O"}
N_PERM, SEED, ALPHA = 1000, 20260823, 0.05


def decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_step14b():
    """Load only the saved refined-92D without-sEMG-control curves."""
    with h5py.File(SOURCE, "r") as h5:
        curves = np.asarray(h5["rsa_curves"][...], float)
        times = np.asarray(h5["time_axis"][...], float)
        tasks = decode(h5["tasks"][...])
        models = decode(h5["model_names"][...])
        subjects = decode(h5["subject_ids"][...])
        model_set = h5["model_set_name"][()]
        model_set = model_set.decode() if isinstance(model_set, bytes) else str(model_set)
    if curves.shape != (3, 30, 3, 50) or times.shape != (50,):
        raise ValueError(f"Unexpected Step 14B shape: curves={curves.shape}, times={times.shape}")
    if tasks != TASKS or models != MODELS or subjects != [f"S{i:02d}" for i in range(1, 31)]:
        raise ValueError("Step 14B task/model/subject order mismatch")
    if model_set != "modelset_v3_92d_refined":
        raise ValueError(f"Wrong model set: {model_set}")
    if not np.isfinite(curves).all():
        raise ValueError("NaN/inf in saved Step 14B curves")
    forbidden = {"sEMG_time_matched", "phonology_onehot_v1", "orthography_yang2009_inspired_270d"}
    if forbidden.intersection(models):
        raise ValueError(f"Forbidden model present: {forbidden.intersection(models)}")
    return curves, times, tasks, models, subjects, model_set


def cluster_indices(cluster):
    indices = cluster[0] if isinstance(cluster, tuple) else cluster
    return np.asarray(indices, dtype=int).ravel()


def direction_fields(task_a, task_b, mean_difference):
    if mean_difference > 0:
        return f"{INITIAL[task_a]} > {INITIAL[task_b]}", f"{task_a} stronger than {task_b}"
    return f"{INITIAL[task_b]} > {INITIAL[task_a]}", f"{task_b} stronger than {task_a}"


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
    local_packages = ROOT / "external_models/python_packages"
    if local_packages.is_dir() and str(local_packages) not in sys.path:
        sys.path.insert(0, str(local_packages))
    import mne
    from mne.stats import permutation_cluster_1samp_test

    for path in (H5_OUT.parent, STATS_CSV.parent, LOG.parent):
        path.mkdir(parents=True, exist_ok=True)
    log = [
        "Step 20E: EEG task-state paired differences without sEMG control",
        f"source={SOURCE}",
        "Step 14B partial RSA is read only and is not recomputed or modified.",
        f"mne_version={mne.__version__}",
        f"mne_path={mne.__file__}",
    ]
    try:
        curves, time_axis, tasks, models, subjects, model_set = load_step14b()
        threshold = float(t.ppf(1 - 0.05 / 2, len(subjects) - 1))
        results, cluster_rows, summary_rows, point_rows = [], [], [], []
        for model_index, model in enumerate(models):
            log.append(f"{model}: original_curve_shape={curves[:, :, model_index, :].shape}")
            for task_a, task_b, comparison in COMPARISONS:
                task_a_index, task_b_index = tasks.index(task_a), tasks.index(task_b)
                difference = curves[task_a_index, :, model_index, :] - curves[task_b_index, :, model_index, :]
                T_obs, clusters, p_values, H0 = permutation_cluster_1samp_test(
                    difference, threshold=threshold, tail=0, n_permutations=N_PERM,
                    out_type="indices", seed=SEED, verbose=False,
                )
                masks = np.zeros((len(clusters), len(time_axis)), dtype=np.uint8)
                significant_mask = np.zeros(len(time_axis), dtype=np.int8)
                directions, significant_windows = [], []
                for cluster_id, (cluster, p_value) in enumerate(zip(clusters, p_values)):
                    indices = cluster_indices(cluster)
                    masks[cluster_id, indices] = 1
                    mean_difference = float(difference[:, indices].mean())
                    peak_index = int(indices[np.argmax(np.abs(T_obs[indices]))])
                    significant = bool(p_value < ALPHA)
                    direction_label, interpretation = direction_fields(task_a, task_b, mean_difference)
                    if significant:
                        significant_mask[indices] = 1 if mean_difference > 0 else -1
                        directions.append(direction_label)
                        significant_windows.append(f"{time_axis[indices[0]]:.6f}-{time_axis[indices[-1]]:.6f}s")
                    cluster_rows.append({
                        "target_model": model, "comparison": comparison, "task_A": task_a, "task_B": task_b,
                        "difference_direction": "task_A_minus_task_B", "n_subjects": len(subjects), "n_times": len(time_axis),
                        "statistical_test": "mne.stats.permutation_cluster_1samp_test", "tail": "two-sided",
                        "cluster_forming_threshold": threshold, "n_permutations": N_PERM, "seed": SEED,
                        "n_clusters": len(clusters), "cluster_id": cluster_id,
                        "cluster_start_time": float(time_axis[indices[0]]), "cluster_end_time": float(time_axis[indices[-1]]),
                        "cluster_size_n_timepoints": len(indices), "cluster_p_value": float(p_value),
                        "cluster_sign": "positive" if mean_difference > 0 else "negative", "cluster_mean_diff": mean_difference,
                        "cluster_peak_diff": float(difference[:, peak_index].mean()), "peak_t": float(T_obs[peak_index]),
                        "peak_time": float(time_axis[peak_index]), "significant_cluster": significant,
                        "direction_label": direction_label, "direction_interpretation": interpretation,
                        "notes": "cluster-corrected within this model-specific task-pair curve across time",
                    })
                mean = difference.mean(axis=0)
                sem = difference.std(axis=0, ddof=1) / np.sqrt(len(subjects))
                peak = int(np.argmax(np.abs(mean)))
                summary_rows.append({
                    "target_model": model, "comparison": comparison, "task_A": task_a, "task_B": task_b,
                    "significant_difference": bool(significant_windows), "n_significant_clusters": len(significant_windows),
                    "significant_windows": "; ".join(significant_windows),
                    "direction_summary": "; ".join(directions) if directions else "no significant difference",
                    "peak_abs_diff": float(abs(mean[peak])), "peak_abs_diff_time": float(time_axis[peak]),
                    "min_cluster_p": float(np.min(p_values)) if len(p_values) else np.nan,
                    "summary_note": "two-sided MNE cluster correction within each curve across time; no global model/comparison correction",
                })
                for time_index, time_value in enumerate(time_axis):
                    memberships = [str(index) for index in range(len(clusters)) if masks[index, time_index]]
                    point_rows.append({
                        "target_model": model, "comparison": comparison, "time": float(time_value),
                        "mean_difference": float(mean[time_index]), "sem_difference": float(sem[time_index]),
                        "sd_difference": float(difference[:, time_index].std(ddof=1)), "t_obs": float(T_obs[time_index]),
                        "in_significant_cluster": bool(significant_mask[time_index]), "cluster_id_if_any": ";".join(memberships),
                    })
                results.append({"model": model, "comparison": comparison, "task_a": task_a, "task_b": task_b,
                                "difference": difference, "T_obs": T_obs, "p_values": np.asarray(p_values),
                                "H0": np.asarray(H0), "masks": masks, "significant_mask": significant_mask})
                log.append(f"{model}/{comparison}: difference_shape={difference.shape}, candidate_clusters={len(clusters)}, significant={len(significant_windows)}, windows={significant_windows or 'none'}, directions={directions or 'none'}")

        cluster_fields = ["target_model", "comparison", "task_A", "task_B", "difference_direction", "n_subjects", "n_times", "statistical_test", "tail", "cluster_forming_threshold", "n_permutations", "seed", "n_clusters", "cluster_id", "cluster_start_time", "cluster_end_time", "cluster_size_n_timepoints", "cluster_p_value", "cluster_sign", "cluster_mean_diff", "cluster_peak_diff", "peak_t", "peak_time", "significant_cluster", "direction_label", "direction_interpretation", "notes"]
        summary_fields = ["target_model", "comparison", "task_A", "task_B", "significant_difference", "n_significant_clusters", "significant_windows", "direction_summary", "peak_abs_diff", "peak_abs_diff_time", "min_cluster_p", "summary_note"]
        point_fields = ["target_model", "comparison", "time", "mean_difference", "sem_difference", "sd_difference", "t_obs", "in_significant_cluster", "cluster_id_if_any"]
        write_csv(STATS_CSV, cluster_rows, cluster_fields)
        write_csv(SUMMARY_CSV, summary_rows, summary_fields)
        write_csv(TIMEPOINT_CSV, point_rows, point_fields)

        text_dtype = h5py.string_dtype("utf-8")
        metadata = {
            "source_step": "Step 14B", "source_analysis": "EEG partial RSA without sEMG control",
            "model_set": "semantic + AlexNet conv2 + refined 92D phonetic features", "sEMG_control": False,
            "statistical_test": "mne.stats.permutation_cluster_1samp_test", "tail": "two-sided",
            "n_permutations": N_PERM, "seed": SEED, "cluster_forming_alpha_two_sided": 0.05,
            "cluster_corrected_alpha": 0.05,
            "correction_scope": "within each model-specific task-pair difference curve across time",
            "global_model_task_correction": False, "pchip_visualization_only": True,
        }
        with h5py.File(H5_OUT, "w") as h5:
            h5.create_dataset("difference_curves", data=np.stack([result["difference"] for result in results]))
            h5.create_dataset("original_curves", data=curves)
            h5.create_dataset("subject_ids", data=np.asarray(subjects, dtype=object), dtype=text_dtype)
            h5.create_dataset("model_names", data=np.asarray(models, dtype=object), dtype=text_dtype)
            h5.create_dataset("task_names", data=np.asarray(tasks, dtype=object), dtype=text_dtype)
            h5.create_dataset("task_comparisons", data=np.asarray([result["comparison"] for result in results], dtype=object), dtype=text_dtype)
            h5.create_dataset("task_A", data=np.asarray([result["task_a"] for result in results], dtype=object), dtype=text_dtype)
            h5.create_dataset("task_B", data=np.asarray([result["task_b"] for result in results], dtype=object), dtype=text_dtype)
            h5.create_dataset("target_models", data=np.asarray([result["model"] for result in results], dtype=object), dtype=text_dtype)
            h5.create_dataset("difference_definition", data="task_A_minus_task_B", dtype=text_dtype)
            h5.create_dataset("time_axis", data=time_axis)
            h5.create_dataset("T_obs", data=np.stack([result["T_obs"] for result in results]))
            h5.create_dataset("significant_cluster_masks", data=np.stack([result["significant_mask"] for result in results]))
            h5.create_dataset("metadata_json", data=json.dumps(metadata, ensure_ascii=False), dtype=text_dtype)
            cluster_group, p_group, h0_group = h5.create_group("clusters"), h5.create_group("cluster_p_values"), h5.create_group("H0")
            for index, result in enumerate(results):
                cluster_group.create_dataset(str(index), data=result["masks"])
                p_group.create_dataset(str(index), data=result["p_values"])
                h0_group.create_dataset(str(index), data=result["H0"])
        with h5py.File(H5_OUT, "r") as h5:
            assert h5["difference_curves"].shape == (9, 30, 50)
            assert h5["original_curves"].shape == (3, 30, 3, 50)
        log += [
            "Step 14B refined-92D curves loaded successfully: YES", "subject order consistent across tasks: YES",
            f"model_set={model_set}", "sEMG model/control included: NO", "Step 19 source read: NO",
            "MNE function=mne.stats.permutation_cluster_1samp_test", "tail=0 (two-sided)",
            f"threshold=t.ppf(1 - 0.05/2, df=29)={threshold}", f"n_permutations={N_PERM}", f"seed={SEED}",
            "correction_scope=within each model-specific task-pair difference curve across time",
            "global model/comparison correction=NO", "QC: 9 curves, shape=9x30x50, NaN/inf=0, pairing PASS",
            f"outputs={H5_OUT}, {STATS_CSV}, {SUMMARY_CSV}, {TIMEPOINT_CSV}", "OVERALL: PASS",
        ]
    except Exception:
        log += [traceback.format_exc(), "OVERALL: FAILED"]
        LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
        raise
    LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
