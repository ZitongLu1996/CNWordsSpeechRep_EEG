#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation-Based Speech Decoding Latency Analysis
==================================================
Core Functions:
  1. Load decoding accuracy & permutation null distribution from HDF5 files
  2. Automatically detect significant clusters & extract onset/peak latencies
  3. Compute descriptive statistics (mean ± SE) for latencies across tasks
  4. Perform repeated-measures ANOVA for task effects
  5. Run pairwise post-hoc tests with Bonferroni correction
  6. Generate publication-ready bar plots with individual lines & significance markers
"""
import os
import h5py
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# ==================== Configuration ====================
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.paths import resolve_paths
save_dir = resolve_paths()["output_root"] / "decoding" / "latency"
os.makedirs(save_dir, exist_ok=True)

sub_ids = [f"S{i:02d}" for i in range(1, 31)]
tasks = ["imagined_speech", "silent_speech", "overt_speech"]
task_labels = ["Imagined", "Silent", "Overt"]

task_colors = {
    "imagined_speech": "#4C72B0",
    "overt_speech": "#C44E52",
    "silent_speech": "#55A868"
}
colors = [task_colors[t] for t in tasks]


# ==================== Core Analysis Functions ====================
def find_clusters(sig, min_len=2):
    """Identify contiguous significant clusters of at least ``min_len`` samples."""
    clusters = []
    start = None
    for i, val in enumerate(sig):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                clusters.append((start, i))
            start = None
    if start is not None and len(sig) - start >= min_len:
        clusters.append((start, len(sig)))
    return clusters


def extract_onset_peak(acc_true, perm_accs, times, k=2):
    """Calculate the 95th-percentile permutation threshold and onset/peak latencies."""
    perm_95 = np.percentile(perm_accs, 95, axis=0)
    sig = acc_true >= perm_95
    sig[times < 0] = False  # Restrict the search to times at or after stimulus onset.
    clusters = find_clusters(sig, min_len=k)

    if len(clusters) == 0:
        return {
            "onset_time": np.nan, "onset_idx": -1,
            "peak_time": np.nan, "peak_idx": -1, "peak_acc": np.nan,
            "sig_mask": np.zeros_like(acc_true, dtype=bool)
        }

    # Onset is the first sample of the earliest significant cluster.
    onset_idx = clusters[0][0]
    onset_time = times[onset_idx]

    # Construct the complete significance mask.
    sig_mask = np.zeros_like(acc_true, dtype=bool)
    for s, e in clusters:
        sig_mask[s:e] = True

    # Peak is the maximum true accuracy within all significant clusters.
    valid_acc = acc_true.copy()
    valid_acc[~sig_mask] = np.nan
    if np.all(np.isnan(valid_acc)):
        return {
            "onset_time": onset_time, "onset_idx": onset_idx,
            "peak_time": np.nan, "peak_idx": -1, "peak_acc": np.nan,
            "sig_mask": sig_mask
        }

    peak_idx = np.nanargmax(valid_acc)
    peak_time = times[peak_idx]
    peak_acc = acc_true[peak_idx]

    return {
        "onset_time": onset_time, "onset_idx": onset_idx,
        "peak_time": peak_time, "peak_idx": peak_idx, "peak_acc": peak_acc,
        "sig_mask": sig_mask
    }


def run_anova(df, col):
    """Run repeated-measures ANOVA on participants complete in all three modes."""
    df_wide = df.pivot(index="sub", columns="task", values=col)[tasks].dropna()
    df_long = df_wide.reset_index().melt(
        id_vars="sub", value_vars=tasks, var_name="task", value_name=col
    )
    aov = AnovaRM(df_long, depvar=col, subject="sub", within=["task"]).fit()
    return aov, df_wide


def paired_ttests_bonferroni(df_wide):
    """Run post-hoc paired t tests with Bonferroni correction."""
    results = []
    for t1, t2 in itertools.combinations(tasks, 2):
        paired_data = df_wide[[t1, t2]].dropna()
        if len(paired_data) < 2:
            continue

        t_stat, p_val = stats.ttest_rel(paired_data[t1], paired_data[t2])
        results.append({
            "Condition1": t1,
            "Condition2": t2,
            "n": len(paired_data),
            "t": t_stat,
            "p": p_val
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        reject, pvals_corrected, _, _ = multipletests(
            df_res["p"].values, method='bonferroni'
        )
        df_res["p_bonferroni"] = pvals_corrected
    return df_res


# ==================== Data extraction and statistics ====================
print(">> Extracting onset and peak latencies for all participants...")
results = []
for task in tasks:
    for sub in sub_ids:
        fpath = os.path.join(save_dir, f"{sub}_{task}_accs.h5")
        if not os.path.exists(fpath):
            continue
        try:
            with h5py.File(fpath, "r") as f:
                acc_true = np.array(f["accs"]).squeeze()
                acc_perm = np.array(f["perm_accs"])
                times = np.array(f.attrs["times"])
            res = extract_onset_peak(acc_true, acc_perm, times)
            results.append({
                "sub": sub,
                "task": task,
                "onset_time": res["onset_time"],
                "peak_time": res["peak_time"]
            })
        except Exception as e:
            print(f"[Warning] Error while reading {fpath}: {e}")

df = pd.DataFrame(results)
csv_save_path = os.path.join(save_dir, "all_latency_results.csv")
df.to_csv(csv_save_path, index=False)
print(f">> Data extraction complete; results saved to: {csv_save_path}")

anova_onset, onset_wide = run_anova(df, "onset_time")
anova_peak, peak_wide = run_anova(df, "peak_time")

print("\n" + "=" * 50)
print("Onset Latency rm-ANOVA")
print("=" * 50)
print(anova_onset.summary())

print("\n" + "=" * 50)
print("Peak Latency rm-ANOVA")
print("=" * 50)
print(anova_peak.summary())

# Post-hoc pairwise comparisons
post_onset = paired_ttests_bonferroni(onset_wide)
post_peak = paired_ttests_bonferroni(peak_wide)
print("\nOnset Pairwise Comparisons (Bonferroni-corrected):\n", post_onset.round(4))
print("\nPeak Pairwise Comparisons (Bonferroni-corrected):\n", post_peak.round(4))


# ==================== Visualization functions ====================
def plot_latency_bar(ax, df_wide, label, post_hoc_df):
    means, ses = [], []
    for t in tasks:
        data = df_wide[t].values
        means.append(np.nanmean(data))
        ses.append(np.nanstd(data) / np.sqrt(np.sum(~np.isnan(data))))

    # 1. Bar plot
    ax.bar(
        task_labels, means,
        yerr=ses,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
        capsize=6,
        width=0.55,
        error_kw={"linewidth": 1.2}
    )

    # 2. Participant-level connecting lines
    x_pos = [0, 1, 2]
    for sub_idx in range(len(df_wide)):
        y_vals = df_wide.iloc[sub_idx].values
        if not np.any(np.isnan(y_vals)):
            ax.plot(
                x_pos, y_vals,
                color="gray",
                alpha=0.35,
                linewidth=0.8,
                marker="o",
                markersize=4,
                markeredgecolor="black",
                markerfacecolor="white",
                zorder=3
            )

    max_mean = max(means)

    # 3. Y-axis limits and tick settings
    # if "Onset" in label:
    #     ax.set_ylim(0, 1.0)
    #     y_base = max_mean + 0.35
    #     gap = 0.08
    #     ax.set_yticks(np.arange(0, 1.01, 0.2))
    # else:
    #     ax.set_ylim(0, 1.5)
    #     y_base = max_mean + 0.25
    #     gap = 0.12
    #     ax.set_yticks(np.arange(0, 1.51, 0.2))
    
    # Use consistent Y-axis limits and ticks across both panels.
    ax.set_ylim(0, 1.5)
    ax.set_yticks(np.arange(0, 1.6, 0.2))
    y_base = max_mean + 0.15
    gap = 0.08

    ax.set_ylabel("Latency (s)", fontsize=14, labelpad=8)
    ax.set_title(label, fontsize=16, pad=12, fontweight="bold")

    # 4. Significance annotations
    def add_sig(x1, x2, h, p):
        if np.isnan(p):
            return
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        ax.plot([x1, x2], [h, h], "k-", lw=1.2)
        text_y_offset = 0.015 if sig == "n.s." else - 0.01
        font_sz = 11 if sig == "n.s." else 14
        ax.text(
            (x1 + x2) / 2, h + text_y_offset, sig,
            ha="center", va="bottom",
            fontsize=font_sz, fontweight="bold"
        )

    # Map condition pairs to p values explicitly to avoid positional-index mismatches.
    p_map = {}
    for _, row in post_hoc_df.iterrows():
        pair = (row["Condition1"], row["Condition2"])
        p_map[pair] = row["p_bonferroni"]
        p_map[(pair[1], pair[0])] = row["p_bonferroni"]

    # Draw significance brackets.
    h1 = y_base + gap
    h2 = y_base + 2 * gap
    h3 = y_base + 3 * gap

    # Imagined (0) vs Overt (1)
    # Imagined (0) vs Silent (1)
    if ("imagined_speech", "silent_speech") in p_map:
        add_sig(0, 1, h1, p_map[("imagined_speech", "silent_speech")])

    # Silent (1) vs Overt (2)
    if ("silent_speech", "overt_speech") in p_map:
        add_sig(1, 2, h2, p_map[("silent_speech", "overt_speech")])

    # Place the widest Imagined-versus-Overt bracket at the highest level (h3).
    if ("imagined_speech", "overt_speech") in p_map:
        add_sig(0, 2, h3, p_map[("imagined_speech", "overt_speech")])

    # 5. Closed axes and border styling
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.tick_params(width=1.2, length=4, colors="black")
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)


# ==================== Plotting and export ====================
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

plot_latency_bar(ax1, onset_wide, "Onset Latency", post_onset)
plot_latency_bar(ax2, peak_wide, "Peak Latency", post_peak)

plt.tight_layout()

# Save before displaying to avoid clearing the figure canvas.
out_fig_path = os.path.join(save_dir, "latency_combined_final_bar_with_individual_lines.png")
fig.savefig(out_fig_path, dpi=300, bbox_inches="tight")
print(f">> High-resolution figure saved to: {out_fig_path}")

plt.show()
