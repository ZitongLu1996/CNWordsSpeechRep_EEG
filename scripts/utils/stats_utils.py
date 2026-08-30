#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:24:32 2026

"""
import numpy as np
from scipy.stats import ttest_1samp, t
from scipy.ndimage import label
from mne.stats import permutation_cluster_1samp_test
from decimal import Decimal
import matplotlib.pyplot as plt


 
def cluster_permutation_1samp(
    data,
    chance=0.0,
    n_perm=1000,
    p_cluster=0.05,
    tail="two",  # "two", "pos", "neg"
    seed=None,
    return_clusters=False
):
    """
    Cluster-based permutation test (one-sample, sign-flip)

    Parameters
    ----------
    data : array (n_subs, n_timepoints)
    chance : float
        chance level (e.g., 0.25 for decoding, 0 for difference)
    n_perm : int
        number of permutations
    p_cluster : float
        cluster-forming threshold (typically 0.05)
    tail : str
        "two" (recommended), "pos", or "neg"
    seed : int or None
    return_clusters : bool
        whether to return full cluster info

    Returns
    -------
    sig_idx : array
        significant time indices (FWE corrected, sorted ascending)
    clusters_info : list (optional)
        [{'indices': ..., 'mass': ..., 'pval': ..., 'sign': ...}, ...]
    """
    # 
   
    # Random-number generator
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)

    # Input validation
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_subs, n_timepoints)")
    if not isinstance(n_perm, int) or n_perm <= 0:
        raise ValueError("n_perm must be a positive integer")
    if not (0 < p_cluster < 1):
        raise ValueError("p_cluster must be between 0 and 1")
    if tail not in ("two", "pos", "neg"):
        raise ValueError("tail must be 'two', 'pos', or 'neg'")

    n_subs, n_tp = data.shape
    df = n_subs - 1

    # =========================
    # Extract clusters and calculate cluster mass.
    # =========================
    def get_clusters_and_masses(t_vals):
        if tail == "two":
            mask = np.abs(t_vals) > t_thresh
        elif tail == "pos":
            mask = t_vals > t_thresh
        else:
            mask = t_vals < -t_thresh

        labeled, n_clusters = label(mask)
        indices = []
        masses = []

        for c in range(1, n_clusters + 1):
            idx = np.where(labeled == c)[0]
            indices.append(idx)
            if tail == "two":
                mass = np.sum(np.abs(t_vals[idx]))
            else:
                mass = np.sum(t_vals[idx])
            masses.append(mass)

        return indices, masses

    # =========================
    # Observed t statistics
    # =========================
    t_obs, _ = ttest_1samp(data, chance, axis=0)

    # =========================
    # Cluster-forming threshold
    # =========================
    if tail == "two":
        t_thresh = t.ppf(1 - p_cluster / 2, df=df)
    else:
        t_thresh = t.ppf(1 - p_cluster, df=df)

    # =========================
    # Observed clusters
    # =========================
    clusters, cluster_masses = get_clusters_and_masses(t_obs)

    # =========================
    # Permutation distribution of the maximum cluster mass
    # =========================
    centered = data - chance
    max_masses = np.zeros(n_perm)

    for pi in range(n_perm):
        signs = rng.choice([1, -1], size=(n_subs, 1))
        perm_data = centered * signs
        t_perm, _ = ttest_1samp(perm_data, 0, axis=0)
        _, perm_masses = get_clusters_and_masses(t_perm)
        max_masses[pi] = max(perm_masses) if perm_masses else 0.0

    # =========================
    # Calculate corrected p values and significant indices.
    # =========================
    sig_idx = []
    clusters_info = []

    for c, m in zip(clusters, cluster_masses):
        pval = (np.sum(max_masses >= m) + 1) / (n_perm + 1)
        sign = np.sign(np.mean(t_obs[c]))

        if pval < p_cluster:
            sig_idx.extend(c)

        clusters_info.append({
            "indices": np.array(c),
            "mass": m,
            "pval": pval,
            "sign": sign  # 1 = significant positive cluster; -1 = significant negative cluster
        })

    sig_idx = np.unique(sig_idx)  # Return sorted unique indices.

    if return_clusters:
        return sig_idx, clusters_info
    else:
        return sig_idx
    
    
    
def summarize_clusters(
    clusters_info,
    t_vals,
    time_axis,
    alpha=0.05,
    effect_name="decoding accuracy",
    chance=0.25
):
    """
    Convert ``clusters_info`` into a manuscript-ready results description.

    Parameters
    ----------
    clusters_info : list
        Output from ``cluster_permutation_1samp(return_clusters=True)``.
    t_vals : array (n_timepoints,)
        Original t statistics used to locate the peak.
    time_axis : array
        Time axis in seconds.
    alpha : float
        Significance level.
    effect_name : str
        Name of the variable being described.
    chance : float
        Chance level used to describe above- or below-chance effects.

    Returns
    -------
    text : str
        English description suitable for a manuscript results section.
    """

    sig_clusters = [c for c in clusters_info if c["pval"] < alpha]

    if len(sig_clusters) == 0:
        return f"No significant clusters were found (cluster-based permutation test, p > {alpha})."

    sentences = []

    for c in sig_clusters:
        idx = c["indices"]
        pval = c["pval"]

        # onset / offset
        t_start = time_axis[idx[0]] * 1000
        t_end   = time_axis[idx[-1]] * 1000

        # Define the peak by the maximum absolute t statistic.
        peak_idx = idx[np.argmax(np.abs(t_vals[idx]))]
        t_peak = time_axis[peak_idx] * 1000

        # effect direction
        sign = c.get("sign", np.sign(np.mean(t_vals[idx])))

        if sign > 0:
            direction = "above chance"
        else:
            direction = "below chance"

        sentence = (
            f"A significant cluster was observed from {t_start:.0f} to {t_end:.0f} ms "
            f"(peak at {t_peak:.0f} ms, p = {pval:.3f}, {direction})."
        )

        sentences.append(sentence)

    return " ".join(sentences)


def plot_ct_diff_decoding_acc_2sided(
        acc1, acc2,
        start_timex=0, end_timex=1,
        start_timey=0, end_timey=1,
        time_intervalx=0.01, time_intervaly=0.01,
        p=0.05, cbpt=True, clusterp=0.05,
        stats_timex=[0, 1], stats_timey=[0, 1],
        xlim=[0, 1], ylim=[0, 1],
        clim=None,
        xlabel='Training Time (s)',
        ylabel='Test Time (s)',
        clabel='Decoding Difference (A1 - A2)',
        figsize=[6.4, 4.8],
        cmap="RdBu_r",
        ticksize=12,
        fontsize=16,
        title=None,
        title_fontsize=16):


    # =========================
    # difference
    # =========================
    acc = acc1 - acc2
    nsubs, nx, ny = acc.shape

    # =========================
    # time resolution check
    # =========================
    tstepx = float(Decimal((end_timex - start_timex) / nx).quantize(Decimal(str(time_intervalx))))
    tstepy = float(Decimal((end_timey - start_timey) / ny).quantize(Decimal(str(time_intervaly))))

    if tstepx != time_intervalx or tstepy != time_intervaly:
        return "Invalid input!"

    # =========================
    # index conversion (robust)
    # =========================
    stats_timex1 = int(np.round((stats_timex[0] - start_timex) / tstepx))
    stats_timex2 = int(np.round((stats_timex[1] - start_timex) / tstepx))
    stats_timey1 = int(np.round((stats_timey[0] - start_timey) / tstepy))
    stats_timey2 = int(np.round((stats_timey[1] - start_timey) / tstepy))

    # =========================
    # cluster-based permutation (STRICT 2-sided)
    # =========================
    ps = np.zeros((nx, ny))

    if cbpt:


        X = acc[:, stats_timex1:stats_timex2, stats_timey1:stats_timey2]

        T_obs, clusters, cluster_p, H0 = permutation_cluster_1samp_test(
            X,
            n_permutations=1000,
            threshold=None,
            tail=0,          # IMPORTANT: 2-sided
            out_type="mask",
            seed=42
        )

        ps_stats = np.zeros(X.shape[1:], dtype=int)

        for cl, pval in zip(clusters, cluster_p):
            if pval < clusterp:
                # Use the mean sign: positive clusters = 1, negative clusters = -1.
                mean_val = np.mean(X[:, cl])
                if mean_val > 0:
                    ps_stats[cl] = 1
                else:
                    ps_stats[cl] = -1

        ps[stats_timex1:stats_timex2, stats_timey1:stats_timey2] = ps_stats

    else:
        # =========================
        # STRICT pixel-level test (NO split p/2)
        # =========================
        for t1 in range(nx):
            for t2 in range(ny):

                if (stats_timex1 <= t1 < stats_timex2 and
                    stats_timey1 <= t2 < stats_timey2):

                    tval, pval = ttest_1samp(acc[:, t1, t2], 0)

                    if pval < p:
                        ps[t1, t2] = np.sign(tval)
                    else:
                        ps[t1, t2] = 0

    # =========================
    # plotting grid
    # =========================
    newps = np.zeros((nx + 2, ny + 2))
    newps[1:nx + 1, 1:ny + 1] = ps

    x = np.linspace(start_timex - 0.5 * tstepx, end_timex + 0.5 * tstepx, nx + 2)
    y = np.linspace(start_timey - 0.5 * tstepy, end_timey + 0.5 * tstepy, ny + 2)
    X, Y = np.meshgrid(x, y)

    # =========================
    # figure
    # =========================
    fig = plt.gcf()
    fig.set_size_inches(figsize)

    avg = np.mean(acc, axis=0).T

    # =========================
    # main map (CENTERED at 0)
    # =========================
    if clim is None:
        vmax = np.max(np.abs(avg))
        clim = [-vmax, vmax]

    im = plt.imshow(
        avg,
        extent=(start_timex, end_timex, start_timey, end_timey),
        cmap=cmap,
        origin="lower",
        vmin=clim[0],
        vmax=clim[1]
    )

    # =========================
    # significance overlay (clean)
    # =========================
    plt.contour(X, Y, np.transpose(newps), levels=[-0.5, 0.5], colors="silver", linewidths=3, linestyles="dashed")

    # =========================
    # colorbar
    # =========================
    cb = plt.colorbar(im)
    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(clabel, fontsize=ticksize + 2)

    # =========================
    # labels
    # =========================
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=title_fontsize)

    plt.show()

    return ps

