"""Corrected RSA-specific one-dimensional cluster sign-flip permutation tests."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label
from scipy.stats import t as t_dist,ttest_1samp

def _runs(mask):
    labeled,n=label(np.asarray(mask,bool))
    return [np.flatnonzero(labeled==i) for i in range(1,n+1)]

def _clusters(t_values,threshold,tail):
    """Extract sign-separated clusters with positive-valued t mass."""
    out=[]
    if tail in ("pos","two"):
        for idx in _runs(t_values>threshold): out.append(dict(indices=idx,sign="pos",mass=float(np.sum(t_values[idx]))))
    if tail in ("neg","two"):
        for idx in _runs(t_values < -threshold): out.append(dict(indices=idx,sign="neg",mass=float(np.sum(np.abs(t_values[idx])))))
    return sorted(out,key=lambda x:int(x["indices"][0]))

def cluster_permutation_1samp_v2(data,chance=0.0,n_perm=1000,p_cluster=0.05,tail="pos",seed=42,nan_policy="raise",return_null=True):
    """One-sample 1D maximum-cluster-mass test using subject-level sign flips.

    Two-sided positive and negative clusters are formed separately. Correction is
    across time within the supplied curve, not across separate curves.
    """
    x=np.asarray(data,dtype=float)
    if x.ndim!=2: raise ValueError(f"data must be subjects x times; got {x.shape}")
    if x.shape[0]<2 or x.shape[1]<1: raise ValueError("data needs >=2 subjects and >=1 time point")
    if tail not in {"pos","neg","two"}: raise ValueError("tail must be 'pos', 'neg', or 'two'")
    if not isinstance(n_perm,(int,np.integer)) or n_perm<=0: raise ValueError("n_perm must be a positive integer")
    if not 0<p_cluster<1: raise ValueError("p_cluster must be between 0 and 1")
    if nan_policy not in {"raise","omit"}: raise ValueError("nan_policy must be 'raise' or 'omit'")
    finite=np.isfinite(x)
    n_omitted=0
    if not finite.all():
        if nan_policy=="raise": raise ValueError("data contains NaN or infinite values")
        keep=finite.all(axis=1); n_omitted=int((~keep).sum()); x=x[keep]
        if x.shape[0]<2: raise ValueError("fewer than two complete subjects remain after nan_policy='omit'")
    n_subjects,n_times=x.shape; df=n_subjects-1
    threshold=float(t_dist.ppf(1-p_cluster/2 if tail=="two" else 1-p_cluster,df))
    t_obs=np.asarray(ttest_1samp(x,chance,axis=0).statistic,float)
    if not np.isfinite(t_obs).all(): raise ValueError("observed t statistics contain NaN/inf (possibly zero-variance data)")
    observed=_clusters(t_obs,threshold,tail); centered=x-chance; rng=np.random.default_rng(seed); h0=np.zeros(n_perm,float)
    for pi in range(n_perm):
        signs=rng.choice((-1.0,1.0),size=(n_subjects,1)); tp=np.asarray(ttest_1samp(centered*signs,0,axis=0).statistic,float)
        if not np.isfinite(tp).all(): raise ValueError("permutation t statistics contain NaN/inf")
        masses=[c["mass"] for c in _clusters(tp,threshold,tail)]; h0[pi]=max(masses,default=0.0)
    pos=np.zeros(n_times,bool); neg=np.zeros(n_times,bool); records=[]
    for c in observed:
        idx=c["indices"]; p=float((np.sum(h0>=c["mass"])+1)/(n_perm+1)); record=dict(indices=idx.copy(),start_index=int(idx[0]),end_index=int(idx[-1]),sign=c["sign"],mass=float(c["mass"]),p_value=p,signed_t_values=t_obs[idx].copy()); records.append(record)
        if p<p_cluster:
            (pos if c["sign"]=="pos" else neg)[idx]=True
    return dict(t_obs=t_obs,clusters=records,cluster_p_values=np.asarray([c["p_value"] for c in records]),cluster_masses=np.asarray([c["mass"] for c in records]),cluster_signs=np.asarray([c["sign"] for c in records],dtype=object),significant_mask_pos=pos,significant_mask_neg=neg,significant_mask_any=pos|neg,h0_max_cluster_mass=h0 if return_null else None,threshold=threshold,tail=tail,n_perm=int(n_perm),p_cluster=float(p_cluster),seed=seed,metadata=dict(n_subjects=n_subjects,n_times=n_times,df=df,chance=float(chance),nan_policy=nan_policy,n_subjects_omitted=n_omitted,cluster_definition="contiguous indices; positive and negative masks separate",cluster_mass="sum(t) for pos; sum(abs(t)) for neg",p_value_formula="(count(H0_max >= observed_mass)+1)/(n_perm+1)",correction_scope="within-curve time-wise maximum cluster mass"))
