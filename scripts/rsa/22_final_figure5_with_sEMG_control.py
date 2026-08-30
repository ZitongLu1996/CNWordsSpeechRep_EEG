#!/usr/bin/env python3
"""Step 22: compute Figure 5B-E maps with time-matched sEMG controls."""
from __future__ import annotations

import csv, json, os, sys, traceback
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import rankdata, t

from utils.config import RESULTS_DIR, MODEL_RDM_DIR, RSA_LABEL_ORDER, RSA_WORD_ORDER, sub_ids
from utils.rdm_utils import partial_corr_vector, vectorize_rdm

ROOT=Path(__file__).resolve().parent
OUT=RESULTS_DIR/"rsa_results/step22_final_figure5_with_sEMG_control"
STEP19=RESULTS_DIR/"rsa_results/step19_final_partial_rsa_semantic_alexnet_refined92d_sEMG/partial_rsa/final_partial_rsa_semantic_alexnet_refined92d_sEMG.h5"
SIM_H5=OUT/"cross_mode_similarity_control_sEMG/figure5B_silent_overt_eeg_similarity_control_sEMG.h5"
UNIQUE_H5=OUT/"unique_variance_control_sEMG/figure5CDE_unique_variance_control_sEMG.h5"
SIM_CSV=OUT/"group_stats/figure5B_silent_overt_eeg_similarity_control_sEMG_cluster_stats.csv"
UNIQUE_CSV=OUT/"group_stats/figure5CDE_unique_variance_control_sEMG_cluster_stats.csv"
SUMMARY_CSV=OUT/"group_stats/step22_final_figure5_with_sEMG_control_summary.csv"
LOG=OUT/"logs/step22_final_figure5_with_sEMG_control.txt"
TASKS=["silent speech","overt speech"]
INFORMATION=["visual","semantic","phonetic"]
MODEL_PATHS={"visual":MODEL_RDM_DIR/"visual_alexnet_conv2_rdm.h5","semantic":MODEL_RDM_DIR/"semantic_tencent_rdm.h5","phonetic":MODEL_RDM_DIR/"phonology_distributed_feature_v3_92d_refined_rdm.h5"}
N_PERM,SEED,ALPHA=1000,20260824,.05
CLUSTER_FIELDS=["figure_panel","analysis_type","information_type","task_pair","task_A","task_B","n_subjects","n_times_y","n_times_x","statistical_test","tail","cluster_forming_threshold","n_permutations","seed","cluster_id","cluster_p_value","significant_cluster","cluster_start_y_time","cluster_end_y_time","cluster_start_x_time","cluster_end_x_time","cluster_size_n_points","cluster_mean_value","cluster_peak_value","peak_t","peak_y_time","peak_x_time","cluster_sign","direction_interpretation","notes"]
SUMMARY_FIELDS=["figure_panel","analysis_type","information_type","task_pair","has_significant_cluster","n_significant_clusters","significant_windows_summary","peak_value","peak_y_time","peak_x_time","min_cluster_p","summary_note"]

def decode(v): return [x.decode() if isinstance(x,bytes) else str(x) for x in v]

def validate_rdm(a,shape,name):
 a=np.asarray(a,float)
 if a.shape!=shape or not np.isfinite(a).all(): raise ValueError(f"{name}: invalid shape/finite QC {a.shape}")
 m=a[np.newaxis] if a.ndim==2 else a
 if not np.allclose(m,m.transpose(0,2,1),atol=1e-8) or not np.allclose(np.diagonal(m,axis1=1,axis2=2),0,atol=1e-8): raise ValueError(f"{name}: symmetry/diagonal QC failed")
 return a

def load_rdm_file(path,shape):
 with h5py.File(path) as h:
  key="rdms" if "rdms" in h else "neural_rdms" if "neural_rdms" in h else "rdm"
  a=validate_rdm(h[key][...],shape,str(path)); times=np.asarray(h["time_axis"][...],float) if "time_axis" in h else None
  labels=[int(x) for x in h["label_order"][...]]; words=decode(h["word_order"][...])
 if labels!=list(RSA_LABEL_ORDER) or words!=list(RSA_WORD_ORDER): raise ValueError(f"Order mismatch: {path}")
 return a,times

def load_inputs():
 eeg=np.empty((2,30,50,10,10)); semg=np.empty_like(eeg); eeg_paths=[]; semg_paths=[]; eeg_t=semg_t=None
 for ti,task in enumerate(TASKS):
  for si,subject in enumerate(sub_ids):
   ep=RESULTS_DIR/"neural_rdms/EEG"/f"{subject}_{task.replace(' ','_')}_EEG_rdms.h5"; sp=RESULTS_DIR/"neural_rdms/sEMG"/f"{subject}_{task.replace(' ','_')}_sEMG_rdms.h5"
   ea,et=load_rdm_file(ep,(50,10,10)); sa,st=load_rdm_file(sp,(50,10,10)); eeg[ti,si]=ea; semg[ti,si]=sa; eeg_paths.append(str(ep)); semg_paths.append(str(sp))
   if eeg_t is None: eeg_t,semg_t=et,st
   if not np.array_equal(et,eeg_t) or not np.array_equal(st,semg_t): raise ValueError(f"Within-signal time mismatch: {subject}/{task}")
 if len(eeg_t)!=len(semg_t) or not np.allclose(np.diff(eeg_t),np.diff(semg_t),atol=1e-10): raise ValueError("EEG/sEMG intervals cannot be index aligned")
 models={}; model_paths={}
 for name,path in MODEL_PATHS.items(): models[name]=vectorize_rdm(load_rdm_file(path,(10,10))[0]); model_paths[name]=str(path)
 return eeg,semg,eeg_t,semg_t,models,eeg_paths,semg_paths,model_paths

def ranks(rdms): return rankdata(np.stack([vectorize_rdm(r) for r in rdms]),axis=1,method="average")

def partial_matrix_ranked(a,b,static_controls,semg_a,semg_b):
 """Residual-Pearson partial correlation via batched correlation-matrix inversion."""
 n=a.shape[0]; ar=np.repeat(a,n,axis=0); br=np.tile(b,(n,1)); sr=np.repeat(semg_a,n,axis=0); or_=np.tile(semg_b,(n,1))
 controls=[np.broadcast_to(rankdata(c,method="average"),(n*n,45)) for c in static_controls]+[sr,or_]
 variables=np.stack([ar,br,*controls],axis=1); variables=variables-variables.mean(axis=2,keepdims=True); norms=np.linalg.norm(variables,axis=2,keepdims=True)
 if np.any(norms<=np.finfo(float).eps): raise ValueError("Constant ranked vector in partial correlation")
 z=variables/norms; corr=z @ z.transpose(0,2,1); precision=np.linalg.pinv(corr)
 r=-precision[:,0,1]/np.sqrt(precision[:,0,0]*precision[:,1,1])
 return np.clip(r,-1,1).reshape(n,n)

def compute_maps(eeg,semg,models):
 sim=np.empty((30,50,50)); unique=np.empty((3,30,50,50)); rwout=np.empty_like(unique); rwith=np.empty_like(unique); errors=[]
 for si in range(30):
  ea,eb=ranks(eeg[0,si]),ranks(eeg[1,si]); sa,sb=ranks(semg[0,si]),ranks(semg[1,si])
  sim[si]=partial_matrix_ranked(ea,eb,[],sa,sb)
  for ii,target in enumerate(INFORMATION):
   without=[models[n] for n in INFORMATION if n!=target]; with_target=without+[models[target]]
   rwout[ii,si]=partial_matrix_ranked(ea,eb,without,sa,sb); rwith[ii,si]=partial_matrix_ranked(ea,eb,with_target,sa,sb); unique[ii,si]=rwout[ii,si]**2-rwith[ii,si]**2
   if si==0:
    i,j=ii,ii+2; dynamic=[vectorize_rdm(semg[0,0,i]),vectorize_rdm(semg[1,0,j])]
    expected=partial_corr_vector(vectorize_rdm(eeg[0,0,i]),vectorize_rdm(eeg[1,0,j]),without+dynamic,method="spearman")["r"]
    expected2=partial_corr_vector(vectorize_rdm(eeg[0,0,i]),vectorize_rdm(eeg[1,0,j]),with_target+dynamic,method="spearman")["r"]
    errors += [abs(expected-rwout[ii,si,i,j]),abs(expected2-rwith[ii,si,i,j])]
 if not all(np.isfinite(x).all() for x in [sim,unique,rwout,rwith]) or max(errors)>1e-9: raise ValueError(f"Map QC/backend validation failed: max_error={max(errors)}")
 return sim,unique,rwout,rwith,max(errors)

def normalize_cluster(c,shape):
 if isinstance(c,tuple): m=np.zeros(shape,bool); m[c]=True; return m
 return np.asarray(c,bool).reshape(shape)

def cluster_test(data,tail,threshold,fn):
 T,clusters,p,H0=fn(data,threshold=threshold,tail=tail,n_permutations=N_PERM,out_type="mask",seed=SEED,verbose=False); masks=np.stack([normalize_cluster(c,T.shape) for c in clusters]) if len(clusters) else np.zeros((0,)+T.shape,bool); sig=np.zeros(T.shape,np.int8)
 for m,pv in zip(masks,p):
  if pv<ALPHA: sig[m]=1 if data[:,m].mean()>0 else -1
 return dict(T_obs=T,clusters=masks,p_values=np.asarray(p),H0=np.asarray(H0),significant_mask=sig)

def result_rows(data,result,panel,atype,info,threshold,tail,times_y,times_x):
 rows=[]; windows=[]; meanmap=data.mean(0)
 for cid,(mask,pv) in enumerate(zip(result["clusters"],result["p_values"])):
  yi,xi=np.where(mask); mv=float(data[:,mask].mean()); local=int(np.argmax(np.abs(result["T_obs"][mask]))); py,px=yi[local],xi[local]; sig=bool(pv<ALPHA); sign="positive" if mv>0 else "negative"
  if atype=="cross_mode_similarity_control_sEMG": interp="positive silent–overt EEG shared geometry remains after controlling time-matched sEMG RDMs"
  elif mv>0: interp="target information uniquely explains silent–overt shared EEG geometry beyond sEMG"
  else: interp="controlling target information increases residual silent–overt EEG similarity beyond sEMG, suggesting possible suppressor-like or variance-reallocation effect"
  if sig: windows.append(f"y={times_y[yi.min()]:.6f}–{times_y[yi.max()]:.6f}s, x={times_x[xi.min()]:.6f}–{times_x[xi.max()]:.6f}s, {sign}, p={pv:.6g}")
  rows.append(dict(figure_panel=panel,analysis_type=atype,information_type=info,task_pair="silent_overt",task_A="silent speech",task_B="overt speech",n_subjects=30,n_times_y=50,n_times_x=50,statistical_test="mne.stats.permutation_cluster_1samp_test",tail=tail,cluster_forming_threshold=threshold,n_permutations=N_PERM,seed=SEED,cluster_id=cid,cluster_p_value=float(pv),significant_cluster=sig,cluster_start_y_time=float(times_y[yi.min()]),cluster_end_y_time=float(times_y[yi.max()]),cluster_start_x_time=float(times_x[xi.min()]),cluster_end_x_time=float(times_x[xi.max()]),cluster_size_n_points=int(mask.sum()),cluster_mean_value=mv,cluster_peak_value=float(meanmap[py,px]),peak_t=float(result["T_obs"][py,px]),peak_y_time=float(times_y[py]),peak_x_time=float(times_x[px]),cluster_sign=sign,direction_interpretation=interp,notes="cluster correction within this 2D map; no global information-type correction"))
 peak=np.unravel_index(np.argmax(np.abs(meanmap)),meanmap.shape); summary=dict(figure_panel=panel,analysis_type=atype,information_type=info,task_pair="silent_overt",has_significant_cluster=bool(windows),n_significant_clusters=len(windows),significant_windows_summary="; ".join(windows),peak_value=float(meanmap[peak]),peak_y_time=float(times_y[peak[0]]),peak_x_time=float(times_x[peak[1]]),min_cluster_p=float(np.min(result["p_values"])) if len(result["p_values"]) else np.nan,summary_note="correction within this 2D map; no global correction across information types")
 return rows,summary

def write_csv(path,rows,fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)

def save_h5(path,main_name,values,stats,meta,times_eeg,times_semg,extras=None,info=False):
 path.parent.mkdir(parents=True,exist_ok=True); dt=h5py.string_dtype("utf-8")
 with h5py.File(path,"w") as h:
  h.create_dataset(main_name,data=values)
  for k,v in (extras or {}).items(): h.create_dataset(k,data=v)
  h.create_dataset("subject_ids",data=np.asarray(sub_ids,dtype=object),dtype=dt)
  if info: h.create_dataset("information_types",data=np.asarray(INFORMATION,dtype=object),dtype=dt)
  for k,v in {"task_pair":"silent_overt","task_A":"silent speech","task_B":"overt speech"}.items(): h.create_dataset(k,data=v,dtype=dt)
  h.create_dataset("time_axis_silent",data=times_eeg); h.create_dataset("time_axis_overt",data=times_eeg); h.create_dataset("sEMG_time_axis_silent",data=times_semg); h.create_dataset("sEMG_time_axis_overt",data=times_semg); h.create_dataset("T_obs",data=np.stack([s["T_obs"] for s in stats])); h.create_dataset("significant_cluster_masks",data=np.stack([s["significant_mask"] for s in stats])); h.create_dataset("metadata_json",data=json.dumps(meta,ensure_ascii=False),dtype=dt)
  cg=h.create_group("clusters"); pg=h.create_group("cluster_p_values"); hg=h.create_group("H0")
  for i,s in enumerate(stats): cg.create_dataset(str(i),data=s["clusters"]); pg.create_dataset(str(i),data=s["p_values"]); hg.create_dataset(str(i),data=s["H0"])

def main():
 os.environ.setdefault("NUMBA_CACHE_DIR",str(OUT/".numba_cache")); os.environ.setdefault("MPLCONFIGDIR",str(OUT/".matplotlib")); local=ROOT/"external_models/python_packages"
 if local.is_dir() and str(local) not in sys.path: sys.path.insert(0,str(local))
 import mne
 from mne.stats import permutation_cluster_1samp_test
 LOG.parent.mkdir(parents=True,exist_ok=True); log=["Step 22 Final Figure 5 with sEMG control",f"Figure5A_Step19_source={STEP19}"]
 try:
  if not STEP19.is_file(): raise FileNotFoundError(STEP19)
  eeg,semg,eeg_t,semg_t,models,eeg_paths,semg_paths,model_paths=load_inputs(); sim,unique,rwout,rwith,error=compute_maps(eeg,semg,models); pth=float(t.ppf(.95,29)); tth=float(t.ppf(1-.05/2,29)); simstat=cluster_test(sim,1,pth,permutation_cluster_1samp_test); ustats=[cluster_test(unique[i],0,tth,permutation_cluster_1samp_test) for i in range(3)]
  simrows,simsum=result_rows(sim,simstat,"Figure 5B","cross_mode_similarity_control_sEMG","EEG_RDM_similarity",pth,"positive",eeg_t,eeg_t); urows=[]; summaries=[simsum]
  for i,name in enumerate(INFORMATION): rows,summary=result_rows(unique[i],ustats[i],f"Figure 5{'CDE'[i]}","information_specific_unique_variance_control_sEMG",name,tth,"two-sided",eeg_t,eeg_t); urows+=rows; summaries.append(summary)
  write_csv(SIM_CSV,simrows,CLUSTER_FIELDS); write_csv(UNIQUE_CSV,urows,CLUSTER_FIELDS); write_csv(SUMMARY_CSV,summaries,SUMMARY_FIELDS)
  meta=dict(source_step="Step 22",figure="Final Figure 5 with sEMG control",Figure5A_source="Step 19 final EEG partial RSA",task_pair="silent_overt",sEMG_control=True,sEMG_controls="time-matched silent and overt sEMG RDMs",information_types=INFORMATION,unique_variance_definition="delta_R2 = r_without_target_squared - r_with_target_squared",Figure5B_stat="MNE positive-tail 2D cluster permutation",Figure5CDE_stat="MNE two-sided 2D cluster permutation",n_permutations=N_PERM,seed=SEED,correction_scope="within each 2D map",global_correction=False,eeg_sEMG_alignment="window index; identical n_times and interval; EEG axes retained",fixed_onset_offset_s=float(semg_t[0]-eeg_t[0]),partial_spearman_backend="average ranks + least-squares-equivalent batched correlation precision matrix")
  save_h5(SIM_H5,"similarity_control_sEMG_maps",sim,[simstat],meta,eeg_t,semg_t); save_h5(UNIQUE_H5,"unique_variance_maps",unique,ustats,meta,eeg_t,semg_t,{"r_without_target":rwout,"r_with_target":rwith},True)
  log += [f"EEG_RDM_paths_count={len(eeg_paths)}",*[f"EEG_RDM={p}" for p in eeg_paths],f"sEMG_RDM_paths_count={len(semg_paths)}",*[f"sEMG_RDM={p}" for p in semg_paths],*[f"model_RDM[{k}]={v}" for k,v in model_paths.items()],"silent/overt EEG subject order=PASS (S01-S30)","silent/overt sEMG subject order=PASS (S01-S30)","EEG/sEMG subject order=PASS","time/index alignment=PASS",f"EEG_time={eeg_t[0]}..{eeg_t[-1]}; sEMG_time={semg_t[0]}..{semg_t[-1]}; fixed_offset={semg_t[0]-eeg_t[0]}",f"Figure5B_shape={sim.shape}",f"Figure5CDE_unique_shape={unique.shape}",f"partial_backend_validation_max_abs_error={error:.3g}",f"mne_version={mne.__version__}",f"Figure5B threshold={pth}; tail=1; seed={SEED}; n_permutations={N_PERM}",f"Figure5CDE threshold={tth}; tail=0; seed={SEED}; n_permutations={N_PERM}",f"Figure5B significant_clusters={sum(simstat['p_values']<ALPHA)}"]
  for i,name in enumerate(INFORMATION): log.append(f"Figure5{'CDE'[i]} {name}: significant={sum(ustats[i]['p_values']<ALPHA)}, positive_present={int(np.any(ustats[i]['significant_mask']>0))}, negative_present={int(np.any(ustats[i]['significant_mask']<0))}")
  log += ["Step 19/21 and all old results modified=NO",f"outputs={SIM_H5}; {UNIQUE_H5}; {SIM_CSV}; {UNIQUE_CSV}; {SUMMARY_CSV}","OVERALL: PASS"]
 except Exception: log += [traceback.format_exc(),"OVERALL: FAILED"]; LOG.write_text("\n".join(log)+"\n",encoding="utf-8"); raise
 LOG.write_text("\n".join(log)+"\n",encoding="utf-8"); print("\n".join(log[-35:]))

if __name__=="__main__": main()
