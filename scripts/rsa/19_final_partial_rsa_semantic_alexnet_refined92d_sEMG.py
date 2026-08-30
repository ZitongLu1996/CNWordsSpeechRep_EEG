#!/usr/bin/env python3
"""Step 19 final EEG partial RSA: semantic, AlexNet conv2, refined 92D, and sEMG."""
import csv,importlib.util,json,os,traceback
import h5py,numpy as np
from scipy.stats import ttest_1samp
from utils.config import RESULTS_DIR,MODEL_RDM_DIR,RSA_LABEL_ORDER,RSA_ROOT,RSA_WORD_ORDER,sub_ids
from utils.rdm_utils import compute_temporal_model_partial_rsas,validate_square_rdm

SEM="semantic_tencent"; VIS="visual_alexnet_conv2"; PH="phonology_distributed_feature_v3_92d_refined"; SEMG="sEMG_time_matched"
TASKS=["imagined speech","silent speech","overt speech"]
ORDERS={TASKS[0]:[SEM,VIS,PH],TASKS[1]:[SEM,VIS,PH,SEMG],TASKS[2]:[SEM,VIS,PH,SEMG]}
MODEL_PATHS={SEM:MODEL_RDM_DIR/"semantic_tencent_rdm.h5",VIS:MODEL_RDM_DIR/"visual_alexnet_conv2_rdm.h5",PH:MODEL_RDM_DIR/"phonology_distributed_feature_v3_92d_refined_rdm.h5"}
OUT=RESULTS_DIR/"rsa_results"/"step19_final_partial_rsa_semantic_alexnet_refined92d_sEMG"; PART=OUT/"partial_rsa"; STATS=OUT/"group_stats"; LOG=OUT/"logs"/"step19_final_partial_rsa.txt"; CONS=OUT/"logs"/"step19_consistency_with_step12.txt"
STATS_UTILS=RSA_ROOT.parent/"utils"/"stats_utils.py"; ANALYSIS="final_EEG_partial_RSA_semantic_alexnet_refined92d_sEMG"
os.environ.setdefault("NUMBA_CACHE_DIR",str(RESULTS_DIR/"logs"/"numba_cache"))
os.environ.setdefault("MPLCONFIGDIR",str(RESULTS_DIR/"logs"/".matplotlib"))
def dec(x): return [v.decode() if isinstance(v,bytes) else str(v) for v in x]
def load(path):
 with h5py.File(path) as h:
  key="rdms" if "rdms" in h else "neural_rdms" if "neural_rdms" in h else "rdm"; r=np.asarray(h[key],float); labels=[int(v) for v in h["label_order"][:]]; words=dec(h["word_order"][:]); times=np.asarray(h["time_axis"],float) if "time_axis" in h else None
 if labels!=RSA_LABEL_ORDER or words!=RSA_WORD_ORDER or not np.isfinite(r).all(): raise ValueError(f"RDM/order/finite QC failed: {path}")
 if r.ndim==2:
  validate_square_rdm(r,str(path));
  if r.shape!=(10,10) or not np.allclose(r,r.T,atol=1e-8) or not np.allclose(np.diag(r),0,atol=1e-8): raise ValueError(f"Static QC failed: {path}")
 elif r.ndim==3:
  if r.shape!=(50,10,10) or times.shape!=(50,) or not np.allclose(r,r.transpose(0,2,1),atol=1e-8) or not np.allclose(np.diagonal(r,axis1=1,axis2=2),0,atol=1e-8): raise ValueError(f"Temporal QC failed: {path}")
 else: raise ValueError(f"Unsupported RDM shape: {r.shape}")
 return r,times,key
def statsfn():
 s=importlib.util.spec_from_file_location("stats19",STATS_UTILS); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.cluster_permutation_1samp
def compute(subject,task,static):
 eeg,et,_=load(RESULTS_DIR/"neural_rdms"/"EEG"/f"{subject}_{task.replace(' ','_')}_EEG_rdms.h5"); order=ORDERS[task]; controls={m:[x for x in order if x!=m] for m in order}; base={m:static[m] for m in order if m!=SEMG}; offset=0.
 if SEMG not in order: values,returned,_=compute_temporal_model_partial_rsas(eeg,base,controls,method="spearman")
 else:
  sg,st,_=load(RESULTS_DIR/"neural_rdms"/"sEMG"/f"{subject}_{task.replace(' ','_')}_sEMG_rdms.h5"); offset=float(st[0]-et[0])
  if len(et)!=len(st) or not np.allclose(np.diff(et),np.diff(st),atol=1e-10,rtol=0) or abs(offset)>1/250: raise ValueError(f"EEG/sEMG time mismatch: {offset}")
  values=np.empty((50,len(order))); returned=order
  for ti in range(50):
   one,ret,_=compute_temporal_model_partial_rsas(eeg[ti:ti+1],{**base,SEMG:sg[ti]},controls,method="spearman")
   if ret!=order: raise ValueError("Model order changed")
   values[ti]=one[0]
 if returned!=order or values.shape!=(50,len(order)) or not np.isfinite(values).all(): raise ValueError("Partial RSA QC failed")
 return values.T,et,offset
def save(payload,offsets):
 path=PART/"final_partial_rsa_semantic_alexnet_refined92d_sEMG.h5"; dt=h5py.string_dtype("utf-8"); meta=dict(analysis_name=ANALYSIS,backend="utils.rdm_utils.compute_temporal_model_partial_rsas; custom partial Spearman",imagined_has_no_sEMG_model=True,silent_overt_use_time_matched_subject_specific_sEMG_RDM=True,sEMG_time_matching="window index; fixed onset offset logged; no interpolation",model_set_by_task=ORDERS,model_paths={k:str(v) for k,v in MODEL_PATHS.items()},correction_method="cluster permutation across time within each curve",cluster_forming_p=.05,cluster_corrected_p=.05,n_perm=1000,tail="positive",word_order=RSA_WORD_ORDER,label_order=RSA_LABEL_ORDER,eeg_semg_onset_offsets_s=offsets)
 with h5py.File(path,"w") as h:
  rg=h.create_group("rsa_curves"); mg=h.create_group("model_names_by_task"); sg=h.create_group("subject_ids"); tg=h.create_group("time_axis"); setg=h.create_group("model_set_by_task")
  for task,(v,t,order) in payload.items(): key=task.replace(" ","_"); rg.create_dataset(key,data=v); mg.create_dataset(key,data=np.asarray(order,dtype=object),dtype=dt); sg.create_dataset(key,data=np.asarray(sub_ids,dtype=object),dtype=dt); tg.create_dataset(key,data=t); setg.create_dataset(key,data=json.dumps(order),dtype=dt)
  h.create_dataset("tasks",data=np.asarray(TASKS,dtype=object),dtype=dt); h.create_dataset("analysis_name",data=ANALYSIS,dtype=dt); h.create_dataset("metadata_json",data=json.dumps(meta,ensure_ascii=False),dtype=dt)
 return path
def consistency(payload):
 lines=["Read-only numerical comparison with saved Step 12 refined-v3 partial RSA."]; allclose=True
 for task,(new,t,order) in payload.items():
  p=RESULTS_DIR/"rsa_results"/"partial_rsa_phonology_v3_92d_refined"/f"{task.replace(' ','_')}_EEG_v3_92d_refined_partial_rsa.h5"
  if not p.is_file(): lines.append(f"{task}: Step 12 not found: {p}"); allclose=False; continue
  with h5py.File(p) as h: old=np.asarray(h["rsa_values"],float); oo=dec(h["model_order"][:]); ot=np.asarray(h["time_axis"],float)
  if not np.allclose(t,ot,atol=1e-10,rtol=0): lines.append(f"{task}: time_axis mismatch"); allclose=False; continue
  for m in order:
   d=np.abs(new[:,order.index(m)]-old[:,oo.index(m)]); ok=bool(np.allclose(d,0,atol=1e-12,rtol=0)); allclose &= ok; lines.append(f"{task}/{m}: max_abs_difference={d.max():.17g}; mean_abs_difference={d.mean():.17g}; allclose_atol_1e-12={ok}")
 lines.append(f"OVERALL_MATCH={allclose}"); CONS.write_text("\n".join(lines)+"\n",encoding="utf-8")
 if not allclose: raise ValueError("Step 19 does not match Step 12; inspect consistency log for model order/path/backend/time-axis/sEMG matching")
def main():
 for d in [PART,STATS,OUT/"figures"/"individual_curves",OUT/"logs"]: d.mkdir(parents=True,exist_ok=True)
 logs=["Step 19 final integrated model; existing RDMs read, none rebuilt."]; static={}
 for m,p in MODEL_PATHS.items(): static[m],_,key=load(p); logs.append(f"model PASS: {m}; path={p}; dataset={key}")
 payload={}; offsets={}; summary=[]
 for task in TASKS:
  arrays=[]; ref=None; offsets[task]=[]
  for subject in sub_ids:
   v,t,off=compute(subject,task,static); arrays.append(v); offsets[task].append(off)
   if ref is None: ref=t
   elif not np.allclose(ref,t,atol=1e-10,rtol=0): raise ValueError("EEG axes differ")
  group=np.stack(arrays); expected=(30,len(ORDERS[task]),50)
  if group.shape!=expected or not np.isfinite(group).all(): raise ValueError(f"Group QC failed: {task} {group.shape}")
  payload[task]=(group,ref,ORDERS[task]); logs.append(f"{task}: PASS shape={group.shape}; EEG_sEMG_onset_offset={offsets[task][0] if offsets[task] else 'NA'}")
 path=save(payload,offsets); consistency(payload); fn=statsfn(); rows=[]; records={}
 for task,(values,times,order) in payload.items():
  for mi,target in enumerate(order):
   data=values[:,mi]; sig,clusters=fn(data=data,chance=0,n_perm=1000,p_cluster=.05,tail="pos",seed=20260819,return_clusters=True); mask=np.zeros(50,np.uint8); mask[np.asarray(sig,int)]=1; cs=[dict(id=i,indices=np.asarray(c["indices"],int).tolist(),mass=float(c["mass"]),pval=float(c["pval"]),sign=float(c["sign"])) for i,c in enumerate(clusters)]; good=[c for c in cs if c["pval"]<.05]; mean=data.mean(0); tv=ttest_1samp(data,0,axis=0).statistic; peak=int(np.argmax(mean)); ctl=[x for x in order if x!=target]
   rows.append(dict(task=task,target_model=target,control_models=";".join(ctl),n_subjects=30,n_times=50,cluster_found=bool(good),n_clusters=len(good),significant_cluster_ids=";".join(str(c["id"]) for c in good),cluster_p_values=";".join(f'{c["pval"]:.6g}' for c in good),cluster_start_time=";".join(f'{times[c["indices"][0]]:.9f}' for c in good),cluster_end_time=";".join(f'{times[c["indices"][-1]]:.9f}' for c in good),peak_r=float(mean[peak]),peak_t=float(tv[peak]),peak_time=float(times[peak]),notes="cluster-corrected; n_perm=1000; p_cluster=.05; tail=pos; seed=20260819")); records[(task,target)]=(mask,cs)
 cp=STATS/"step19_final_partial_rsa_stats.csv"
 with cp.open("w",encoding="utf-8-sig",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 dt=h5py.string_dtype("utf-8")
 with h5py.File(cp.with_suffix(".h5"),"w") as h:
  h.attrs.update(n_perm=1000,p_cluster=.05,tail="pos",seed=20260819,correction="maximum cluster mass across time")
  for task,(values,times,order) in payload.items():
   tg=h.create_group(task.replace(" ","_")); tg.create_dataset("time_axis",data=times)
   for m in order: g=tg.create_group(m); g.create_dataset("significance_mask",data=records[(task,m)][0]); g.create_dataset("clusters_json",data=json.dumps(records[(task,m)][1]),dtype=dt)
 if len(rows)!=11: raise ValueError("Expected 11 group statistics")
 logs += [f"saved={path}","partial shapes="+str({k:v[0].shape for k,v in payload.items()}),"n_nan=0; n_inf=0","Step12 consistency=PASS; exact/near-exact within atol=1e-12","group_stats=11/11","OVERALL: PASS"]; LOG.write_text("\n".join(logs)+"\n",encoding="utf-8")
if __name__=="__main__":
 try: main()
 except Exception: LOG.parent.mkdir(parents=True,exist_ok=True); LOG.write_text("FATAL\n"+traceback.format_exc(),encoding="utf-8"); raise
