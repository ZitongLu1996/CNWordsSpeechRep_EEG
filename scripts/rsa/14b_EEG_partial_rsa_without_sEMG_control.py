#!/usr/bin/env python3
"""Step 14B: EEG partial RSA excluding sEMG from all model sets."""
import csv,importlib.util,inspect,json,os,time,traceback
import h5py,numpy as np
from scipy.stats import ttest_1samp
from utils.config import RESULTS_DIR,MODEL_RDM_DIR,RSA_ROOT,RSA_LABEL_ORDER,RSA_WORD_ORDER,sub_ids
from utils.rdm_utils import partial_corr_vector,vectorize_rdm

TASKS=["imagined speech","silent speech","overt speech"]
MODEL_SETS={"modelset_v3_92d_refined":["semantic_tencent","visual_alexnet_conv2","phonology_distributed_feature_v3_92d_refined"]}
MODEL_PATHS={"semantic_tencent":MODEL_RDM_DIR/"semantic_tencent_rdm.h5","visual_alexnet_conv2":MODEL_RDM_DIR/"visual_alexnet_conv2_rdm.h5","phonology_distributed_feature_v3_92d_refined":MODEL_RDM_DIR/"phonology_distributed_feature_v3_92d_refined_rdm.h5"}
OUT=RESULTS_DIR/"rsa_results"/"step14b_EEG_partial_without_sEMG_control"; STATS=OUT/"group_stats"; LOG=OUT/"logs"/"step14b_EEG_partial_without_sEMG_control.txt"; STATS_UTILS=RSA_ROOT.parent/"utils"/"stats_utils.py"
os.environ.setdefault("NUMBA_CACHE_DIR",str(RESULTS_DIR/"logs"/"numba_cache")); os.environ.setdefault("MPLCONFIGDIR",str(RESULTS_DIR/"logs"/"matplotlib_cache"))
def dec(x): return [v.decode() if isinstance(v,bytes) else str(v) for v in x]
def load_rdm(path,temporal=False):
    with h5py.File(path) as h:
        key="rdms" if "rdms" in h else "neural_rdms" if "neural_rdms" in h else "rdm"; r=np.asarray(h[key][...],float); labels=[int(v) for v in h["label_order"][...]]; words=dec(h["word_order"][...]); times=np.asarray(h["time_axis"][...],float) if "time_axis" in h else None
    expected=(50,10,10) if temporal else (10,10)
    if r.shape!=expected or labels!=list(RSA_LABEL_ORDER) or words!=list(RSA_WORD_ORDER) or not np.isfinite(r).all(): raise ValueError(f"Shape/order/value QC failed: {path}")
    if temporal:
        if times.shape!=(50,) or not np.allclose(r,r.transpose(0,2,1),atol=1e-8) or not np.allclose(np.diagonal(r,axis1=1,axis2=2),0,atol=1e-8): raise ValueError(f"Temporal RDM QC failed: {path}")
    elif not np.allclose(r,r.T,atol=1e-8) or not np.allclose(np.diag(r),0,atol=1e-8): raise ValueError(f"Static RDM QC failed: {path}")
    return r,times
def curve(neural,target,controls):
    x=vectorize_rdm(target); cs=[vectorize_rdm(c) for c in controls]; out=np.asarray([partial_corr_vector(vectorize_rdm(r),x,cs,method="spearman")["r"] for r in neural])
    if out.shape!=(50,) or not np.isfinite(out).all(): raise ValueError("Malformed/non-finite partial RSA curve")
    return out
def save_set(setname,values,times,models):
    folder=OUT/setname; folder.mkdir(parents=True,exist_ok=True); suffix="onehot" if setname=="modelset_onehot" else "v3_92d_refined"; path=folder/f"EEG_partial_rsa_without_sEMG_{suffix}.h5"; dt=h5py.string_dtype("utf-8"); meta=dict(model_set_name=setname,analysis_name="EEG_partial_spearman_without_sEMG_control",axis_order=["tasks","subjects","models","times"],method="custom partial Spearman",sEMG_included=False,tasks=TASKS,model_names=models,model_paths={m:str(MODEL_PATHS[m]) for m in models},word_order=RSA_WORD_ORDER,label_order=RSA_LABEL_ORDER)
    with h5py.File(path,"w") as h:
        h.create_dataset("rsa_curves",data=values); h.create_dataset("subject_ids",data=np.asarray(sub_ids,dtype=object),dtype=dt); h.create_dataset("tasks",data=np.asarray(TASKS,dtype=object),dtype=dt); h.create_dataset("model_names",data=np.asarray(models,dtype=object),dtype=dt); h.create_dataset("time_axis",data=times); h.create_dataset("model_set_name",data=setname,dtype=dt); h.create_dataset("analysis_name",data="EEG_partial_spearman_without_sEMG_control",dtype=dt); h.create_dataset("metadata_json",data=json.dumps(meta,ensure_ascii=False),dtype=dt)
    with h5py.File(path) as h:
        if h["rsa_curves"].shape!=(3,30,3,50): raise ValueError("HDF5 reread QC failed")
    return path
def stats_fn():
    spec=importlib.util.spec_from_file_location("stats14b",STATS_UTILS); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.cluster_permutation_1samp
def main():
    start=time.time(); STATS.mkdir(parents=True,exist_ok=True); LOG.parent.mkdir(parents=True,exist_ok=True); logs=[]; models={}; summary=[]; payload={}
    for name,path in MODEL_PATHS.items(): models[name],_=load_rdm(path); logs.append(f"model {name}: PASS; {path}")
    for setname,order in MODEL_SETS.items():
        task_values=[]; common=None
        for task in TASKS:
            subjects=[]
            for subject in sub_ids:
                try:
                    eeg,times=load_rdm(RESULTS_DIR/"neural_rdms"/"EEG"/f"{subject}_{task.replace(' ','_')}_EEG_rdms.h5",True)
                    if common is None: common=times
                    elif not np.allclose(common,times,atol=1e-10,rtol=0): raise ValueError("EEG time axes differ")
                    curves=np.stack([curve(eeg,models[target],[models[c] for c in order if c!=target]) for target in order]); subjects.append(curves)
                    for mi,target in enumerate(order):
                        c=curves[mi]; peak=int(np.argmax(c)); summary.append(dict(model_set=setname,task=task,subject=subject,target_model=target,status="success",n_times=50,n_nan=0,n_inf=0,mean_r=float(c.mean()),max_r=float(c[peak]),peak_time=float(times[peak]),error_message=""))
                except Exception as exc:
                    for target in order: summary.append(dict(model_set=setname,task=task,subject=subject,target_model=target,status="failed",n_times="",n_nan="",n_inf="",mean_r="",max_r="",peak_time="",error_message=str(exc)))
                    logs.append(f"FAILED {setname}/{task}/{subject}\n{traceback.format_exc()}")
            if len(subjects)!=30: raise RuntimeError(f"{setname}/{task}: {len(subjects)}/30 succeeded")
            task_values.append(np.stack(subjects)); logs.append(f"{setname}/{task}: 30/30 PASS")
        values=np.stack(task_values)
        if values.shape!=(3,30,3,50) or not np.isfinite(values).all(): raise ValueError(f"Group QC failed: {values.shape}")
        save_set(setname,values,common,order); payload[setname]=(values,common,order)
    with (OUT/"step14b_EEG_partial_without_sEMG_summary.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    fn=stats_fn(); logs.append(f"stats signature={inspect.signature(fn)}"); rows=[]; records={}
    for setname,(values,times,order) in payload.items():
        for ti,task in enumerate(TASKS):
            for mi,target in enumerate(order):
                data=values[ti,:,mi]; sig,clusters=fn(data=data,chance=0,n_perm=1000,p_cluster=.05,tail="pos",seed=20260819,return_clusters=True); mask=np.zeros(50,np.uint8); mask[np.asarray(sig,int)]=1; cs=[dict(id=i,indices=np.asarray(c["indices"],int).tolist(),mass=float(c["mass"]),pval=float(c["pval"]),sign=float(c["sign"])) for i,c in enumerate(clusters)]; good=[c for c in cs if c["pval"]<.05]; mean=data.mean(0); tv=ttest_1samp(data,0,axis=0).statistic; peak=int(np.argmax(mean)); controls=[m for m in order if m!=target]
                rows.append(dict(model_set=setname,task=task,target_model=target,control_models=";".join(controls),n_subjects=30,n_times=50,cluster_found=bool(good),n_clusters=len(good),significant_cluster_ids=";".join(str(c["id"]) for c in good),cluster_p_values=";".join(f'{c["pval"]:.6g}' for c in good),cluster_start_time=";".join(f'{times[c["indices"][0]]:.9f}' for c in good),cluster_end_time=";".join(f'{times[c["indices"][-1]]:.9f}' for c in good),peak_r=float(mean[peak]),peak_t=float(tv[peak]),peak_time=float(times[peak]),notes="actual cluster p-values; n_perm=1000; tail=pos; no sEMG model/control")); records[(setname,task,target)]=(mask,cs)
    sp=STATS/"step14b_EEG_partial_without_sEMG_stats.csv"
    with sp.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    dt=h5py.string_dtype("utf-8")
    with h5py.File(sp.with_suffix(".h5"),"w") as h:
        h.attrs.update(n_perm=1000,p_cluster=.05,tail="pos")
        for setname,(values,times,order) in payload.items():
            sg=h.create_group(setname); sg.create_dataset("time_axis",data=times)
            for task in TASKS:
                tg=sg.create_group(task.replace(" ","_"))
                for target in order:
                    g=tg.create_group(target); mask,cs=records[(setname,task,target)]; g.create_dataset("significance_mask",data=mask); g.create_dataset("clusters_json",data=json.dumps(cs),dtype=dt)
    if len(summary)!=540 or len(rows)!=18: raise ValueError("Final count QC failed")
    logs += ["subject-model curves=540","group stats=18/18",f"elapsed_seconds={time.time()-start:.3f}","OVERALL: PASS"]; LOG.write_text("\n".join(logs)+"\n",encoding="utf-8")
if __name__=="__main__":
    try: main()
    except Exception: LOG.parent.mkdir(parents=True,exist_ok=True); LOG.write_text((LOG.read_text(encoding="utf-8") if LOG.exists() else "")+"\nFATAL\n"+traceback.format_exc(),encoding="utf-8"); raise
