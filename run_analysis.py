#!/usr/bin/env python3
"""Run one preserved analysis script with portable path overrides."""
from __future__ import annotations
import argparse, os, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.utils.paths import add_path_arguments, resolve_paths

SCRIPTS = {
 "within-mode-decoding": "scripts/decoding/within_mode_decoding.py",
 "latency": "scripts/decoding/latency_permutation_decoding.py",
 "temporal-generalization": "scripts/decoding/within_mode_temporal_generalization.py",
 "cross-mode-decoding": "scripts/decoding/directional_cross_mode_decoding.py",
 "neural-rdms": "scripts/rsa/06_build_neural_rdms_all_subjects.py",
 "within-mode-rsa": "scripts/rsa/14b_EEG_partial_rsa_without_sEMG_control.py",
 "semg-control-rsa": "scripts/rsa/19_final_partial_rsa_semantic_alexnet_refined92d_sEMG.py",
 "mode-differences": "scripts/rsa/20e_task_state_paired_difference_without_sEMG_control_refined92d.py",
 "cross-mode-rsa": "scripts/rsa/21_cross_mode_eeg_rdm_similarity_and_unique_variance.py",
 "semg-control-cross-mode": "scripts/rsa/22_final_figure5_with_sEMG_control.py",
}
parser = add_path_arguments(argparse.ArgumentParser(description=__doc__))
parser.add_argument("analysis", choices=sorted(SCRIPTS))
args = parser.parse_args(); paths = resolve_paths(args)
sys.path.insert(0, str(ROOT / "scripts" / "rsa"))
os.environ["CWR_DATA_ROOT"] = str(paths["data_root"])
os.environ["CWR_OUTPUT_ROOT"] = str(paths["output_root"])
os.environ["CWR_MODEL_RDM_ROOT"] = str(paths["model_rdm_root"])
runpy.run_path(str(ROOT / SCRIPTS[args.analysis]), run_name="__main__")
