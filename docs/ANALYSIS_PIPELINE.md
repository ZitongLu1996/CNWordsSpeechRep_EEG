# Analysis pipeline

| Order | Step | Script | Input | Output | Figure |
|---:|---|---|---|---|---|
| 1 | Validate public epochs | `check_data.py` | `data/` and `model_rdms/` | console report | — |
| 2 | Within-mode word decoding | `scripts/decoding/within_mode_decoding.py` | EEG `.set/.fdt` | `outputs/decoding/within_mode/` | Fig. 1A |
| 3 | Onset/peak null generation and extraction | `latency_permutation_decoding.py`; `extract_onset_peak_latency.py` | EEG epochs | `outputs/decoding/latency/` | Fig. 1B–C |
| 4 | Within-mode temporal generalization | `within_mode_temporal_generalization.py` | EEG epochs | `outputs/decoding/temporal_generalization/` | Fig. 1D |
| 5 | Directional cross-mode decoding | `directional_cross_mode_decoding.py` | paired-mode EEG epochs | `outputs/decoding/cross_mode/` | Fig. S1; bidirectional average supports Fig. 2 |
| 6 | EEG/sEMG RDM construction | `scripts/rsa/06_build_neural_rdms_all_subjects.py` | EEG and Silent/Overt sEMG epochs | `outputs/neural_rdms/` | Figs. 3–5 inputs |
| 7 | Within-mode partial RSA | `14b_EEG_partial_rsa_without_sEMG_control.py` | EEG RDMs + three model RDMs | `outputs/rsa_results/step14b_*` | Fig. 3A |
| 8 | Paired mode differences | `20e_task_state_paired_difference_without_sEMG_control_refined92d.py` | Step 14B curves | `outputs/rsa_results/step20e_*` | Fig. 3B |
| 9 | Cross-mode similarity and unique ΔR² | `21_cross_mode_eeg_rdm_similarity_and_unique_variance.py` | EEG + model RDMs | `outputs/rsa_results/step21_*` | Fig. 4 |
| 10 | Within-mode sEMG-control partial RSA | `19_final_partial_rsa_semantic_alexnet_refined92d_sEMG.py` | EEG, sEMG, model RDMs | `outputs/rsa_results/step19_*` | Fig. 5A |
| 11 | Silent–Overt sEMG-controlled similarity and unique ΔR² | `22_final_figure5_with_sEMG_control.py` | EEG, sEMG, model RDMs | `outputs/rsa_results/step22_*` | Fig. 5B–E |

Run a named step through `python run_analysis.py <analysis>`. Path priority is CLI, `config/config.yaml`, then repository-relative defaults.
