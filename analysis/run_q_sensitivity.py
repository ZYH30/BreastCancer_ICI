#!/usr/bin/env python3
"""Fresh-stream single Q-interface repair confirmation; 400 datasets."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
from pathlib import Path
import run_calibration as core

core.ENTROPY=202609160602
core.DATASET_PREFIX='QR_'
core.LAYER_METHODS={'L2':['M1_all_inputs','M3_actual_Q_closure','M1_restricted_Q_fixed_C','M3_restricted_Q_fixed_C']}
core.LABEL_MAP={('L2','M1_restricted_Q_fixed_C'):'L2_M1_QR',('L2','M3_restricted_Q_fixed_C'):'L2_M3_QR'}
core.ALL_METHODS=['L2_M1_S','L2_M3_S','L2_M1_QR','L2_M3_QR','complete_oracle','true_e_clip_operator']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reps',type=int,default=100);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    args.extra_sources=[Path(__file__),core.ROOT/'docs/reproducibility.md']
    args.extra_protocol=dict(repair='restricted_Q_original_C_fixed',new_confirmation_entropy=202609160602,
        source_primary_entropy=202609160601,source_primary_datasets=800,new_datasets=4*args.reps,
        only_Q_and_necessary_matched_g_change=True,M1_receives_identical_Q_repair=True,
        method_comparison_paired=True,no_second_performance_repair_planned=True)
    core.main(args)
