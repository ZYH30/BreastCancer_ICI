#!/usr/bin/env python3
"""Frozen full confirmation summaries and explicitly paired repair contrasts."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from tacb_bci.provenance import file_digest,write_immutable_json
from summarize_calibration import main as summarize_main

def run(args):
    summarize_main(args)
    out=args.output;data=pd.read_csv(out/'K2_dataset_results.csv',float_precision='round_trip');data['squared_error']=(data.estimate-data.truth)**2
    for field,col in [('score_reject','p_score'),('delta_reject','p_two_sided'),('score_BH_reject','BH2_score')]:data[field]=(data[col]<.05).astype(float)
    rows=[]
    for key,d in data.groupby(['n','signal','candidate']):
        for new,old in [('L2_M3_QR','L2_M3_S'),('L2_M1_QR','L2_M1_S'),('L2_M3_QR','L2_M1_QR'),('L2_M3_QR','complete_oracle')]:
            for metric in ['squared_error','Q_MSE','e_MSE','m_MSE','g_MSE','estimate','covers_fieller','covers_delta','score_reject','score_BH_reject','fieller_length']:
                v=d.pivot(index='dataset_id',columns='method',values=metric)[[new,old]].dropna().astype(float);v=v[np.isfinite(v).all(axis=1)]
                if not len(v):continue
                diff=v[new]-v[old];se=diff.std(ddof=1)/np.sqrt(len(diff))
                rows.append(dict(zip(['n','signal','candidate'],key),comparison=new+'_minus_'+old,metric=metric,paired_success=len(diff),mean_difference=diff.mean(),paired_MCSE=se,MC_normal_low=diff.mean()-1.96*se,MC_normal_high=diff.mean()+1.96*se))
    pd.DataFrame(rows).to_csv(out/'K2_QR_paired_comparisons.csv',index=False)
    write_immutable_json(out/'repair_summary_completion.json',dict(independent_datasets=400,not_pooled_with_primary800=True,
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.glob('*.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
