#!/usr/bin/env python3
"""One-repair benefits and costs, generated only from delivered CSVs."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True);sources=out/'source_data';sources.mkdir(exist_ok=True)
    for name in ['K1_QR_results.csv', 'K2_QR_paired_comparisons.csv', 'K1_stage_swap_results.csv']:
        src=args.source_tables/name
        if src.resolve()!=(sources/name).resolve():shutil.copy2(src,sources/name)
    qr=pd.read_csv(sources/'K1_QR_results.csv');old=pd.read_csv(sources/'K1_stage_swap_results.csv');pairs=pd.read_csv(sources/'K2_QR_paired_comparisons.csv')
    fig,axs=plt.subplots(1,3,figsize=(17,6),gridspec_kw={'width_ratios':[1.7,1,1]});plt.rcParams['svg.fonttype']='none'
    ids=[f'{trial}_{part}' for trial in ['NeoTRIP','ISPY2'] for part in ['original','0','1','2']]
    for i,identity in enumerate(ids):
        before=old[(old.run_id==identity)&(old.method=='M3')&(old.form=='spline')&(old.combination=='C11')].iloc[0]
        after=qr[(qr.run_id==identity)&(qr.method=='M3')&(qr.form=='spline')].iloc[0]
        for row,offset,color,label in [(before,-.13,'#305c83','Common-bound full Q'),(after,.13,'#b2412e','Q-only repair')]:
            axs[0].errorbar(row.estimate_SD,i+offset,xerr=[[row.estimate_SD-row.low_SD],[row.high_SD-row.estimate_SD]],fmt='o',color=color,capsize=3,label=label if i==0 else None)
    axs[0].set_yticks(range(8),[x.replace('_original',' primary').replace('_',' split ') for x in ids]);axs[0].invert_yaxis();axs[0].axvline(0,color='grey',ls='--');axs[0].set_xlabel('Clinical study-SD coefficient (95% delta CI)');axs[0].legend(loc='upper right',fontsize=8);axs[0].set_title('A  Clinical cost: wider / weaker across splits')
    for ax,metric,title in [(axs[1],'Q_MSE','B  Q prediction error change'),(axs[2],'score_reject','C  Signal power change')]:
        sub=pairs[(pairs.candidate=='B')&(pairs.metric==metric)]
        cells=[(85,0),(85,1),(241,0),(241,1)] if metric=='Q_MSE' else [(85,1),(241,1)]
        for j,(n,signal) in enumerate(cells):
            for method,offset,color in [('M1',-.08,'#305c83'),('M3',.08,'#b2412e')]:
                row=sub[(sub.n==n)&(sub.signal==signal)&(sub.comparison==f'L2_{method}_QR_minus_L2_{method}_S')].iloc[0]
                ax.errorbar(row.mean_difference,j+offset,xerr=1.96*row.paired_MCSE,fmt='o',color=color,capsize=3,label=method if j==0 else None)
        ax.set_yticks(range(len(cells)),[f'n={n}\nsignal={s}' for n,s in cells]);ax.invert_yaxis();ax.axvline(0,color='grey',ls='--');ax.set_xlabel('Repair minus original\n(±1.96 paired Monte Carlo SE)');ax.set_title(title);ax.legend(fontsize=8)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One Q-only repair: fresh 100 datasets/cell; same e, m, graphs and folds',fontsize=13);fig.tight_layout(rect=[0,0,1,.95])
    for suffix in ['svg','png']:fig.savefig(out/f'PlanB_Q_repair_benefits_and_costs.{suffix}',dpi=180,bbox_inches='tight')
    dest=out/'plotting_scripts';dest.mkdir(exist_ok=True);shutil.copy2(Path(__file__),dest/Path(__file__).name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-tables',type=Path,required=True);run(p.parse_args())
