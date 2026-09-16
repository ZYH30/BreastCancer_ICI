#!/usr/bin/env python3
"""Regenerate K1/K2 figures exclusively from distributable aggregate CSVs."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ifn18_mpl')
import argparse,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(k1,k2,out):
    out.mkdir(parents=True,exist_ok=False);source=out/'source_data';source.mkdir()
    for root,name in [(k1,'K1_stage_swap_results.csv'),(k1,'K1_parameter_change_decomposition.csv'),(k2,'K2_calibration_summary.csv')]:shutil.copy2(root/name,source/name)
    dest=out/'plotting_scripts';dest.mkdir();shutil.copy2(Path(__file__),dest/Path(__file__).name)
    plot(source,out)


def plot(source,out):
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    data=pd.read_csv(source/'K1_stage_swap_results.csv');data=data[(data.form=='spline')&data.combination.isin(['C00','C11'])]
    ordering=['NeoTRIP_original','NeoTRIP_0','NeoTRIP_1','NeoTRIP_2','ISPY2_original','ISPY2_0','ISPY2_1','ISPY2_2']
    fig,axes=plt.subplots(1,2,figsize=(13,6.3),gridspec_kw={'width_ratios':[1.5,1]})
    for method,color,marker in [('M3','#a33422','o'),('M1','#215d94','s')]:
        for combo,offset,alpha in [('C00',-.14,.42),('C11',.14,1.)]:
            d=data[(data.method==method)&(data.combination==combo)].set_index('run_id')
            for j,run_id in enumerate(ordering):
                if run_id not in d.index:continue
                row=d.loc[run_id];yy=j+offset+(0.07 if method=='M1' else -.07)
                axes[0].errorbar(row.estimate_SD,yy,xerr=[[row.estimate_SD-row.low_SD],[row.high_SD-row.estimate_SD]],fmt=marker,color=color,alpha=alpha,ms=5,capsize=2,
                    label=f'{method} {combo}' if run_id==('ISPY2_original' if method=='M1' else 'NeoTRIP_original') else None)
    axes[0].axvline(0,color='.5',lw=.8);axes[0].set_yticks(range(8),[v.replace('_original',' primary').replace('_',' split ') for v in ordering]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Signed benefit coefficient per original study SD (95% delta CI)')
    axes[0].set_title('A  Matched clinical stabilization');axes[0].legend(loc='upper right',ncol=2,fontsize=8)
    dd=pd.read_csv(source/'K1_parameter_change_decomposition.csv');dd=dd[(dd.form=='spline')&(dd.trial=='ISPY2')]
    dd=dd.copy();dd['display_order']=dd.run_id.map({'ISPY2_original':0,'ISPY2_0':1,'ISPY2_1':2,'ISPY2_2':3})
    dd=dd.sort_values(['method','display_order']);yy=np.arange(len(dd))
    axes[1].barh(yy-.15,dd.outer_phi_change_SD,height=.28,color='#4f7896',label='Outer score contribution')
    axes[1].barh(yy+.15,dd.inner_g_change_SD,height=.28,color='#b65d37',label='Inner g contribution')
    axes[1].set_yticks(yy,[f'{r.method}: '+r.run_id.replace('ISPY2_','').replace('original','primary') for r in dd.itertuples()])
    axes[1].axvline(0,color='.5',lw=.8);axes[1].invert_yaxis();axes[1].legend(fontsize=8)
    axes[1].set_title('B  Exact decomposition of C11 - C00');axes[1].set_xlabel('Change in coefficient per original SD')
    fig.suptitle('CXCL9 beyond IFNG: fixed patients, folds, Q, m and graphs',fontsize=13)
    fig.tight_layout();fig.savefig(out/'K1_clinical_matched_check.svg');fig.savefig(out/'K1_clinical_matched_check.png',dpi=200);plt.close(fig)
    d=pd.read_csv(source/'K2_calibration_summary.csv');methods=['L2_M1_S','L2_M3_S','L1_M3_S','complete_oracle'];colors=['#215d94','#a33422','#699547','#777777']
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5))
    panels=[('A  Null rejection: includes H when B has signal','score_reject',[(85,0,'B'),(85,0,'H'),(85,1,'H'),(241,0,'B'),(241,0,'H'),(241,1,'H')],.05),
        ('B  Fieller coverage of the true B parameter','Fieller_coverage',[(85,0,'B'),(85,1,'B'),(241,0,'B'),(241,1,'B')],.95),
        ('C  Power: nonzero B, same zero-score test','score_reject',[(85,1,'B'),(241,1,'B')],None),
        ('D  Delta coverage of the true B parameter','delta_coverage',[(85,0,'B'),(85,1,'B'),(241,0,'B'),(241,1,'B')],.95)]
    for ax,(title,metric,cells,reference) in zip(axes.flat,panels):
        for mi,(method,color) in enumerate(zip(methods,colors)):
            x=[];y=[];lower=[];upper=[]
            for ci,(n,signal,candidate) in enumerate(cells):
                row=d[(d.n==n)&(d.signal==signal)&(d.candidate==candidate)&(d.method==method)].iloc[0]
                x.append(ci+(mi-1.5)*.13);y.append(row[metric]);lower.append(row[metric]-row[metric+'_Wilson_low']);upper.append(row[metric+'_Wilson_high']-row[metric])
            ax.errorbar(x,y,yerr=[lower,upper],fmt='o',ms=4,capsize=3,color=color,label=method)
        if reference is not None:ax.axhline(reference,color='.5',ls='--',lw=.8)
        ax.set_xticks(range(len(cells)),[f'n={n}\n'+('B signal' if signal else 'constant')+f'\n{candidate}' for n,signal,candidate in cells],fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_ylabel('Rate (Wilson 95% Monte Carlo interval)')
        ax.set_ylim((0,.35) if title.startswith('A') else ((.65,1.01) if 'coverage' in title else (0,1.01)))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('F2: 200 independent datasets per cell; two topologies balanced',fontsize=13)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/'K2_calibration_and_power.svg');fig.savefig(out/'K2_calibration_and_power.png',dpi=200);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k1',type=Path);p.add_argument('--k2',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-tables',type=Path)
    a=p.parse_args()
    if a.source_tables:a.output.mkdir(parents=True,exist_ok=False);plot(a.source_tables,a.output)
    else:run(a.k1,a.k2,a.output)
