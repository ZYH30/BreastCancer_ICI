#!/usr/bin/env python3
"""Dataset-denominator K2 calibration, paired comparisons and complete families."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tacb_bci.provenance import file_digest,write_immutable_json


def binary_rate(values):
    x=np.asarray(values,float);n=len(x);count=int(x.sum());p=float(x.mean());z=norm.ppf(.975)
    den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(count=count,denominator=n,rate=p,MCSE=np.sqrt(p*(1-p)/n),Wilson_low=center-half,Wilson_high=center+half)


def summarize(data,planned,out):
    data=data.copy();fdp=[]
    for (identity,method),idx in data.groupby(['dataset_id','method']).groups.items():
        d=data.loc[idx];assert sorted(d.candidate)==['B','H']
        for name,col in [('score','p_score'),('delta','p_two_sided')]:
            valid=d.fit_status.eq('SUCCESS') & d[col].notna()
            # p=1 exists only internally for complete-family non-discovery.
            p=np.where(valid,d[col],1.);q=multipletests(p,method='fdr_bh')[1]
            data.loc[idx,'BH2_'+name]=np.where(valid,q,np.nan)
            discoveries=q<.05;false=discoveries & np.isclose(d.truth.to_numpy(),0,atol=1e-12)
            fdp.append(dict(dataset_id=identity,n=d.n.iloc[0],signal=d.signal.iloc[0],topology=d.topology.iloc[0],
                method=method,inference=name,family_size=2,valid_p_values=int(valid.sum()),
                discoveries=int(discoveries.sum()),false_discoveries=int(false.sum()),
                FDP=float(false.sum()/max(1,discoveries.sum())),any_false=bool(false.any())))
    family=pd.DataFrame(fdp);family.to_csv(out/'K2_family_FDP.csv',index=False)
    family_summary=[]
    for keys,d in family.groupby(['n','signal','method','inference']):
        row=dict(zip(['n','signal','method','inference'],keys));den=len(planned[(planned.n==keys[0])&(planned.signal==keys[1])])
        assert len(d)==den
        row.update(N_planned=den,N_generated=d.dataset_id.nunique(),FDR=d.FDP.mean(),FDP_MCSE=d.FDP.std(ddof=1)/np.sqrt(len(d)),
            mean_discoveries=d.discoveries.mean(),mean_valid_p_values=d.valid_p_values.mean())
        row.update({'FWER_'+k:v for k,v in binary_rate(d.any_false).items()});family_summary.append(row)
    pd.DataFrame(family_summary).to_csv(out/'K2_family_calibration.csv',index=False)
    data.to_csv(out/'K2_dataset_results.csv',index=False)
    summary=[];rates=[]
    keys=['n','signal','method','candidate']
    for key,d in data.groupby(keys):
        ident=dict(zip(keys,key));den=len(planned[(planned.n==key[0])&(planned.signal==key[1])]);assert len(d)==den
        ok=d[d.fit_status.eq('SUCCESS')];available=ok[ok.p_score.notna()];bounded=available[available.bounded.eq(True)]
        row=dict(**ident,N_planned=den,N_generated=d.dataset_id.nunique(),N_point_success=len(ok),N_interval_available=len(available),
            N_bounded=len(bounded),N_failed=den-len(ok),truth=float(d.truth.iloc[0]))
        if len(ok):
            err=ok.estimate-ok.truth
            row.update(bias=err.mean(),bias_MCSE=err.std(ddof=1)/np.sqrt(len(err)),RMSE=np.sqrt(np.mean(err**2)),
                absolute_error_q90=np.quantile(abs(err),.9),absolute_error_q99=np.quantile(abs(err),.99),maximum_absolute_error=np.max(abs(err)),
                mean_SE=ok.se.mean(),mean_D=ok.D.mean(),mean_residual_square_ESS=ok.residual_square_ESS.mean(),
                bounded_mean_length=bounded.fieller_length.mean(),bounded_median_length=bounded.fieller_length.median())
            for c in ['Q_MSE','e_MSE','m_MSE','g_MSE','graph_valid_all_folds','graph_valid_fold_fraction','C_size','true_e_outside_bounds']:
                if c in ok:row['mean_'+c]=ok[c].mean()
        for kind in ['bounded','all','empty','halfline','outer']:row['Fieller_'+kind+'_count']=int((available.fieller_kind==kind).sum())
        for name,values,population in [
            ('score_reject',available.p_score<.05,'successful_interval'),
            ('delta_reject',available.p_two_sided<.05,'successful_interval'),
            ('score_BH_reject',available.BH2_score<.05,'successful_interval'),
            ('delta_BH_reject',available.BH2_delta<.05,'successful_interval'),
            ('Fieller_coverage',available.covers_fieller,'successful_interval'),
            ('delta_coverage',available.covers_delta,'successful_interval'),
            ('bounded',available.bounded,'successful_interval'),
            ('covered_and_bounded',available.covered_and_bounded,'successful_interval'),
            ('score_discovery_output',d.p_score.fillna(1)<.05,'all_planned'),
            ('delta_discovery_output',d.p_two_sided.fillna(1)<.05,'all_planned'),
            ('score_BH_discovery_output',d.BH2_score.fillna(1)<.05,'all_planned'),
            ('covered_bounded_output',d.covered_and_bounded.fillna(False),'all_planned'),
            ('failed_output',~d.fit_status.eq('SUCCESS'),'all_planned')]:
            if not len(values):continue
            rr=binary_rate(values);rates.append(dict(**ident,metric=name,population=population,**rr))
            row[name]=rr['rate'];row[name+'_MCSE']=rr['MCSE'];row[name+'_Wilson_low']=rr['Wilson_low'];row[name+'_Wilson_high']=rr['Wilson_high']
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'K2_calibration_summary.csv',index=False)
    pd.DataFrame(rates).to_csv(out/'K2_binary_rate_details.csv',index=False)
    paired=[]
    comparators=['L2_M1_S','L1_M3_S','complete_oracle']
    work=data.copy();work['squared_error']=(work.estimate-work.truth)**2
    work['score_reject']=(work.p_score<.05).astype(float);work['delta_reject']=(work.p_two_sided<.05).astype(float)
    for key,d in work[work.fit_status=='SUCCESS'].groupby(['n','signal','candidate']):
        for metric in ['squared_error','estimate','covers_fieller','covers_delta','score_reject','delta_reject','fieller_length']:
            pivot=d.pivot(index='dataset_id',columns='method',values=metric)
            if 'L2_M3_S' not in pivot:continue
            for comparator in comparators:
                if comparator not in pivot:continue
                vals=pivot[['L2_M3_S',comparator]].dropna().astype(float)
                if metric=='fieller_length':vals=vals[np.isfinite(vals).all(axis=1)]
                dif=vals['L2_M3_S']-vals[comparator]
                paired.append(dict(zip(['n','signal','candidate'],key),metric=metric,comparison='L2_M3_S_minus_'+comparator,
                    paired_success=len(dif),mean_difference=dif.mean(),paired_MCSE=dif.std(ddof=1)/np.sqrt(len(dif))))
    pd.DataFrame(paired).to_csv(out/'K2_paired_comparisons.csv',index=False)
    return pd.DataFrame(summary)


def main(args):
    root=args.input.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not (root/'completion.json').exists():raise ValueError('full_registered_batch_required_no_interim_selection')
    d=pd.read_csv(root/'all_results.csv',float_precision='round_trip');planned=pd.read_csv(root/'planned_datasets.csv')
    result=summarize(d,planned,out)
    folds=[];diagnostics=[];failures=[];index=[]
    for path in sorted(root.glob('datasets/*')):
        meta=json.loads((path/'truth_definition.json').read_text());identity=path.name;n=int(identity.split('_n')[1].split('_')[0]);signal=int(identity.split('_signal')[1].split('_')[0])
        for failure in json.loads((path/'failures.json').read_text()):failures.append(dict(dataset_id=identity,**failure))
        if (path/'fit/roles.json').exists():
            for role in json.loads((path/'fit/roles.json').read_text()):
                folds.append(dict(dataset_id=identity,n=n,signal=signal,topology=meta['topology'],**role))
        if (path/'stage_diagnostics.csv').exists():
            dd=pd.read_csv(path/'stage_diagnostics.csv');dd['dataset_id']=identity;dd['n']=n;dd['signal']=signal;diagnostics.append(dd)
        index.append(dict(dataset_id=identity,seed_key=str(meta['seed_key']),fit_seed=meta['fit_seed'],topology=meta['topology'],
            known_truth_sha256=file_digest(path/'known_truth.npz'),completion_sha256=file_digest(path/'completion.json')))
    pd.DataFrame(folds).to_csv(out/'K2_fold_roles.csv',index=False)
    if diagnostics:pd.concat(diagnostics,ignore_index=True).to_csv(out/'K2_nuisance_diagnostics.csv',index=False)
    pd.DataFrame(failures,columns=sorted(set(k for row in failures for k in row)) or ['dataset_id','stage','error','traceback']).to_csv(out/'K2_failure_and_root_cause.csv',index=False)
    pd.DataFrame(index).to_csv(out/'K2_dataset_identity.csv',index=False)
    write_immutable_json(out/'completion.json',dict(distinct_datasets=len(index),failure_records=len(failures),
        source_completion_sha256=file_digest(root/'completion.json'),
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))
    print(result[['n','signal','method','candidate','N_point_success','RMSE','score_reject','Fieller_coverage','delta_coverage']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
