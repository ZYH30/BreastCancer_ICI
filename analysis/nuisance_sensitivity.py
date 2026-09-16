#!/usr/bin/env python3
"""K2 Plan B diagnosis on completed data: strictly one nuisance at a time.

No clinical result is fitted here. Replacements are not implementable methods.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from tacb_bci.route_ab.core_checks import score_summary
from tacb_bci.route_ab.conditional_benefit import context_mean
from tacb_bci.causal_ifn.orthogonal_score import aipw
from tacb_bci.route_ab.continuous import sklearn_seed
from tacb_bci.provenance import write_immutable_json,file_digest

ROOT=Path(__file__).resolve().parents[1]


def worker(task):
    source,destination,method_labels=task;source=Path(source);out=Path(destination)/source.name;out.mkdir()
    t=np.load(source/'known_truth.npz',allow_pickle=True);meta=json.loads((source/'truth_definition.json').read_text())
    fit=np.load(source/'fit/arrays.npz');w=pd.DataFrame(t['W'],columns=t['columns']);a,y=t['A'],t['Y'];seed=meta['fit_seed']
    n=len(a);sig=int(source.name.split('_signal')[1].split('_')[0]);rows=[];sets=[];saved={};models=[]
    for label in method_labels:
        layer=label[:2];method='M1_all_inputs' if 'M1' in label else 'M3_actual_Q_closure';prefix=layer+'__'+method+'__'
        q,e,phi,g=[fit[prefix+field] for field in ['Q','e','phi','g']]
        matched={kind:np.zeros(n) for kind in ['only_e_true_matched_g','only_Q_true_matched_g']}
        for fold in range(3):
            z=np.load(source/f'fit/fold{fold}_{layer}_{method}_inner.npz');tr,te=z['train'],z['test']
            for kind in matched:
                iq=z['Q'] if kind.startswith('only_e') else t['Q'][tr]
                ie=t['e'][tr] if kind.startswith('only_e') else z['e']
                ip=aipw(y[tr],a[tr],iq[:,0],iq[:,1],ie)
                matched[kind][te],record=context_mean(w.G.to_numpy()[tr],ip,w.G.to_numpy()[te],seed=sklearn_seed(seed,15,fold),form='spline')
                models.append(dict(method=label,diagnostic=kind,fold=fold,g=record))
                saved[f'{label}__{kind}__fold{fold}_inner_phi']=ip
        configurations=[('baseline',phi,g,None),
            ('only_e_true_matched_g',aipw(y,a,q[:,0],q[:,1],t['e']),matched['only_e_true_matched_g'],None),
            ('only_Q_true_matched_g',aipw(y,a,t['Q'][:,0],t['Q'][:,1],e),matched['only_Q_true_matched_g'],None),
            ('only_m_true',phi,g,'true'),('only_g_true',phi,t['g_true'],None),
            ('learned_Q_true_context_unclipped_true_e',aipw(y,a,q[:,0],q[:,1],t['e']),t['g_true'],'true'),
            ('learned_Q_true_context_clipped_true_e',aipw(y,a,q[:,0],q[:,1],np.clip(t['e'],.05,.95)),t['g_true'],'true')]
        # Exact conditional mean score-bias uses the same fitted Q/e, not refits.
        conditional=(t['e']-e)*((t['Q'][:,1]-q[:,1])/e+(t['Q'][:,0]-q[:,0])/(1-e))
        saved[label+'__conditional_score_bias']=conditional
        for name,pp,gg,mm in configurations:
            saved[label+'__'+name+'__phi']=pp;saved[label+'__'+name+'__g']=gg
            for candidate in ['B','H']:
                m=t['m_'+candidate] if mm=='true' else fit['m__'+candidate]
                row,fi,parts=score_summary(w[candidate],pp,m,gg,meta['population_SD'][candidate],meta['truth'][candidate])
                r=w[candidate].to_numpy()-m
                row.update(dataset_id=source.name,n=n,signal=sig,topology=meta['topology'],candidate=candidate,
                    method=label,diagnostic=name,conditional_bias_projection=np.mean(r*conditional)/np.mean(r*r),
                    mean_m_error_squared=np.mean((m-t['m_'+candidate])**2),mean_g_error_squared=np.mean((gg-t['g_true'])**2))
                rows.append(row);sets.append(dict(method=label,diagnostic=name,candidate=candidate,**fi))
                for k,v in parts.items():saved[label+'__'+name+'__'+candidate+'__'+k]=v
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False);np.savez_compressed(out/'arrays.npz',**saved)
    write_immutable_json(out/'full_Fieller_sets.json',sets);write_immutable_json(out/'g_models.json',models)
    write_immutable_json(out/'completion.json',dict(new_datasets=0,source_known_truth_sha256=file_digest(source/'known_truth.npz'),
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))
    return rows


def main(args):
    source=args.input.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not (source/'completion.json').exists():raise ValueError('finish primary batch before root-cause analysis')
    paths=sorted(source.glob('datasets/*'))
    write_immutable_json(out/'protocol_before_results.json',dict(reason=args.reason,methods=args.methods,
        completed_source=str(source),source_completion_sha256=file_digest(source/'completion.json'),
        single_component_replacements=True,only_Q_or_e_retrains_matched_g=True,new_datasets=0,
        methods_not_clinically_implementable=True,source_sha256=file_digest(Path(__file__))))
    shutil.copy2(Path(__file__),out/Path(__file__).name);rows=[];tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,(str(path),str(out),args.methods)):path.name for path in paths}
        for idx,future in enumerate(as_completed(pending)):
            rows+=future.result()
            if (idx+1)%20==0:print('component diagnosis',idx+1,'/',len(paths),'elapsed',round(time.perf_counter()-tick,1),flush=True)
    data=pd.DataFrame(rows);data.to_csv(out/'all_results.csv',index=False);summary=[]
    for keys,d in data.groupby(['n','signal','method','diagnostic','candidate']):
        err=d.estimate-d.truth
        summary.append(dict(zip(['n','signal','method','diagnostic','candidate'],keys),datasets=len(d),bias=err.mean(),
            RMSE=np.sqrt(np.mean(err**2)),score_reject=(d.p_score<.05).mean(),delta_reject=(d.p_two_sided<.05).mean(),
            Fieller_coverage=d.covers_fieller.mean(),delta_coverage=d.covers_delta.mean(),
            mean_m_MSE=d.mean_m_error_squared.mean(),mean_g_MSE=d.mean_g_error_squared.mean()))
    pd.DataFrame(summary).to_csv(out/'component_summary.csv',index=False)
    write_immutable_json(out/'completion.json',dict(reused_datasets=len(paths),new_datasets=0,
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['L1_M3_S','L2_M3_S','L2_M1_S']);p.add_argument('--reason',required=True)
    p.add_argument('--workers',type=int,default=6);main(p.parse_args())
