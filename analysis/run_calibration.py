#!/usr/bin/env python3
"""K2 preregistered F2, 800 new actual-TLCD full-pipeline datasets."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,shutil,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from tacb_bci.route_ab.enhancement import draw_truth,fit_pipeline
from tacb_bci.route_ab.core_checks import score_summary,stage_diagnostics
from tacb_bci.route_ab.continuous import sklearn_seed
from tacb_bci.route_ab.native import UPSTREAM
from tacb_bci.causal_ifn.orthogonal_score import aipw
from tacb_bci.provenance import file_digest,write_immutable_json

ROOT=Path(__file__).resolve().parents[1]
ENTROPY=202609160601
LAYER_METHODS={'L2':['M1_all_inputs','M3_actual_Q_closure'],'L1':['M3_actual_Q_closure']}
ALL_METHODS=['L2_M1_S','L2_M3_S','L1_M3_S','complete_oracle','true_e_clip_operator']
LABEL_MAP={}
DATASET_PREFIX=''


def worker(task):
    n,signal,rep,root=task
    identity=DATASET_PREFIX+f'F2_n{n}_signal{signal}_rep{rep:04d}'
    out=Path(root)/'datasets'/identity;out.mkdir(parents=True,exist_ok=False)
    key=[ENTROPY,2,n,signal,rep];seed=sklearn_seed(ENTROPY,2000+n,signal*10000+rep)
    d=draw_truth(n,'F2',bool(signal),rep%2,key)
    np.savez_compressed(out/'known_truth.npz',W=d['W'].to_numpy(),columns=np.asarray(d['W'].columns),
        A=d['A'],Y=d['Y'],Q=d['Q'],e=d['e'],tau=d['tau'],g_true=d['g_true'],
        m_B=d['m_true']['B'],m_H=d['m_true']['H'],seed_key=np.asarray(key))
    write_immutable_json(out/'truth_definition.json',dict(graph=d['graph'],truth=d['truth'],
        population_SD=d['population_SD'],residual_variance=d['residual_variance'],seed_key=key,
        topology=rep%2,fit_seed=seed,truth_method='unchanged 128-node Gaussian quadrature'))
    tick=time.perf_counter();rows=[];sets=[];diagnostics=[];errors=[];extra={}
    try:
        _,roles,failures=fit_pipeline(d['W'],d['A'],d['Y'],['B','H'],'G',out/'fit',seed,
            true=d,layers=tuple(LAYER_METHODS),product=True,selected_methods=LAYER_METHODS['L2'],
            layer_methods=LAYER_METHODS,propensity_clip=.05,save_inner_details=True)
        errors+=failures
        fitted=np.load(out/'fit/arrays.npz');records=json.loads((out/'fit/model_records.json').read_text())
        for layer,methods in LAYER_METHODS.items():
            for method in methods:
                label=LABEL_MAP.get((layer,method),layer+('_M1_S' if method.startswith('M1') else '_M3_S'));prefix=layer+'__'+method+'__'
                if prefix+'phi' not in fitted:continue
                q,e,phi,g=[fitted[prefix+k] for k in ['Q','e','phi','g']]
                selected=[r for r in records if r.get('layer')==layer and r.get('method')==method and r['stage']=='Q_e_g']
                raw_e=np.zeros(n)
                for r in selected:
                    f=r['fold'];test=np.asarray(r['outer']['test_rows']);raw_e[test]=r['outer']['score_probability_control']['raw_predictions']
                    z=np.load(out/f'fit/fold{f}_{layer}_{method}_inner.npz')
                    diagnostics.append(dict(method=label,stage='inner',fold=f,
                        **stage_diagnostics(d['A'][z['train']],z['raw_e'],z['e'],z['phi']),
                        g_alpha=r['g']['selected_alpha']))
                extra[label+'__raw_e']=raw_e
                diagnostics.append(dict(method=label,stage='outer',fold=-1,**stage_diagnostics(d['A'],raw_e,e,phi,g)))
                role=[r for r in roles if r['layer']==layer and r['method']==method]
                for candidate in ['B','H']:
                    row,fi,parts=score_summary(d['W'][candidate],phi,fitted['m__'+candidate],g,
                        d['population_SD'][candidate],d['truth'][candidate])
                    row.update(method=label,candidate=candidate,
                        graph_valid_all_folds=all(r['true_graph_actual_Q_valid'] for r in role),
                        graph_valid_fold_fraction=np.mean([r['true_graph_actual_Q_valid'] for r in role]),
                        C_size=np.mean([len(r['C']) for r in role]),Q_MSE=np.mean((q-d['Q'])**2),
                        e_MSE=np.mean((e-d['e'])**2),m_MSE=np.mean((fitted['m__'+candidate]-d['m_true'][candidate])**2),
                        g_MSE=np.mean((g-d['g_true'])**2),true_e_outside_bounds=np.mean((d['e']<.05)|(d['e']>.95)))
                    rows.append(row);sets.append(dict(method=label,candidate=candidate,**fi))
    except Exception as exc:
        errors.append(dict(stage='pipeline',error=str(exc),traceback=traceback.format_exc()))
    for label,e in [('complete_oracle',d['e']),('true_e_clip_operator',np.clip(d['e'],.05,.95))]:
        phi=aipw(d['Y'],d['A'],d['Q'][:,0],d['Q'][:,1],e)
        extra[label+'__phi']=phi
        for candidate in ['B','H']:
            row,fi,parts=score_summary(d['W'][candidate],phi,d['m_true'][candidate],d['g_true'],
                d['population_SD'][candidate],d['truth'][candidate])
            row.update(method=label,candidate=candidate)
            rows.append(row);sets.append(dict(method=label,candidate=candidate,**fi))
            for field,value in parts.items():extra[label+'__'+candidate+'__'+field]=value
    existing={(r['method'],r['candidate']) for r in rows}
    for method in ALL_METHODS:
        for candidate in ['B','H']:
            if (method,candidate) not in existing:
                rows.append(dict(method=method,candidate=candidate,fit_status='FAILED',interval_available=False,
                    truth=d['truth'][candidate],truth_SD=d['truth'][candidate]*d['population_SD'][candidate],
                    failure_reason=';'.join(e['error'] for e in errors)))
    for row in rows:row.update(dataset_id=identity,n=n,signal=signal,replicate=rep,topology=rep%2,seed_key=str(key))
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'stage_diagnostics.csv',index=False)
    np.savez_compressed(out/'additional_arrays.npz',**extra)
    write_immutable_json(out/'full_Fieller_sets.json',sets);write_immutable_json(out/'failures.json',errors)
    write_immutable_json(out/'completion.json',dict(dataset_id=identity,elapsed_seconds=time.perf_counter()-tick,
        planned_method_candidate_rows=2*len(ALL_METHODS),success_rows=sum(r['fit_status']=='SUCCESS' for r in rows),
        failed_rows=sum(r['fit_status']!='SUCCESS' for r in rows),
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))
    return rows,errors


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'analysis').glob('*.py')) + [ROOT/'docs/reproducibility.md']
    sources+=list((ROOT/'src/tacb_bci/route_ab').glob('*.py'))
    sources+=[ROOT/'src/tacb_bci/causal_ifn/orthogonal_score.py',ROOT/'src/tacb_bci/provenance.py']
    sources+=getattr(args,'extra_sources',[])
    for path in sources:
        target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    tasks=[(n,sig,rep,str(out)) for rep in range(args.reps) for n in [85,241] for sig in [0,1]]
    write_immutable_json(out/'protocol_before_results.json',dict(command=sys.argv,entropy=ENTROPY,
        planned_datasets=len(tasks),per_cell=args.reps,topology='rep%2 balanced',family='F2',p=10,
        signal0_ATE=.12,probability_bound=.05,no_g_projection=True,methods=ALL_METHODS,
        clinical_patients=0,independent_of_old_entropy=202609160101,
        extra_protocol=getattr(args,'extra_protocol',{}),
        source_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in sources},
        upstream_commit='993ed47266539545b7eb14249f3e3ccfd75d1f25',
        upstream_sha256={str(p.relative_to(UPSTREAM)):file_digest(p) for p in UPSTREAM.rglob('*.py')}))
    pd.DataFrame([dict(dataset_id=DATASET_PREFIX+f'F2_n{n}_signal{s}_rep{r:04d}',n=n,signal=s,replicate=r,topology=r%2,seed_key=str([ENTROPY,2,n,s,r]))
                  for n,s,r,_ in tasks]).to_csv(out/'planned_datasets.csv',index=False)
    truth=draw_truth(10,'F2',True,0,[ENTROPY,999])['truth']['B']
    integral,error=quad(lambda r:.20*r*np.tanh(r)*norm.pdf(r,scale=.8)/.64,-np.inf,np.inf,epsabs=1e-13,epsrel=1e-13)
    assert abs(integral-truth)<1e-11
    write_immutable_json(out/'independent_truth_check.json',dict(Hermite128=truth,adaptive_integral=integral,
        integral_absolute_error_estimate=error,absolute_difference=abs(integral-truth),population_SD_B=np.sqrt(.89)))
    tick=time.perf_counter();rows=[];done_count=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,t):t for t in tasks}
        for future in as_completed(pending):
            rr,ff=future.result();rows+=rr;done_count+=1
            print('K2',done_count,'/',len(tasks),pending[future][:3],'failures',len(ff),'elapsed',round(time.perf_counter()-tick,1),flush=True)
    pd.DataFrame(rows).to_csv(out/'all_results.csv',index=False)
    write_immutable_json(out/'completion.json',dict(planned_datasets=len(tasks),generated_datasets=done_count,
        result_rows=len(rows),elapsed_seconds=time.perf_counter()-tick,
        success_rows=sum(r['fit_status']=='SUCCESS' for r in rows),
        dataset_completion_sha256={str(p.relative_to(out)):file_digest(p) for p in out.glob('datasets/*/completion.json')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reps',type=int,default=200);parser.add_argument('--workers',type=int,default=6)
    main(parser.parse_args())
