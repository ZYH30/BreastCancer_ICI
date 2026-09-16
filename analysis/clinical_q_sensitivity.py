#!/usr/bin/env python3
"""One-Q-component clinical sensitivity. No new graphs, patients or e/m fits."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,shutil,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from tacb_bci.route_ab.candidate_role_lift import CandidateRoleLift
from tacb_bci.route_ab.enhancement import role_specs
from tacb_bci.route_ab.score_surface_roles import require_score_surface_certificate
from tacb_bci.route_ab.conditional_benefit import context_mean
from tacb_bci.route_ab.continuous import sklearn_seed
from tacb_bci.route_ab.core_checks import score_summary,stage_diagnostics
from tacb_bci.causal_ifn.orthogonal_score import aipw
from tacb_bci.provenance import file_digest,write_immutable_json
from tacb_bci.route_ab.nuisance import nested_q_fit_predict
from clinical_stabilization import ROOT,BASE,E3,A11,GENES

K1=Path(os.environ.get('BREAST_ICI_STABILIZATION_RESULTS', ROOT/'private_inputs/clinical_stabilization')).resolve()


def worker(task):
    trial,partition,outroot=task;name=f'{trial}_{partition}';out=Path(outroot)/name;out.mkdir()
    old=K1/name;arr=np.load(old/'LOCAL_arrays.npz',allow_pickle=True)
    w=pd.DataFrame(arr['W'],columns=arr['columns']);a,y,fold=arr['A'],arr['Y'],arr['fold'];n=len(y)
    p=pd.read_csv(BASE/'cxcl9_ifng_benefit_increment_v1'/trial/'patient_scores.csv',float_precision='round_trip')
    oldrec=json.loads((old/'LOCAL_model_records.json').read_text())
    original=partition=='original';forms=['linear','spline'] if original else ['spline']
    methods=['M3','M1'] if trial=='ISPY2' else ['M3'];q=np.zeros((n,2))
    gsaved={method:{form:np.zeros(n) for form in forms} for method in methods}
    saved=dict(W=w.to_numpy(),columns=np.asarray(w.columns),A=a,Y=y,fold=fold)
    records=[];diagnostics=[];checks=[]
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        rec=next(r for r in oldrec if r['fold']==f and r['method']=='M3' and r['form']=='spline')
        np.testing.assert_array_equal(tr,rec['outer_train']);np.testing.assert_array_equal(te,rec['outer_test'])
        if original and trial=='ISPY2':
            graphpath=A11/f'fold{f}_extended_roles.json';z=json.loads(graphpath.read_text());native=z['native']
            required=native['required_candidates'];cl=CandidateRoleLift(native['graph'],required,native['raw_C'],native['raw_S'],available=list(w)).greedy_close()
            qset=set(cl['C'])|set(cl['S_prec']);cols=[c for c in w if c in qset]
            cached=BASE/'two_legal_nuisance_interfaces_v1/ISPY2';ca=np.load(cached/'estimation_arrays.npz')
            cr=json.loads((cached/f'fold{f}.json').read_text());assert cols==cr['Q_columns']
            q[te]=ca['Q'][te];iq=ca[f'fold{f}__Q'];np.testing.assert_array_equal(tr,ca[f'fold{f}__train'])
            qr=cr['Q_fit'];iqr=[r['Q'] for r in cr['inner']]
        else:
            if original:
                graphpath=BASE/'route_a_first_cycle_v1/NeoTRIP'/f'roles_fold{f}.json';z=json.loads(graphpath.read_text());native=z['roles']
                assert z['train_hashes']==p.patient_hash.iloc[tr].tolist() and z['test_hashes']==p.patient_hash.iloc[te].tolist()
                # Missing four added assay nodes remain unresolved, never invented edges.
                required=sorted(set(native['required_candidates'])|{'IFNG','CXCR3','PTPRC','EPCAM'})
                cl=CandidateRoleLift(native['graph'],required,native['raw_C'],native['raw_S'],available=list(w)).greedy_close()
                qset=set(cl['C'])|set(cl['S_prec']);cols=[c for c in w if c in qset]
                innerseed=sklearn_seed(202609150090,10,f)
                outerseeds=[2026091510+f*2+arm for arm in [0,1]]
                innerseeds=lambda j:[sklearn_seed(innerseed,1,j*2+arm) for arm in [0,1]]
            else:
                fit=E3/f'{trial}_split{partition}_all/fit';graphpath=fit/f'fold{f}_graph.json';z=json.loads(graphpath.read_text());native=z['native']
                assert z['train']==tr.tolist() and z['test']==te.tolist()
                required=['CXCL9','IFNG'];spec=role_specs(native['graph'],list(w),required,native['raw_C'],native['raw_S'],native)['M2_candidate_restricted_Q'];cols=spec['Q']
                seed=json.loads((fit.parent/'job_before_results.json').read_text())['seed']
                outerseeds=[sklearn_seed(sklearn_seed(seed,13,f),1,arm) for arm in [0,1]]
                innerseeds=lambda j:[sklearn_seed(sklearn_seed(seed,14,f*3+j),1,arm) for arm in [0,1]]
            qr=[];iq=np.zeros((len(tr),2));iqr=[]
            for arm in [0,1]:
                use=tr[a[tr]==arm];q[te,arm],r=nested_q_fit_predict(w.iloc[use][cols],y[use],w.iloc[te][cols],outerseeds[arm]);qr.append(r)
            for j,split in enumerate(rec['inner_splits']):
                it=np.asarray(split['train']);iv=np.asarray(split['test']);positions=np.searchsorted(tr,iv);models=[]
                for arm in [0,1]:
                    use=it[a[it]==arm];iq[positions,arm],r=nested_q_fit_predict(w.iloc[use][cols],y[use],w.iloc[iv][cols],innerseeds(j)[arm]);models.append(r)
                iqr.append(models)
        for method in methods:
            mr=next(r for r in oldrec if r['fold']==f and r['method']==method and r['form']=='spline')
            inner=np.load(old/f'LOCAL_fold{f}_{method}_inner.npz');np.testing.assert_array_equal(inner['train'],tr)
            # Keep precisely the existing e probabilities / C. Randomized half is
            # a working design assumption, not a certificate from an incomplete graph.
            if trial=='ISPY2':certificate=require_score_surface_certificate(native['graph'],mr['C'],required,cols,available=list(w))
            else:certificate=None;assert np.all(inner['e']==.5) and np.all(arr[method+'__e']==.5)
            phi=aipw(y[tr],a[tr],iq[:,0],iq[:,1],inner['e'])
            for form in forms:
                used=next(r for r in oldrec if r['fold']==f and r['method']==method and r['form']==form)
                g,gm=context_mean(w.IFNG.to_numpy()[tr],phi,w.IFNG.to_numpy()[te],seed=used['g_seed'],form=form)
                gsaved[method][form][te]=g
                records.append(dict(fold=f,method=method,form=form,C=mr['C'],Q=cols,Q_models=qr,inner_Q_models=iqr,g=gm,
                    graph_sha256=file_digest(graphpath),required=required,certificate=certificate,
                    known_e_working_randomization=trial=='NeoTRIP',missing_graph_nodes=sorted(set(required)-set(native['graph'])),
                    e_m_graph_folds_unchanged=True,train=tr.tolist(),test=te.tolist(),g_seed=used['g_seed']))
            np.savez_compressed(out/f'LOCAL_fold{f}_{method}_inner.npz',train=tr,test=te,fold=inner['fold'],Q=iq,
                raw_e=inner['raw_e'],e=inner['e'],phi=phi)
            diagnostics.append(dict(run_id=name,method=method,fold=f,stage='inner',**stage_diagnostics(a[tr],inner['raw_e'],inner['e'],phi)))
            checks.append(dict(run_id=name,fold=f,method=method,e_reused=True,C_reused=True,m_reused=True,graph_reused=True,Q_count=len(cols),previous_Q_count=len(w),certificate=certificate))
    rows=[];sets=[];family=[]
    residuals=np.load(BASE/'ifng_partial_family_audit_v1'/f'{trial}_family_arrays.npz') if original else None
    for method in methods:
        phi=aipw(y,a,q[:,0],q[:,1],arr[method+'__e']);saved[method+'__phi']=phi;saved[method+'__e']=arr[method+'__e']
        for form in forms:
            g=gsaved[method][form];m=arr['m__'+form];saved[method+'__'+form+'__g']=g;saved['m__'+form]=m
            st,fi,parts=score_summary(w.CXCL9.to_numpy(),phi,m,g,w.CXCL9.std(ddof=0))
            row=dict(run_id=name,trial=trial,partition=str(partition),method=method,form=form,**st)
            oldstat,_,_=score_summary(w.CXCL9.to_numpy(),arr[method+'__phi1'],m,arr[method+'__'+form+'__g1'],w.CXCL9.std(ddof=0))
            row.update(baseline_C11_estimate_SD=oldstat['estimate_SD'],change_from_C11_SD=st['estimate_SD']-oldstat['estimate_SD']);rows.append(row)
            sets.append(dict(run_id=name,method=method,form=form,**fi))
            for key,v in parts.items():saved[f'{method}__{form}__{key}']=v
            if original:
                for gene in GENES:
                    b=w[gene].to_numpy();mm=b-residuals[form+'__'+gene+'__rB'];ss,ff,_=score_summary(b,phi,mm,g,b.std())
                    family.append(dict(trial=trial,method=method,form=form,gene=gene,**ss));sets.append(dict(run_id=name,method=method,form=form,gene=gene,**ff))
    saved['Q']=q;np.savez_compressed(out/'LOCAL_arrays.npz',**saved)
    for filename,data in [('results',rows),('family18',family),('diagnostics',diagnostics),('fixed_component_checks',checks)]:pd.DataFrame(data).to_csv(out/(filename+'.csv'),index=False)
    write_immutable_json(out/'full_Fieller_sets.json',sets);write_immutable_json(out/'LOCAL_model_records.json',records)
    write_immutable_json(out/'completion.json',dict(new_patients=0,new_graphs=0,new_e_or_m_fits=0,
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))
    return name


def main(args):
    if not K1.is_dir():raise FileNotFoundError('Set BREAST_ICI_STABILIZATION_RESULTS to the completed clinical stabilization output')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'analysis').glob('*.py'))+list((ROOT/'src').rglob('*.py'))
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write_immutable_json(out/'protocol_before_results.json',dict(command=sys.argv,one_Q_repair=True,
        original_Neo_graph='old49 native nodes; four added required nodes explicitly unresolved; working known e=.5 unchanged, not a graph-based selected-population certificate',
        source_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in sources}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker,(trial,part,str(out))) for trial in ['ISPY2','NeoTRIP'] for part in ['original','0','1','2']]):print('clinical QR complete',f.result(),flush=True)
    for name in ['results','diagnostics','fixed_component_checks']:
        pd.concat([pd.read_csv(p) for p in out.glob('*/'+name+'.csv')],ignore_index=True).to_csv(out/('K1_QR_'+name+'.csv'),index=False)
    d=pd.concat([pd.read_csv(p) for p in out.glob('*_original/family18.csv')],ignore_index=True)
    for _,ix in d.groupby(['trial','method','form']).groups.items():
        for name,col in [('delta','p_two_sided'),('score','p_score')]:d.loc[ix,'BH18_'+name]=multipletests(d.loc[ix,col],method='fdr_bh')[1]
    d.to_csv(out/'K1_QR_family18.csv',index=False);average=[]
    for (form,gene),v in d[d.method=='M3'].groupby(['form','gene']):
        beta=v.estimate_SD.mean();vv=(.5*v.SE_SD.to_numpy())**2;se=np.sqrt(vv.sum());df=vv.sum()**2/np.sum(vv**2/(v.n.to_numpy()-1));cut=student_t.ppf(.975,df)
        average.append(dict(form=form,gene=gene,estimate_SD=beta,SE_SD=se,low_SD=beta-cut*se,high_SD=beta+cut*se,p_delta=2*student_t.sf(abs(beta/se),df),df=df,conjunction_delta_p=v.p_two_sided.max(),conjunction_score_p=v.p_score.max()))
    av=pd.DataFrame(average)
    for _,ix in av.groupby('form').groups.items():
        for name in ['p_delta','conjunction_delta_p','conjunction_score_p']:av.loc[ix,name+'_BH18']=multipletests(av.loc[ix,name],method='fdr_bh')[1]
    av.to_csv(out/'K1_QR_two_study_average.csv',index=False)
    write_immutable_json(out/'completion.json',dict(new_patients=0,new_graphs=0,completed_runs=8,
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.glob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',default=2,type=int);main(p.parse_args())
