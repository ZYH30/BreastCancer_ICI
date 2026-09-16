#!/usr/bin/env python3
"""K1: saved-fold clinical replay and paired inner/outer score stabilization."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,shutil,sys,time,hashlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold,KFold
from tacb_bci.route_ab.enhancement import fit_nuisances
from tacb_bci.route_ab.native import fit_propensity_adapter
from tacb_bci.route_ab.score_surface_roles import score_surface_closure
from tacb_bci.route_ab.conditional_benefit import context_mean
from tacb_bci.route_ab.continuous import sklearn_seed
from tacb_bci.route_ab.core_checks import score_summary,stage_diagnostics
from tacb_bci.causal_ifn.orthogonal_score import aipw
from tacb_bci.provenance import file_digest,write_immutable_json
from tacb_bci.route_ab.nuisance import nested_q_fit_predict
from tacb_bci.route_ab.constants import GENES

ROOT=Path(__file__).resolve().parents[1]
CACHE_ROOT=Path(os.environ.get('BREAST_ICI_CLINICAL_CACHE', ROOT/'private_inputs')).resolve()
BASE=CACHE_ROOT/'results/route_ab'
E3=CACHE_ROOT/'results/route_a_enhancement/E3_full_spline_stability_v1'
A13=BASE/'role_calibrated_final_family_v1';A11=BASE/'ispy2_extended_role_context_v2'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run_one(task):
    trial,partition,output=task;original=partition=='original'
    identity=trial+'_'+str(partition);out=Path(output)/identity;out.mkdir()
    source=BASE/'cxcl9_ifng_benefit_increment_v1'/trial
    w=pd.read_csv(source/'extended_W.csv',float_precision='round_trip')
    p=pd.read_csv(source/'patient_scores.csv',float_precision='round_trip')
    a=p.arm.to_numpy(int);y=p.y.to_numpy(int);b=w.CXCL9.to_numpy();z=w.IFNG.to_numpy();n=len(y);sd=b.std()
    forms=['linear','spline'] if original else ['spline']
    methods=['M3','M1'] if trial=='ISPY2' else ['M3']
    if original:
        old=pd.read_csv(BASE/'common_nuisance_controls_v1'/trial/'patient_scores.csv',float_precision='round_trip').set_index('patient_hash').loc[p.patient_hash]
        fold=old.fold.to_numpy(int)
        residuals=np.load(BASE/'ifng_partial_family_audit_v1'/f'{trial}_family_arrays.npz')
        m={form:b-residuals[form+'__CXCL9__rB'] for form in forms}
        quals=np.load(A13/'qualification_arrays.npz')
        q=np.column_stack([p.q0_new,p.q1_new])
        e=np.load(A11/'all_arrays.npz')['e_closed'] if trial=='ISPY2' else np.full(n,.5)
        oldg={form:quals[f'{trial}__{form}__g'] for form in forms}
        phi_old=quals[trial+'__phi']
        oldnested=json.loads((BASE/'cxcl9_nonlinear_IFNG_context_v1'/trial/'nested_models.json').read_text())
        seed=None
    else:
        cache=E3/f'{trial}_split{partition}_all';meta=json.loads((cache/'job_before_results.json').read_text())
        axis=pd.read_csv(cache/'LOCAL_patient_axis.csv');assert axis.patient_hash.tolist()==p.patient_hash.tolist()
        np.testing.assert_array_equal(axis[['arm','y']],p[['arm','y']])
        seed=meta['seed'];fitroot=cache/'fit';saved=np.load(fitroot/'arrays.npz',allow_pickle=True)
        np.testing.assert_allclose(saved['W'],w.to_numpy(),atol=1e-12,rtol=0)
        assert list(saved['columns'])==list(w)
        fold=saved['fold'];m={'spline':saved['m__CXCL9']}
        prefix='L2__M3_actual_Q_closure__'
        q,e,phi_old=[saved[prefix+k] for k in ['Q','e','phi']];oldg={'spline':saved[prefix+'g']}
        oldrecords=json.loads((fitroot/'model_records.json').read_text())
        oldroles=json.loads((fitroot/'roles.json').read_text())
    replays=[];rows=[];diagnostics=[];decomp=[];ratios=[];records=[];arrays=dict(W=w.to_numpy(),columns=np.asarray(w.columns),A=a,Y=y,fold=fold,Q=q)
    model={method:dict(e=e.copy(),g0={f:np.zeros(n) for f in forms},g1={f:np.zeros(n) for f in forms}) for method in methods}
    def check(name,actual,expected):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
        replays.append(dict(run_id=identity,field=name,max_absolute_error=error,passed=True))
    check('original_outer_phi',aipw(y,a,q[:,0],q[:,1],e),phi_old)
    write_immutable_json(out/'identity_before_results.json',dict(run_id=identity,trial=trial,partition=partition,
        n=n,SD=sd,SD_definition='original empirical gene SD ddof=0',g_projection='none',
        patient_axis_hash=digest(p.patient_hash.tolist()),outer_fold_hash=digest(fold.tolist()),
        input_sha256={str(path.relative_to(CACHE_ROOT)):file_digest(path) for path in [source/'extended_W.csv',source/'patient_scores.csv']},
        old_primary_unchanged=True,new_patients=0))
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);cachefit={}
        if original:
            rec0=oldnested[f]
            assert rec0['train_hashes']==p.patient_hash.iloc[tr].tolist() and rec0['test_hashes']==p.patient_hash.iloc[te].tolist()
            if trial=='ISPY2':
                rolepath=A11/f'fold{f}_extended_roles.json';role=json.loads(rolepath.read_text())
                assert role['train_hashes']==p.patient_hash.iloc[tr].tolist() and role['test_hashes']==p.patient_hash.iloc[te].tolist()
                spec=score_surface_closure(role['native']['graph'],role['native']['required_candidates'],list(w),role['native']['raw_C'],role['native']['raw_S'],available=list(w))
                assert spec['C']==role['closed']['C'];c=spec['C'];inner_records=role['inner_training_phi']
                old_g_records={r['form']:r for r in role['g_models']}
            else:
                rolepath=None;c=[];inner_records=rec0['inner_training_phi_only']
                old_g_records={r['form']:r['g'] for r in rec0['background_models']}
            inner_seed=sklearn_seed(202609150090,11 if trial=='ISPY2' else 10,f)
            gs=sklearn_seed(202609150090,21,f)
            generated=list(StratifiedKFold(3,shuffle=True,random_state=inner_seed).split(tr,2*a[tr]+y[tr]))
            iq=np.zeros((len(tr),2));ifolds=np.full(len(tr),-1);qrecords=[]
            for j,ir in enumerate(inner_records):
                it=np.asarray(ir['train']);iv=np.asarray(ir['test'])
                np.testing.assert_array_equal(it,generated[j][0]);np.testing.assert_array_equal(iv,generated[j][1]);ifolds[iv]=j
                qr=[]
                for arm in [0,1]:
                    use=tr[it[a[tr[it]]==arm]]
                    iq[iv,arm],rr=nested_q_fit_predict(w.iloc[use],y[use],w.iloc[tr[iv]],sklearn_seed(inner_seed,1,j*2+arm));qr.append(rr)
                # Preserve exact fitting records, not just matching predictions.
                assert digest(qr)==digest(ir['Q'])
                qrecords.append(qr)
        else:
            rolepath=fitroot/f'fold{f}_graph.json';graph=json.loads(rolepath.read_text())
            np.testing.assert_array_equal(tr,graph['train']);np.testing.assert_array_equal(te,graph['test'])
            spec=next(r for r in oldroles if r['fold']==f and r['method']=='M3_actual_Q_closure')
            c=spec['C'];oldrec=next(r for r in oldrecords if r['fold']==f and r['stage']=='Q_e_g')
            old_g_records={'spline':oldrec['g']};inner_records=oldrec['inner'];gs=sklearn_seed(seed,15,f)
            inner_seed=sklearn_seed(seed,11,f)
            generated=list(StratifiedKFold(3,shuffle=True,random_state=inner_seed).split(tr,2*a[tr]+y[tr]))
            iq=np.zeros((len(tr),2));ifolds=np.full(len(tr),-1);qrecords=[]
            for j,ir in enumerate(inner_records):
                it,iv=generated[j];np.testing.assert_array_equal(tr[it],ir['train_rows']);np.testing.assert_array_equal(tr[iv],ir['test_rows']);ifolds[iv]=j
                qq,ee,rr=fit_nuisances(w,a,y,tr[it],tr[iv],spec,sklearn_seed(seed,14,f*3+j),False,cachefit,.5 if trial=='NeoTRIP' else None)
                iq[iv]=qq;assert digest(rr['Q'])==digest(ir['Q']);qrecords.append(rr['Q'])
        for method in methods:
            cc=c if method=='M3' else sorted(w)
            if method=='M1':
                frame=w.iloc[tr].copy();frame['T']=a[tr];frame['Y']=y[tr]
                es=2026091511+f if original else sklearn_seed(sklearn_seed(seed,13,f),2)
                em,er=fit_propensity_adapter(frame,cc,seed=es)
                model[method]['e'][te]=em.predict(w.iloc[te][sorted(cc)])
            ie=np.zeros(len(tr));ers=[]
            for j,(it,iv) in enumerate(generated):
                if trial=='NeoTRIP':ie[iv]=.5;er=dict(known_working_e=.5)
                elif original:
                    frame=w.iloc[tr[it]].copy();frame['T']=a[tr[it]];frame['Y']=y[tr[it]]
                    em,er=fit_propensity_adapter(frame,cc,seed=sklearn_seed(inner_seed,2,j))
                    ie[iv]=em.predict(w.iloc[tr[iv]][sorted(cc)])
                    if method=='M3':assert digest(er)==digest(inner_records[j]['e'])
                else:
                    sp=dict(spec,C=cc)
                    qq,ee,rr=fit_nuisances(w,a,y,tr[it],tr[iv],sp,sklearn_seed(seed,14,f*3+j),False,cachefit,None)
                    check(f'{method}_fold{f}_inner{j}_Q_shared',qq,iq[iv]);ie[iv]=ee;er=rr['e']
                    if method=='M3':assert digest(er)==digest(inner_records[j]['e'])
                ers.append(er)
            inner_phi0=aipw(y[tr],a[tr],iq[:,0],iq[:,1],ie)
            inner_phi1=aipw(y[tr],a[tr],iq[:,0],iq[:,1],np.clip(ie,.05,.95))
            if not original and method=='M3':
                oldinner=np.load(fitroot/f'fold{f}_L2_M3_actual_Q_closure_inner.npz')
                check(f'fold{f}_inner_phi',inner_phi0,oldinner['phi']);np.testing.assert_array_equal(tr,oldinner['train'])
            for form in forms:
                gg0,gr0=context_mean(z[tr],inner_phi0,z[te],seed=gs,form=form)
                gg1,gr1=context_mean(z[tr],inner_phi1,z[te],seed=gs,form=form)
                if method=='M3':
                    check(f'fold{f}_{form}_g',gg0,oldg[form][te]);assert digest(gr0)==digest(old_g_records[form])
                model[method]['g0'][form][te]=gg0;model[method]['g1'][form][te]=gg1
                records.append(dict(method=method,fold=f,form=form,g_seed=gs,g0=gr0,g1=gr1,
                    g_selection_indices=[dict(train=t.tolist(),test=v.tolist()) for t,v in KFold(3,shuffle=True,random_state=gs).split(tr)],
                    outer_train=tr.tolist(),outer_test=te.tolist(),inner_splits=[dict(train=tr[t].tolist(),test=tr[v].tolist()) for t,v in generated],
                    C=cc,actual_Q_columns=list(w) if original else sorted(w),actual_Q_raw_sources=list(w),derived_Q_columns=[],
                    graph_sha256=file_digest(rolepath) if rolepath else 'inherited_randomized_half_no_new_graph',
                    Q_models=qrecords,e_models=ers,inner_probability_clip=.05,g_projection='none'))
            for label,ev,pv in [('raw',ie,inner_phi0),('stabilized',np.clip(ie,.05,.95),inner_phi1)]:
                diagnostics.append(dict(run_id=identity,trial=trial,method=method,stage='inner',fold=f,version=label,
                    **stage_diagnostics(a[tr],ie,ev,pv)))
            np.savez_compressed(out/f'LOCAL_fold{f}_{method}_inner.npz',train=tr,test=te,fold=ifolds,Q=iq,raw_e=ie,
                e=np.clip(ie,.05,.95),phi0=inner_phi0,phi1=inner_phi1)
        print('K1 reconstructed',identity,'fold',f,flush=True)
    for method,obj in model.items():
        phi0=aipw(y,a,q[:,0],q[:,1],obj['e']);phi1=aipw(y,a,q[:,0],q[:,1],np.clip(obj['e'],.05,.95))
        arrays.update({method+'__phi0':phi0,method+'__phi1':phi1,method+'__raw_e':obj['e'],method+'__e':np.clip(obj['e'],.05,.95)})
        for form in forms:
            g0,g1=obj['g0'][form],obj['g1'][form];r=b-m[form];D=np.mean(r*r);results={}
            arrays.update({method+'__'+form+'__g0':g0,method+'__'+form+'__g1':g1,'m__'+form:m[form]})
            for combo,ph,gg in [('C00',phi0,g0),('C10',phi1,g0),('C01',phi0,g1),('C11',phi1,g1)]:
                row,fi,parts=score_summary(b,ph,m[form],gg,sd);results[combo]=row
                row.update(run_id=identity,trial=trial,partition=str(partition),method=method,form=form,combination=combo,n=n)
                rows.append(row);ratios.append(dict(run_id=identity,method=method,form=form,combination=combo,**fi))
                for k,v in parts.items():arrays[f'{method}__{form}__{combo}__{k}']=v
                diagnostics.append(dict(run_id=identity,trial=trial,method=method,form=form,stage='outer',fold=-1,version=combo,
                    **stage_diagnostics(a,obj['e'],obj['e'] if combo[1]=='0' else np.clip(obj['e'],.05,.95),ph,gg)))
            outer=np.mean(r*(phi1-phi0))/D;inner=-np.mean(r*(g1-g0))/D
            total=results['C11']['estimate']-results['C00']['estimate']
            identity_error=total-outer-inner;interaction=results['C11']['estimate']-results['C10']['estimate']-results['C01']['estimate']+results['C00']['estimate']
            assert abs(identity_error)<1e-10 and abs(interaction)<1e-10
            decomp.append(dict(run_id=identity,trial=trial,method=method,form=form,change_raw=total,change_SD=total*sd,
                outer_phi_change_SD=outer*sd,inner_g_change_SD=inner*sd,change_over_original_SE=total/results['C00']['se'],
                exact_decomposition_error=identity_error,interaction_zero_error=interaction,
                changed_inner_alpha=sum(r['g0']['selected_alpha']!=r['g1']['selected_alpha'] for r in records if r['method']==method and r['form']==form)))
            if method=='M3':
                if original:
                    rr=pd.read_csv(A13/'role_calibrated_all18_partial_benefit.csv',float_precision='round_trip')
                    rr=rr[(rr.trial==trial)&(rr.form==form)&(rr.gene=='CXCL9')].iloc[0]
                    for new,old in [('estimate_SD','estimate_SD'),('SE_SD','SE_SD'),('low_SD','low_SD'),('high_SD','high_SD'),('p_two_sided','p_delta'),('p_score','p_score')]:check('A13_'+form+'_'+new,results['C00'][new],rr[old])
                else:
                    rr=pd.read_csv(cache/'clinical_results.csv',float_precision='round_trip').iloc[0]
                    for key in ['estimate_SD','low_SD','high_SD','p_two_sided']:check('E3_'+key,results['C00'][key],rr[key])
        if trial=='NeoTRIP':
            check('NeoTRIP_phi_invariance',phi1,phi0)
            for form in forms:check('NeoTRIP_'+form+'_g_invariance',obj['g1'][form],obj['g0'][form])
    np.savez_compressed(out/'LOCAL_arrays.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'stage_swap_results.csv',index=False);pd.DataFrame(replays).to_csv(out/'original_replay.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'nuisance_diagnostics.csv',index=False);pd.DataFrame(decomp).to_csv(out/'parameter_change_decomposition.csv',index=False)
    write_immutable_json(out/'full_Fieller_sets.json',ratios);write_immutable_json(out/'LOCAL_model_records.json',records)
    write_immutable_json(out/'completion.json',dict(original_replays_passed=True,new_patients=0,new_graphs=0,
        output_sha256={str(path.relative_to(out)):file_digest(path) for path in out.rglob('*') if path.is_file()}))
    return rows


def family(out,destination=None):
    destination=destination or out
    rows=[];sets=[]
    for trial in ['NeoTRIP','ISPY2']:
        z=np.load(out/(trial+'_original')/'LOCAL_arrays.npz',allow_pickle=True)
        w=pd.DataFrame(z['W'],columns=z['columns'])
        residuals=np.load(BASE/'ifng_partial_family_audit_v1'/f'{trial}_family_arrays.npz')
        for method in (['M3','M1'] if trial=='ISPY2' else ['M3']):
            for form in ['linear','spline']:
                for combo in ['C00','C10','C01','C11']:
                    phi=z[method+'__phi'+combo[1]];g=z[method+'__'+form+'__g'+combo[2]]
                    for gene in GENES:
                        b=w[gene].to_numpy();m=b-residuals[form+'__'+gene+'__rB']
                        st,fi,_=score_summary(b,phi,m,g,b.std())
                        rows.append(dict(trial=trial,method=method,form=form,combination=combo,gene=gene,**st))
                        sets.append(dict(trial=trial,method=method,form=form,combination=combo,gene=gene,**fi))
    d=pd.DataFrame(rows)
    for _,idx in d.groupby(['trial','method','form','combination']).groups.items():
        for name,col in [('delta','p_two_sided'),('score','p_score')]:d.loc[idx,'BH18_'+name]=multipletests(d.loc[idx,col],method='fdr_bh')[1]
    average=[]
    for (form,combo,gene),v in d[d.method=='M3'].groupby(['form','combination','gene']):
        beta=v.estimate_SD.mean();vv=(.5*v.SE_SD.to_numpy())**2;se=np.sqrt(vv.sum());df=vv.sum()**2/np.sum(vv**2/(v.n.to_numpy()-1));cut=student_t.ppf(.975,df)
        average.append(dict(form=form,combination=combo,gene=gene,estimate_SD=beta,SE_SD=se,low_SD=beta-cut*se,high_SD=beta+cut*se,
            p_delta=2*student_t.sf(abs(beta/se),df),df=df,conjunction_delta_p=v.p_two_sided.max(),conjunction_score_p=v.p_score.max()))
    av=pd.DataFrame(average)
    for _,ix in av.groupby(['form','combination']).groups.items():
        for name in ['p_delta','conjunction_delta_p','conjunction_score_p']:av.loc[ix,name+'_BH18']=multipletests(av.loc[ix,name],method='fdr_bh')[1]
    d.to_csv(destination/'K1_family18.csv',index=False);av.to_csv(destination/'K1_two_study_average.csv',index=False)
    write_immutable_json(destination/'K1_family18_full_Fieller_sets.json',sets)
    old=pd.read_csv(A13/'two_study_average_and_conjunction.csv',float_precision='round_trip')
    checks=av[av.combination=='C00'].merge(old,on=['form','gene'])
    np.testing.assert_allclose(checks.estimate_SD,checks.equal_study_average,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(checks.p_delta_BH18,checks.p_BH18,rtol=1e-10,atol=1e-10)


def main(args):
    if not BASE.is_dir():raise FileNotFoundError('Set BREAST_ICI_CLINICAL_CACHE to the authorized frozen cache root; see docs/data_contract.md')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'analysis').glob('*.py'))+list((ROOT/'src').rglob('*.py'))
    for path in sources:
        dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write_immutable_json(out/'protocol_before_results.json',dict(command=sys.argv,probability_bound=.05,original_g_projection='none',
        graph_relearning=False,Q_m_fixed=True,rebuild_only_missing_inner_Q_e=True,new_patients=0,
        source_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in sources}))
    jobs=[(trial,part,str(out)) for trial in ['ISPY2','NeoTRIP'] for part in ['original',0,1,2]]
    failed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_one,job):job for job in jobs}
        for future in as_completed(pending):
            try:future.result()
            except Exception as exc:
                fail=dict(task=pending[future],error=str(exc),traceback=traceback.format_exc());failed.append(fail)
                write_immutable_json(out/(str(pending[future][0])+'_'+str(pending[future][1]))/'failure.json',fail)
                print('FAILED',pending[future][:2],str(exc),flush=True)
    write_immutable_json(out/'failures.json',failed)
    if failed:raise RuntimeError('K1 replay failed; preserve outputs, diagnose before new version')
    family(out)
    for source,target in [('original_replay','K1_original_replay'),('stage_swap_results','K1_stage_swap_results'),('nuisance_diagnostics','K1_nuisance_diagnostics'),('parameter_change_decomposition','K1_parameter_change_decomposition')]:
        pd.concat([pd.read_csv(out/(t+'_'+str(p))/(source+'.csv')) for t,p,_ in jobs],ignore_index=True).to_csv(out/(target+'.csv'),index=False)
    write_immutable_json(out/'completion.json',dict(clinical_runs=8,new_patients=0,new_graphs=0,all_replays_passed=True,
        output_sha256={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=2)
    main(p.parse_args())
