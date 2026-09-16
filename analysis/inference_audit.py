"""Independent numerical audit for the bundled final synthetic evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, norm
from scipy.special import expit
from scipy.integrate import quad

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);sortedq=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    q=np.empty(len(p));q[order]=np.minimum(sortedq,1.);return q

def independent_moments(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x);N=np.sum(x)/n;D=np.sum(y)/n
    beta=N/D;v=(x-beta*y)/D;se=np.sqrt(np.sum((v-v.mean())**2)/(n*(n-1)));cut=t_dist.ppf(.975,n-1)
    centered=np.column_stack([x-N,y-D]);cov=centered.T@centered/(n*(n-1))
    coeff=np.array([D*D-cut**2*cov[1,1],2*(cut**2*cov[0,1]-N*D),N*N-cut**2*cov[0,0]])
    aa,bb,cc=coeff;tol=1e-12*max(np.abs(coeff).max(),1e-12)
    if abs(aa)<=tol:
        if abs(bb)<=tol:kind='all' if cc<=0 else 'empty';intervals=[[None,None]] if cc<=0 else []
        else:kind='halfline';intervals=[[None,float(-cc/bb)]] if bb>0 else [[float(-cc/bb),None]]
    else:
        discriminant=bb*bb-4*aa*cc
        if discriminant<0:kind='all' if aa<0 else 'empty';intervals=[[None,None]] if aa<0 else []
        else:
            # Independent polynomial root implementation (not production formula).
            roots=sorted(np.roots(coeff).real.tolist())
            kind='bounded' if aa>0 else 'outer';intervals=[roots] if aa>0 else [[None,roots[0]],[roots[1],None]]
    scorese=np.sqrt(cov[0,0]);ps=2*t_dist.sf(abs(N/scorese),n-1) if scorese else (1. if N==0 else 0.)
    return dict(N=N,D=D,estimate=beta,se=se,ci_low=beta-cut*se,ci_high=beta+cut*se,
        p_two_sided=2*t_dist.sf(abs(beta/se),n-1),p_score=ps,coeff=coeff,kind=kind,intervals=intervals)

def check_set(result,saved):
    assert result['kind']==saved['kind']
    np.testing.assert_allclose(result['coeff'],saved['quadratic_coefficients'],rtol=1e-9,atol=1e-11)
    assert len(result['intervals'])==len(saved['intervals'])
    for a,b in zip(result['intervals'],saved['intervals']):
        for u,v in zip(a,b):
            if u is None or v is None:assert u is None and v is None
            else:np.testing.assert_allclose(u,v,rtol=1e-8,atol=1e-10)

def check_row(x,y,row,sd=1.):
    st=independent_moments(x,y)
    for field in ['N','D','estimate','se','ci_low','ci_high','p_two_sided','p_score']:
        if field in row:np.testing.assert_allclose(st[field],row[field],rtol=1e-9,atol=1e-10)
    if 'estimate_SD' in row:np.testing.assert_allclose(st['estimate']*sd,row['estimate_SD'],rtol=1e-9,atol=1e-10)
    return st

def check_hashes(root,record):
    count=0
    for name,value in record.get('output_sha256',{}).items():
        assert sha(root/name)==value,(root,name);count+=1
    return count

def audit_k2(root,summary,*,entropy=202609160601,reps=200,repair=False,check_history=False):
    planned=pd.read_csv(root/'planned_datasets.csv');data=pd.read_csv(summary/'K2_dataset_results.csv',float_precision='round_trip')
    fdp=pd.read_csv(summary/'K2_family_FDP.csv',float_precision='round_trip');hashes=0;moments=0;folds=0;native_runs=0;keys=[];raw_hashes=[]
    expected_truth=quad(lambda r:.20*r*np.tanh(r)*norm.pdf(r,scale=.8)/.64,-np.inf,np.inf,epsabs=1e-13)[0]
    assert len(planned)==4*reps
    for _,v in planned.groupby(['n','signal']):assert len(v)==reps and v.topology.value_counts().to_dict()=={0:reps//2,1:reps//2}
    for identity,d in data.groupby('dataset_id'):
        path=root/'datasets'/identity;truth=np.load(path/'known_truth.npz',allow_pickle=True);w=pd.DataFrame(truth['W'],columns=truth['columns'])
        definition=json.loads((path/'truth_definition.json').read_text());keys.append(tuple(definition['seed_key']))
        assert definition['seed_key'][0]==entropy and definition['seed_key'][0]!=202609160101
        rng=np.random.default_rng(np.random.SeedSequence(definition['seed_key']));n=len(w)
        u,v,g,h,noise=rng.normal(size=(5,n));b=.5*g+.8*noise
        expected=pd.DataFrame(dict(U=u,V=v,G=g,B=b,H=h))
        for j in range(5):expected['X'+str(j)]=.8*expected[['U','V','G'][j%3]]+rng.normal(0,.7,n)
        np.testing.assert_array_equal(expected.to_numpy(),w.to_numpy())
        etrue=expit(1.2*expected['U' if definition['topology']==0 else 'X0'].to_numpy())
        signal=int(d.signal.iloc[0]);tau=.12+(.20*np.tanh(.8*noise) if signal else np.zeros(n))
        q0=.32+.12*np.tanh(v)+.04*np.tanh(g)+.08*np.tanh(b);qt=np.column_stack([q0,q0+tau])
        at=rng.binomial(1,etrue);yt=rng.binomial(1,qt[np.arange(n),at])
        for actual,expect in [(truth['e'],etrue),(truth['Q'],qt),(truth['A'],at),(truth['Y'],yt),(truth['tau'],tau)]:np.testing.assert_array_equal(actual,expect)
        np.testing.assert_allclose(definition['truth']['B'],expected_truth if signal else 0.,atol=1e-12,rtol=0)
        assert definition['truth']['H']==0 and np.isclose(definition['population_SD']['B'],np.sqrt(.89))
        raw_hashes.append(hashlib.sha256(w.to_numpy().tobytes()+at.tobytes()+yt.tobytes()).hexdigest())
        sets=json.loads((path/'full_Fieller_sets.json').read_text());fitted=np.load(path/'fit/arrays.npz');extra=np.load(path/'additional_arrays.npz')
        # Same actual full-Q columns, folds and seeds must share exact Q fits.
        prefixes=['L2__M1_all_inputs__','L2__M3_actual_Q_closure__']+([] if repair else ['L1__M3_actual_Q_closure__'])
        for prefix in prefixes[1:]:np.testing.assert_array_equal(fitted[prefix+'Q'],fitted[prefixes[0]+'Q'])
        if repair:
            np.testing.assert_array_equal(fitted['L2__M1_restricted_Q_fixed_C__Q'],fitted['L2__M3_restricted_Q_fixed_C__Q'])
            for new,old in [('M1_restricted_Q_fixed_C','M1_all_inputs'),('M3_restricted_Q_fixed_C','M3_actual_Q_closure')]:
                np.testing.assert_array_equal(fitted['L2__'+new+'__e'],fitted['L2__'+old+'__e'])
                for f in range(3):
                    ni=np.load(path/f'fit/fold{f}_L2_{new}_inner.npz');oi=np.load(path/f'fit/fold{f}_L2_{old}_inner.npz')
                    for field in ['e','raw_e','train','test','fold']:np.testing.assert_array_equal(ni[field],oi[field])
        records=json.loads((path/'fit/model_records.json').read_text())
        roles=json.loads((path/'fit/roles.json').read_text())
        for rec in records:
            if rec['stage']!='Q_e_g':continue
            columns=rec['outer']['Q']['actual_columns']
            sources=set(rec['outer']['Q']['raw_sources'])
            assert set(columns)==sources|({'derived__B_times_U'} if {'B','U'}<=sources else set())
            if not repair or not rec['method'].endswith('fixed_C'):assert sources==set(w.columns)
            tr=set(rec['outer']['train_rows']);te=set(rec['outer']['test_rows'])
            assert not tr&te and len(tr|te)==n
            for inner in rec['inner']:
                it,iv=set(inner['train_rows']),set(inner['test_rows'])
                assert not it&iv and it|iv==tr and not (it|iv)&te
            layer,method=rec['layer'],rec['method'];f=rec['fold'];prefix=layer+'__'+method+'__'
            raw=np.asarray(rec['outer']['score_probability_control']['raw_predictions']);idx=rec['outer']['test_rows']
            np.testing.assert_array_equal(np.clip(raw,.05,.95),fitted[prefix+'e'][idx])
            for other in records:
                if other['stage']!='Q_e_g' or other['fold']!=f:continue
                myrole=next(r for r in roles if r['layer']==layer and r['method']==method and r['fold']==f)
                their=next(r for r in roles if r['layer']==other['layer'] and r['method']==other['method'] and r['fold']==f)
                if myrole['C']==their['C']:
                    np.testing.assert_array_equal(raw,other['outer']['score_probability_control']['raw_predictions'])
        for _,row in d[d.fit_status=='SUCCESS'].iterrows():
            method,candidate=row['method'],row['candidate'];sd=definition['population_SD'][candidate]
            if method in ['complete_oracle','true_e_clip_operator']:
                r=w[candidate].to_numpy()-truth['m_'+candidate];phi=extra[method+'__phi'];g=truth['g_true']
            else:
                layer=method[:2];mm='M1_all_inputs' if 'M1' in method else 'M3_actual_Q_closure'
                if method.endswith('_QR'):mm='M1_restricted_Q_fixed_C' if 'M1' in method else 'M3_restricted_Q_fixed_C'
                prefix=layer+'__'+mm+'__'
                r=w[candidate].to_numpy()-fitted['m__'+candidate];phi=fitted[prefix+'phi'];g=fitted[prefix+'g']
                q,e=fitted[prefix+'Q'],fitted[prefix+'e'];a,y=truth['A'],truth['Y']
                independent=q[:,1]-q[:,0]+a*(y-q[:,1])/e-(1-a)*(y-q[:,0])/(1-e)
                np.testing.assert_allclose(independent,phi,rtol=1e-12,atol=1e-12)
            st=check_row(r*(phi-g),r*r,row,sd);fi=next(v for v in sets if v['method']==method and v['candidate']==candidate)
            check_set(st,fi['raw']);check_set(independent_moments(r*(phi-g)*sd,r*r),fi['SD']);moments+=1
            val=definition['truth'][candidate];poly=np.polyval(st['coeff'],val)
            if abs(poly)>1e-10:assert bool(poly<=0)==bool(row.covers_fieller)
        for method,v in d.groupby('method'):
            for inf,col in [('score','p_score'),('delta','p_two_sided')]:
                valid=v.fit_status.eq('SUCCESS')&v[col].notna();pp=np.where(valid,v[col],1.);q=bh(pp);discover=q<.05
                false=discover & (np.abs(v.truth)<1e-12);expected=false.sum()/max(1,discover.sum())
                saved=fdp[(fdp.dataset_id==identity)&(fdp.method==method)&(fdp.inference==inf)].iloc[0]
                assert abs(expected-saved.FDP)<1e-12 and false.sum()==saved.false_discoveries
                np.testing.assert_allclose(q[valid],v.loc[valid,'BH2_'+inf],rtol=1e-12,atol=1e-12)
        for pathgraph in sorted((path/'fit').glob('fold*_graph.json')):
            rec=json.loads(pathgraph.read_text());tr=set(rec['train']);te=set(rec['test']);assert not tr&te and len(tr|te)==len(w)
            assert rec['native']['actual_target_runs']>0;native_runs+=rec['native']['actual_target_runs'];folds+=1
        for npz in (path/'fit').glob('fold*_inner.npz'):
            inn=np.load(npz);tr=set(inn['train']);te=set(inn['test']);assert not tr&te
            for f in range(3):assert set(inn['train'][inn['fold']==f])<=tr
            assert np.all(inn['fold']>=0)
            np.testing.assert_array_equal(np.clip(inn['raw_e'],.05,.95),inn['e'])
        hashes+=check_hashes(path,json.loads((path/'completion.json').read_text()))
    assert len(keys)==4*reps and len(set(keys))==4*reps and len(set(raw_hashes))==4*reps and folds==12*reps
    if check_history:raise ValueError('Historical experiments are outside this release; use check_history=False')
    old_hashes=[]
    return dict(distinct_datasets=len(keys),independent_moment_set_replays=moments,fold_graph_checks=folds,
        recorded_actual_native_target_calls=native_runs,sha256_checks=hashes,BH_and_FDP_replayed=True,
        balanced_topologies=True,new_seed_keys_disjoint_from_E1=True,independent_DGP_replays=len(keys),distinct_W_A_Y_hashes=len(raw_hashes),
        prior_raw_data_count=len(old_hashes),prior_raw_data_hashes_disjoint=True if check_history else 'not_rechecked_without_historical_inputs',only_Q_repair_e_and_inner_folds_checked=repair)
