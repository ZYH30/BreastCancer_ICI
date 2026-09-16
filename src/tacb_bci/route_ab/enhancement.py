"""Cross-fitted candidate-role closure and the manuscript F2 calibration pipeline."""
import json
import time
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, roots_hermitenorm
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .native import fit_propensity_adapter
from .roles import discover_roles
from .candidate_role_lift import CandidateRoleLift
from .score_surface_roles import score_surface_closure
from .conditional_benefit import context_mean
from .continuous import sklearn_seed, weighted_derivative
from .ratio_inference import ratio_score_set, set_contains
from ..causal_ifn.orthogonal_score import aipw
from ..provenance import write_immutable_json
from graph_roles import find_confounder_sets, precision_variables

METHODS = ['M1_all_inputs', 'M2_candidate_restricted_Q', 'M3_actual_Q_closure']


def nx_graph(graph):
    g = nx.DiGraph()
    g.add_nodes_from(graph)
    g.add_edges_from((p, c) for c, ps in graph.items() for p in ps)
    return g


def draw_truth(n, family, signal, topology, seed_key, p=10):
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    u, v, g, h, error = rng.normal(size=(5, n))
    sign = 1 if topology == 0 else -1
    if family == 'F1':
        residual = u + sign*v + .8*error
        var = 2.64
    elif family == 'F2':
        residual = .8*error
        var = .64
    else:
        raise ValueError('undeclared_family')
    b = .5*g + residual
    w = pd.DataFrame(dict(U=u, V=v, G=g, B=b, H=h))
    graph = dict(U=[], V=[], G=[], H=[], B=['G']+(['U','V'] if family=='F1' else []))
    for j in range(p-5):
        parent = ['U','V','G'][j % 3]
        w[f'X{j}'] = .8*w[parent]+rng.normal(0, .7, n)
        graph[f'X{j}'] = [parent]
    # Two graph topologies: direct U assignment vs assignment through X0.
    treatment_parent = 'U' if topology == 0 else 'X0'
    e = expit(1.2*w[treatment_parent].to_numpy())
    q0 = .32 + .12*np.tanh(v) + .04*np.tanh(g)
    if family == 'F2':
        q0 += .08*np.tanh(b)
    tau = .12 + (.20*np.tanh(residual) if signal else 0.)
    tau = np.broadcast_to(tau, (n,)).copy()
    q = np.column_stack([q0, q0+tau])
    if not (np.all(q > 0) and np.all(q < 1)):
        raise AssertionError('invalid_binary_SCM')
    a = rng.binomial(1, e)
    y = rng.binomial(1, q[np.arange(n), a])
    graph['T'] = [treatment_parent]
    graph['Y'] = ['T','G','V'] + (['B'] if family=='F2' or signal else [])
    # In F1 signal tau is a function of B and G; all its parents are represented.
    nodes, weights = roots_hermitenorm(128)
    r = np.sqrt(var)*nodes
    truth = float(.20*np.sum(weights*r*np.tanh(r))/(np.sqrt(2*np.pi)*var)) if signal else 0.
    return dict(W=w, A=a, Y=y, Q=q, e=e, tau=tau, graph=graph,
                m_true={'B':.5*g, 'H':np.zeros(n)}, g_true=np.full(n,.12),
                truth={'B':truth, 'H':0.}, population_SD={'B':np.sqrt(var+.25),'H':1.},
                residual_variance=var, treatment_parent=treatment_parent)


def raw_roles(graph):
    c = sorted(find_confounder_sets(graph, 'T', 'Y')['proximal'])
    s = sorted(precision_variables(graph, 'T', 'Y', c))
    return c, s


def role_specs(graph, available, required, raw_c=None, raw_s=None, learned=None):
    available, required = sorted(available), sorted(set(required))
    if raw_c is None:
        raw_c, raw_s = raw_roles(graph)
    base = CandidateRoleLift(graph, required, raw_c, raw_s, available=available).greedy_close()
    broad = score_surface_closure(graph, required, available, raw_c, raw_s, available=available)
    values = [
        ('M1_all_inputs', available, [], available),
        ('M2_candidate_restricted_Q', base['C'], base['S_prec'], sorted(set(base['C']) | set(base['S_prec']))),
        ('M3_actual_Q_closure', broad['C'], broad['S_prec'], available),
    ]
    output = {}
    for method, c, s, q in values:
        lift = CandidateRoleLift(graph, set(required)|set(q), available=available)
        preserved = set(required) <= set(c)|set(q)
        output[method] = dict(C=sorted(c), S=sorted(set(s)-set(c)), Q=sorted(q),
            required=required, required_preserved=preserved,
            actual_Q_certified=bool(lift.unresolved<=set(c) and lift.certificate(c)),
            interpretation='working_graph_interface')
    return output


def truth_role_checks(graph, spec, available):
    lift = CandidateRoleLift(graph, set(spec['required'])|set(spec['Q']), available=available)
    bd = nx_graph(graph)
    bd.remove_edges_from(list(bd.out_edges('T')))
    return dict(true_graph_actual_Q_valid=lift.certificate(spec['C']),
                true_graph_C_backdoor=nx.is_d_separator(bd, {'T'}, {'Y'}, set(spec['C'])),
                true_graph_CQ_backdoor=nx.is_d_separator(bd, {'T'}, {'Y'}, set(spec['C'])|set(spec['Q'])))


def design(w, columns, product=False):
    x = w[sorted(columns)].copy()
    if product and {'B','U'} <= set(columns):
        x['derived__B_times_U'] = w.B*w.U
    return x


def fit_nuisances(w, a, y, train, test, spec, seed, product, cache, known_e=None):
    # Identical columns/rows/seed => identical fits shared across methods.
    from .nuisance import nested_q_fit_predict
    qkey = ('Q',tuple(spec['Q']),tuple(train),tuple(test),seed,product)
    ekey = ('e',tuple(spec['C']),tuple(train),tuple(test),seed,known_e)
    if qkey not in cache:
        x = design(w, spec['Q'], product)
        q = np.zeros((len(test),2)); records=[]
        for arm in [0,1]:
            tr = train[a[train]==arm]
            q[:,arm], record = nested_q_fit_predict(x.iloc[tr],y[tr],x.iloc[test],sklearn_seed(seed,1,arm))
            records.append(record)
        cache[qkey] = (q, dict(raw_sources=spec['Q'],actual_columns=list(x),models=records))
    if ekey not in cache:
        frame=w.iloc[train].copy();frame['T']=a[train];frame['Y']=y[train]
        model, record=fit_propensity_adapter(frame,spec['C'],seed=sklearn_seed(seed,2),known_e=known_e)
        cache[ekey] = (model.predict(w.iloc[test][spec['C']]),record)
    q, qr=cache[qkey];e,er=cache[ekey]
    return q,e,dict(Q=qr,e=er,actual_Q_certified=spec['actual_Q_certified'],
        train_rows=train.tolist(),test_rows=test.tolist())


def weight_diagnostics(a, e, phi):
    ww=np.where(a,1/e,1/(1-e))
    return dict(propensity_ess=float(ww.sum()**2/np.sum(ww**2)),
        max_weight=float(ww.max()),e_min=float(e.min()),e_max=float(e.max()),
        max_abs_phi=float(np.abs(phi).max()),A_logloss=float(log_loss(a,e,labels=[0,1])))


def fit_pipeline(w, a, y, candidates, context, out, seed, *, true=None,
                 layers=('L2',), outer_folds=3, product=False, anchors=None,
                 selected_methods=None, known_e=None, reuse_graph_dir=None, propensity_clip=None,
                 layer_methods=None, save_inner_details=False):
    """Fully out-of-fold score; inner g never sees outer-test outcomes.

    Inner nuisance fits reuse the outer-training graph as a learned design.
    Their scores are NOT reported as independent inner graph validation.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected_methods=selected_methods or METHODS
    methods_for=lambda layer: (layer_methods or {}).get(layer, selected_methods)
    def active_e(e,record):
        if propensity_clip is None:return e,record
        if not 0 < propensity_clip < .5:raise ValueError('invalid_common_probability_bound')
        record=dict(record);clipped=np.clip(e,propensity_clip,1-propensity_clip)
        record['score_probability_control']=dict(bound=propensity_clip,raw_predictions=e.tolist(),
            clipped_count=int(np.sum(clipped!=e)),raw_min=float(e.min()),raw_max=float(e.max()),
            same_estimand_no_patient_trimming=True,possible_truncation_bias=True)
        return clipped,record
    n=len(y);required=sorted(set(candidates)|{context});anchors=anchors or required
    splits=list(StratifiedKFold(outer_folds,shuffle=True,random_state=seed).split(w,2*a+y))
    fold_ids=np.full(n,-1); m={b:np.zeros(n) for b in candidates}
    fits={};records=[];failures=[];role_records=[]
    discovery_seconds=0.;estimation_seconds=0.
    for layer in layers:
        for method in methods_for(layer):
            fits[(layer,method)]={k:np.zeros((n,2)) if k=='Q' else np.zeros(n) for k in ['Q','e','g']}
    for f,(tr,te) in enumerate(splits):
        fold_ids[te]=f
        frame=w.iloc[tr].copy();frame['T']=a[tr];frame['Y']=y[tr]
        checkpoint=out/f'fold{f}_graph.json';tick=time.perf_counter()
        if 'L2' in layers:
            cached=Path(reuse_graph_dir)/f'fold{f}_graph.json' if reuse_graph_dir else checkpoint
            if cached.exists():
                r=json.loads(cached.read_text());assert r['train']==tr.tolist() and r['test']==te.tolist()
                learned=r['native']
            else:
                learned=discover_roles(frame,anchors,seed=sklearn_seed(seed,10,f))
            if not checkpoint.exists():
                write_immutable_json(checkpoint,dict(train=tr.tolist(),test=te.tolist(),native=learned))
        discovery_seconds+=time.perf_counter()-tick
        tick=time.perf_counter()
        cache={};inner_splits=list(StratifiedKFold(3,shuffle=True,random_state=sklearn_seed(seed,11,f)).split(tr,2*a[tr]+y[tr]))
        for b in candidates:
            m[b][te],rec=context_mean(w[context].to_numpy()[tr],w[b].to_numpy()[tr],w[context].to_numpy()[te],
                seed=sklearn_seed(seed,12,f),form='spline')
            records.append(dict(fold=f,stage='m',candidate=b,record=rec))
        for layer in layers:
            graph=true['graph'] if layer=='L1' else learned['graph']
            rc,rs=raw_roles(graph) if layer=='L1' else (learned['raw_C'],learned['raw_S'])
            specs=role_specs(graph,list(w),required,rc,rs,learned if layer=='L2' else None)
            # K2 optional one-component repair: reduce Q only, retaining the
            # original propensity design. Both baselines get the identical Q.
            for label,base_method in [('M3_restricted_Q_fixed_C','M3_actual_Q_closure'),
                                      ('M1_restricted_Q_fixed_C','M1_all_inputs')]:
                if label in methods_for(layer):
                    restricted=dict(specs[base_method]);restricted['Q']=specs['M2_candidate_restricted_Q']['Q']
                    check=CandidateRoleLift(graph,set(required)|set(restricted['Q']),available=list(w))
                    restricted['actual_Q_certified']=bool(check.unresolved<=set(restricted['C']) and check.certificate(restricted['C']))
                    if not restricted['actual_Q_certified']:raise ValueError('restricted_Q_repair_not_certified')
                    restricted['repair']='only_Q_sources_reduced_original_C_unchanged'
                    specs[label]=restricted
            for method in methods_for(layer):
                spec=specs[method];rr=dict(layer=layer,method=method,fold=f,**spec)
                if true is not None:rr.update(truth_role_checks(true['graph'],spec,list(w)))
                role_records.append(rr)
                key=(layer,method)
                if key not in fits:continue
                try:
                    q,e,outerrec=fit_nuisances(w,a,y,tr,te,spec,sklearn_seed(seed,13,f),product,cache,known_e)
                    e,outerrec=active_e(e,outerrec)
                    inner_phi=np.zeros(len(tr));inner_records=[]
                    inner_q=np.zeros((len(tr),2));inner_e=np.zeros(len(tr));inner_raw_e=np.zeros(len(tr));inner_fold=np.full(len(tr),-1)
                    for j,(itr,ite) in enumerate(inner_splits):
                        iq,ie,irec=fit_nuisances(w,a,y,tr[itr],tr[ite],spec,sklearn_seed(seed,14,f*3+j),product,cache,known_e)
                        inner_raw_e[ite]=ie
                        ie,irec=active_e(ie,irec)
                        inner_q[ite]=iq;inner_e[ite]=ie;inner_fold[ite]=j
                        ip=aipw(y[tr[ite]],a[tr[ite]],iq[:,0],iq[:,1],ie);inner_phi[ite]=ip
                        irec.update(weight_diagnostics(a[tr[ite]],ie,ip));irec['fold']=j
                        inner_records.append(irec)
                    gg,grec=context_mean(w[context].to_numpy()[tr],inner_phi,w[context].to_numpy()[te],
                        seed=sklearn_seed(seed,15,f),form='spline')
                    fits[key]['Q'][te]=q;fits[key]['e'][te]=e;fits[key]['g'][te]=gg
                    details=dict(Q=inner_q,e=inner_e,raw_e=inner_raw_e,fold=inner_fold,test=te) if save_inner_details else {}
                    np.savez_compressed(out/f'fold{f}_{layer}_{method}_inner.npz',train=tr,phi=inner_phi,g_test=gg,**details)
                    records.append(dict(fold=f,layer=layer,method=method,stage='Q_e_g',outer=outerrec,
                        inner=inner_records,g=grec,inner_graph_design='outer_training_graph_reused_not_inner_validation'))
                except Exception as exc:
                    failures.append(dict(layer=layer,method=method,fold=f,error=str(exc),traceback=traceback.format_exc()))
                    del fits[key]
        estimation_seconds+=time.perf_counter()-tick
        write_immutable_json(out/f'fold{f}_progress.json',dict(fold=f,roles=[r for r in role_records if r['fold']==f],
            failures=failures,discovery_seconds=discovery_seconds,estimation_seconds=estimation_seconds))
    rows=[];arrays=dict(W=w.to_numpy(),columns=np.asarray(w.columns),A=a,Y=y,fold=fold_ids)
    ratios=[]
    def evaluate(layer,method,phi,gg,q=None,e=None,diagnostic=False):
        matching=[r for r in role_records if r['layer']==layer and r['method']==method]
        for b in candidates:
            stats,parts=weighted_derivative(w[b].to_numpy(),phi,m[b],gg)
            fi=ratio_score_set(parts['numerator'],parts['denominator'])
            row=dict(layer=layer,method=method,candidate=b,context=context,**stats,
                p_value=fi['null_score_p'],interval_type='Fieller_working',bounded_interval=fi['kind']=='bounded',
                fieller_low=fi['intervals'][0][0] if fi['kind']=='bounded' else np.nan,
                fieller_high=fi['intervals'][0][1] if fi['kind']=='bounded' else np.nan,
                fit_status='SUCCESS',failure_reason='',native_direct_diagnostic=diagnostic,
                residual_denominator=parts['denominator'].mean(),C_size=np.mean([len(r['C']) for r in matching]) if matching else np.nan,
                S_size=np.mean([len(r['S']) for r in matching]) if matching else np.nan,
                required_preserved=all(r['required_preserved'] for r in matching) if matching else False,
                candidate_context_retained=all({b,context}<=set(r['C'])|set(r['Q']) for r in matching) if matching else False,
                actual_Q_certified=all(r['actual_Q_certified'] for r in matching) if matching else False)
            if true is not None:
                truth=true['truth'][b];row.update(truth=truth,bias=stats['estimate']-truth,
                    covers_truth=set_contains(fi,truth),delta_covers=stats['ci_low']<=truth<=stats['ci_high'],
                    ATE_bias=float(phi.mean()-np.mean(true['tau'])),
                    true_graph_valid_fraction=np.mean([r['true_graph_actual_Q_valid'] for r in matching]) if matching else np.nan)
            if e is not None:
                row.update(weight_diagnostics(a,e,phi));row['Q_logloss']=float(log_loss(y,q[np.arange(n),a],labels=[0,1]))
                ir=[r for r in records if r.get('layer')==layer and r.get('method')==method and r['stage']=='Q_e_g']
                row['inner_min_ESS']=min(i['propensity_ess'] for r in ir for i in r['inner'])
                row['inner_max_weight']=max(i['max_weight'] for r in ir for i in r['inner'])
            rows.append(row);ratios.append(dict(layer=layer,method=method,candidate=b,**fi))
            prefix=f'{layer}__{method}__{b}__'
            arrays.update({prefix+k:v for k,v in parts.items()})
        arrays[f'{layer}__{method}__phi']=phi;arrays[f'{layer}__{method}__g']=gg
    for (layer,method),obj in fits.items():
        phi=aipw(y,a,obj['Q'][:,0],obj['Q'][:,1],obj['e'])
        evaluate(layer,method,phi,obj['g'],obj['Q'],obj['e'])
        arrays[f'{layer}__{method}__Q']=obj['Q'];arrays[f'{layer}__{method}__e']=obj['e']
        # Oracle diagnostics use the SAME fitted inputs and data, never count extra cohorts.
        if true is not None:
            for name,qq,ee in [('true_Q_learned_e',true['Q'],obj['e']),('learned_Q_true_e',obj['Q'],true['e'])]:
                pp=aipw(y,a,qq[:,0],qq[:,1],ee)
                # Use true context explicitly; isolates nuisance pathways, not full method claim.
                for b in candidates:
                    st,_=weighted_derivative(w[b].to_numpy(),pp,true['m_true'][b],true['g_true'])
                    rows.append(dict(layer=layer,method=method+'__'+name,candidate=b,context=context,
                        **st,truth=true['truth'][b],bias=st['estimate']-true['truth'][b],
                        fit_status='SUCCESS',failure_reason='',interval_type='delta_oracle_context_diagnostic',
                        p_value=st['p_two_sided'],bounded_interval=True))
    for layer in layers:
        for method in methods_for(layer):
            if (layer,method) not in fits:
                for b in candidates:
                    rows.append(dict(layer=layer,method=method,candidate=b,context=context,fit_status='FAILED',
                        failure_reason='; '.join(v['error'] for v in failures if v['layer']==layer and v['method']==method),
                        truth=true['truth'][b] if true else np.nan))
    arrays.update({'m__'+b:v for b,v in m.items()})
    np.savez_compressed(out/'arrays.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    write_immutable_json(out/'roles.json',role_records);write_immutable_json(out/'model_records.json',records)
    write_immutable_json(out/'failures.json',failures);write_immutable_json(out/'Fieller_sets.json',ratios)
    write_immutable_json(out/'timing.json',dict(discovery_seconds=discovery_seconds,estimation_seconds=estimation_seconds))
    return rows,role_records,failures
