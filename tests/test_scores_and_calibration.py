import numpy as np
from tacb_bci.route_ab.core_checks import score_summary,stage_diagnostics
from tacb_bci.route_ab.enhancement import draw_truth
from tacb_bci.causal_ifn.orthogonal_score import aipw


def test_stage_swap_exact_identity():
    rng=np.random.default_rng(8241);b,m,p0,p1,g0,g1=rng.normal(size=(6,85))
    r=b-m;D=np.mean(r*r)
    values={}
    for i,p in enumerate([p0,p1]):
        for j,g in enumerate([g0,g1]):values[i,j]=score_summary(b,p,m,g)[0]['estimate']
    np.testing.assert_allclose(values[1,1]-values[0,0],np.mean(r*(p1-p0))/D-np.mean(r*(g1-g0))/D,atol=1e-14)
    assert abs(values[1,1]-values[1,0]-values[0,1]+values[0,0])<1e-14


def test_probability_half_is_invariant_and_not_phi_clipping():
    a=np.arange(20)%2;y=1-a;q=np.full((20,2),.5);e=np.full(20,.5)
    before=aipw(y,a,q[:,0],q[:,1],e);after=aipw(y,a,q[:,0],q[:,1],np.clip(e,.05,.95))
    np.testing.assert_array_equal(before,after)
    diag=stage_diagnostics(a,e,np.clip(e,.05,.95),after)
    assert diag['clip_count']==0 and diag['weight_ESS']==20


def test_frozen_population_sd_is_not_residual_sd():
    d=draw_truth(85,'F2',False,0,[202609160601,999,0])
    assert d['truth']['B']==0 and np.all(d['tau']==.12)
    assert np.isclose(d['population_SD']['B'],np.sqrt(.89))
    assert not np.isclose(d['population_SD']['B'],np.sqrt(d['residual_variance']))


def test_sd_conversion_leaves_score_p_and_set_coverage():
    rng=np.random.default_rng(95);b=rng.normal(size=85);phi=.1*b+rng.normal(size=85)
    s,f,_=score_summary(b,phi,np.zeros(85),np.zeros(85),3.,.1)
    assert abs(f['SD']['null_score_p']-f['raw']['null_score_p'])<1e-14
    np.testing.assert_allclose(f['SD']['intervals'],np.array(f['raw']['intervals'])*3)
    assert np.isclose(s['truth_SD'],.3)


def test_independent_ratio_auditor():
    from inference_audit import independent_moments,check_set
    from tacb_bci.route_ab.ratio_inference import ratio_score_set
    rng=np.random.default_rng(948)
    for shift in [0.,.2,3.]:
        denominator=rng.normal(shift,1.,85);numerator=.1*denominator+rng.normal(size=85)
        check_set(independent_moments(numerator,denominator),ratio_score_set(numerator,denominator))


def test_dataset_fdr_not_candidate_rejection_and_failure_denominator(tmp_path):
    import pandas as pd
    from summarize_calibration import summarize
    rows=[]
    for rep in range(2):
        for candidate in ['B','H']:
            x=np.linspace(-1,1,85)
            row,_,_=score_summary(x,x*x+.1*x,np.zeros(85),np.zeros(85),truth=.1 if candidate=='B' else 0.)
            row.update(dataset_id='fixture'+str(rep),n=85,signal=1,topology=rep,method='L2_M3_S',candidate=candidate)
            row['p_score']=row['p_two_sided']=.001 if rep==0 else .6
            if rep==1 and candidate=='H':
                row.update(fit_status='FAILED',p_score=np.nan,p_two_sided=np.nan,estimate=np.nan,interval_available=False,
                    covers_fieller=np.nan,covers_delta=np.nan,covered_and_bounded=np.nan,bounded=np.nan)
            rows.append(row)
    planned=pd.DataFrame(dict(n=[85,85],signal=[1,1]))
    result=summarize(pd.DataFrame(rows),planned,tmp_path)
    fdr=pd.read_csv(tmp_path/'K2_family_calibration.csv')
    assert np.allclose(fdr.FDR,.25)  # FDP .5 then 0; not H's successful rejection rate 1.
    assert np.allclose(fdr.FWER_rate,.5)
    assert np.allclose(fdr.FDP_MCSE,.25)
    h=result[result.candidate=='H'].iloc[0]
    assert h.N_planned==2 and h.N_failed==1 and h.N_interval_available==1
    assert h.score_discovery_output==.5 and h.score_reject==1.


def test_only_Q_restriction_preserves_original_C_certificate():
    from tacb_bci.route_ab.enhancement import role_specs,raw_roles,design
    from tacb_bci.route_ab.candidate_role_lift import CandidateRoleLift
    for topology in [0,1]:
        d=draw_truth(85,'F2',True,topology,[202609160602,999,topology])
        graph=d['graph'];rc,rs=raw_roles(graph);spec=role_specs(graph,list(d['W']),['B','H','G'],rc,rs)
        q=spec['M2_candidate_restricted_Q']['Q'];assert set(q)<=set(d['W'])
        for label in ['M1_all_inputs','M3_actual_Q_closure']:
            original_C=spec[label]['C'].copy();check=CandidateRoleLift(graph,set(['B','H','G'])|set(q),available=list(d['W']))
            assert check.unresolved<=set(original_C) and check.certificate(original_C)
            assert spec[label]['C']==original_C
        expected=set(q)|({'derived__B_times_U'} if {'B','U'}<=set(q) else set())
        assert set(design(d['W'],q,True))==expected
