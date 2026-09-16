import numpy as np
import pytest
from tacb_bci.route_ab.candidate_role_lift import CandidateRoleLift
from tacb_bci.route_ab.score_surface_roles import score_surface_closure,require_score_surface_certificate


def test_ATE_and_candidate_certificate_does_not_cover_extra_Q_input():
    g={'Z':[],'B':[],'T':['Z'],'Y':['T']};available=['B','Z']
    old=CandidateRoleLift(g,['B'],available=available).greedy_close()
    assert old['C']==[]
    with pytest.raises(ValueError,match='actual_Q'):
        require_score_surface_certificate(g,old['C'],['B'],['B','Z'],available=available)
    new=score_surface_closure(g,['B'],['B','Z'],available=available)
    assert new['C']==['Z'] and new['S_prec']==['B']
    assert require_score_surface_certificate(g,new['C'],['B'],['B','Z'],available=available)
    b=np.array([-1,-1,1,1]);z=np.array([-1,1,-1,1]);e=.5+.3*z
    q0=np.full(4,.3);q1=np.full(4,.5);s0=q0+.1*b*z;s1=q1+.1*b*z
    def score(est):return s1-s0+e*(q1-s1)/est-(1-e)*(q0-s0)/(1-est)
    bad=score(.5);good=score(e)
    np.testing.assert_allclose(bad,.2-.12*b,atol=1e-14)
    assert np.isclose(bad.mean(),.2)
    assert np.isclose(np.mean(b*bad),-.12)
    np.testing.assert_allclose(good,.2,atol=1e-14)


def test_unresolved_Q_measurement_is_not_certified_as_isolated():
    g={'B':[],'T':[],'Y':['T']};available=['B','comparator']
    result=score_surface_closure(g,['B'],['B','comparator'],available=available)
    assert result['C']==['comparator']
    with pytest.raises(ValueError):
        require_score_surface_certificate(g,[],['B'],['B','comparator'],available=available)
