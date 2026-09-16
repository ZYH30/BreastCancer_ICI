import numpy as np
from scipy.stats import t as student_t
from tacb_bci.route_ab.ratio_inference import ratio_score_set,set_contains


def test_constant_denominator_reduces_to_mean_interval():
    x=np.array([1.,2.,3.,4.,5.]);r=ratio_score_set(x,np.ones(5)*2)
    half=student_t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5)/2
    np.testing.assert_allclose(r['intervals'],[[1.5-half,1.5+half]])


def test_weak_denominator_is_not_forced_bounded():
    rng=np.random.default_rng(7);x=rng.normal(1,.1,40);y=rng.normal(0,1,40)
    r=ratio_score_set(x,y);assert r['kind'] in ['outer','all','halfline']


def test_quadratic_set_matches_direct_score_inversion():
    rng=np.random.default_rng(8);x=rng.normal(size=100);y=np.exp(rng.normal(size=100))
    r=ratio_score_set(x,y)
    for b in np.linspace(-2,2,501):
        residual=x-b*y
        direct=abs(residual.mean())<=student_t.ppf(.975,99)*residual.std(ddof=1)/10
        assert set_contains(r,b)==direct


def test_null_score_and_zero_membership_agree():
    rng=np.random.default_rng(12)
    for shift in [0,.3,1]:
        r=ratio_score_set(rng.normal(shift,1,100),np.exp(rng.normal(size=100)))
        assert r['zero_in_set']==(r['null_score_p']>=.05)
