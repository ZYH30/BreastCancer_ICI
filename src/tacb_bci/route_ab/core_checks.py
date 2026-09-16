"""K1/K2 numerical summaries; no new identification or inference claims."""
import numpy as np
from .continuous import weighted_derivative
from .ratio_inference import ratio_score_set, set_contains


def score_summary(b, phi, m, g, sd=1., truth=None):
    stat, parts = weighted_derivative(b, phi, m, g)
    fi = ratio_score_set(parts['numerator'], parts['denominator'])
    fi_sd = ratio_score_set(parts['numerator']*sd, parts['denominator'])
    den = parts['denominator']; influence = parts['influence']
    row = dict(**stat, N=float(parts['numerator'].mean()), D=float(den.mean()),
        estimate_SD=stat['estimate']*sd, SE_SD=stat['se']*sd,
        low_SD=stat['ci_low']*sd, high_SD=stat['ci_high']*sd, scale_SD=sd,
        p_score=fi['null_score_p'], fieller_kind=fi['kind'],
        bounded=fi['kind']=='bounded', interval_available=True, fit_status='SUCCESS',
        residual_square_ESS=float(den.sum()**2/np.sum(den**2)),
        IF_max_fraction=float(np.max(influence**2)/np.sum(influence**2)),
        IF_top5_fraction=float(np.sort(influence**2)[-5:].sum()/np.sum(influence**2)),
        r_min=float(parts['residual_t'].min()),r_max=float(parts['residual_t'].max()))
    row['fieller_length']=fi['intervals'][0][1]-fi['intervals'][0][0] if row['bounded'] else np.inf
    if truth is not None:
        row.update(truth=truth, truth_SD=truth*sd, error=stat['estimate']-truth,
            covers_fieller=set_contains(fi,truth), covers_delta=stat['ci_low']<=truth<=stat['ci_high'],
            covered_and_bounded=set_contains(fi,truth) and row['bounded'])
    return row, dict(raw=fi,SD=fi_sd),parts


def distribution(x,prefix):
    x=np.asarray(x,float)
    return {prefix+'_'+name:float(value) for name,value in zip(
        ['min','q01','q25','median','q75','q99','max','SD','max_abs'],
        [*np.quantile(x,[0,.01,.25,.5,.75,.99,1]),x.std(ddof=1),np.abs(x).max()])}


def stage_diagnostics(a,raw_e,used_e,phi,g=None):
    raw_e=np.asarray(raw_e);used_e=np.asarray(used_e)
    weight=np.where(a,1/used_e,1/(1-used_e))
    result=dict(clip_fraction=float(np.mean(raw_e!=used_e)),clip_count=int(np.sum(raw_e!=used_e)),
        weight_ESS=float(weight.sum()**2/np.sum(weight**2)),max_weight=float(weight.max()),
        **distribution(raw_e,'raw_e'),**distribution(used_e,'used_e'),**distribution(phi,'phi'))
    if g is not None:result.update(distribution(g,'raw_g'),**distribution(g,'used_g'),g_projection_fraction=0.)
    return result
