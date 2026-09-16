"""Score-inverted Fieller-type sets, without pretending weak ratios are precise.

This is classical ratio inference, not a new causal-identification theorem.
Studentization is a working approximation outside joint normal/large samples.
"""
import numpy as np
from scipy.stats import t as student_t


def ratio_score_set(numerator,denominator,alpha=.05):
    x,y=np.asarray(numerator,float),np.asarray(denominator,float)
    if x.shape!=y.shape or x.ndim!=1 or len(x)<4 or not np.isfinite([x,y]).all():
        raise ValueError('aligned_finite_ratio_moments_required')
    n=len(x);a0,b0=x.mean(),y.mean()
    cov=np.cov(np.vstack([x,y]),ddof=1)/n;crit=float(student_t.ppf(1-alpha/2,n-1))
    aa=float(b0*b0-crit*crit*cov[1,1]);bb=float(-2*a0*b0+2*crit*crit*cov[0,1]);cc=float(a0*a0-crit*crit*cov[0,0])
    disc=bb*bb-4*aa*cc;tol=1e-12*max(abs(aa),abs(bb),abs(cc),1e-12)
    if abs(aa)<=tol:
        if abs(bb)<=tol:kind='all' if cc<=0 else 'empty';intervals=[[None,None]] if cc<=0 else []
        elif bb>0:kind='halfline';intervals=[[None,float(-cc/bb)]]
        else:kind='halfline';intervals=[[float(-cc/bb),None]]
    elif disc<0:
        kind='all' if aa<0 else 'empty';intervals=[[None,None]] if aa<0 else []
    else:
        roots=sorted([float((-bb-np.sqrt(disc))/(2*aa)),float((-bb+np.sqrt(disc))/(2*aa))])
        if aa>0:kind='bounded';intervals=[roots]
        else:kind='outer';intervals=[[None,roots[0]],[roots[1],None]]
    se0=float(np.sqrt(cov[0,0]));p0=float(2*student_t.sf(abs(a0/se0),n-1)) if se0 else (1. if a0==0 else 0.)
    return dict(kind=kind,intervals=intervals,quadratic_coefficients=[aa,bb,cc],
        null_score_p=p0,zero_in_set=bool(cc<=0),N=float(a0),D=float(b0),n=n,
        denominator_mean_SE=float(np.sqrt(cov[1,1])),exact_finite_sample_coverage_claimed=False)


def set_contains(result,value):
    a,b,c=result['quadratic_coefficients'];v=float(value)
    return bool(a*v*v+b*v+c<=1e-12*max(abs(a*v*v),abs(b*v),abs(c),1e-12))
