"""Seed mapping and residualized signed-benefit moments."""
import numpy as np
from scipy.stats import t as student_t

def sklearn_seed(root, stream, fold=0):
    """Map a recorded large entropy root to the legacy sklearn uint32 range."""
    return int(np.random.SeedSequence([root, stream, fold]).generate_state(1, dtype=np.uint32)[0])

def weighted_derivative(treatment, y, m, ell):
    """Hines et al. least-squares Psi and conditional covariance numerator.

    This is a graph-conditional working inference diagnostic. The caller owns
    exchangeability, measurement, smoothness and graph-selection assumptions.
    No Gaussian-treatment or globally linear dose-response claim is made.
    """
    t, y, m, ell = [np.asarray(x, float) for x in [treatment,y,m,ell]]
    if not (t.shape == y.shape == m.shape == ell.shape) or not np.isfinite([t,y,m,ell]).all():
        raise ValueError('finite_aligned_continuous_interface_required')
    rt, ry = t-m, y-ell
    numerator = rt*ry
    denominator = rt*rt
    nv, dv = float(numerator.mean()), float(denominator.mean())
    if dv <= 1e-12 or len(y) < 4:
        raise ValueError('insufficient_continuous_residual_support')
    beta = nv/dv
    influence = (numerator-beta*denominator)/dv
    se = float(influence.std(ddof=1)/np.sqrt(len(y)))
    crit = float(student_t.ppf(.975,len(y)-1))
    ns = float(numerator.std(ddof=1)/np.sqrt(len(y)))
    return dict(estimate=beta,se=se,ci_low=beta-crit*se,ci_high=beta+crit*se,
        p_two_sided=float(2*student_t.sf(abs(beta/se),len(y)-1)) if se else (1. if beta==0 else 0.),
        tilt_derivative_numerator=nv, numerator_ci_low=nv-crit*ns,numerator_ci_high=nv+crit*ns,
        remaining_variance=dv, remaining_fraction=float(dv/np.var(t)),
        maximal_abs_influence=float(np.max(np.abs(influence))),n=len(y)),dict(
            residual_t=rt,residual_y=ry,numerator=numerator,denominator=denominator,influence=influence)
