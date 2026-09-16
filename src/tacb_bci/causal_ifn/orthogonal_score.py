"""Augmented inverse-probability treatment-benefit score."""
import numpy as np

def aipw(y, a, q0, q1, e):
    y, a, q0, q1, e = np.broadcast_arrays(*[np.asarray(v, float) for v in [y,a,q0,q1,e]])
    if y.ndim != 1 or not all(np.isfinite(v).all() for v in [y,a,q0,q1,e]):
        raise ValueError('finite_aligned_vectors_required')
    if not np.isin(a,[0,1]).all() or not np.isin(y,[0,1]).all() or np.any((e<=0)|(e>=1)):
        raise ValueError('binary_A_Y_and_positive_propensity_required')
    if np.any((q0<0)|(q0>1)|(q1<0)|(q1>1)):
        raise ValueError('valid_nuisance_probabilities_required')
    return q1-q0+a*(y-q1)/e-(1-a)*(y-q0)/(1-e)
