"""Strict role contract for every input actually supplied to the Q models."""
from .candidate_role_lift import CandidateRoleLift


def score_surface_closure(graph, required, q_columns, raw_c=(), raw_s=(), *, available):
    retained=set(required)|set(q_columns)
    lift=CandidateRoleLift(graph,retained,raw_c,raw_s,available=available)
    result=lift.greedy_close()
    result.update(biomarker_required=sorted(set(required)),actual_Q_columns=sorted(set(q_columns)),
        role_contract='candidate_and_actual_Q_surface',graph_truth_not_established=True)
    return result


def require_score_surface_certificate(graph, c, required, q_columns, *, available):
    lift=CandidateRoleLift(graph,set(required)|set(q_columns),available=available)
    if not lift.unresolved<=set(c) or not lift.certificate(c):
        raise ValueError('C_does_not_certify_actual_Q_information_surface')
    return True
