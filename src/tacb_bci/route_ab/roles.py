"""Training-only TLCD expansion and transparent candidate-role safeguards."""
from dataclasses import asdict
import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .native import ProbabilityRegressor
from tlcd.tlcd_runner import TLCDRunConfig, TLCDRunner, merge_graphs
from graph_roles import find_confounder_sets, precision_variables, d_separated


def discover_roles(frame, anchors, *, seed, min_leaf=10, depth=1):
    """Baseline timing is known; baseline molecular arrows remain provisional.

    Search both endpoints with ancestry; then query candidate neighborhoods not
    already visited. A forced candidate is an estimand input, NOT a fabricated
    response-parent edge. All selected candidates receive the role check.
    """
    cols = [c for c in frame.columns if c not in ['T', 'Y']]
    # IO is a separate comparator measurement; no biological orientation.
    graph_cols = [c for c in cols if c != 'IO_cont' and frame[c].nunique() > 1]
    observed = graph_cols + ['T', 'Y']
    order = {c: 0 for c in graph_cols}
    order.update(T=1, Y=2)
    config = TLCDRunConfig(direction_strategy='task_adaptive', max_search_depth=depth,
        max_neighbors=8, max_corr_neighbors=6, max_importance_neighbors=6,
        max_condition_candidates=4, max_condition_set_size=1, lgb_estimators=40,
        lgb_min_child_samples=min_leaf, min_samples=30, random_state=seed)
    results, visited = [], set()
    for target in ['T', 'Y'] + list(anchors):
        if target in visited or target not in observed:
            continue
        conf = TLCDRunConfig(**asdict(config))
        if target not in ['T', 'Y']:
            conf.max_search_depth = 0  # targeted frontier; no unrestricted global graph
        runner = TLCDRunner(conf, variable_order=order)
        result = runner.run(frame[observed], target)
        result['cache_counts'] = dict(residual_fits=len(runner.residual_cache),
                                      importance_fits=len(runner.importance_cache))
        results.append(result)
        visited.add(target)
        visited.update(d['node'] for d in result['edge_decisions'])
    merged = merge_graphs(results, nodes=observed, prefer_edges={('T', 'Y')})
    graph = merged['graph']
    # Treatment effect is the prespecified query, not an empirically proven arrow.
    graph['Y'] = sorted(set(graph['Y']) | {'T'})
    raw_c = find_confounder_sets(graph, 'T', 'Y')['proximal']
    raw_s = sorted(precision_variables(graph, 'T', 'Y', raw_c))
    required = set(anchors) & set(cols)
    required.update(c for c in ['mp', 'batch_2', 'batch_3', 'IO_cont'] if c in cols)
    c = set(raw_c)
    s = (set(raw_s) | required) - c
    transitions = []
    # Structural neutrality must be checked for the entire requested candidate
    # interface, not only Y parents. A missing graph node is unresolved.
    changed = True
    while changed:
        changed = False
        for variable in sorted(s):
            if variable not in graph or not d_separated(graph, 'T', variable, c):
                s.remove(variable)
                c.add(variable)
                transitions.append(dict(variable=variable, reason='graph_non_neutral_or_unresolved'))
                changed = True
    # A-only conditional predictive check. It is a safeguard, not a test that
    # can certify independence. No pCR is used in this stage.
    a = frame['T'].to_numpy(int)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+1).split(frame, a))

    def losses(features):
        x = frame[sorted(features)].to_numpy()
        pred = np.zeros(len(a))
        for tr, te in splits:
            pred[te] = ProbabilityRegressor(.1).fit(x[tr], a[tr]).predict(x[te])
        pred = np.clip(pred, 1e-12, 1-1e-12)  # log evaluation only
        return -(a*np.log(pred)+(1-a)*np.log1p(-pred))

    before = losses(c)
    checks = []
    for variable in sorted(s):
        gain = before-losses(c | {variable})
        se = float(gain.std(ddof=1)/np.sqrt(len(gain)))
        promote = float(gain.mean()) > max(se, .005)
        checks.append(dict(variable=variable, mean_A_logloss_gain=float(gain.mean()),
                           descriptive_se=se, promote=promote, based_on_same_initial_C=True))
    for check in checks:
        if check['promote']:
            variable = check['variable']
            c.add(variable)
            s.remove(variable)
            transitions.append(dict(variable=variable, reason='A_only_oof_prediction_safeguard'))
    # Joint linear treatment information may survive individual checks.
    joint_before, joint_after = losses(c), losses(c | s)
    delta = joint_before-joint_after
    joint_se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    if s and delta.mean() > max(joint_se, .005):
        transitions.extend(dict(variable=v, reason='joint_A_only_oof_gain_unresolved_source') for v in sorted(s))
        c |= s
        s = set()
    return dict(C=sorted(c), S_prec=sorted(s), raw_C=raw_c, raw_S=raw_s,
                excluded=sorted(set(cols)-c-s), required_candidates=sorted(required),
                graph=graph, native_runs=results, merge_audit=merged, transitions=transitions,
                individual_A_checks=checks, joint_A_gain=float(delta.mean()), joint_A_gain_se=joint_se,
                graph_is_provisional=True, neutrality_not_proven_by_nonrejection=True,
                stop_reason='bounded_endpoint_ancestry_then_unvisited_candidate_frontier',
                actual_target_runs=len(results), visited_nodes=sorted(visited),
                total_CI_queries=sum(len(d.get('ci_tests', [])) for r in results for d in r['edge_decisions']))
