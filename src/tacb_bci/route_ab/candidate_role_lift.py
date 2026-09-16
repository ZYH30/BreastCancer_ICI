"""Candidate-leaf reduction of baseline C/S role certificates to d-separation.

The leaf readouts are formal graph devices, never additional biological data.
General separator primitives are NetworkX implementations of existing methods.
"""
import networkx as nx
from .native import UPSTREAM  # register immutable upstream import path
from graph_roles import complete_graph


class CandidateRoleLift:
    def __init__(self, graph, required, raw_c=(), raw_s=(), *, available):
        self.available = set(available)
        self.required, raw_c, raw_s = set(required), set(raw_c), set(raw_s)
        if self.available & {'T', 'Y'} or not (self.required|raw_c|raw_s) <= self.available:
            raise ValueError('measured_baseline_variable_sets_required')
        self.unresolved = self.required-set(graph)
        self.graph = complete_graph(graph, self.available|{'T', 'Y'})
        self.original = nx.DiGraph()
        self.original.add_nodes_from(self.graph)
        self.original.add_edges_from((parent, child) for child, parents in self.graph.items() for parent in parents)
        if not nx.is_directed_acyclic_graph(self.original):
            raise ValueError('DAG_required_for_role_lift')
        descendants = nx.descendants(self.original, 'T')
        if self.available & descendants:
            raise ValueError('role_lift_available_domain_must_be_pre_treatment')
        parents = set(self.graph['T'])
        if not parents <= self.available:
            raise ValueError('unmeasured_treatment_parent')
        self.retain = self.required|raw_s
        self.initial_c = parents|raw_c|self.unresolved
        self.lifted = self.original.copy()
        self.lifted.remove_edges_from(list(self.lifted.out_edges('T')))
        self.targets = {'Y'}
        for r in sorted(self.retain):
            leaf = ('__candidate_role_readout__', r)
            if leaf in self.lifted:
                raise ValueError('formal_leaf_namespace_collision')
            self.lifted.add_edge(r, leaf)
            self.targets.add(leaf)

    def certificate(self, c):
        c = set(c)
        if not c <= self.available:
            raise ValueError('conditioning_set_outside_available_baseline')
        return bool(nx.is_d_separator(self.lifted, {'T'}, self.targets, c))

    def greedy_close(self):
        c = self.initial_c.copy()
        if not self.certificate(c):
            raise ValueError('initial_candidate_interface_not_certified')
        steps, changed = [], True
        while changed:
            changed = False
            for variable in sorted(c-self.unresolved):
                reduced = c-{variable}
                accepted = self.certificate(reduced)
                steps.append(dict(variable=variable, removed=accepted))
                if accepted:
                    c = reduced
                    changed = True
        return dict(C=sorted(c), S_prec=sorted(self.retain-c), reduction_steps=steps,
                    unresolved_required_forced_C=sorted(self.unresolved),
                    formal_leaves=len(self.retain), single_treatment_baseline_DAG_only=True)

    def existing_minimal_separator(self):
        c = nx.find_minimal_d_separator(self.lifted, {'T'}, self.targets,
            included=self.unresolved, restricted=self.initial_c)
        if c is None:
            raise ValueError('no_constrained_separator')
        if not self.certificate(c):
            raise AssertionError('separator_primitive_returned_invalid_role_set')
        minimal = nx.is_minimal_d_separator(self.lifted, {'T'}, self.targets, c,
            included=self.unresolved, restricted=self.initial_c)
        if not minimal:
            raise AssertionError('separator_primitive_failed_minimality')
        return dict(C=sorted(c), S_prec=sorted(self.retain-set(c)),
                    inclusion_minimal_relative_to_forced_nodes=True,
                    cardinality_or_variance_minimum_not_claimed=True)
