"""Causal-graph operations and role-separated adjustment-set extraction."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Iterable


Graph = dict[str, list[str]]


def complete_graph(graph: Graph, nodes: Iterable[str] = ()) -> Graph:
    all_nodes = set(nodes) | set(graph)
    for node_parents in graph.values():
        all_nodes.update(node_parents)
    return {
        node: sorted(set(graph.get(node, [])) - {node})
        for node in sorted(all_nodes)
    }


def get_parents(graph: Graph, node: str) -> set[str]:
    return set(graph.get(node, []))


def invert_graph(graph: Graph) -> dict[str, list[str]]:
    graph = complete_graph(graph)
    children = {node: [] for node in graph}
    for child, node_parents in graph.items():
        for parent in node_parents:
            children[parent].append(child)
    return {node: sorted(values) for node, values in children.items()}


def get_descendants(children: dict[str, list[str]], node: str) -> set[str]:
    found: set[str] = set()
    queue = deque(children.get(node, []))
    while queue:
        current = queue.popleft()
        if current in found or current == node:
            continue
        found.add(current)
        queue.extend(children.get(current, []))
    return found


def get_ancestors(graph: Graph, node: str) -> set[str]:
    found: set[str] = set()
    queue = deque(graph.get(node, []))
    while queue:
        current = queue.popleft()
        if current in found or current == node:
            continue
        found.add(current)
        queue.extend(graph.get(current, []))
    return found


def has_directed_path_without(
    source: str,
    target: str,
    blocked: str,
    children: dict[str, list[str]],
) -> bool:
    if source == blocked:
        return False
    queue = deque([source])
    visited = {source}
    while queue:
        current = queue.popleft()
        for child in children.get(current, []):
            if child == blocked:
                continue
            if child == target:
                return True
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return False


def d_separated(
    graph: Graph,
    source: str,
    target: str,
    conditioned: set[str],
) -> bool:
    """Check d-separation by ancestral moralization."""

    graph = complete_graph(graph)
    ancestral_nodes = {source, target} | set(conditioned)
    for node in tuple(ancestral_nodes):
        ancestral_nodes.update(get_ancestors(graph, node))

    undirected = {node: set() for node in ancestral_nodes}
    for child in ancestral_nodes:
        node_parents = [parent for parent in graph.get(child, []) if parent in ancestral_nodes]
        for parent in node_parents:
            undirected[parent].add(child)
            undirected[child].add(parent)
        for left, right in combinations(node_parents, 2):
            undirected[left].add(right)
            undirected[right].add(left)

    if source in conditioned or target in conditioned:
        return True
    queue = deque([source])
    visited = {source} | set(conditioned)
    while queue:
        current = queue.popleft()
        for neighbor in undirected.get(current, set()):
            if neighbor == target:
                return False
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return True


def find_c_total(graph: Graph, treatment: str, outcome: str) -> set[str]:
    graph = complete_graph(graph)
    children = invert_graph(graph)
    common_ancestors = get_ancestors(graph, treatment) & get_ancestors(graph, outcome)
    return {
        node
        for node in common_ancestors
        if has_directed_path_without(node, outcome, treatment, children)
    }


def find_c_root(graph: Graph, c_total: set[str]) -> set[str]:
    return {
        node
        for node in c_total
        if not (get_parents(graph, node) & c_total)
    }


def _valid_proximal_set(
    graph: Graph,
    c_total: set[str],
    treatment: str,
    candidate: set[str],
) -> bool:
    distal = c_total - candidate
    return all(
        d_separated(graph, treatment, node, candidate)
        for node in distal
    )


def _greedy_minimize_proximal_set(
    graph: Graph,
    c_total: set[str],
    treatment: str,
    initial: set[str],
) -> set[str]:
    candidate = set(initial)
    changed = True
    while changed:
        changed = False
        for variable in sorted(candidate):
            reduced = candidate - {variable}
            if _valid_proximal_set(
                graph,
                c_total,
                treatment,
                reduced,
            ):
                candidate = reduced
                changed = True
    return candidate


def find_proximal_set(
    graph: Graph,
    c_total: set[str],
    treatment: str,
    exact_search_max_size: int = 18,
) -> set[str]:
    """Find a d-separation-valid graph-proximal adjustment set.

    Exact minimum-cardinality search is retained for small local graphs. For
    larger common-ancestor sets, the method first tests treatment parents and
    then greedily removes redundant variables while preserving the screening
    criterion. This avoids exponential search without returning an unverified
    adjustment set.
    """

    ordered = sorted(c_total)
    if len(ordered) <= exact_search_max_size:
        for size in range(len(ordered) + 1):
            for subset in combinations(ordered, size):
                selected = set(subset)
                if _valid_proximal_set(
                    graph,
                    c_total,
                    treatment,
                    selected,
                ):
                    return selected
        return set(c_total)

    treatment_parents = get_parents(graph, treatment) & c_total
    if _valid_proximal_set(
        graph,
        c_total,
        treatment,
        treatment_parents,
    ):
        return _greedy_minimize_proximal_set(
            graph,
            c_total,
            treatment,
            treatment_parents,
        )
    return _greedy_minimize_proximal_set(
        graph,
        c_total,
        treatment,
        c_total,
    )


def find_confounder_sets(
    graph: Graph,
    treatment: str = "T",
    outcome: str = "Y",
) -> dict[str, list[str]]:
    graph = complete_graph(graph)
    c_total = find_c_total(graph, treatment, outcome)
    return {
        "total": sorted(c_total),
        "proximal": sorted(find_proximal_set(graph, c_total, treatment)),
        "root": sorted(find_c_root(graph, c_total)),
    }


def precision_variables(
    graph: Graph,
    treatment: str,
    outcome: str,
    c_set: Iterable[str],
) -> set[str]:
    graph = complete_graph(graph)
    descendants_t = get_descendants(invert_graph(graph), treatment)
    return (
        get_parents(graph, outcome)
        - descendants_t
        - {treatment}
        - set(c_set)
    )


def causal_role_sets(
    graph: Graph,
    treatment: str = "T",
    outcome: str = "Y",
) -> dict[str, set[str]]:
    confounders = find_confounder_sets(graph, treatment, outcome)
    c_proximal = set(confounders["proximal"])
    return {
        "C_Total": set(confounders["total"]),
        "C_Proximal": c_proximal,
        "C_Root": set(confounders["root"]),
        "S_Prec": precision_variables(graph, treatment, outcome, c_proximal),
    }
