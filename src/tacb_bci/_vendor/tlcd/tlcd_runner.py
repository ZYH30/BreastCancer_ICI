"""Callable TLCD runner for project-local graph learning experiments.

This module intentionally keeps LLM calls disabled by default. It reuses the
project-local LightGBM/KAN residual learners while adding experiment-friendly
features: bounded candidate search, residual caching, explicit edge logs, and
JSON-serializable outputs.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, norm, spearmanr
from sklearn.preprocessing import LabelEncoder

TLCD_DIR = os.path.dirname(os.path.abspath(__file__))
if TLCD_DIR not in sys.path:
    sys.path.insert(0, TLCD_DIR)

try:
    from .direction_methods import dependency_degrees, post_treatment_score, task_adaptive_direction
except ImportError:  # pragma: no cover - supports direct execution from tlcd/
    from direction_methods import dependency_degrees, post_treatment_score, task_adaptive_direction

@dataclass
class TLCDRunConfig:
    model: str = "lgb"
    cor_theta: float | None = None
    cor_quantile: float = 0.75
    min_abs_corr: float = 0.03
    ci_theta: float = 0.05
    direct_theta: float = 0.10
    anm_margin: float = 0.03
    max_condition_set_size: int = 2
    max_neighbors: int = 8
    max_corr_neighbors: int = 8
    max_importance_neighbors: int = 8
    max_condition_candidates: int = 8
    max_search_depth: int = 3
    min_samples: int = 50
    use_optuna: bool = False
    lgb_estimators: int = 80
    lgb_max_depth: int = 3
    lgb_num_leaves: int = 15
    lgb_min_child_samples: int = 30
    llm_mode: str = "off"
    direction_strategy: str = "anm_only"
    random_state: int = 2026
    standardize: bool = True
    verbose: bool = False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def complete_graph(graph: dict[str, list[str]], nodes: list[str] | set[str] | None = None) -> dict[str, list[str]]:
    all_nodes = set([] if nodes is None else list(nodes))
    all_nodes.update(graph.keys())
    for parents in graph.values():
        all_nodes.update(parents)
    completed = {node: sorted(set(graph.get(node, [])) - {node}) for node in sorted(all_nodes)}
    return completed


def project_graph_to_dag(
    graph: dict[str, list[str]],
    nodes: list[str] | set[str] | None = None,
    edge_priorities: dict[tuple[str, str], tuple[float, ...]] | None = None,
    protected_edges: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Retain high-priority directions while rejecting edges that close a cycle."""

    completed = complete_graph(graph, nodes)
    edge_priorities = edge_priorities or {}
    protected_edges = protected_edges or set()
    edges = {
        (parent, child)
        for child, parents in completed.items()
        for parent in parents
        if parent != child
    }
    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            0 if edge in protected_edges else 1,
            *edge_priorities.get(edge, (999.0, 0.0)),
            edge[0],
            edge[1],
        ),
    )

    dag = nx.DiGraph()
    dag.add_nodes_from(completed)
    rejected = []
    for parent, child in ordered_edges:
        if nx.has_path(dag, child, parent):
            rejected.append(
                {
                    "parent": parent,
                    "child": child,
                    "priority": list(edge_priorities.get((parent, child), (999.0, 0.0))),
                    "protected": (parent, child) in protected_edges,
                    "reason": "would_create_cycle",
                }
            )
            continue
        dag.add_edge(parent, child)

    projected = {
        node: sorted(dag.predecessors(node))
        for node in sorted(dag.nodes)
    }
    return {
        "graph": projected,
        "rejected_edges": rejected,
        "retained_edges": sorted([list(edge) for edge in dag.edges]),
    }


def merge_graphs(
    graph_results: list[dict[str, Any]],
    nodes: list[str] | set[str] | None = None,
    prefer_edges: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge target-wise TLCD graphs and remove reciprocal conflicts."""

    prefer_edges = prefer_edges or set()
    edge_scores: dict[tuple[str, str], tuple[float, int]] = {}
    merged: dict[str, set[str]] = {}
    conflicts = []

    for result in graph_results:
        for child, parents in result.get("graph", {}).items():
            merged.setdefault(child, set()).update(parents)
        for decision in result.get("edge_decisions", []):
            if decision.get("accepted_edge") and decision.get("edge_parent") and decision.get("edge_child"):
                edge = (decision["edge_parent"], decision["edge_child"])
                score = float(decision.get("orientation_confidence", 0.0) or 0.0)
                depth = int(decision.get("node_depth", 999) if not pd.isna(decision.get("node_depth", 999)) else 999)
                current_score, current_depth = edge_scores.get(edge, (-1.0, 999))
                if depth < current_depth or (depth == current_depth and score > current_score):
                    edge_scores[edge] = (score, depth)

    all_edges = {(parent, child) for child, parents in merged.items() for parent in parents if parent != child}
    removed = set()
    for parent, child in sorted(all_edges):
        if (child, parent) not in all_edges or (parent, child) in removed or (child, parent) in removed:
            continue
        score_forward, depth_forward = edge_scores.get((parent, child), (0.0, 999))
        score_backward, depth_backward = edge_scores.get((child, parent), (0.0, 999))
        if (parent, child) in prefer_edges:
            keep_forward = True
        elif (child, parent) in prefer_edges:
            keep_forward = False
        elif depth_forward != depth_backward:
            keep_forward = depth_forward < depth_backward
        else:
            keep_forward = score_forward >= score_backward

        if keep_forward:
            removed.add((child, parent))
            kept = (parent, child)
        else:
            removed.add((parent, child))
            kept = (child, parent)
        conflicts.append(
            {
                "edge_a": [parent, child],
                "edge_b": [child, parent],
                "score_a": score_forward,
                "score_b": score_backward,
                "depth_a": depth_forward,
                "depth_b": depth_backward,
                "kept": list(kept),
            }
        )

    cleaned: dict[str, list[str]] = {}
    for child, parents in merged.items():
        cleaned[child] = sorted(
            parent
            for parent in parents
            if parent != child and (parent, child) not in removed
        )

    projection = project_graph_to_dag(
        cleaned,
        nodes=nodes,
        edge_priorities={
            edge: (float(depth), -float(score))
            for edge, (score, depth) in edge_scores.items()
        },
        protected_edges=prefer_edges,
    )
    return {
        "graph": projection["graph"],
        "conflicts": conflicts,
        "removed_edges": [list(edge) for edge in sorted(removed)],
        "cycle_rejections": projection["rejected_edges"],
    }


class TLCDRunner:
    def __init__(self, config: TLCDRunConfig | None = None, variable_order: dict[str, int] | None = None):
        self.config = config or TLCDRunConfig()
        if self.config.llm_mode != "off":
            raise ValueError("This runner does not call LLMs. Use llm_mode='off' for synthetic experiments.")
        if self.config.direction_strategy not in {"anm_only", "default_parent", "task_adaptive"}:
            raise ValueError("direction_strategy must be 'anm_only', 'default_parent', or 'task_adaptive'.")
        if self.config.model not in {"lgb", "kan"}:
            raise ValueError("model must be 'lgb' or 'kan'.")

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.var_types: dict[str, str] = {}
        self.correlation_matrix: pd.DataFrame | None = None
        self.importance_cache: dict[str, pd.Series] = {}
        self.residual_cache: dict[tuple[str, tuple[str, ...], str], np.ndarray] = {}
        self.direction_degrees: dict[str, float] | None = None
        self.post_treatment_cache: dict[str, dict[str, float]] = {}
        self.edge_decisions: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.start_time = time.time()
        self.variable_order = variable_order or {}

    def prepare_data(self, data: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
        exclude_cols = exclude_cols or []
        df = data.drop(columns=[c for c in exclude_cols if c in data.columns]).copy()

        for col in df.select_dtypes(exclude=["number"]).columns:
            mask = df[col].isna()
            encoder = LabelEncoder()
            df.loc[~mask, col] = encoder.fit_transform(df.loc[~mask, col].astype(str))
            df.loc[mask, col] = -9990

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True)).fillna(0)

        var_types = {}
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            unique_ratio = nunique / max(len(df), 1)
            if nunique <= 2 or (unique_ratio < 0.01 and nunique <= 20):
                var_types[col] = "discrete"
                df[col] = df[col].round().astype(np.int32)
            else:
                var_types[col] = "continuous"
                df[col] = df[col].astype(np.float32)

        if self.config.standardize:
            for col, var_type in var_types.items():
                if var_type != "continuous":
                    continue
                std = float(df[col].std())
                mean = float(df[col].mean())
                if std > 1e-12:
                    df[col] = (df[col] - mean) / std

        self.raw_df = data.copy()
        self.df = df
        self.var_types = var_types
        self.correlation_matrix = self.compute_type_aware_correlation(df, var_types)
        if self.config.direction_strategy == "task_adaptive":
            self.direction_degrees = dependency_degrees(df)
        return df

    @staticmethod
    def _known_baseline_post_score() -> dict[str, float]:
        return {
            "r2_t": 0.0,
            "r2_y": 0.0,
            "r2_ty": 0.0,
            "gain_t_given_y": 0.0,
            "gain_y_given_t": 0.0,
            "is_post": False,
        }

    def _post_treatment_score(self, variable: str) -> dict[str, float]:
        if variable in self.post_treatment_cache:
            return self.post_treatment_cache[variable]
        if self.df is None:
            raise RuntimeError("Data must be prepared before direction scoring.")

        treatment_order = self.variable_order.get("T")
        variable_order = self.variable_order.get(variable)
        if (
            variable not in {"T", "Y"}
            and treatment_order is not None
            and variable_order is not None
            and variable_order < treatment_order
        ):
            score = self._known_baseline_post_score()
        else:
            score = post_treatment_score(
                self.df,
                variable,
                treatment="T",
                outcome="Y",
                var_types=self.var_types,
                random_state=self.config.random_state,
                n_estimators=self.config.lgb_estimators,
            )
        self.post_treatment_cache[variable] = score
        return score

    def compute_type_aware_correlation(self, df: pd.DataFrame, var_types: dict[str, str]) -> pd.DataFrame:
        variables = list(df.columns)
        cor = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables, dtype=float)
        for i, var1 in enumerate(variables):
            for var2 in variables[i + 1 :]:
                value = self._pairwise_association(df[var1], df[var2], var_types[var1], var_types[var2])
                cor.loc[var1, var2] = value
                cor.loc[var2, var1] = value
        return cor

    def _pairwise_association(self, x: pd.Series, y: pd.Series, x_type: str, y_type: str) -> float:
        if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
            return 0.0
        try:
            if x_type == "discrete" and y_type == "discrete":
                table = pd.crosstab(x, y)
                chi2 = chi2_contingency(table)[0]
                n = table.to_numpy().sum()
                if n <= 1:
                    return 0.0
                phi2 = chi2 / n
                r, k = table.shape
                phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) ** 2) / (n - 1)
                kcorr = k - ((k - 1) ** 2) / (n - 1)
                denom = min(kcorr - 1, rcorr - 1)
                return float(math.sqrt(phi2corr / denom)) if denom > 0 else 0.0
            corr, _ = spearmanr(x, y)
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception as exc:
            self.logs.append(f"association_failed: {x.name}-{y.name}: {exc}")
            return 0.0

    def run(self, data: pd.DataFrame, target: str, exclude_cols: list[str] | None = None) -> dict[str, Any]:
        self.prepare_data(data, exclude_cols=exclude_cols)
        if self.df is None or self.correlation_matrix is None:
            raise RuntimeError("Data preparation failed.")
        if target not in self.df.columns:
            raise ValueError(f"target {target!r} not found in data.")

        queue = [(target, 0)]
        processed: set[str] = set()
        graph: dict[str, list[str]] = {}
        descendants: dict[str, list[str]] = {}

        while queue:
            node, depth = queue.pop(0)
            if node in processed:
                continue
            parents, children = self.identify_parents(node, depth)
            graph[node] = sorted(set(graph.get(node, [])) | set(parents))
            for child in children:
                graph[child] = sorted(set(graph.get(child, [])) | {node})
            descendants[node] = children
            processed.add(node)
            if depth >= self.config.max_search_depth:
                continue
            for parent in parents:
                queued_nodes = {queued_node for queued_node, _ in queue}
                if parent not in processed and parent not in queued_nodes:
                    queue.append((parent, depth + 1))

        completed = complete_graph(graph, self.df.columns)
        edge_priorities: dict[tuple[str, str], tuple[float, ...]] = {}
        for decision in self.edge_decisions:
            if not decision.get("accepted_edge"):
                continue
            parent = decision.get("edge_parent")
            child = decision.get("edge_child")
            if not parent or not child:
                continue
            edge = (parent, child)
            priority = (
                float(decision.get("node_depth", 999)),
                -float(decision.get("orientation_confidence", 0.0) or 0.0),
            )
            if edge not in edge_priorities or priority < edge_priorities[edge]:
                edge_priorities[edge] = priority

        projection = project_graph_to_dag(
            completed,
            nodes=self.df.columns,
            edge_priorities=edge_priorities,
        )
        retained_edges = {
            tuple(edge)
            for edge in projection["retained_edges"]
        }
        for decision in self.edge_decisions:
            if decision.get("accepted_edge"):
                edge = (decision.get("edge_parent"), decision.get("edge_child"))
                decision["dag_retained"] = edge in retained_edges

        result = {
            "target": target,
            "graph": projection["graph"],
            "descendants": descendants,
            "var_types": self.var_types,
            "correlation_matrix": self.correlation_matrix.to_dict(),
            "edge_decisions": self.edge_decisions,
            "cycle_rejections": projection["rejected_edges"],
            "logs": self.logs,
            "config": asdict(self.config),
            "runtime_sec": time.time() - self.start_time,
        }
        return result

    def _auto_threshold(self, node: str) -> float:
        if self.config.cor_theta is not None:
            return float(self.config.cor_theta)
        assert self.correlation_matrix is not None
        vals = self.correlation_matrix[node].drop(labels=[node], errors="ignore").abs()
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return self.config.min_abs_corr
        return max(self.config.min_abs_corr, float(vals.quantile(self.config.cor_quantile)))

    def candidate_neighbors(self, node: str) -> list[str]:
        assert self.correlation_matrix is not None
        threshold = self._auto_threshold(node)
        scores = self.correlation_matrix[node].drop(labels=[node], errors="ignore").abs()
        scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
        if self.variable_order and node in self.variable_order:
            node_order = self.variable_order[node]
            scores = scores[[candidate for candidate in scores.index if self.variable_order.get(candidate, 0) <= node_order]]

        corr_selected = scores[scores >= threshold].sort_values(ascending=False)
        if self.config.max_corr_neighbors > 0:
            corr_selected = corr_selected.head(self.config.max_corr_neighbors)

        importance = self.feature_importance_scores(node)
        if self.variable_order and node in self.variable_order:
            node_order = self.variable_order[node]
            importance = importance[
                [candidate for candidate in importance.index if self.variable_order.get(candidate, 0) <= node_order]
            ]
        imp_selected = importance[importance > 0].sort_values(ascending=False)
        if self.config.max_importance_neighbors > 0:
            imp_selected = imp_selected.head(self.config.max_importance_neighbors)

        selected_names = set(corr_selected.index) | set(imp_selected.index)
        combined = pd.Series(0.0, index=sorted(selected_names), dtype=float)
        if not corr_selected.empty:
            corr_norm = corr_selected / (corr_selected.max() + 1e-12)
            combined.loc[corr_norm.index] = np.maximum(combined.loc[corr_norm.index], corr_norm)
        if not imp_selected.empty:
            imp_norm = imp_selected / (imp_selected.max() + 1e-12)
            combined.loc[imp_norm.index] = np.maximum(combined.loc[imp_norm.index], imp_norm)

        selected = combined.sort_values(ascending=False)
        if self.config.max_neighbors > 0:
            selected = selected.head(self.config.max_neighbors)
        self.logs.append(
            f"{node}: cor_threshold={threshold:.4f}, corr_candidates={list(corr_selected.index)}, "
            f"importance_candidates={list(imp_selected.index)}, candidates={list(selected.index)}"
        )
        return list(selected.index)

    def feature_importance_scores(self, node: str) -> pd.Series:
        if node in self.importance_cache:
            return self.importance_cache[node]
        assert self.df is not None

        feature_cols = [col for col in self.df.columns if col != node]
        if self.variable_order and node in self.variable_order:
            node_order = self.variable_order[node]
            feature_cols = [
                col for col in feature_cols
                if self.variable_order.get(col, 0) <= node_order
            ]
        if not feature_cols:
            out = pd.Series(dtype=float)
            self.importance_cache[node] = out
            return out

        x = self.df[feature_cols]
        y = self.df[node].to_numpy()
        y_type = self.var_types.get(node, "continuous")
        try:
            if y_type == "continuous":
                model = LGBMRegressor(
                    n_estimators=self.config.lgb_estimators,
                    learning_rate=0.05,
                    max_depth=self.config.lgb_max_depth,
                    num_leaves=self.config.lgb_num_leaves,
                    min_child_samples=self.config.lgb_min_child_samples,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=self.config.random_state,
                    n_jobs=1,
                    verbose=-1,
                )
                model.fit(x, y)
            else:
                y_encoded = LabelEncoder().fit_transform(y)
                if len(np.unique(y_encoded)) <= 1:
                    out = pd.Series(0.0, index=feature_cols)
                    self.importance_cache[node] = out
                    return out
                model = LGBMClassifier(
                    n_estimators=self.config.lgb_estimators,
                    learning_rate=0.05,
                    max_depth=self.config.lgb_max_depth,
                    num_leaves=self.config.lgb_num_leaves,
                    min_child_samples=self.config.lgb_min_child_samples,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=self.config.random_state,
                    n_jobs=1,
                    verbose=-1,
                )
                model.fit(x, y_encoded)
            importances = pd.Series(model.feature_importances_, index=feature_cols, dtype=float)
        except Exception as exc:
            self.logs.append(f"feature_importance_failed: {node}: {exc}")
            importances = pd.Series(0.0, index=feature_cols)

        self.importance_cache[node] = importances.sort_values(ascending=False)
        return self.importance_cache[node]

    def identify_parents(self, node: str, depth: int = 0) -> tuple[list[str], list[str]]:
        candidates = self.candidate_neighbors(node)
        active: set[str] = set(candidates)
        removed_by_ci: set[str] = set()
        ci_records: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}

        for candidate in sorted(candidates, key=lambda c: abs(self.correlation_matrix.loc[node, c])):
            if candidate not in active:
                continue
            condition_pool = [
                z for z in active
                if z != candidate and z != node
            ]
            if self.variable_order:
                max_order = max(
                    self.variable_order.get(node, 0),
                    self.variable_order.get(candidate, 0),
                )
                condition_pool = [
                    z for z in condition_pool
                    if self.variable_order.get(z, 0) <= max_order
                ]
            condition_pool = sorted(
                condition_pool,
                key=lambda z: max(
                    abs(float(self.correlation_matrix.loc[node, z])),
                    abs(float(self.correlation_matrix.loc[candidate, z])),
                ),
                reverse=True,
            )[: self.config.max_condition_candidates]

            independent = False
            for size in range(1, self.config.max_condition_set_size + 1):
                if len(condition_pool) < size:
                    break
                for cond_set in combinations(condition_pool, size):
                    stat, p_value = self.gcm_test(node, candidate, cond_set)
                    record = {
                        "conditioning_set": list(cond_set),
                        "statistic": stat,
                        "p_value": p_value,
                    }
                    ci_records[candidate].append(record)
                    if p_value > self.config.ci_theta:
                        independent = True
                        break
                if independent:
                    break
            if independent:
                active.remove(candidate)
                removed_by_ci.add(candidate)

        parents: list[str] = []
        children: list[str] = []

        for candidate in sorted(active):
            relation, orientation = self.orient_edge(node, candidate)
            accepted_edge = True
            if relation == "parent":
                parents.append(candidate)
                edge_parent, edge_child = candidate, node
            elif relation == "child":
                children.append(candidate)
                edge_parent, edge_child = node, candidate
            else:
                parents.append(candidate)
                accepted_edge = True
                edge_parent, edge_child = candidate, node

            self.edge_decisions.append(
                {
                    "node": node,
                    "node_depth": depth,
                    "candidate": candidate,
                    "abs_corr": abs(float(self.correlation_matrix.loc[node, candidate])),
                    "ci_removed": False,
                    "ci_tests": ci_records.get(candidate, []),
                    "final_relation": relation,
                    "direction_method": orientation.get("method"),
                    "p_candidate_to_node": orientation.get("p_candidate_to_node"),
                    "p_node_to_candidate": orientation.get("p_node_to_candidate"),
                    "orientation_confidence": orientation.get("confidence", 0.0),
                    "accepted_edge": accepted_edge,
                    "edge_parent": edge_parent,
                    "edge_child": edge_child,
                }
            )

        for candidate in sorted(removed_by_ci):
            best_p = max((r["p_value"] for r in ci_records.get(candidate, [])), default=np.nan)
            self.edge_decisions.append(
                {
                    "node": node,
                    "node_depth": depth,
                    "candidate": candidate,
                    "abs_corr": abs(float(self.correlation_matrix.loc[node, candidate])),
                    "ci_removed": True,
                    "ci_tests": ci_records.get(candidate, []),
                    "best_ci_p_value": best_p,
                    "final_relation": "non_adjacent",
                    "direction_method": "gcm",
                    "accepted_edge": False,
                    "edge_parent": None,
                    "edge_child": None,
                }
            )

        return sorted(set(parents)), sorted(set(children))

    def residualize(self, variable: str, cond_set: tuple[str, ...]) -> np.ndarray:
        assert self.df is not None
        cond_set = tuple(sorted(cond_set))
        key = (variable, cond_set, self.config.model)
        if key in self.residual_cache:
            return self.residual_cache[key]

        y = self.df[variable].to_numpy()
        if len(cond_set) == 0:
            residual = y.astype(float)
            self.residual_cache[key] = residual
            return residual

        x = self.df.loc[:, list(cond_set)].copy()
        y_type = self.var_types.get(variable, "continuous")
        if self.config.model == "lgb":
            residual = self._fit_lgb_residual(x, y, y_type)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    from .kan_models import kan_optuna
                except ImportError:  # pragma: no cover - direct execution fallback
                    from kan_models import kan_optuna

                _, _, residual, _ = kan_optuna(
                    x,
                    y,
                    y_type,
                    random_state=self.config.random_state,
                    is_optM=self.config.use_optuna,
                )

        residual = np.asarray(residual, dtype=float).reshape(-1)
        residual = np.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
        self.residual_cache[key] = residual
        return residual

    def _fit_lgb_residual(self, x: pd.DataFrame, y: np.ndarray, y_type: str) -> np.ndarray:
        if y_type == "continuous":
            model = LGBMRegressor(
                n_estimators=self.config.lgb_estimators,
                learning_rate=0.05,
                max_depth=self.config.lgb_max_depth,
                num_leaves=self.config.lgb_num_leaves,
                min_child_samples=self.config.lgb_min_child_samples,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=self.config.random_state,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(x, y)
            pred = model.predict(x)
            return y.astype(float) - np.asarray(pred, dtype=float)

        y_encoded = LabelEncoder().fit_transform(y)
        n_classes = len(np.unique(y_encoded))
        if n_classes <= 1:
            return np.zeros_like(y_encoded, dtype=float)
        model = LGBMClassifier(
            n_estimators=self.config.lgb_estimators,
            learning_rate=0.05,
            max_depth=self.config.lgb_max_depth,
            num_leaves=self.config.lgb_num_leaves,
            min_child_samples=self.config.lgb_min_child_samples,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=self.config.random_state,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x, y_encoded)
        proba = model.predict_proba(x)
        if n_classes == 2:
            return y_encoded.astype(float) - proba[:, 1]
        onehot = np.zeros((len(y_encoded), n_classes))
        onehot[np.arange(len(y_encoded)), y_encoded] = 1.0
        return np.linalg.norm(onehot - proba, axis=1)

    def gcm_test(self, x_var: str, y_var: str, cond_set: tuple[str, ...]) -> tuple[float, float]:
        rx = self.residualize(x_var, tuple(cond_set))
        ry = self.residualize(y_var, tuple(cond_set))
        if len(rx) < self.config.min_samples or len(ry) < self.config.min_samples:
            return 0.0, 1.0
        product = rx * ry
        mean_product = float(np.mean(product))
        var_product = float(np.mean(product**2) - mean_product**2)
        if var_product <= 1e-12:
            return 0.0, 1.0
        statistic = math.sqrt(len(product)) * mean_product / math.sqrt(var_product)
        p_value = float(2 * norm.sf(abs(statistic)))
        return float(statistic), p_value

    def orient_edge(self, node: str, candidate: str) -> tuple[str, dict[str, Any]]:
        if self.variable_order:
            node_order = self.variable_order.get(node)
            candidate_order = self.variable_order.get(candidate)
            if node_order is not None and candidate_order is not None and node_order != candidate_order:
                if candidate_order < node_order:
                    return "parent", {
                        "method": "temporal_order",
                        "confidence": float(abs(node_order - candidate_order)),
                    }
                return "child", {
                    "method": "temporal_order",
                    "confidence": float(abs(node_order - candidate_order)),
                }

        if self.config.direction_strategy == "task_adaptive":
            result = task_adaptive_direction(
                self.df,
                candidate,
                node,
                treatment="T",
                outcome="Y",
                var_types=self.var_types,
                random_state=self.config.random_state,
                max_samples=1500,
                n_estimators=self.config.lgb_estimators,
                precomputed_degrees=self.direction_degrees,
                precomputed_post_scores={
                    candidate: self._post_treatment_score(candidate),
                    node: self._post_treatment_score(node),
                },
                data_is_standardized=True,
            )
            if result.relation == "x_to_y":
                relation = "parent"
            elif result.relation == "y_to_x":
                relation = "child"
            else:
                relation = "parent"
            return relation, {
                "method": "task_adaptive",
                "p_candidate_to_node": result.score_xy,
                "p_node_to_candidate": result.score_yx,
                "confidence": result.confidence,
                "details": result.details,
            }

        if self.config.direction_strategy == "default_parent":
            return "parent", {"method": "default_parent", "confidence": 0.0}

        p_candidate_to_node = self.anm_score(cause=candidate, effect=node)
        p_node_to_candidate = self.anm_score(cause=node, effect=candidate)
        diff = p_candidate_to_node - p_node_to_candidate
        confidence = abs(diff)

        if (
            p_candidate_to_node > self.config.direct_theta
            and diff > self.config.anm_margin
        ):
            relation = "parent"
        elif (
            p_node_to_candidate > self.config.direct_theta
            and -diff > self.config.anm_margin
        ):
            relation = "child"
        else:
            relation = "parent"

        return relation, {
            "method": "anm",
            "p_candidate_to_node": p_candidate_to_node,
            "p_node_to_candidate": p_node_to_candidate,
            "confidence": confidence,
        }

    def anm_score(self, cause: str, effect: str) -> float:
        residual = self.residualize(effect, (cause,))
        cause_values = self.df[cause].to_numpy()
        cause_type = self.var_types.get(cause, "continuous")
        return self.independence_pvalue(cause_values, residual, cause_type)

    def independence_pvalue(self, variable: np.ndarray, residual: np.ndarray, var_type: str) -> float:
        variable = np.asarray(variable).reshape(-1)
        residual = np.asarray(residual).reshape(-1)
        try:
            if var_type == "discrete":
                values = np.unique(variable)
                groups = [residual[variable == value] for value in values]
                groups = [g for g in groups if len(g) >= 3]
                if len(groups) < 2:
                    return 1.0
                if len(groups) == 2:
                    _, p_value = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                else:
                    _, p_value = kruskal(*groups)
                return float(p_value) if not np.isnan(p_value) else 0.0
            corr, p_value = spearmanr(variable, residual)
            if np.isnan(p_value):
                return 0.0
            return float(p_value)
        except Exception as exc:
            self.logs.append(f"independence_failed: {exc}")
            return 0.0


def run_tlcd(
    data: pd.DataFrame,
    target: str,
    output_dir: str | None = None,
    exclude_cols: list[str] | None = None,
    variable_order: dict[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    config = TLCDRunConfig(**kwargs)
    runner = TLCDRunner(config, variable_order=variable_order)
    result = runner.run(data, target=target, exclude_cols=exclude_cols)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_json(result["graph"], os.path.join(output_dir, f"tlcd_graph_target_{target}.json"))
        save_json(result["descendants"], os.path.join(output_dir, f"tlcd_descendants_target_{target}.json"))
        save_json(result["config"], os.path.join(output_dir, f"tlcd_config_target_{target}.json"))
        pd.DataFrame(result["edge_decisions"]).to_csv(
            os.path.join(output_dir, f"tlcd_edge_decisions_target_{target}.csv"),
            index=False,
        )
    return result
