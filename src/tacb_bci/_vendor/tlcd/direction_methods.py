"""Data-driven causal direction scores used by the TLCD synthetic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import norm, spearmanr
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


@dataclass
class DirectionResult:
    relation: str
    method: str
    score_xy: float
    score_yx: float
    confidence: float
    context: list[str]
    details: dict


def infer_var_types(df: pd.DataFrame) -> dict[str, str]:
    out = {}
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)
        if nunique <= 2 or (ratio < 0.01 and nunique <= 20):
            out[col] = "discrete"
        else:
            out[col] = "continuous"
    return out


def standardize_frame(df: pd.DataFrame, var_types: dict[str, str] | None = None) -> pd.DataFrame:
    var_types = var_types or infer_var_types(df)
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True)).fillna(0)
    for col, var_type in var_types.items():
        if col not in out.columns:
            continue
        if var_type == "continuous":
            std = float(out[col].std())
            if std > 1e-12:
                out[col] = (out[col] - float(out[col].mean())) / std
        else:
            out[col] = out[col].round().astype(int)
    return out


def association(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return 0.0
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0
    return abs(float(rho))


def dependency_degrees(df: pd.DataFrame) -> dict[str, float]:
    cols = list(df.columns)
    out = {}
    for col in cols:
        out[col] = float(sum(association(df[col].to_numpy(), df[other].to_numpy()) for other in cols if other != col))
    return out


def make_lgb_model(y_type: str, random_state: int, n_estimators: int = 120):
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    if y_type == "continuous":
        return LGBMRegressor(**params)
    return LGBMClassifier(**params)


def fit_residual(
    df: pd.DataFrame,
    effect: str,
    predictors: Iterable[str],
    var_types: dict[str, str],
    random_state: int = 2026,
    n_estimators: int = 120,
    discrete_residual: str = "quantile",
) -> np.ndarray:
    predictors = list(predictors)
    y = df[effect].to_numpy()
    if not predictors:
        return y.astype(float)

    x = df[predictors]
    y_type = var_types.get(effect, "continuous")
    if y_type == "continuous":
        model = make_lgb_model("continuous", random_state, n_estimators)
        model.fit(x, y)
        pred = model.predict(x)
        return y.astype(float) - np.asarray(pred, dtype=float)

    y_encoded = LabelEncoder().fit_transform(y)
    if len(np.unique(y_encoded)) <= 1:
        return np.zeros_like(y_encoded, dtype=float)
    model = make_lgb_model("discrete", random_state, n_estimators)
    model.fit(x, y_encoded)
    proba = model.predict_proba(x)
    if proba.shape[1] == 2:
        if discrete_residual == "quantile":
            rng = np.random.default_rng(random_state)
            p1 = np.clip(proba[:, 1], 1e-6, 1 - 1e-6)
            u = np.empty(len(y_encoded), dtype=float)
            mask0 = y_encoded == 0
            u[mask0] = rng.uniform(1e-6, 1 - p1[mask0])
            u[~mask0] = rng.uniform(1 - p1[~mask0], 1 - 1e-6)
            return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        return y_encoded.astype(float) - proba[:, 1]
    onehot = np.zeros((len(y_encoded), proba.shape[1]))
    onehot[np.arange(len(y_encoded)), y_encoded] = 1.0
    return np.linalg.norm(onehot - proba, axis=1)


def subsample_arrays(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 1500,
    random_state: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = len(x)
    if n <= max_samples:
        return x, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return x[idx], y[idx]


def _rbf_kernel(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    sq = (values - values.T) ** 2
    upper = sq[np.triu_indices_from(sq, k=1)]
    bandwidth = np.sqrt(0.5 * np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        bandwidth = 1.0
    return np.exp(-sq / (2 * bandwidth**2))


def hsic_statistic(x: np.ndarray, y: np.ndarray, max_samples: int = 1500, random_state: int = 2026) -> float:
    x, y = subsample_arrays(x, y, max_samples=max_samples, random_state=random_state)
    n = len(x)
    if n < 5 or len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return 0.0
    kx = _rbf_kernel(x)
    ky = _rbf_kernel(y)
    h = np.eye(n) - np.ones((n, n)) / n
    kxc = h @ kx @ h
    kyc = h @ ky @ h
    return float(np.sum(kxc * kyc) / ((n - 1) ** 2))


def dependence_score(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "hsic",
    max_samples: int = 1500,
    random_state: int = 2026,
) -> float:
    if method == "spearman":
        rho, _ = spearmanr(x, y)
        return abs(float(rho)) if not np.isnan(rho) else 0.0
    if method == "hsic":
        return hsic_statistic(x, y, max_samples=max_samples, random_state=random_state)
    raise ValueError(f"Unknown dependence method: {method}")


def select_context(
    df: pd.DataFrame,
    x: str,
    y: str,
    max_context: int = 3,
    forbidden: set[str] | None = None,
) -> list[str]:
    forbidden = forbidden or set()
    candidates = [c for c in df.columns if c not in {x, y} and c not in forbidden]
    scores = []
    xv = df[x].to_numpy()
    yv = df[y].to_numpy()
    for z in candidates:
        zv = df[z].to_numpy()
        score = min(association(zv, xv), association(zv, yv))
        if score > 0:
            scores.append((score, z))
    scores.sort(reverse=True)
    return [z for _, z in scores[:max_context]]


def anm_direction(
    df: pd.DataFrame,
    x: str,
    y: str,
    var_types: dict[str, str] | None = None,
    context: list[str] | None = None,
    dep_method: str = "hsic",
    random_state: int = 2026,
    max_samples: int = 1500,
    n_estimators: int = 120,
    margin_ratio: float = 0.05,
) -> DirectionResult:
    var_types = var_types or infer_var_types(df)
    context = context or []
    df = standardize_frame(df[[x, y] + context], var_types)

    residual_y = fit_residual(df, y, [x] + context, var_types, random_state, n_estimators)
    residual_x_given_context = fit_residual(df, x, context, var_types, random_state, n_estimators)
    score_xy = dependence_score(
        residual_x_given_context,
        residual_y,
        method=dep_method,
        max_samples=max_samples,
        random_state=random_state,
    )

    residual_x = fit_residual(df, x, [y] + context, var_types, random_state, n_estimators)
    residual_y_given_context = fit_residual(df, y, context, var_types, random_state, n_estimators)
    score_yx = dependence_score(
        residual_y_given_context,
        residual_x,
        method=dep_method,
        max_samples=max_samples,
        random_state=random_state,
    )

    denom = max(score_xy, score_yx, 1e-12)
    rel_gap = abs(score_xy - score_yx) / denom
    if rel_gap < margin_ratio:
        relation = "ambiguous"
    elif score_xy < score_yx:
        relation = "x_to_y"
    else:
        relation = "y_to_x"

    return DirectionResult(
        relation=relation,
        method=f"anm_{dep_method}",
        score_xy=float(score_xy),
        score_yx=float(score_yx),
        confidence=float(rel_gap),
        context=context,
        details={},
    )


def _oof_loss(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    var_types: dict[str, str],
    random_state: int,
    n_estimators: int,
) -> float:
    y = df[target].to_numpy()
    if not predictors:
        if var_types.get(target, "continuous") == "continuous":
            pred = np.full(len(y), np.mean(y))
            return float(mean_squared_error(y, pred))
        y_encoded = LabelEncoder().fit_transform(y)
        counts = np.bincount(y_encoded)
        proba = counts / counts.sum()
        pred = np.tile(proba, (len(y_encoded), 1))
        return float(log_loss(y_encoded, pred, labels=np.arange(len(proba))))

    x = df[predictors]
    y_type = var_types.get(target, "continuous")
    model = make_lgb_model(y_type, random_state, n_estimators)
    if y_type == "continuous":
        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)
        return float(mean_squared_error(y, pred))

    y_encoded = LabelEncoder().fit_transform(y)
    if len(np.unique(y_encoded)) <= 1:
        return 0.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    proba = cross_val_predict(model, x, y_encoded, cv=cv, method="predict_proba", n_jobs=None)
    return float(log_loss(y_encoded, proba, labels=np.arange(proba.shape[1])))


def predictive_gain_direction(
    df: pd.DataFrame,
    x: str,
    y: str,
    var_types: dict[str, str] | None = None,
    context: list[str] | None = None,
    random_state: int = 2026,
    n_estimators: int = 80,
    margin_ratio: float = 0.05,
) -> DirectionResult:
    var_types = var_types or infer_var_types(df)
    context = context or []
    df = standardize_frame(df[[x, y] + context], var_types)

    loss_y_base = _oof_loss(df, y, context, var_types, random_state, n_estimators)
    loss_y_with_x = _oof_loss(df, y, context + [x], var_types, random_state, n_estimators)
    loss_x_base = _oof_loss(df, x, context, var_types, random_state, n_estimators)
    loss_x_with_y = _oof_loss(df, x, context + [y], var_types, random_state, n_estimators)

    gain_xy = max(0.0, (loss_y_base - loss_y_with_x) / (loss_y_base + 1e-12))
    gain_yx = max(0.0, (loss_x_base - loss_x_with_y) / (loss_x_base + 1e-12))
    denom = max(gain_xy, gain_yx, 1e-12)
    rel_gap = abs(gain_xy - gain_yx) / denom
    if rel_gap < margin_ratio:
        relation = "ambiguous"
    elif gain_xy > gain_yx:
        relation = "x_to_y"
    else:
        relation = "y_to_x"
    return DirectionResult(
        relation=relation,
        method="predictive_gain",
        score_xy=float(gain_xy),
        score_yx=float(gain_yx),
        confidence=float(rel_gap),
        context=context,
        details={
            "loss_y_base": loss_y_base,
            "loss_y_with_x": loss_y_with_x,
            "loss_x_base": loss_x_base,
            "loss_x_with_y": loss_x_with_y,
        },
    )


def _cv_r2_continuous(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    var_types: dict[str, str],
    random_state: int,
    n_estimators: int,
) -> float:
    predictors = [p for p in predictors if p != target and p in df.columns]
    if not predictors or var_types.get(target, "continuous") != "continuous":
        return 0.0
    y = df[target].to_numpy()
    x = df[predictors]
    model = make_lgb_model("continuous", random_state, n_estimators)
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)
    return float(r2_score(y, pred))


def post_treatment_score(
    df: pd.DataFrame,
    variable: str,
    treatment: str,
    outcome: str,
    var_types: dict[str, str],
    random_state: int = 2026,
    n_estimators: int = 80,
) -> dict[str, float]:
    if variable in {treatment, outcome} or var_types.get(variable, "continuous") != "continuous":
        return {
            "r2_t": 0.0,
            "r2_y": 0.0,
            "r2_ty": 0.0,
            "gain_t_given_y": 0.0,
            "gain_y_given_t": 0.0,
            "is_post": False,
        }
    r2_t = _cv_r2_continuous(df, variable, [treatment], var_types, random_state, n_estimators)
    r2_y = _cv_r2_continuous(df, variable, [outcome], var_types, random_state, n_estimators)
    r2_ty = _cv_r2_continuous(df, variable, [treatment, outcome], var_types, random_state, n_estimators)
    gain_t_given_y = r2_ty - r2_y
    gain_y_given_t = r2_ty - r2_t
    is_post = (
        r2_ty >= 0.50
        and gain_y_given_t >= 0.10
        and (r2_t >= 0.05 or gain_t_given_y >= 0.20)
    )
    return {
        "r2_t": r2_t,
        "r2_y": r2_y,
        "r2_ty": r2_ty,
        "gain_t_given_y": gain_t_given_y,
        "gain_y_given_t": gain_y_given_t,
        "is_post": bool(is_post),
    }


def task_adaptive_direction(
    df: pd.DataFrame,
    x: str,
    y: str,
    treatment: str = "T",
    outcome: str = "Y",
    var_types: dict[str, str] | None = None,
    random_state: int = 2026,
    max_samples: int = 1500,
    n_estimators: int = 80,
    precomputed_degrees: dict[str, float] | None = None,
    precomputed_post_scores: dict[str, dict[str, float]] | None = None,
    data_is_standardized: bool = False,
) -> DirectionResult:
    var_types = var_types or infer_var_types(df)
    if not data_is_standardized:
        df = standardize_frame(df, var_types)
    degrees = precomputed_degrees or dependency_degrees(df)
    post_scores = precomputed_post_scores or {}
    post_x = post_scores.get(x)
    if post_x is None:
        post_x = post_treatment_score(
            df,
            x,
            treatment,
            outcome,
            var_types,
            random_state,
            n_estimators,
        )
    post_y = post_scores.get(y)
    if post_y is None:
        post_y = post_treatment_score(
            df,
            y,
            treatment,
            outcome,
            var_types,
            random_state,
            n_estimators,
        )

    base = anm_direction(
        df,
        x,
        y,
        var_types=var_types,
        context=[],
        dep_method="spearman",
        random_state=random_state,
        max_samples=max_samples,
        n_estimators=n_estimators,
        margin_ratio=0.03,
    )

    reason = "base_spearman_quantile_anm"
    relation = base.relation
    confidence = base.confidence

    if {x, y} == {treatment, outcome}:
        relation = "x_to_y" if x == treatment else "y_to_x"
        confidence = 1.0
        reason = "known_treatment_outcome_order"
    elif x in {treatment, outcome} and post_y["is_post"]:
        relation = "x_to_y"
        confidence = max(confidence, 0.8)
        reason = "post_treatment_detector"
    elif y in {treatment, outcome} and post_x["is_post"]:
        relation = "y_to_x"
        confidence = max(confidence, 0.8)
        reason = "post_treatment_detector"
    elif x == treatment and y != outcome and not post_y["is_post"]:
        if degrees[y] + 0.05 < degrees[x]:
            relation = "y_to_x"
            confidence = max(confidence, min(1.0, (degrees[x] - degrees[y]) / (degrees[x] + 1e-12)))
            reason = "source_sink_treatment_parent"
    elif y == treatment and x != outcome and not post_x["is_post"]:
        if degrees[x] + 0.05 < degrees[y]:
            relation = "x_to_y"
            confidence = max(confidence, min(1.0, (degrees[y] - degrees[x]) / (degrees[y] + 1e-12)))
            reason = "source_sink_treatment_parent"
    elif not post_x["is_post"] and not post_y["is_post"]:
        gap = abs(degrees[x] - degrees[y])
        threshold = max(0.15, 0.10 * max(degrees[x], degrees[y], 1e-12))
        if degrees[x] + threshold < degrees[y]:
            relation = "x_to_y"
            confidence = min(1.0, (degrees[y] - degrees[x]) / (degrees[y] + 1e-12))
            reason = "source_sink_general"
        elif degrees[y] + threshold < degrees[x]:
            relation = "y_to_x"
            confidence = min(1.0, (degrees[x] - degrees[y]) / (degrees[x] + 1e-12))
            reason = "source_sink_general"

    return DirectionResult(
        relation=relation,
        method="task_adaptive",
        score_xy=base.score_xy,
        score_yx=base.score_yx,
        confidence=float(confidence),
        context=[],
        details={
            "reason": reason,
            "base": base.__dict__,
            "degree_x": degrees.get(x),
            "degree_y": degrees.get(y),
            "post_x": post_x,
            "post_y": post_y,
        },
    )


def ensemble_direction(
    df: pd.DataFrame,
    x: str,
    y: str,
    var_types: dict[str, str] | None = None,
    context: list[str] | None = None,
    random_state: int = 2026,
    max_samples: int = 1500,
    n_estimators: int = 100,
) -> DirectionResult:
    hsic = anm_direction(
        df,
        x,
        y,
        var_types=var_types,
        context=context,
        dep_method="hsic",
        random_state=random_state,
        max_samples=max_samples,
        n_estimators=n_estimators,
        margin_ratio=0.03,
    )
    gain = predictive_gain_direction(
        df,
        x,
        y,
        var_types=var_types,
        context=context,
        random_state=random_state,
        n_estimators=max(50, n_estimators // 2),
        margin_ratio=0.05,
    )

    votes = []
    weights = []
    for result in [hsic, gain]:
        if result.relation == "x_to_y":
            votes.append(1.0)
            weights.append(max(result.confidence, 1e-6))
        elif result.relation == "y_to_x":
            votes.append(-1.0)
            weights.append(max(result.confidence, 1e-6))

    if not votes:
        relation = "ambiguous"
        confidence = 0.0
    else:
        score = float(np.average(votes, weights=weights))
        confidence = abs(score)
        if abs(score) < 0.15:
            relation = "ambiguous"
        elif score > 0:
            relation = "x_to_y"
        else:
            relation = "y_to_x"

    return DirectionResult(
        relation=relation,
        method="ensemble_hsic_gain",
        score_xy=hsic.score_xy,
        score_yx=hsic.score_yx,
        confidence=confidence,
        context=context or [],
        details={
            "hsic": hsic.__dict__,
            "gain": gain.__dict__,
        },
    )
