"""Binary-outcome nuisance adapters and the vendored native graph backend."""
from pathlib import Path
import sys
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

UPSTREAM = Path(__file__).resolve().parents[1] / '_vendor'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

class ProbabilityRegressor(RegressorMixin, BaseEstimator):
    """Regressor API returning P(Y=1), including honest constant-input fallback."""
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, int)
        if not np.isin(y, [0, 1]).all():
            raise ValueError('binary_favorable_outcome_required')
        self.mean_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-10] = 1.
        self.constant_ = float(np.average(y, weights=sample_weight))
        self.model_ = None
        if len(np.unique(y)) == 2 and np.any(X.std(axis=0) > 1e-10):
            self.model_ = LogisticRegression(C=self.C, solver='lbfgs', l1_ratio=0,
                                              max_iter=10000, tol=1e-7, random_state=17)
            self.model_.fit((X-self.mean_)/self.sd_, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if self.model_ is None:
            return np.full(len(X), self.constant_)
        return self.model_.predict_proba((X-self.mean_)/self.sd_)[:, 1]

class NamedPropensity:
    def __init__(self, columns, model=None, constant=None):
        self.feature_names = list(columns)
        self.model = model
        self.constant = constant

    def predict(self, features):
        if hasattr(features, 'columns') and list(features.columns) != self.feature_names:
            raise ValueError('propensity_role_columns_changed')
        if self.constant is not None:
            return np.full(len(features), self.constant)
        return self.model.predict(np.asarray(features))

def fit_propensity_adapter(data, c, *, seed, known_e=None, grid=(.001, .01, .1, 1., 10.)):
    a = data['T'].to_numpy(int)
    c = sorted(set(c))
    if not np.isin(a, [0, 1]).all() or len(np.unique(a)) != 2:
        raise ValueError('two_binary_treatment_arms_required')
    if known_e is not None:
        if not 0 < known_e < 1:
            raise ValueError('positive_design_probability_required')
        return NamedPropensity(c, constant=known_e), dict(mode='declared_working_design', columns=c, known_e=known_e)
    if not c:
        return NamedPropensity(c, constant=float(a.mean())), dict(mode='empty_C_empirical', columns=[])
    x = data[c].to_numpy()
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(x, a))
    losses = []
    for penalty in grid:
        pred = np.zeros(len(a))
        for tr, te in splits:
            pred[te] = ProbabilityRegressor(penalty).fit(x[tr], a[tr]).predict(x[te])
        losses.append(float(log_loss(a, pred, labels=[0, 1])))
    best = int(np.argmin(losses))
    model = ProbabilityRegressor(grid[best]).fit(x, a)
    return NamedPropensity(c, model=model), dict(mode='A_only_nested_logloss', columns=c,
        grid=list(grid), loss=losses, selected_C=grid[best],
        inner_folds=[dict(train=tr.tolist(), test=te.tolist()) for tr, te in splits])
