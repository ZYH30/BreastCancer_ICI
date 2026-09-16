"""Nested, arm-specific probability-regression selection."""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from .native import ProbabilityRegressor

def nested_q_fit_predict(train_x, train_y, test_x, seed, grid=(.001, .01, .1, 1.)):
    y = np.asarray(train_y, int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        raise ValueError('insufficient_events_for_nested_nuisance_selection')
    splits = list(StratifiedKFold(min(3, counts.min()), shuffle=True, random_state=seed).split(train_x, y))
    loss = []
    for c in grid:
        p = np.zeros(len(y))
        for tr, te in splits:
            p[te] = ProbabilityRegressor(c).fit(train_x.iloc[tr], y[tr]).predict(train_x.iloc[te])
        loss.append(float(log_loss(y, p, labels=[0, 1])))
    best = int(np.argmin(loss))
    model = ProbabilityRegressor(grid[best]).fit(train_x, y)
    return model.predict(test_x), dict(grid=list(grid), inner_logloss=loss, selected_C=grid[best],
        inner_folds=[dict(train=tr.tolist(), test=te.tolist()) for tr, te in splits])
