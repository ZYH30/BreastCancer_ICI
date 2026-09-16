"""Signed benefit information remaining beyond a specified continuous context."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer,StandardScaler


def _fit(g,y,test,penalty,form):
    g=np.asarray(g,float).reshape(-1,1);test=np.asarray(test,float).reshape(-1,1)
    if form=='linear':x,xt=g,test
    elif form=='spline':
        transform=SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='linear')
        x=transform.fit_transform(g);xt=transform.transform(test)
    else:raise ValueError('undeclared_context_basis')
    scale=StandardScaler().fit(x);model=Ridge(alpha=penalty).fit(scale.transform(x),y)
    return model.predict(scale.transform(xt))


def context_mean(g,y,test,*,seed,form,grid=(.1,1.,10.,100.)):
    g=np.asarray(g,float);y=np.asarray(y,float);test=np.asarray(test,float)
    if g.ndim!=1 or g.shape!=y.shape or not np.isfinite(g).all() or not np.isfinite(y).all():raise ValueError('finite_scalar_context_required')
    splits=list(KFold(3,shuffle=True,random_state=seed).split(g));loss=[]
    for penalty in grid:
        pred=np.zeros(len(y))
        for tr,te in splits:pred[te]=_fit(g[tr],y[tr],g[te],penalty,form)
        loss.append(float(np.mean((pred-y)**2)))
    best=int(np.argmin(loss))
    return _fit(g,y,test,grid[best],form),dict(form=form,grid=list(grid),MSE=loss,selected_alpha=grid[best],
        knots_and_scaling_refit_inside_each_inner_training_fold=True)
