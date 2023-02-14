import numpy as np
from sksurv.metrics import concordance_index_censored


def format_survival_outcome(event, fu):
    return np.array(
        list(zip(event, fu)), 
        dtype=[('censor','?'),('time','<f8')]
    )

def concordance_index(estimator, X, y):
    return concordance_index_censored(
        y['censor'], 
        y['time'], 
        estimator.predict(X)
    )[0]