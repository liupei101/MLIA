import pandas as pd
import numpy as np
from scipy import stats

def value_counts(Ser, frequency=False):
    """
    Ser: Pandas Series object.
    """
    N = len(Ser)
    res = dict(Ser.value_counts())
    if frequency:
        return res
    for k in res.keys():
        res[k] = 1.0 * res[k] / N
    return res

def df_tranform(df, inplace=True):
    xcols = df.columns
    df['pred'] = 0
    for idx in df.index:
        df.loc[idx] /= sum(df.loc[idx].values)
        df.loc[idx, 'pred'] = df.loc[idx].argmax()
    return df

class NaiveBayesModel(object):
    """ Naive Bayesian Model"""
    def __init__(self):
        # Laplace smoothing
        self.alpha = 1
        self.n_samples = 0
        # Number of class
        self.n_class = 0
        # (value: probilities) of class
        self.class_pr = dict()
        # {value: probilities} of one feature for every class
        self.prior_discrete_pr = dict()
        # {col: number of different values} of all features.
        self.prior_discrete_stats = dict()
        # (mean, std) of one feature for every class
        self.prior_continuous_pr = dict()

    def fit(self, X, y, discrete_cols=[], continuous_cols=[]):
        """
        X: Pandas DataFrame object, corresponding to covariates.
        y: Pandas Series object, corresponding to outcome variable.
        """
        self.n_samples = len(X)
        res = y.value_counts()
        self.n_class = len(res)
        self.class_pr = value_counts(y)
        for col in discrete_cols:
            self.prior_discrete_stats[col] = len(X[col].value_counts())
        # learn prior distribution from data
        # for a specific class
        for cls in res.index:
            # for sub_data
            sub_X = X[y==cls]
            # discrete_cols
            dt = dict()
            for col in discrete_cols:
                dt[col] = value_counts(sub_X[col], frequency=True)
            self.prior_discrete_pr[cls] = dt
            # continuous_cols
            dt = dict()
            for col in continuous_cols:
                dt[col] = [sub_X[col].mean(), sub_X[col].std()]
            self.prior_continuous_pr[cls] = dt
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols

    def calc_conditional_proba(self, y, col, value):
        """Calculate conditional probability.
        y: int64, value of class.
        col: string, name of variable.
        value: int64 or float32.
        """
        if col in self.discrete_cols:
            pr_table = self.prior_discrete_pr[y][col]
            N = sum([v for v in pr_table.values()])
            cnt = pr_table[value] if value in pr_table else 0
            return 1.0 * (cnt + self.alpha) / (N + self.prior_discrete_stats[col] * self.alpha);
        elif col in self.continuous_cols:
            [mean, std] = self.prior_continuous_pr[y][col]
            return stats.norm.pdf(value, loc=mean, scale=std)
        else:
            raise NotImplementedError('Name of column not recognized.')

    def predict_proba(self, X, y):
        """Predict probility of singleton.
        X: Pandas DataFrame object.
        y: int64 value.
        """
        p = self.class_pr[y]
        for col in X.index:
            p *= self.calc_conditional_proba(y, col, X[col])
        return p

    def predict(self, X):
        """
        X: Pandas DataFrame object.
        """
        res = dict()
        for k, v in self.class_pr.iteritems():
            L = []
            for s in X.index:
                instance = X.iloc[s]
                L.append(self.predict_proba(instance, k))
            res[k] = L
        df_res = pd.DataFrame(res)
        return df_res