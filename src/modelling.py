"""Baseline (WoE logistic scorecard) and advanced (LightGBM) models."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .config import OOT_START, SEED, TARGET, ID_COL


def oot_split(f: pd.DataFrame):
    """Train on 2022-23 applications, test on 2024 (out-of-time)."""
    is_test = f["application_date"] >= pd.Timestamp(OOT_START)
    return f[~is_test].copy(), f[is_test].copy()


# --------------------------------------------------------------------------- #
# Baseline: Weight-of-Evidence logistic regression (industry scorecard)
# --------------------------------------------------------------------------- #
class WoELogit(BaseEstimator, ClassifierMixin):
    """Bins each feature on the training data, replaces values by their WoE,
    then fits a regularised logistic regression. Missing values get their own bin.
    """

    def __init__(self, n_bins=10, C=1.0):
        self.n_bins = n_bins
        self.C = C

    def _bin(self, col, s):
        if col in self.edges_:
            b = pd.cut(s, self.edges_[col], include_lowest=True).astype(str)
        else:
            b = s.astype(str)
        return b.where(s.notna(), "MISSING")

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.edges_, self.woe_ = {}, {}
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > self.n_bins:
                e = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, self.n_bins + 1)))
                e[0], e[-1] = -np.inf, np.inf
                self.edges_[c] = e
            b = self._bin(c, s)
            t = pd.crosstab(b, y) + 0.5
            t.columns = ["non", "rep"]
            self.woe_[c] = np.log((t["rep"] / t["rep"].sum()) / (t["non"] / t["non"].sum())).to_dict()
        self.columns_ = list(X.columns)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(self.transform(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def transform(self, X):
        return pd.DataFrame({c: self._bin(c, X[c]).map(self.woe_[c]).fillna(0.0).astype(float)
                             for c in self.columns_}, index=X.index)

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.transform(X))

    def predict(self, X):
        return self.lr_.predict(self.transform(X))

    def coefficients(self) -> pd.Series:
        return pd.Series(self.lr_.coef_[0], index=self.columns_).sort_values(key=abs, ascending=False)


# --------------------------------------------------------------------------- #
# Advanced: LightGBM with customer-grouped CV tuning
# --------------------------------------------------------------------------- #
PARAM_SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 800],
    "min_child_samples": [50, 100, 300, 1000],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
}


def make_lgbm(**params):
    return lgb.LGBMClassifier(random_state=SEED, subsample_freq=1, verbose=-1, **params)


def tune_lgbm(X, y, groups, n_iter=15, n_splits=5, seed=SEED):
    """Random search; GroupKFold keeps every customer_id in one fold."""
    rng = np.random.default_rng(seed)
    cv = GroupKFold(n_splits=n_splits)
    results = []
    for i in range(n_iter):
        params = {k: v[rng.integers(len(v))] for k, v in PARAM_SPACE.items()}
        oof = cross_val_predict(make_lgbm(**params), X, y, groups=groups, cv=cv,
                                method="predict_proba")[:, 1]
        results.append({**params, "cv_auc": roc_auc_score(y, oof)})
    res = pd.DataFrame(results).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in res.iloc[0].drop("cv_auc").items()}
    for k in ["num_leaves", "n_estimators", "min_child_samples"]:
        best[k] = int(best[k])
    return best, res


def grouped_cv_auc(model, X, y, groups, n_splits=5):
    oof = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof), oof


# --------------------------------------------------------------------------- #
# Positive control: can this pipeline find signal when signal exists?
# --------------------------------------------------------------------------- #
def synthetic_labels(f: pd.DataFrame, strength: float, base_rate=0.30, seed=SEED) -> np.ndarray:
    """Labels driven by plausible behaviour (higher score, more transactions,
    recent repayment, longer tenure -> more likely to repeat).

    `strength` scales the log-odds effect; 0 means pure noise.
    """
    rng = np.random.default_rng(seed)
    z = lambda s: ((s - s.mean()) / s.std()).fillna(0)
    logit = strength * (0.8 * z(f["credit_score"]) + 0.6 * z(f["nb_txn_cnt"])
                        - 0.5 * z(f["days_since_last_txn_credit_repayment_24month"])
                        + 0.3 * z(f["tenure_days"]) - 0.4 * f["has_complaint"])
    # shift intercept so the base rate stays ~30%
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(logit + mid)))).mean() < base_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + mid)))
    return (rng.random(len(f)) < p).astype(int)
