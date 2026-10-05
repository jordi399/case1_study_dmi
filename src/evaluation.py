"""Model evaluation: discrimination, calibration, lift, signal tests, drift."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def score_metrics(y, p, name="model") -> dict:
    auc = roc_auc_score(y, p)
    return {
        "model": name,
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_stat(y, p),
        "PR_AUC": average_precision_score(y, p),
        "PR_AUC_random": float(np.mean(y)),
        "Brier": brier_score_loss(y, p),
        "Brier_base_rate": brier_score_loss(y, np.full(len(y), np.mean(y))),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def bootstrap_auc_ci(y, p, n_boot=500, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def label_permutation_test(y, p, n_perm=1000, seed=0):
    """Is the observed AUC better than what random labels would give?

    Shuffles the outcome n_perm times and recomputes AUC against the same
    scores. p-value = share of shuffles at least as good as the real AUC.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    observed = roc_auc_score(y, p)
    null = np.array([roc_auc_score(rng.permutation(y), p) for _ in range(n_perm)])
    p_value = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return observed, null, float(p_value)


# --------------------------------------------------------------------------- #
# Lift / gains
# --------------------------------------------------------------------------- #
def gains_table(y, p, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    # rank first so ties (flat scores) still split into equal deciles
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins,
                           labels=range(1, n_bins + 1))
    g = df.groupby("decile", observed=True).agg(n=("y", "size"), repeaters=("y", "sum"),
                                                avg_score=("p", "mean"))
    g["repeat_rate"] = g["repeaters"] / g["n"]
    g["lift"] = g["repeat_rate"] / df["y"].mean()
    g["cum_capture"] = g["repeaters"].cumsum() / g["repeaters"].sum()
    g["cum_population"] = g["n"].cumsum() / g["n"].sum()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Univariate strength: Information Value
# --------------------------------------------------------------------------- #
def information_value(x: pd.Series, y: pd.Series, bins=10) -> tuple:
    """IV with quantile bins for numerics; missing gets its own bin."""
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        b = pd.qcut(x, bins, duplicates="drop").astype(str)
    else:
        b = x.astype(str)
    b = b.where(x.notna(), "MISSING")
    t = pd.crosstab(b, y)
    t.columns = ["non", "rep"]
    t = t + 0.5  # smoothing for empty cells
    dist_rep, dist_non = t["rep"] / t["rep"].sum(), t["non"] / t["non"].sum()
    woe = np.log(dist_rep / dist_non)
    iv = float(((dist_rep - dist_non) * woe).sum())
    return iv, pd.DataFrame({"bin": t.index, "woe": woe.values, "n": (t["rep"] + t["non"] - 1).values,
                             "repeat_rate": ((t["rep"] - 0.5) / (t["rep"] + t["non"] - 1)).values})


def iv_strength(iv: float) -> str:
    """Standard credit-scoring rule of thumb (Siddiqi)."""
    if iv < 0.02:
        return "Not predictive"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    return "Strong"


def univariate_tests(f: pd.DataFrame, numeric, categorical, target) -> pd.DataFrame:
    rows = []
    for c in numeric:
        m = f[c].notna()
        r, pv = stats.pointbiserialr(f.loc[m, target], f.loc[m, c])
        iv, _ = information_value(f[c], f[target])
        rows.append({"feature": c, "type": "numeric", "test": "point-biserial r",
                     "effect": r, "p_value": pv, "IV": iv})
    for c in categorical:
        t = pd.crosstab(f[c].astype(str), f[target])
        chi2, pv, _, _ = stats.chi2_contingency(t)
        cramers_v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
        iv, _ = information_value(f[c], f[target])
        rows.append({"feature": c, "type": "categorical", "test": "chi-square (Cramér's V)",
                     "effect": cramers_v, "p_value": pv, "IV": iv})
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg: we run many tests, so control the false discovery rate
    out = out.sort_values("p_value").reset_index(drop=True)
    m = len(out)
    out["p_adj_BH"] = np.minimum.accumulate((out["p_value"] * m / (np.arange(m) + 1))[::-1])[::-1].clip(upper=1)
    out["IV_strength"] = out["IV"].apply(iv_strength)
    return out.sort_values("IV", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def psi(expected, actual, bins=10) -> float:
    """Population Stability Index between two samples of one variable."""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > bins:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    else:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True).reindex(e.index, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def psi_label(v: float) -> str:
    return "Stable" if v < 0.1 else ("Monitor" if v < 0.25 else "Shifted")


# --------------------------------------------------------------------------- #
# Label audit
# --------------------------------------------------------------------------- #
def label_audit(f: pd.DataFrame, target: str, id_col: str) -> dict:
    """Does the label behave like a real 12-month repeat flag?"""
    g = f.sort_values([id_col, "application_date"]).copy()
    nxt = g.groupby(id_col)["application_date"].shift(-1)
    gap = (nxt - g["application_date"]).dt.days
    g["id_reapplies_within_12m"] = np.where(gap.isna(), "No later application",
                                     np.where(gap <= 365, "Re-applies within 12m", "Re-applies after 12m"))
    by_reapply = g.groupby("id_reapplies_within_12m")[target].agg(["mean", "size"])

    by_year = f.groupby(f["application_date"].dt.year)[target].agg(["mean", "size"])
    last_date = f["application_date"].max()
    censored = (f["application_date"] > last_date - pd.Timedelta(days=365)).mean()
    return {"by_reapply": by_reapply, "by_year": by_year, "data_end": last_date,
            "share_without_full_12m_window": float(censored)}
