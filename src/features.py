"""Feature engineering. Every feature is available at application time."""
import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FLAG_FEATURES, NUMERIC_FEATURES


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = d.copy()
    tenure = (f["application_date"] - f["registration_date"]).dt.days
    f["tenure_days"] = tenure.where(tenure >= 0)            # impossible ordering -> missing
    f["approval_ratio"] = f["approved_amount"] / f["loan_amount"]
    f["loan_to_income"] = f["loan_amount"] / f["income_clean"]
    f["app_month"] = f["application_date"].dt.month
    f["app_year"] = f["application_date"].dt.year

    # Number of EARLIER applications under the same ID (no look-ahead)
    f = f.sort_values(["customer_id", "application_date"])
    f["prior_apps_same_id"] = f.groupby("customer_id").cumcount()
    f = f.sort_index()

    # Segmentation helpers (not model inputs)
    f["credit_band"] = pd.cut(f["credit_score"], [0, 600, 700, 750, 900],
                              labels=["<600", "600-699", "700-749", "750+"], right=False)
    f["credit_band"] = f["credit_band"].cat.add_categories("No score").fillna("No score")
    f["income_band"] = pd.qcut(f["income_clean"], 3, labels=["Low", "Mid", "High"])
    f["ticket_band"] = pd.qcut(f["approved_amount"], 3, labels=["Small", "Medium", "Large"])
    return f


def model_matrix(f: pd.DataFrame, numeric=None, flags=None, cats=None) -> pd.DataFrame:
    """Columns used by the models; categoricals as pandas 'category' dtype."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    flags = FLAG_FEATURES if flags is None else flags
    cats = CATEGORICAL_FEATURES if cats is None else cats
    X = f[numeric + flags + cats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X
