"""Data cleaning with an auditable data-quality log.

Every fix is (a) counted in the quality report and (b) leaves a flag column
behind, so the model can learn from "data was messy here" if that carries signal.
"""
import numpy as np
import pandas as pd

from .config import RAW_PATH, TARGET

CITY_MAP = {"Bangalore": "Bengaluru"}
DATE_FMT = "%d/%m/%y"


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame):
    """Return (clean_df, quality_report_df)."""
    d = df.copy()
    n = len(d)
    log = []

    def note(issue, rows, action):
        log.append({"issue": issue, "rows": int(rows), "pct": rows / n, "action": action})

    # --- identifiers --------------------------------------------------------
    dup_ids = d.duplicated("customer_id", keep=False).sum()
    note("customer_id appears on more than one row", dup_ids,
         "Kept rows (applications); split train/test by customer to prevent leakage")
    multi = d.groupby("customer_id")["age"].nunique()
    inconsistent = d["customer_id"].isin(multi[multi > 1].index).sum()
    note("same customer_id with different age", inconsistent,
         "IDs are not reliable person keys - flagged as a data-lineage question")

    # --- dates --------------------------------------------------------------
    d["application_date"] = pd.to_datetime(d["application_date"], format=DATE_FMT, errors="coerce")
    d["registration_date"] = pd.to_datetime(d["registration_timestamp"], format=DATE_FMT, errors="coerce")
    note("unparseable dates", d["application_date"].isna().sum() + d["registration_date"].isna().sum(),
         "Parsed as DD/MM/YY")
    after = d["registration_date"] > d["application_date"]
    d["flag_registered_after_application"] = after.astype(int)
    note("registration date after application date", after.sum(),
         "Impossible ordering - flagged; tenure set to missing")

    # --- gender -------------------------------------------------------------
    blank = d["gender"].isna().sum()
    unknown = (d["gender"] == "Unknown").sum()
    d["gender_clean"] = d["gender"].fillna("Unknown")
    note("gender blank or 'Unknown'", blank + unknown,
         "Merged into 'Unknown'; gender excluded from model (fair lending)")

    # --- city ---------------------------------------------------------------
    alias = d["city"].isin(CITY_MAP.keys()).sum()
    d["city_clean"] = d["city"].str.strip().replace(CITY_MAP)
    note("city spelling variants (Bangalore/Bengaluru)", alias, "Standardised to 'Bengaluru'")

    # --- income: mixed units -------------------------------------------------
    small = d["income"] < 1_000
    d["income_clean"] = np.where(small, d["income"] * 1_000, d["income"]).astype(float)
    d["flag_income_rescaled"] = small.astype(int)
    note("income < 1,000 (reported in ₹ thousands)", small.sum(), "Multiplied by 1,000; flagged")

    # --- age ----------------------------------------------------------------
    minor = d["age"] < 18
    d["flag_age_invalid"] = minor.astype(int)
    d.loc[minor, "age"] = np.nan
    note("age under 18 (not a legal borrower)", minor.sum(), "Set to missing; flagged for source check")

    # --- credit score -------------------------------------------------------
    cs_missing = d["credit_score"].isna()
    d["flag_credit_score_missing"] = cs_missing.astype(int)
    note("credit_score missing (thin-file / no-hit)", cs_missing.sum(),
         "Kept as missing (trees) / own WoE bin (scorecard); flagged")

    # --- approved vs applied ------------------------------------------------
    neg = d["approved_amount"] <= 0
    d["flag_approved_invalid"] = neg.astype(int)
    d.loc[neg, "approved_amount"] = np.nan
    note("approved_amount negative", neg.sum(), "Impossible value - set to missing; flagged")
    over = d["approved_amount"] > d["loan_amount"]
    d["flag_approved_gt_applied"] = over.astype(int)
    note("approved_amount > loan_amount (manual adjustment)", over.sum(),
         "Kept; flagged - approval above ask needs a policy explanation")

    # --- device, complaint --------------------------------------------------
    note("device_type blank", d["device_type"].isna().sum(), "Set to 'Unknown'")
    d["device_type_clean"] = d["device_type"].fillna("Unknown")
    note("text_complaint empty", d["text_complaint"].isna().sum(), "Treated as 'no complaint'")
    d["complaint_type"] = d["text_complaint"].fillna("none").str.strip().str.lower()
    d["has_complaint"] = (d["complaint_type"] != "none").astype(int)

    # --- days since last repayment: negative values are impossible ------------
    c = "days_since_last_txn_credit_repayment_24month"
    neg_days = d[c] < 0
    d["flag_days_negative"] = neg_days.astype(int)
    d.loc[neg_days, c] = np.nan
    note("days_since_last_repayment negative", neg_days.sum(), "Impossible value - set to missing; flagged")

    # --- outliers: winsorise heavy-tailed counts at 1st/99th pct -------------
    for c in ["nb_txn_cnt", "days_since_last_txn_credit_repayment_24month"]:
        lo, hi = d[c].quantile([0.01, 0.99])
        clipped = ((d[c] < lo) | (d[c] > hi)).sum()
        d[c] = d[c].clip(lo, hi)
        note(f"{c} outside 1st-99th percentile", clipped, f"Winsorised to [{lo:.0f}, {hi:.0f}]")

    assert d[TARGET].isin([0, 1]).all(), "target must be binary"
    report = pd.DataFrame(log)
    return d, report
