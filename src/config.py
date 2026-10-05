"""Paths, column groups, and business assumptions in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "finfast_100k_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "repeat_loan"
ID_COL = "customer_id"
SEED = 42

# Out-of-time split: train on 2022-2023 applications, test on 2024
OOT_START = "2024-01-01"

# Columns never used as model inputs
#  - gender: excluded on fair-lending grounds (kept only for bias checks)
#  - approved_amount is set by the credit decision. It is a fair input here
#    because we score customers AFTER their first loan is disbursed (repeat
#    propensity), but it would be leakage in an application-time model.
PROTECTED = ["gender"]

NUMERIC_FEATURES = [
    "age", "income_clean", "credit_score", "loan_amount", "approved_amount",
    "nb_txn_cnt", "days_since_last_txn_credit_repayment_24month",
    "tenure_days", "approval_ratio", "loan_to_income", "app_month",
    "prior_apps_same_id",
]
FLAG_FEATURES = [
    "flag_income_rescaled", "flag_credit_score_missing", "flag_age_invalid",
    "flag_registered_after_application", "flag_approved_gt_applied", "flag_approved_invalid", "flag_days_negative",
    "has_complaint",
]
CATEGORICAL_FEATURES = [
    "city_clean", "employment_type", "application_channel", "loan_purpose",
    "device_type_clean", "complaint_type",
]

# --------------------------------------------------------------------------- #
# Business assumptions (illustrative, stated openly; change here, not in code)
# --------------------------------------------------------------------------- #
BUSINESS = {
    # Average repeat-loan ticket: taken from the data (mean approved amount)
    "avg_ticket_inr": None,  # filled from data at runtime
    # Net contribution per ₹ disbursed on a short-tenure loan after cost of
    # funds, opex and expected credit loss (typical digital NBFC range 4-8%)
    "net_margin_on_disbursal": 0.06,
    # Cost to acquire a NEW customer vs reactivating an existing one
    "new_customer_cac_inr": 1_500,
    "retention_offer_cost_inr": 150,
    # Share of targeted would-be repeaters whose repeat loan is *caused* by
    # the offer (incremental uplift) - to be measured by A/B test
    "offer_incremental_uplift": 0.20,
    # Share of the scored base the business can target each cycle
    "target_share": 0.30,
}
