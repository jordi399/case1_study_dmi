"""Turn model performance into rupees. All assumptions come from config.BUSINESS."""
import numpy as np
import pandas as pd
from scipy import stats


def contribution_per_repeat(b: dict) -> float:
    """Net contribution from one incremental repeat loan."""
    return b["avg_ticket_inr"] * b["net_margin_on_disbursal"]


def campaign_value(capture_rate: float, n_customers: int, base_rate: float, b: dict) -> dict:
    """Value of targeting the top `target_share` of customers with a retention offer.

    capture_rate: share of all true repeaters that fall inside the targeted group
    (this is the model's recall at that cut-off).
    """
    n_target = int(n_customers * b["target_share"])
    repeaters = n_customers * base_rate
    reached = repeaters * capture_rate
    incremental_loans = reached * b["offer_incremental_uplift"]
    gross = incremental_loans * contribution_per_repeat(b)
    cost = n_target * b["retention_offer_cost_inr"]
    return {"capture_rate": capture_rate, "targeted": n_target, "repeaters_reached": reached,
            "incremental_loans": incremental_loans,
            "incremental_disbursal_inr": incremental_loans * b["avg_ticket_inr"],
            "gross_contribution_inr": gross, "offer_cost_inr": cost, "net_value_inr": gross - cost}


def capture_at_share_for_auc(auc: float, base_rate: float, share: float, n=200_000, seed=0) -> float:
    """Recall at a top-`share` cut-off for a model of a given AUC.

    Uses the standard binormal model: scores ~ N(0,1) for non-repeaters and
    N(d,1) for repeaters, with d = sqrt(2) * Phi^-1(AUC).
    """
    rng = np.random.default_rng(seed)
    d = np.sqrt(2) * stats.norm.ppf(auc)
    y = rng.random(n) < base_rate
    s = rng.normal(0, 1, n) + d * y
    cut = np.quantile(s, 1 - share)
    return float((s[y] >= cut).mean())


def value_of_model_quality(aucs, n_customers, base_rate, b) -> pd.DataFrame:
    rows = []
    for auc in aucs:
        cap = capture_at_share_for_auc(auc, base_rate, b["target_share"])
        v = campaign_value(cap, n_customers, base_rate, b)
        rows.append({"AUC": auc, **v})
    out = pd.DataFrame(rows)
    out["uplift_vs_random_inr"] = out["net_value_inr"] - out.loc[out["AUC"] == 0.5, "net_value_inr"].iloc[0]
    return out


def recall_drop_impact(recall_from, recall_to, n_customers, base_rate, b) -> pd.Series:
    """What a fall in recall (at the same targeting volume) costs the business."""
    repeaters = n_customers * base_rate
    missed = repeaters * (recall_from - recall_to)
    contrib = contribution_per_repeat(b)
    return pd.Series({
        "true_repeaters_in_base": repeaters,
        "recall_from": recall_from,
        "recall_to": recall_to,
        "additional_repeaters_missed": missed,
        "disbursal_at_stake_inr": missed * b["avg_ticket_inr"],
        "contribution_at_stake_inr": missed * contrib,
        "expected_lost_contribution_inr": missed * b["offer_incremental_uplift"] * contrib,
        "cac_to_replace_lost_loans_inr": missed * b["offer_incremental_uplift"] * b["new_customer_cac_inr"],
        "offer_spend_wasted_inr": missed * b["retention_offer_cost_inr"],  # slots go to non-repeaters instead
    })


def ab_sample_size(p0: float, mde_abs: float, alpha=0.05, power=0.8) -> int:
    """Customers per arm to detect an absolute uplift `mde_abs` on base rate p0."""
    p1 = p0 + mde_abs
    pbar = (p0 + p1) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / mde_abs ** 2
    return int(np.ceil(n))
