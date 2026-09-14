"""
taxpayer_framework.py
=====================
Taxpayer Classification and Tax Anomaly Detection Using Unsupervised Machine
Learning with SHAP-Based Explainability.

Four-stage framework
    Stage 1  Taxpayer segmentation (K-Means on an income-composition space)
    Stage 2  Peer-group Isolation Forest (one forest per segment), compared
             with a population-wide Isolation Forest
    Stage 3  Supervised anomaly classification (Random Forest and Histogram
             Gradient Boosting) trained on INJECTED synthetic anomaly labels
    Stage 4  SHAP explainability (surrogate model for the segmentation,
             TreeExplainer for the Isolation Forests and the Random Forest)

Data
    If ../data/2023_sample_file_SA4.csv (the ATO 2022-23 individual sample
    file) is present it is used.  Otherwise a schema-matched synthetic dataset
    is generated so the script runs end-to-end without any external file.

The injected anomaly labels are SYNTHETIC.  They are not real audit outcomes
and nothing in this script identifies fraud or non-compliance.

Run:  python taxpayer_framework.py
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, average_precision_score,
                             calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================================
# 2. SET RANDOM SEED AND CONFIGURATION
# ============================================================================
SEED = 42
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "2023_sample_file_SA4.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
TAB = ROOT / "tables"
for d in (FIG, RES, TAB):
    d.mkdir(parents=True, exist_ok=True)

INJECT_RATE = 0.01                    # share of records perturbed (synthetic anomalies)
CONTAMINATION_GRID = [0.01, 0.02, 0.05, 0.10]
CONTAMINATION = 0.02                  # operating point
K_RANGE = list(range(3, 11))
MIN_SEGMENT_SIZE = 2000               # a segment must support its own forest
SIL_SAMPLE = 20000                    # sample used for silhouette / DB / CH
WARD_SAMPLE = 8000                    # Ward linkage is O(n^2) memory
SPARSITY_MIN = 0.01                   # a variable must be non-zero for >= 1% of taxpayers
CORR_MAX = 0.95                       # redundancy screen
INCOME_FLOOR = 1000.0                 # denominator floor (AUD) for ratios
TEST_SIZE = 0.30
SHAP_PER_SEGMENT = 300                # flagged + unflagged rows per segment for SHAP
SYNTH_N = 60000                       # size of the fallback synthetic dataset

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
PALETTE = sns.color_palette("tab10")

SUMMARY = {}                          # every headline number quoted in the paper
T0 = time.time()
LOG_FILE = RES / "run_log.txt"
LOG_FILE.write_text("", encoding="utf-8")


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def section(title):
    log("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()
    log(f"  saved figure: {name}")


def savetab(df, name, index=True):
    df.to_csv(TAB / name, index=index)
    log(f"  saved table: {name}")


# Column groups from the ATO data dictionary ("Variables in 2022-23 sample file")
ID_COL = "Ind"
DEMOGRAPHIC_COLS = ["Gender", "age_range", "Occ_code", "Partner_status", "SA4",
                    "Lodgment_method", "PHI_Ind"]
INCOME_COLS = ["Sw_amt", "Alow_ben_amt", "ETP_txbl_amt", "Grs_int_amt",
               "Aust_govt_pnsn_allw_amt", "Unfranked_Div_amt", "Frk_Div_amt",
               "Dividends_franking_cr_amt", "Net_rent_amt", "Gross_rent_amt",
               "Other_rent_ded_amt", "Rent_int_ded_amt", "Rent_cap_wks_amt",
               "Net_farm_management_amt", "Net_PP_BI_amt", "Net_NPP_BI_amt",
               "Total_PP_BI_amt", "Total_NPP_BI_amt", "Total_PP_BE_amt",
               "Total_NPP_BE_amt", "Net_CG_amt", "Tot_CY_CG_amt", "Net_PT_PP_dsn",
               "Net_PT_NPP_dsn", "Taxed_othr_pnsn_amt", "Untaxed_othr_pnsn_amt",
               "Other_foreign_inc_amt", "Other_inc_amt", "Tot_IncLoss_amt"]
DEDUCTION_COLS = ["WRE_car_amt", "WRE_trvl_amt", "WRE_uniform_amt", "WRE_self_amt",
                  "WRE_other_amt", "Div_Ded_amt", "Intrst_Ded_amt", "Gift_amt",
                  "Non_emp_spr_amt", "Cost_tax_affairs_amt", "Other_Ded_amt",
                  "Tot_ded_amt"]
DEDUCTION_COMPONENTS = DEDUCTION_COLS[:-1]
LOSS_COLS = ["PP_loss_claimed", "NPP_loss_claimed"]
OTHER_COLS = ["Rep_frng_ben_amt", "Asbl_forgn_source_incm_amt",
              "Net_fincl_invstmt_lss_amt", "Rptbl_Empr_spr_cont_amt",
              "Cr_PAYG_ITI_amt", "TFN_amts_wheld_gr_intst_amt",
              "TFN_amts_wheld_divs_amt", "Hrs_to_prepare_BPI_cnt",
              "Taxable_Income", "Help_debt", "Spr_Emplr_Contr", "Spr_Prsnl_Contr",
              "Spr_Othr_Contr", "Spr_Ttl_Acnt_Bal"]
AMOUNT_COLS = INCOME_COLS + DEDUCTION_COLS + LOSS_COLS + OTHER_COLS
ALL_COLS = [ID_COL] + DEMOGRAPHIC_COLS + AMOUNT_COLS
WRE_COLS = ["WRE_car_amt", "WRE_trvl_amt", "WRE_uniform_amt", "WRE_self_amt", "WRE_other_amt"]
RENT_DED_COLS = ["Other_rent_ded_amt", "Rent_int_ded_amt", "Rent_cap_wks_amt"]
OCC_NAMES = {0: "Not stated", 1: "Managers", 2: "Professionals", 3: "Technicians/trades",
             4: "Community/personal service", 5: "Clerical/admin", 6: "Sales",
             7: "Machinery operators", 8: "Labourers", 9: "Other"}


# ============================================================================
# 3. GENERATE (OR LOAD) TAXPAYER DATA
# ============================================================================
def generate_synthetic_ato_like(n, rng):
    """Schema-matched fallback dataset (used only when the ATO file is absent).
    Amounts are lognormal, identities are enforced exactly as in the ATO file:
    Tot_ded = sum of deduction items; Taxable = max(0, income - ded - losses)."""
    df = pd.DataFrame(0, index=range(n), columns=AMOUNT_COLS, dtype=float)
    kind = rng.choice(["wage", "business", "retiree", "trust", "other", "invest", "property", "none"],
                      size=n, p=[0.72, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.01])
    ln = lambda m, s, k: np.round(rng.lognormal(np.log(m), s, k))
    m = kind == "wage";     df.loc[m, "Sw_amt"] = ln(62000, 0.65, m.sum())
    m = kind == "business"; df.loc[m, "Total_NPP_BI_amt"] = ln(90000, 1.0, m.sum())
    df["Total_NPP_BE_amt"] = np.round(df["Total_NPP_BI_amt"] * rng.beta(4, 5, n))
    df["Net_NPP_BI_amt"] = df["Total_NPP_BI_amt"] - df["Total_NPP_BE_amt"]
    m = kind == "retiree";  df.loc[m, "Aust_govt_pnsn_allw_amt"] = ln(22000, 0.3, m.sum())
    m = kind == "trust";    df.loc[m, "Net_PT_NPP_dsn"] = ln(40000, 1.0, m.sum())
    m = kind == "other";    df.loc[m, "Other_inc_amt"] = ln(30000, 1.0, m.sum())
    m = kind == "invest"
    df.loc[m, "Frk_Div_amt"] = ln(15000, 1.2, m.sum()); df.loc[m, "Grs_int_amt"] = ln(6000, 1.2, m.sum())
    df["Dividends_franking_cr_amt"] = np.round(df["Frk_Div_amt"] * 0.4286)
    m = kind == "property"
    df.loc[m, "Sw_amt"] = ln(80000, 0.5, m.sum()); df.loc[m, "Gross_rent_amt"] = ln(25000, 0.6, m.sum())
    df["Rent_int_ded_amt"] = np.round(df["Gross_rent_amt"] * rng.beta(3, 3, n))
    df["Other_rent_ded_amt"] = np.round(df["Gross_rent_amt"] * rng.beta(2, 5, n))
    df["Net_rent_amt"] = df["Gross_rent_amt"] - df["Rent_int_ded_amt"] - df["Other_rent_ded_amt"] - df["Rent_cap_wks_amt"]
    has_int = rng.random(n) < 0.6
    df.loc[has_int, "Grs_int_amt"] += ln(400, 1.5, has_int.sum())
    for c, p, mean in [("WRE_car_amt", .23, 2500), ("WRE_uniform_amt", .43, 300), ("WRE_other_amt", .55, 900),
                       ("WRE_trvl_amt", .10, 1200), ("WRE_self_amt", .06, 1800), ("Gift_amt", .28, 250),
                       ("Cost_tax_affairs_amt", .39, 250), ("Other_Ded_amt", .06, 1500), ("Non_emp_spr_amt", .04, 8000)]:
        mm = (rng.random(n) < p) & (df["Sw_amt"] + df["Total_NPP_BI_amt"] > 0)
        df.loc[mm, c] = ln(mean, 0.8, mm.sum())
    df["Tot_ded_amt"] = df[DEDUCTION_COMPONENTS].sum(axis=1)
    inc_items = [c for c in INCOME_COLS if c not in ("Tot_IncLoss_amt", "Gross_rent_amt", "Other_rent_ded_amt",
                 "Rent_int_ded_amt", "Rent_cap_wks_amt", "Total_PP_BI_amt", "Total_NPP_BI_amt",
                 "Total_PP_BE_amt", "Total_NPP_BE_amt", "Tot_CY_CG_amt", "Dividends_franking_cr_amt")]
    df["Tot_IncLoss_amt"] = df[inc_items].sum(axis=1) + df["Dividends_franking_cr_amt"]
    df["Taxable_Income"] = (df["Tot_IncLoss_amt"] - df["Tot_ded_amt"]).clip(lower=0)
    df["Spr_Emplr_Contr"] = np.round(df["Sw_amt"] * 0.105)
    df["Spr_Ttl_Acnt_Bal"] = ln(60000, 1.3, n)
    df["Cr_PAYG_ITI_amt"] = np.where(kind == "business", np.round(df["Net_NPP_BI_amt"].clip(lower=0) * 0.25), 0)
    df = df.round(0)
    demo = pd.DataFrame({"Ind": np.arange(1, n + 1), "Gender": rng.integers(0, 2, n),
                         "age_range": rng.integers(0, 12, n), "Occ_code": rng.integers(0, 10, n),
                         "Partner_status": rng.integers(0, 2, n), "SA4": rng.integers(1, 89, n),
                         "Lodgment_method": rng.choice(["A", "S"], n, p=[0.63, 0.37]),
                         "PHI_Ind": rng.integers(0, 2, n)})
    return pd.concat([demo, df], axis=1)[ALL_COLS]


section("SECTION 3 - LOAD DATA")
if DATA_PATH.exists():
    raw = pd.read_csv(DATA_PATH, thousands=",", encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    DATA_SOURCE = "ATO 2022-23 individual sample file (2% sample of individual returns)"
else:
    raw = generate_synthetic_ato_like(SYNTH_N, RNG)
    DATA_SOURCE = "Synthetic schema-matched fallback dataset (ATO file not found)"
missing = [c for c in ALL_COLS if c not in raw.columns]
assert not missing, f"Missing expected columns: {missing}"
raw = raw[ALL_COLS].copy()
for c in AMOUNT_COLS:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(float)
log("Data source :", DATA_SOURCE)
log("Records     :", f"{len(raw):,}", " Columns:", raw.shape[1])
SUMMARY["data_source"] = DATA_SOURCE
SUMMARY["n_records"] = int(len(raw))
SUMMARY["n_columns"] = int(raw.shape[1])

# ============================================================================
# 4. INJECT SYNTHETIC ANOMALIES (in place, documented, labelled)
# ============================================================================
section("SECTION 4 - INJECT SYNTHETIC ANOMALIES")
df = raw.copy()
df["injected"] = 0
df["injected_type"] = "none"
original = raw.copy()   # kept for the before/after audit table


def recompute_identities(d, idx):
    """Re-apply the two exact accounting identities that hold in the ATO file."""
    d.loc[idx, "Tot_ded_amt"] = d.loc[idx, DEDUCTION_COMPONENTS].sum(axis=1)
    base = d.loc[idx, "Tot_IncLoss_amt"] - d.loc[idx, "Tot_ded_amt"] - d.loc[idx, LOSS_COLS].sum(axis=1)
    d.loc[idx, "Taxable_Income"] = base.clip(lower=0)


def inj_wre_inflation(d, idx, rng):
    """A1: work-related expenses raised to 25-45% of salary (peer 99th pct ~ 24%)."""
    target = rng.uniform(0.25, 0.45, len(idx)) * d.loc[idx, "Sw_amt"]
    add = (target - d.loc[idx, WRE_COLS].sum(axis=1)).clip(lower=0)
    d.loc[idx, "WRE_other_amt"] += np.round(add * 0.5)
    d.loc[idx, "WRE_car_amt"] += np.round(add * 0.3)
    d.loc[idx, "WRE_trvl_amt"] += np.round(add * 0.2)
    recompute_identities(d, idx)


def inj_business_expense_inflation(d, idx, rng):
    """A2: business expenses raised to 150-300% of business income (peer 99th pct ~ 220%)."""
    new_be = np.maximum(d.loc[idx, "Total_NPP_BE_amt"],
                        np.round(rng.uniform(1.5, 3.0, len(idx)) * d.loc[idx, "Total_NPP_BI_amt"]))
    delta = new_be - d.loc[idx, "Total_NPP_BE_amt"]
    d.loc[idx, "Total_NPP_BE_amt"] = new_be
    d.loc[idx, "Net_NPP_BI_amt"] -= delta
    d.loc[idx, "Tot_IncLoss_amt"] -= delta
    recompute_identities(d, idx)


def inj_rental_deduction_inflation(d, idx, rng):
    """A3: rental deductions raised to 300-500% of gross rent (peer 99th pct ~ 295%)."""
    cur = d.loc[idx, RENT_DED_COLS].sum(axis=1)
    new_total = np.round(rng.uniform(3.0, 5.0, len(idx)) * d.loc[idx, "Gross_rent_amt"])
    delta = (new_total - cur).clip(lower=0)
    share = d.loc[idx, RENT_DED_COLS].div(cur.replace(0, np.nan), axis=0)
    share = share.fillna(0)
    share.loc[cur == 0, "Rent_int_ded_amt"] = 1.0
    for c in RENT_DED_COLS:
        d.loc[idx, c] += np.round(delta * share[c])
    d.loc[idx, "Net_rent_amt"] -= delta
    d.loc[idx, "Tot_IncLoss_amt"] -= delta
    recompute_identities(d, idx)


def inj_taxable_income_violation(d, idx, rng):
    """A4: taxable income set to 40-70% of (income - deductions - losses); identity broken."""
    base = d.loc[idx, "Tot_IncLoss_amt"] - d.loc[idx, "Tot_ded_amt"] - d.loc[idx, LOSS_COLS].sum(axis=1)
    d.loc[idx, "Taxable_Income"] = np.round(rng.uniform(0.4, 0.7, len(idx)) * base)


def inj_investment_deduction_no_income(d, idx, rng):
    """A5: interest/dividend deductions of 15-35% of income with no investment income reported."""
    inc = d.loc[idx, "Tot_IncLoss_amt"]
    d.loc[idx, "Intrst_Ded_amt"] = np.round(rng.uniform(0.15, 0.35, len(idx)) * inc)
    d.loc[idx, "Div_Ded_amt"] = np.round(rng.uniform(0.0, 0.05, len(idx)) * inc)
    recompute_identities(d, idx)


inv_income = df[["Grs_int_amt", "Unfranked_Div_amt", "Frk_Div_amt"]].sum(axis=1)
rent_ded = df[RENT_DED_COLS].sum(axis=1)
base_ti = df["Tot_IncLoss_amt"] - df["Tot_ded_amt"] - df[LOSS_COLS].sum(axis=1)
ANOMALY_TYPES = [
    ("A1_wre_inflation", "Work-related expenses inflated to 25-45% of salary",
     df["Sw_amt"] > 20000, inj_wre_inflation),
    ("A2_business_expense", "Business expenses inflated to 150-300% of business income",
     (df["Total_NPP_BI_amt"] > 5000) & (df["Total_NPP_BE_amt"] < df["Total_NPP_BI_amt"]), inj_business_expense_inflation),
    ("A3_rental_deduction", "Rental deductions inflated to 300-500% of gross rent",
     (df["Gross_rent_amt"] > 5000) & (rent_ded < 1.5 * df["Gross_rent_amt"]), inj_rental_deduction_inflation),
    ("A4_taxable_identity", "Taxable income reported at 40-70% of income less deductions and losses",
     (df["Tot_IncLoss_amt"] > 30000) & (base_ti > 20000), inj_taxable_income_violation),
    ("A5_invest_ded_no_income", "Investment deductions of 15-35% of income with zero investment income",
     (inv_income == 0) & (df["Tot_IncLoss_amt"] > 30000) & (df["Intrst_Ded_amt"] == 0) & (df["Div_Ded_amt"] == 0),
     inj_investment_deduction_no_income),
]
n_target = int(round(INJECT_RATE * len(df)))
per_type = [n_target // 5 + (1 if i < n_target % 5 else 0) for i in range(5)]
used = np.zeros(len(df), dtype=bool)
inject_rows = []
for (name, desc, elig, fn), k in zip(ANOMALY_TYPES, per_type):
    pool = np.flatnonzero(elig.values & ~used)
    chosen = RNG.choice(pool, size=min(k, len(pool)), replace=False)
    used[chosen] = True
    idx = df.index[chosen]
    fn(df, idx, RNG)
    df.loc[idx, "injected"] = 1
    df.loc[idx, "injected_type"] = name
    inject_rows.append({"type": name, "description": desc, "eligible_pool": int(len(pool)), "n_injected": int(len(chosen))})
    log(f"  {name:26s} eligible={len(pool):>7,}  injected={len(chosen):>5,}  | {desc}")
inject_table = pd.DataFrame(inject_rows)
savetab(inject_table, "table_injection_design.csv", index=False)
SUMMARY["n_injected"] = int(df["injected"].sum())
SUMMARY["injected_pct"] = round(100 * df["injected"].mean(), 3)
log(f"Total injected: {SUMMARY['n_injected']:,} ({SUMMARY['injected_pct']}%)")

# before/after audit of the injection (median ratio among modified records)
aud = []
for name, *_ in ANOMALY_TYPES:
    m = df["injected_type"] == name
    o, n_ = original[m], df[m]
    r = {"type": name}
    r["ded_to_income_before"] = (o["Tot_ded_amt"] / o["Tot_IncLoss_amt"].clip(lower=1)).median()
    r["ded_to_income_after"] = (n_["Tot_ded_amt"] / n_["Tot_IncLoss_amt"].clip(lower=1)).median()
    r["taxable_before"] = o["Taxable_Income"].median(); r["taxable_after"] = n_["Taxable_Income"].median()
    aud.append(r)
savetab(pd.DataFrame(aud).round(3), "table_injection_audit.csv", index=False)

# ============================================================================
# 5. EXPLORE THE DATASET
# ============================================================================
section("SECTION 5 - EXPLORE THE DATASET")
amt = df[AMOUNT_COLS]
explore = pd.DataFrame({
    "nonzero_pct": (amt != 0).mean() * 100, "negative_pct": (amt < 0).mean() * 100,
    "median": amt.median(), "mean": amt.mean(), "p99": amt.quantile(0.99), "max": amt.max(), "min": amt.min()}).round(2)
savetab(explore, "table_descriptive_statistics.csv")
log("Blank cells :", int(df[AMOUNT_COLS].isna().sum().sum()), " Duplicate rows:", int(raw.duplicated().sum()))
log("Lodged via tax agent:", f"{(df['Lodgment_method'] == 'A').mean():.1%}")
log("Taxable income median: $", f"{df['Taxable_Income'].median():,.0f}", " Total income median: $", f"{df['Tot_IncLoss_amt'].median():,.0f}")
SUMMARY["median_total_income"] = float(df["Tot_IncLoss_amt"].median())
SUMMARY["pct_agent_lodged"] = round(100 * (df["Lodgment_method"] == "A").mean(), 2)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, c in zip(axes.ravel(), ["Sw_amt", "Tot_IncLoss_amt", "Tot_ded_amt", "Gross_rent_amt", "Total_NPP_BI_amt", "Frk_Div_amt"]):
    v = df.loc[df[c] > 0, c]
    ax.hist(np.log10(v), bins=60, color=PALETTE[0], alpha=0.85)
    ax.set_title(f"{c}  (non-zero: {100 * (df[c] != 0).mean():.1f}%)")
    ax.set_xlabel("log10(AUD)"); ax.set_ylabel("taxpayers")
fig.suptitle("Fig. 2  Distribution of key financial variables (non-zero values, log scale)")
savefig("fig02_variable_distributions.png")

# ============================================================================
# 6. PREPROCESS DATA
# ============================================================================
section("SECTION 6 - PREPROCESS DATA")
df["lodged_via_agent"] = (df["Lodgment_method"] == "A").astype(int)
df["occupation"] = df["Occ_code"].map(OCC_NAMES)

# 6a sparsity screen (structural zeros are information, not missing values)
nonzero_share = (df[AMOUNT_COLS] != 0).mean()
sparse_dropped = nonzero_share[nonzero_share < SPARSITY_MIN].index.tolist()
kept_amounts = [c for c in AMOUNT_COLS if c not in sparse_dropped]
log(f"Sparsity screen (<{SPARSITY_MIN:.0%} non-zero) removed {len(sparse_dropped)}: {sparse_dropped}")


# 6b signed-log transform: sign(x) * ln(1 + |x|)
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


slog = signed_log(df[kept_amounts])

# 6c redundancy screen |r| > 0.95 (drop the later column of each pair)
corr = slog.corr().abs()
redundant, pairs = [], []
cols = list(corr.columns)
for i, a in enumerate(cols):
    for b in cols[i + 1:]:
        if corr.loc[a, b] > CORR_MAX and a not in redundant and b not in redundant:
            redundant.append(b); pairs.append((a, b, round(corr.loc[a, b], 3)))
kept_amounts = [c for c in kept_amounts if c not in redundant]
log(f"Redundancy screen (|r|>{CORR_MAX}) removed {len(redundant)}: {pairs}")
feature_screen = pd.DataFrame({"variable": AMOUNT_COLS, "nonzero_pct": (nonzero_share * 100).round(2).values})
feature_screen["decision"] = ["dropped: sparse" if c in sparse_dropped else ("dropped: redundant" if c in redundant else "kept")
                              for c in AMOUNT_COLS]
savetab(feature_screen, "table_feature_screening.csv", index=False)
SUMMARY["n_amount_features_kept"] = len(kept_amounts)
SUMMARY["sparse_dropped"] = sparse_dropped
SUMMARY["redundant_dropped"] = [f"{b} (r={r} with {a})" for a, b, r in pairs]

# ============================================================================
# 7. ENGINEER TAX-RELATED FEATURES
# ============================================================================
section("SECTION 7 - FEATURE ENGINEERING")


def safe_ratio(num, den, cap, floor=INCOME_FLOOR):
    """num / max(den, floor), capped.  Never divides by zero."""
    return (num / np.maximum(den, floor)).clip(upper=cap)


inc = df["Tot_IncLoss_amt"]
wre = df[WRE_COLS].sum(axis=1)
rent_ded = df[RENT_DED_COLS].sum(axis=1)
inv_income = df[["Grs_int_amt", "Unfranked_Div_amt", "Frk_Div_amt"]].sum(axis=1)
base_ti = inc - df["Tot_ded_amt"] - df[LOSS_COLS].sum(axis=1)

# --- income-composition (segmentation) features: "what kind of return is this?"
comp = pd.DataFrame(index=df.index)
comp["wage"] = df[["Sw_amt", "Alow_ben_amt", "ETP_txbl_amt"]].sum(axis=1)
comp["business"] = df[["Total_PP_BI_amt", "Total_NPP_BI_amt"]].clip(lower=0).sum(axis=1)
comp["property"] = df["Gross_rent_amt"].clip(lower=0)
comp["investment"] = df[["Grs_int_amt", "Unfranked_Div_amt", "Frk_Div_amt", "Dividends_franking_cr_amt", "Tot_CY_CG_amt"]].clip(lower=0).sum(axis=1)
comp["pension"] = df[["Aust_govt_pnsn_allw_amt", "Taxed_othr_pnsn_amt", "Untaxed_othr_pnsn_amt"]].sum(axis=1)
comp["trust"] = df[["Net_PT_PP_dsn", "Net_PT_NPP_dsn"]].abs().sum(axis=1)
comp["other"] = df[["Other_inc_amt", "Other_foreign_inc_amt", "Net_farm_management_amt"]].abs().sum(axis=1)
gross_activity = comp.sum(axis=1)
SEG_FEATURES = []
for c in comp.columns:
    df[f"share_{c}"] = np.where(gross_activity > 0, comp[c] / gross_activity.replace(0, np.nan), 0.0)
    SEG_FEATURES.append(f"share_{c}")
df["log_gross_activity"] = np.log1p(gross_activity)
df["log_super_balance"] = np.log1p(df["Spr_Ttl_Acnt_Bal"].clip(lower=0))
SEG_FEATURES += ["log_gross_activity", "log_super_balance"]
df["gross_activity"] = gross_activity

# --- behavioural ratio features (anomaly detection): "how does the return look?"
RATIO_FEATURES = {
    "deduction_to_income_ratio": safe_ratio(df["Tot_ded_amt"], inc, 5),
    "wre_to_salary_ratio": safe_ratio(wre, df["Sw_amt"], 5),
    "business_expense_ratio": safe_ratio(df["Total_NPP_BE_amt"], df["Total_NPP_BI_amt"], 10),
    "rental_deduction_ratio": safe_ratio(rent_ded, df["Gross_rent_amt"], 10),
    "taxable_income_ratio": pd.Series(np.where(base_ti > 0, df["Taxable_Income"] / base_ti.clip(lower=1),
                                               np.where(df["Taxable_Income"] == 0, 1.0, 0.0)), index=df.index).clip(0, 2),
    "investment_deduction_ratio": safe_ratio(df["Div_Ded_amt"] + df["Intrst_Ded_amt"], inv_income, 10),
    "gift_to_income_ratio": safe_ratio(df["Gift_amt"], inc, 2),
    "tax_affairs_cost_ratio": safe_ratio(df["Cost_tax_affairs_amt"], inc, 1),
    "personal_super_ratio": safe_ratio(df["Non_emp_spr_amt"] + df["Spr_Prsnl_Contr"], inc, 5),
    "payg_instalment_ratio": safe_ratio(df["Cr_PAYG_ITI_amt"], inc, 2),
    "items_reported": (df[AMOUNT_COLS] != 0).sum(axis=1).astype(float),
    "lodged_via_agent": df["lodged_via_agent"].astype(float),
}
for k_, v in RATIO_FEATURES.items():
    df[k_] = v.astype(float)
RATIO_NAMES = list(RATIO_FEATURES.keys())
AD_AMOUNT_NAMES = [f"log_{c}" for c in kept_amounts]
for c in kept_amounts:
    df[f"log_{c}"] = signed_log(df[c])
AD_FEATURES = AD_AMOUNT_NAMES + RATIO_NAMES
log(f"Segmentation features ({len(SEG_FEATURES)}): {SEG_FEATURES}")
log(f"Anomaly-detection features ({len(AD_FEATURES)}): {len(AD_AMOUNT_NAMES)} signed-log amounts + {len(RATIO_NAMES)} ratios/indicators")
SUMMARY["seg_features"] = SEG_FEATURES
SUMMARY["n_ad_features"] = len(AD_FEATURES)
SUMMARY["ratio_features"] = RATIO_NAMES
ratio_desc = df[RATIO_NAMES].describe(percentiles=[.5, .9, .99]).T.round(3)
savetab(ratio_desc, "table_ratio_features.csv")

# ============================================================================
# 8. STAGE 1 - TAXPAYER SEGMENTATION
# ============================================================================
section("SECTION 8 - STAGE 1: TAXPAYER SEGMENTATION (K-MEANS)")
seg_scaler = StandardScaler()
X_seg = seg_scaler.fit_transform(df[SEG_FEATURES].values)
sil_idx = RNG.choice(len(df), size=min(SIL_SAMPLE, len(df)), replace=False)

# ============================================================================
# 9. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
section("SECTION 9 - CHOOSING THE NUMBER OF CLUSTERS")
sel_rows, km_models = [], {}
for k in K_RANGE:
    t = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_seg)
    lab = km.labels_
    sizes = np.bincount(lab)
    row = {"k": k, "inertia": km.inertia_,
           "silhouette": silhouette_score(X_seg[sil_idx], lab[sil_idx]),
           "davies_bouldin": davies_bouldin_score(X_seg[sil_idx], lab[sil_idx]),
           "calinski_harabasz": calinski_harabasz_score(X_seg[sil_idx], lab[sil_idx]),
           "smallest_segment": int(sizes.min()), "largest_segment_pct": round(100 * sizes.max() / len(lab), 1),
           "seconds": round(time.time() - t, 1)}
    sel_rows.append(row); km_models[k] = km
    log(f"  k={k}: silhouette={row['silhouette']:.4f} DB={row['davies_bouldin']:.3f} CH={row['calinski_harabasz']:.0f} smallest={row['smallest_segment']:,}")
sel = pd.DataFrame(sel_rows)
savetab(sel.round(4), "table_cluster_selection.csv", index=False)
eligible = sel[sel["smallest_segment"] >= MIN_SEGMENT_SIZE]
K_BEST = int(eligible.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]["k"])
log(f"SELECTION RULE: highest silhouette among k whose smallest segment >= {MIN_SEGMENT_SIZE:,}  ->  k = {K_BEST}")
SUMMARY["k_selected"] = K_BEST
SUMMARY["cluster_selection"] = sel.round(4).to_dict("records")

# alternative algorithms at the chosen k (evaluated on the same subsample)
alt_rows = []
Xs = X_seg[sil_idx]
km_alt = km_models[K_BEST]
alt_rows.append({"model": f"K-Means (k={K_BEST})", "silhouette": silhouette_score(Xs, km_alt.labels_[sil_idx]),
                 "davies_bouldin": davies_bouldin_score(Xs, km_alt.labels_[sil_idx]), "smallest_pct": 100 * np.bincount(km_alt.labels_).min() / len(df)})
for cov in ["diag", "full"]:
    t = time.time()
    gm = GaussianMixture(n_components=K_BEST, covariance_type=cov, random_state=SEED, n_init=1).fit(X_seg)
    gl = gm.predict(X_seg)
    alt_rows.append({"model": f"Gaussian Mixture ({cov}, k={K_BEST})", "silhouette": silhouette_score(Xs, gl[sil_idx]),
                     "davies_bouldin": davies_bouldin_score(Xs, gl[sil_idx]), "smallest_pct": 100 * np.bincount(gl).min() / len(df)})
ward_idx = RNG.choice(len(df), size=min(WARD_SAMPLE, len(df)), replace=False)
wl = AgglomerativeClustering(n_clusters=K_BEST, linkage="ward").fit_predict(X_seg[ward_idx])
alt_rows.append({"model": f"Ward agglomerative (k={K_BEST}, {WARD_SAMPLE:,}-row sample)", "silhouette": silhouette_score(X_seg[ward_idx], wl),
                 "davies_bouldin": davies_bouldin_score(X_seg[ward_idx], wl), "smallest_pct": 100 * np.bincount(wl).min() / len(wl)})
alt = pd.DataFrame(alt_rows).round(4)
savetab(alt, "table_clustering_alternatives.csv", index=False)
log(alt.to_string(index=False))
SUMMARY["clustering_alternatives"] = alt.to_dict("records")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(sel["k"], sel["inertia"] / 1e3, "o-"); axes[0].set_title("Elbow: within-cluster SSE (x1000)"); axes[0].set_xlabel("k")
axes[1].plot(sel["k"], sel["silhouette"], "o-", color=PALETTE[1]); axes[1].set_title("Silhouette (20,000-row sample)"); axes[1].set_xlabel("k")
axes[2].plot(sel["k"], sel["davies_bouldin"], "o-", color=PALETTE[2]); axes[2].set_title("Davies-Bouldin (lower is better)"); axes[2].set_xlabel("k")
for ax in axes:
    ax.axvline(K_BEST, ls="--", color="grey"); ax.set_xticks(K_RANGE)
fig.suptitle("Fig. 3  Selecting the number of taxpayer segments")
savefig("fig03_cluster_selection.png")

# final segmentation
kmeans = km_models[K_BEST]
df["segment"] = kmeans.labels_
SEGMENTS = sorted(df["segment"].unique())

# ============================================================================
# 10. VISUALIZE AND PROFILE TAXPAYER SEGMENTS
# ============================================================================
section("SECTION 10 - SEGMENT PROFILES AND VISUALISATION")
SOURCE_LABEL = {"share_wage": "Wage-dominant", "share_business": "Business-dominant", "share_property": "Rental-property",
                "share_investment": "Investment-income", "share_pension": "Pension/retirement",
                "share_trust": "Trust/partnership", "share_other": "Other-income"}
terc = df["gross_activity"].quantile([1 / 3, 2 / 3]).values
prof_rows, SEG_NAME = [], {}
for s in SEGMENTS:
    g = df[df["segment"] == s]
    shares = g[[f"share_{c}" for c in comp.columns]].mean()
    dom = shares.idxmax()
    med_act = g["gross_activity"].median()
    scale = "low" if med_act < terc[0] else ("mid" if med_act < terc[1] else "high")
    mixed = "" if shares.max() >= 0.5 else ", mixed"
    name = f"S{s}: {SOURCE_LABEL[dom]} ({scale} income{mixed})"
    if med_act < INCOME_FLOOR:                      # no material amount in any income source
        name = f"S{s}: No/minimal reported income"
    SEG_NAME[s] = name
    r = {"segment": s, "name": name, "n": len(g), "share_pct": round(100 * len(g) / len(df), 2),
         "median_total_income": g["Tot_IncLoss_amt"].median(), "median_taxable_income": g["Taxable_Income"].median(),
         "median_deductions": g["Tot_ded_amt"].median(), "median_super_balance": g["Spr_Ttl_Acnt_Bal"].median(),
         "pct_agent_lodged": round(100 * g["lodged_via_agent"].mean(), 1), "median_items_reported": g["items_reported"].median(),
         "modal_age_range": int(g["age_range"].mode().iloc[0]), "top_occupation": g["occupation"].mode().iloc[0]}
    for c in comp.columns:
        r[f"mean_share_{c}"] = round(shares[f"share_{c}"], 3)
    prof_rows.append(r)
    log(f"  {name:55s} n={len(g):>7,}  median income=${g['Tot_IncLoss_amt'].median():>9,.0f}")
profile = pd.DataFrame(prof_rows).set_index("segment")
savetab(profile, "table_segment_profiles.csv")
df["segment_name"] = df["segment"].map(SEG_NAME)
SUMMARY["segments"] = profile.reset_index().to_dict("records")

# occupation composition
occ_mix = pd.crosstab(df["segment_name"], df["occupation"], normalize="index").round(3) * 100
savetab(occ_mix, "table_segment_occupation_mix.csv")

# figure: PCA scatter + share heatmap
pca = PCA(n_components=2, random_state=SEED).fit(X_seg[sil_idx])
P = pca.transform(X_seg[sil_idx])
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
for i, s in enumerate(SEGMENTS):
    m = df["segment"].values[sil_idx] == s
    axes[0].scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, color=PALETTE[i % 10], label=SEG_NAME[s])
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var.)"); axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var.)")
axes[0].legend(fontsize=7, markerscale=3, loc="best"); axes[0].set_title("Segments in the first two principal components (20,000 taxpayers)")
heat = profile[[f"mean_share_{c}" for c in comp.columns]].copy()
heat.columns = list(comp.columns); heat.index = [SEG_NAME[s] for s in heat.index]
heat["log income scale"] = profile["median_total_income"].apply(lambda v: np.log10(max(v, 1))).round(2).values
sns.heatmap(heat, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], cbar=False)
axes[1].set_title("Mean income-source share by segment (last column: log10 median income)")
axes[1].tick_params(axis="y", labelsize=8)
fig.suptitle("Fig. 4  Taxpayer segments")
savefig("fig04_segments.png")

# ============================================================================
# 11. STAGE 2 - SEGMENT-BASED ISOLATION FOREST
# ============================================================================
section("SECTION 11 - STAGE 2: ISOLATION FOREST WITHIN EACH SEGMENT")
ad_scaler = StandardScaler()
X_ad = pd.DataFrame(ad_scaler.fit_transform(df[AD_FEATURES].values), columns=AD_FEATURES, index=df.index)
seg_models = {}
df["if_seg_score"] = np.nan          # sklearn score_samples: lower = more anomalous
t = time.time()
for s in SEGMENTS:
    m = df["segment"] == s
    iso = IsolationForest(n_estimators=200, max_samples="auto", contamination=CONTAMINATION, random_state=SEED, n_jobs=-1)
    iso.fit(X_ad.loc[m].values)
    df.loc[m, "if_seg_score"] = iso.score_samples(X_ad.loc[m].values)
    seg_models[s] = iso
log(f"Fitted {len(SEGMENTS)} segment forests in {time.time() - t:.1f}s")

# ============================================================================
# 12. GENERATE ANOMALY SCORES AND LABELS (contamination grid)
# ============================================================================
section("SECTION 12 - ANOMALY SCORES, THRESHOLDS AND LABELS")
df["anomaly_score_seg"] = -df["if_seg_score"]                    # higher = more anomalous
df["anomaly_pct_rank_seg"] = df.groupby("segment")["anomaly_score_seg"].rank(pct=True, method="first")


def flags_from_rank(pct_rank, c):
    return (pct_rank > 1 - c).astype(int)


thr_rows = []
for c in CONTAMINATION_GRID:
    col = f"flag_seg_{int(c * 100)}pct"
    df[col] = flags_from_rank(df["anomaly_pct_rank_seg"], c)
df["flag_seg"] = df[f"flag_seg_{int(CONTAMINATION * 100)}pct"]
for s in SEGMENTS:
    g = df[df["segment"] == s]
    row = {"segment": s, "name": SEG_NAME[s], "n": len(g)}
    for c in CONTAMINATION_GRID:
        flagged = g[g[f"flag_seg_{int(c * 100)}pct"] == 1]
        row[f"threshold_{int(c * 100)}pct"] = round(flagged["anomaly_score_seg"].min(), 4)
        row[f"n_flagged_{int(c * 100)}pct"] = len(flagged)
    thr_rows.append(row)
thr = pd.DataFrame(thr_rows).set_index("segment")
savetab(thr, "table_segment_thresholds.csv")
log(thr[["name", "n", "threshold_2pct", "n_flagged_2pct"]].to_string())
SUMMARY["n_flagged_seg"] = int(df["flag_seg"].sum())
SUMMARY["segment_thresholds"] = thr.reset_index().to_dict("records")

# score distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for i, s in enumerate(SEGMENTS):
    g = df[df["segment"] == s]
    axes[0].hist(g["anomaly_score_seg"], bins=80, histtype="step", lw=1.3, color=PALETTE[i % 10], label=SEG_NAME[s], density=True)
axes[0].set_xlabel("anomaly score (higher = more isolated)"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=7)
axes[0].set_title("Within-segment anomaly score distributions")
q = df["anomaly_score_seg"].quantile(np.linspace(0.5, 0.999, 200))
axes[1].plot(np.linspace(0.5, 0.999, 200) * 100, q.values, color=PALETTE[3])
for c in CONTAMINATION_GRID:
    axes[1].axvline(100 * (1 - c), ls="--", color="grey", lw=0.8)
    axes[1].text(100 * (1 - c), q.max() - 0.02 * CONTAMINATION_GRID.index(c), f"{int(c * 100)}%", fontsize=8, ha="right", va="top")
axes[1].set_xlabel("population percentile of the pooled score"); axes[1].set_ylabel("anomaly score")
axes[1].set_title("Score quantile curve with the four contamination cut-offs")
fig.suptitle("Fig. 5  Isolation Forest anomaly scores")
savefig("fig05_anomaly_scores.png")

# ============================================================================
# 13. COMPARE WITH POPULATION-LEVEL ANOMALY DETECTION
# ============================================================================
section("SECTION 13 - SEGMENTED VERSUS POPULATION-WIDE ISOLATION FOREST")
t = time.time()
iso_global = IsolationForest(n_estimators=200, max_samples="auto", contamination=CONTAMINATION, random_state=SEED, n_jobs=-1).fit(X_ad.values)
df["anomaly_score_glob"] = -iso_global.score_samples(X_ad.values)
df["anomaly_pct_rank_glob"] = df["anomaly_score_glob"].rank(pct=True, method="first")
for c in CONTAMINATION_GRID:
    df[f"flag_glob_{int(c * 100)}pct"] = flags_from_rank(df["anomaly_pct_rank_glob"], c)
df["flag_glob"] = df[f"flag_glob_{int(CONTAMINATION * 100)}pct"]
log(f"Global forest fitted in {time.time() - t:.1f}s; flagged {int(df['flag_glob'].sum()):,}")

y_inj = df["injected"].values
comp_rows = []
for c in CONTAMINATION_GRID:
    for design in ["seg", "glob"]:
        f = df[f"flag_{design}_{int(c * 100)}pct"].values
        comp_rows.append({"contamination": c, "design": "segmented" if design == "seg" else "population-wide",
                          "n_flagged": int(f.sum()), "injected_recall": recall_score(y_inj, f),
                          "injected_precision": precision_score(y_inj, f), "injected_f1": f1_score(y_inj, f),
                          "natural_flags": int(((f == 1) & (y_inj == 0)).sum())})
sweep = pd.DataFrame(comp_rows).round(4)
savetab(sweep, "table_contamination_sweep.csv", index=False)
log(sweep.to_string(index=False))

# recall by injected type at the operating point
type_rows = []
for name, desc, *_ in ANOMALY_TYPES:
    m = df["injected_type"] == name
    type_rows.append({"type": name, "n": int(m.sum()), "recall_segmented": df.loc[m, "flag_seg"].mean(),
                      "recall_population_wide": df.loc[m, "flag_glob"].mean()})
type_recall = pd.DataFrame(type_rows).round(4)
savetab(type_recall, "table_recall_by_anomaly_type.csv", index=False)
log(type_recall.to_string(index=False))

# overlap and income confounding
both = int(((df["flag_seg"] == 1) & (df["flag_glob"] == 1)).sum())
union = int(((df["flag_seg"] == 1) | (df["flag_glob"] == 1)).sum())
rho_designs = spearmanr(df["anomaly_score_seg"], df["anomaly_score_glob"]).statistic
df["income_decile"] = pd.qcut(df["Tot_IncLoss_amt"].rank(method="first"), 10, labels=False) + 1
conf_rows = []
for design, col, score in [("segmented", "flag_seg", "anomaly_score_seg"), ("population-wide", "flag_glob", "anomaly_score_glob")]:
    fl = df[df[col] == 1]
    conf_rows.append({"design": design, "n_flagged": len(fl), "median_income_flagged": fl["Tot_IncLoss_amt"].median(),
                      "income_multiple_of_population_median": fl["Tot_IncLoss_amt"].median() / max(df["Tot_IncLoss_amt"].median(), 1),
                      "pct_flagged_in_top_income_decile": 100 * (fl["income_decile"] == 10).mean(),
                      "pct_flagged_below_median_income": 100 * (fl["Tot_IncLoss_amt"] < df["Tot_IncLoss_amt"].median()).mean(),
                      "spearman_score_vs_income": spearmanr(df[score], df["Tot_IncLoss_amt"]).statistic})
confound = pd.DataFrame(conf_rows).round(3)
savetab(confound, "table_income_confounding.csv", index=False)
log(confound.to_string(index=False))
log(f"Flagged by both designs: {both:,}  Jaccard={both / union:.4f}  Spearman(scores)={rho_designs:.4f}")
SUMMARY.update({"flag_overlap_both": both, "flag_jaccard": round(both / union, 4), "score_spearman_seg_vs_glob": round(rho_designs, 4),
                "contamination_sweep": sweep.to_dict("records"), "recall_by_type": type_recall.to_dict("records"),
                "income_confounding": confound.to_dict("records")})

# flag rate by segment for each design
seg_rate = df.groupby("segment_name").agg(n=("flag_seg", "size"), flagged_segmented=("flag_seg", "sum"),
                                          flagged_population_wide=("flag_glob", "sum"), injected=("injected", "sum"))
seg_rate["rate_segmented_pct"] = (100 * seg_rate["flagged_segmented"] / seg_rate["n"]).round(2)
seg_rate["rate_population_wide_pct"] = (100 * seg_rate["flagged_population_wide"] / seg_rate["n"]).round(2)
seg_rate["injected_rate_pct"] = (100 * seg_rate["injected"] / seg_rate["n"]).round(2)
savetab(seg_rate, "table_flag_rate_by_segment.csv")
log(seg_rate.to_string())
SUMMARY["flag_rate_by_segment"] = seg_rate.reset_index().to_dict("records")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sw = sweep.pivot(index="contamination", columns="design", values="injected_recall")[["segmented", "population-wide"]]
sw.plot(kind="bar", ax=axes[0], color=[PALETTE[0], PALETTE[1]]); axes[0].set_title("Recall of injected anomalies vs review budget")
axes[0].set_xlabel("contamination (review budget)"); axes[0].set_ylabel("recall"); axes[0].set_xticklabels([f"{int(c * 100)}%" for c in sw.index], rotation=0)
dec = df.groupby("income_decile")[["flag_seg", "flag_glob"]].mean() * 100
dec.columns = ["segmented", "population-wide"]
dec.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[1]]); axes[1].set_title("Flag rate by income decile (2% budget)")
axes[1].set_xlabel("income decile (1 = lowest)"); axes[1].set_ylabel("% flagged"); axes[1].tick_params(axis="x", rotation=0)
sr = seg_rate[["rate_segmented_pct", "rate_population_wide_pct"]]
sr.columns = ["segmented", "population-wide"]
sr.plot(kind="barh", ax=axes[2], color=[PALETTE[0], PALETTE[1]]); axes[2].set_title("Flag rate by segment (2% budget)"); axes[2].set_xlabel("% flagged")
axes[2].tick_params(axis="y", labelsize=7); axes[2].set_ylabel("")
fig.suptitle("Fig. 6  Segment-based versus population-wide Isolation Forest")
savefig("fig06_segmented_vs_global.png")

# ---- 13b. Feature-space ablation: does the anomaly feature space matter more than segmentation? ----
section("SECTION 13b - ABLATION: FEATURE SPACE x DESIGN (2% budget)")
log("Pre-specified sensitivity analysis: the same two designs are re-run on the behavioural ratio space only")
log("(no raw amounts), so that scale is carried by segmentation rather than by the detector.")
abl_rows, abl_type_rows = [], []
X_ratio = X_ad[RATIO_NAMES]
for space_name, Xspace in [("amounts + ratios (primary)", X_ad), ("ratios only", X_ratio)]:
    for design in ["segmented", "population-wide"]:
        if space_name.startswith("amounts") and design == "segmented":
            sc = df["anomaly_score_seg"].values; fl = df["flag_seg"].values
        elif space_name.startswith("amounts"):
            sc = df["anomaly_score_glob"].values; fl = df["flag_glob"].values
        else:
            sc = np.zeros(len(df))
            if design == "segmented":
                for s_ in SEGMENTS:
                    m_ = (df["segment"] == s_).values
                    iso_ = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=SEED, n_jobs=-1).fit(Xspace.values[m_])
                    sc[m_] = -iso_.score_samples(Xspace.values[m_])
                pr = pd.Series(sc, index=df.index).groupby(df["segment"]).rank(pct=True, method="first")
            else:
                iso_ = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=SEED, n_jobs=-1).fit(Xspace.values)
                sc = -iso_.score_samples(Xspace.values)
                pr = pd.Series(sc, index=df.index).rank(pct=True, method="first")
            fl = flags_from_rank(pr, CONTAMINATION).values
            df[f"flag_ratio_{'seg' if design == 'segmented' else 'glob'}"] = fl
            df[f"anomaly_score_ratio_{'seg' if design == 'segmented' else 'glob'}"] = sc
        flagged_ = df[fl == 1]
        abl_rows.append({"feature_space": space_name, "design": design, "n_flagged": int(fl.sum()),
                         "injected_recall": recall_score(y_inj, fl), "injected_precision": precision_score(y_inj, fl),
                         "pr_auc_vs_injected": average_precision_score(y_inj, sc),
                         "pct_flagged_in_top_income_decile": 100 * (flagged_["income_decile"] == 10).mean(),
                         "spearman_score_vs_income": spearmanr(sc, df["Tot_IncLoss_amt"]).statistic})
        for name, *_ in ANOMALY_TYPES:
            m_ = (df["injected_type"] == name).values
            abl_type_rows.append({"feature_space": space_name, "design": design, "type": name, "recall": fl[m_].mean()})
ablation = pd.DataFrame(abl_rows).round(4)
ablation_type = pd.DataFrame(abl_type_rows).pivot(index="type", columns=["feature_space", "design"], values="recall").round(4)
ablation_type.columns = [f"{a} | {b}" for a, b in ablation_type.columns]
savetab(ablation, "table_ablation_feature_space.csv", index=False)
savetab(ablation_type, "table_ablation_recall_by_type.csv")
log(ablation.to_string(index=False)); log(ablation_type.to_string())
SUMMARY["ablation"] = ablation.to_dict("records")
ov_ = int(((df["flag_ratio_seg"] == 1) & (df["flag_seg"] == 1)).sum())
SUMMARY["ablation_overlap_ratio_seg_vs_primary_seg"] = ov_
log(f"Overlap between primary segmented flags and ratio-only segmented flags: {ov_:,}")

# ============================================================================
# 14. STAGE 3 - SUPERVISED ANOMALY CLASSIFICATION
# ============================================================================
section("SECTION 14 - STAGE 3: SUPERVISED CLASSIFICATION (injected labels)")
log("Target = injected synthetic anomaly (1) vs unmodified record (0).  This label is NOT a compliance outcome.")
X_sup = X_ad.values
y_sup = df["injected"].values

# ============================================================================
# 15. TRAIN/TEST SPLIT (stratified)
# ============================================================================
section("SECTION 15 - STRATIFIED TRAIN/TEST SPLIT")
idx_all = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx_all, test_size=TEST_SIZE, stratify=y_sup, random_state=SEED)
X_tr, X_te, y_tr, y_te = X_sup[idx_tr], X_sup[idx_te], y_sup[idx_tr], y_sup[idx_te]
log(f"Train {len(idx_tr):,} ({100 * y_tr.mean():.2f}% positive)   Test {len(idx_te):,} ({100 * y_te.mean():.2f}% positive)")
SUMMARY.update({"n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "test_positive_pct": round(100 * y_te.mean(), 3)})

# ============================================================================
# 16. TRAIN RANDOM FOREST (and a gradient-boosting comparator)
# ============================================================================
section("SECTION 16 - TRAIN SUPERVISED MODELS")
MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
    "Hist. Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                                              random_state=SEED),
}
fitted, train_time = {}, {}
for name, mdl in MODELS.items():
    t = time.time(); mdl.fit(X_tr, y_tr); train_time[name] = time.time() - t
    fitted[name] = mdl
    log(f"  {name}: trained in {train_time[name]:.1f}s")
rf = fitted["Random Forest"]

# ============================================================================
# 17. EVALUATE SUPERVISED MODELS
# ============================================================================
section("SECTION 17 - EVALUATION ON THE HOLD-OUT SET")
eval_rows, probas = [], {}
budget_k = int(round(CONTAMINATION * len(idx_te)))
for name, mdl in fitted.items():
    p = mdl.predict_proba(X_te)[:, 1]; probas[name] = p
    pred = (p >= 0.5).astype(int)
    top = np.zeros_like(pred); top[np.argsort(-p)[:budget_k]] = 1     # same 2% review budget as the forests
    cm = confusion_matrix(y_te, pred)
    eval_rows.append({"model": name, "threshold": "0.50", "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
                      "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, p),
                      "pr_auc": average_precision_score(y_te, p), "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
                      "train_seconds": round(train_time[name], 1)})
    eval_rows.append({"model": name, "threshold": f"top {int(CONTAMINATION * 100)}% budget", "accuracy": accuracy_score(y_te, top),
                      "precision": precision_score(y_te, top), "recall": recall_score(y_te, top), "f1": f1_score(y_te, top),
                      "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p),
                      "TN": confusion_matrix(y_te, top)[0, 0], "FP": confusion_matrix(y_te, top)[0, 1],
                      "FN": confusion_matrix(y_te, top)[1, 0], "TP": confusion_matrix(y_te, top)[1, 1], "train_seconds": round(train_time[name], 1)})
sup_eval = pd.DataFrame(eval_rows).round(4)
savetab(sup_eval, "table_supervised_evaluation.csv", index=False)
log(sup_eval.to_string(index=False))

# 5-fold stratified cross-validation (robustness of the hold-out figures)
cv_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, mdl in MODELS.items():
    t = time.time()
    cv = cross_validate(mdl, X_sup, y_sup, cv=skf, scoring=["f1", "roc_auc", "average_precision"], n_jobs=1)
    cv_rows.append({"model": name, "f1_mean": cv["test_f1"].mean(), "f1_sd": cv["test_f1"].std(),
                    "roc_auc_mean": cv["test_roc_auc"].mean(), "roc_auc_sd": cv["test_roc_auc"].std(),
                    "pr_auc_mean": cv["test_average_precision"].mean(), "pr_auc_sd": cv["test_average_precision"].std(),
                    "seconds": round(time.time() - t, 1)})
    log(f"  5-fold CV {name}: F1={cv_rows[-1]['f1_mean']:.4f}+-{cv_rows[-1]['f1_sd']:.4f}  PR-AUC={cv_rows[-1]['pr_auc_mean']:.4f}")
cv_tab = pd.DataFrame(cv_rows).round(4)
savetab(cv_tab, "table_supervised_cv.csv", index=False)
SUMMARY["supervised_eval"] = sup_eval.to_dict("records")
SUMMARY["supervised_cv"] = cv_tab.to_dict("records")

# ============================================================================
# 18. COMPARE SUPERVISED AND UNSUPERVISED RESULTS (same hold-out rows)
# ============================================================================
section("SECTION 18 - SUPERVISED VERSUS UNSUPERVISED ON THE SAME HOLD-OUT ROWS")
te = df.iloc[idx_te]
cmp_rows = []
for label, f in [("Isolation Forest, segmented (2%)", te["flag_seg"].values), ("Isolation Forest, population-wide (2%)", te["flag_glob"].values)]:
    cmp_rows.append({"method": label, "n_flagged": int(f.sum()), "precision": precision_score(y_te, f), "recall": recall_score(y_te, f),
                     "f1": f1_score(y_te, f), "roc_auc": roc_auc_score(y_te, te["anomaly_score_seg"] if "segmented" in label else te["anomaly_score_glob"]),
                     "pr_auc": average_precision_score(y_te, te["anomaly_score_seg"] if "segmented" in label else te["anomaly_score_glob"])})
for name in fitted:
    p = probas[name]; top = np.zeros_like(y_te); top[np.argsort(-p)[:budget_k]] = 1
    cmp_rows.append({"method": f"{name} (top 2% budget)", "n_flagged": int(top.sum()), "precision": precision_score(y_te, top),
                     "recall": recall_score(y_te, top), "f1": f1_score(y_te, top), "roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p)})
sup_vs_unsup = pd.DataFrame(cmp_rows).round(4)
savetab(sup_vs_unsup, "table_supervised_vs_unsupervised.csv", index=False)
log(sup_vs_unsup.to_string(index=False))
SUMMARY["supervised_vs_unsupervised"] = sup_vs_unsup.to_dict("records")

# agreement between RF and the segmented forest on NATURAL (non-injected) flags
p_rf = probas["Random Forest"]
nat = (te["flag_seg"].values == 1) & (y_te == 0)
SUMMARY["natural_seg_flags_in_test"] = int(nat.sum())
SUMMARY["natural_seg_flags_rf_prob_median"] = round(float(np.median(p_rf[nat])), 4) if nat.sum() else None
SUMMARY["natural_seg_flags_rf_positive_pct"] = round(100 * float((p_rf[nat] >= 0.5).mean()), 2) if nat.sum() else None
log(f"Natural (non-injected) segmented flags in test set: {nat.sum():,}; RF calls {SUMMARY['natural_seg_flags_rf_positive_pct']}% of them anomalous")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for name in fitted:
    fpr, tpr, _ = roc_curve(y_te, probas[name]); axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, probas[name]):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, probas[name]); axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, probas[name]):.3f})")
for label, sc in [("IF segmented", te["anomaly_score_seg"]), ("IF population-wide", te["anomaly_score_glob"])]:
    fpr, tpr, _ = roc_curve(y_te, sc); axes[0].plot(fpr, tpr, ls="--", label=f"{label} (AUC={roc_auc_score(y_te, sc):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, sc); axes[1].plot(rc, pr, ls="--", label=f"{label} (AP={average_precision_score(y_te, sc):.3f})")
axes[0].plot([0, 1], [0, 1], color="grey", lw=0.7); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].set_title("ROC curves (hold-out)"); axes[0].legend(fontsize=7)
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-recall curves (hold-out)"); axes[1].legend(fontsize=7)
cm = confusion_matrix(y_te, (p_rf >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[2], xticklabels=["pred normal", "pred anomalous"], yticklabels=["normal", "injected"])
axes[2].set_title("Random Forest confusion matrix (threshold 0.5)")
fig.suptitle("Fig. 7  Supervised classification of injected anomalies")
savefig("fig07_supervised_evaluation.png")

# ============================================================================
# 19. STAGE 4 - SHAP ANALYSIS
# ============================================================================
section("SECTION 19 - STAGE 4: SHAP EXPLAINABILITY")


def to_2d(sv, class_index=None):
    """Normalise shap_values output across shap versions (list vs 3-D array)."""
    if isinstance(sv, list):
        return np.asarray(sv[class_index]) if class_index is not None else np.asarray(sv)
    sv = np.asarray(sv)
    if sv.ndim == 3 and class_index is not None:
        return sv[:, :, class_index]
    return sv


# ---------- Q1: why is a taxpayer in a segment?  (surrogate model + SHAP) ----------
log("Q1 - Segment membership: K-Means has no additive per-feature output, so a surrogate Random Forest is trained to")
log("     reproduce the cluster labels and SHAP is applied to the surrogate.  Surrogate fidelity is reported.")
Xseg_tr, Xseg_te, yseg_tr, yseg_te = train_test_split(df[SEG_FEATURES].values, df["segment"].values, test_size=0.3,
                                                      stratify=df["segment"].values, random_state=SEED)
surrogate = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Xseg_tr, yseg_tr)
surr_acc = accuracy_score(yseg_te, surrogate.predict(Xseg_te))
log(f"     surrogate hold-out accuracy = {surr_acc:.4f}")
SUMMARY["surrogate_accuracy"] = round(surr_acc, 4)
sidx = RNG.choice(len(Xseg_te), size=min(3000, len(Xseg_te)), replace=False)
expl_surr = shap.TreeExplainer(surrogate)
sv_surr = expl_surr.shap_values(Xseg_te[sidx], check_additivity=False)
sv_surr = np.asarray(sv_surr)
if sv_surr.ndim == 3 and sv_surr.shape[0] == len(SEGMENTS) and sv_surr.shape[1] == len(sidx):
    sv_surr = np.transpose(sv_surr, (1, 2, 0))                     # (n, features, classes)
seg_imp = pd.DataFrame({SEG_NAME[s]: np.abs(sv_surr[:, :, i]).mean(axis=0) for i, s in enumerate(surrogate.classes_)}, index=SEG_FEATURES)
seg_imp["overall"] = seg_imp.mean(axis=1)
seg_imp = seg_imp.sort_values("overall", ascending=False).round(4)
savetab(seg_imp, "table_shap_segmentation_importance.csv")
log(seg_imp[["overall"]].to_string())
SUMMARY["shap_segmentation_top"] = seg_imp["overall"].head(5).to_dict()

fig, ax = plt.subplots(figsize=(9, 4.8))
seg_imp.drop(columns="overall").plot(kind="barh", stacked=True, ax=ax, color=PALETTE[:len(SEGMENTS)])
ax.invert_yaxis(); ax.set_xlabel("mean |SHAP| (summed over segments)"); ax.legend(fontsize=7)
ax.set_title("Fig. 8  Which features determine segment membership (SHAP on the surrogate model)")
savefig("fig08_shap_segmentation.png")

# ---------- Q2: why was a taxpayer flagged?  (TreeExplainer on each segment's Isolation Forest) ----------
log("Q2 - Anomaly flags: TreeExplainer is exact for Isolation Forest.  SHAP values explain the average path length;")
log("     a NEGATIVE value shortens the path (more isolated), so 'contribution towards anomaly' = -SHAP.")
shap_parts, valid_rows = [], []
explainers = {}
for s in SEGMENTS:
    g = df[df["segment"] == s]
    fl = g.index[g["flag_seg"] == 1]; nf = g.index[g["flag_seg"] == 0]
    pick = np.concatenate([RNG.choice(fl, size=min(SHAP_PER_SEGMENT, len(fl)), replace=False),
                           RNG.choice(nf, size=min(SHAP_PER_SEGMENT, len(nf)), replace=False)])
    expl = shap.TreeExplainer(seg_models[s]); explainers[s] = expl
    sv = to_2d(expl.shap_values(X_ad.loc[pick].values, check_additivity=False))
    # validity: the explained quantity must track the flagging score
    rho = spearmanr(sv.sum(axis=1), seg_models[s].score_samples(X_ad.loc[pick].values)).statistic
    valid_rows.append({"segment": s, "name": SEG_NAME[s], "n_explained": len(pick), "spearman_shapsum_vs_score": round(rho, 4)})
    part = pd.DataFrame(sv, columns=AD_FEATURES, index=pick); part["segment"] = s
    shap_parts.append(part)
shap_valid = pd.DataFrame(valid_rows)
savetab(shap_valid, "table_shap_validity.csv", index=False)
log(shap_valid.to_string(index=False))
SHAP_IF = pd.concat(shap_parts)
SHAP_ROWS = SHAP_IF.index
contrib = -SHAP_IF[AD_FEATURES]                                    # towards anomaly

# ============================================================================
# 20. GLOBAL SHAP EXPLANATIONS
# ============================================================================
section("SECTION 20 - GLOBAL SHAP EXPLANATIONS")
flag_rows = df.loc[SHAP_ROWS, "flag_seg"] == 1
glob_imp = pd.DataFrame({
    "mean_abs_shap": contrib.abs().mean(),
    "mean_contribution_flagged": contrib[flag_rows.values].mean(),
    "mean_contribution_unflagged": contrib[~flag_rows.values].mean(),
    "corr_feature_value_vs_contribution": [spearmanr(X_ad.loc[SHAP_ROWS, f], contrib[f]).statistic for f in AD_FEATURES]}).sort_values("mean_abs_shap", ascending=False).round(4)
savetab(glob_imp, "table_shap_if_global_importance.csv")
log(glob_imp.head(15).to_string())
TOP_IF = glob_imp.index[:15].tolist()
SUMMARY["shap_if_top10"] = glob_imp["mean_abs_shap"].head(10).to_dict()
SUMMARY["shap_if_top10_share"] = round(float(glob_imp["mean_abs_shap"].head(10).sum() / glob_imp["mean_abs_shap"].sum()), 4)

# display-friendly feature values (raw AUD for amounts, plain ratios)
def display_values(rows):
    out = pd.DataFrame(index=rows)
    for f in AD_FEATURES:
        out[f] = df.loc[rows, f[4:]] if f.startswith("log_") else df.loc[rows, f]
    return out


disp = display_values(SHAP_ROWS)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
plt.sca(axes[0])
shap.summary_plot(contrib[TOP_IF].values, disp[TOP_IF], feature_names=TOP_IF, show=False, max_display=15, plot_size=None, color_bar=True)
axes[0].set_xlabel("SHAP contribution towards 'anomalous' (-SHAP on path length)"); axes[0].set_title("Beeswarm: direction and magnitude (top 15)")
glob_imp["mean_abs_shap"].head(15)[::-1].plot(kind="barh", ax=axes[1], color=PALETTE[0])
axes[1].set_xlabel("mean |SHAP|"); axes[1].set_title("Global importance for the segment forests (pooled)")
fig.suptitle("Fig. 9  SHAP global explanation of the segment-level Isolation Forests")
savefig("fig09_shap_if_global.png")

# empirical flag thresholds for the most influential features
log("Empirical single-feature flag thresholds: smallest feature value above which the flag rate is >= 10%, 25%, 50%")
LEVELS = [0.10, 0.25, 0.50]


def flag_thresholds(frame, rawcol, flag_col):
    cand = np.unique(frame[rawcol].quantile(np.arange(0.50, 1.0, 0.0025)).values)
    out = {f"value_for_flag_rate_ge_{int(l * 100)}pct": np.nan for l in LEVELS}
    for lvl in LEVELS:
        for v in cand:
            m = frame[rawcol] >= v
            if m.sum() >= 20 and frame.loc[m, flag_col].mean() >= lvl:
                out[f"value_for_flag_rate_ge_{int(lvl * 100)}pct"] = v; break
    top1 = frame[rawcol] >= frame[rawcol].quantile(0.99)
    out["flag_rate_in_top1pct_of_feature"] = round(100 * frame.loc[top1, flag_col].mean(), 1) if top1.sum() else np.nan
    out["feature_p99"] = frame[rawcol].quantile(0.99)
    return out


thr_feat_rows, pooled = [], []
for f in TOP_IF[:10]:
    rawcol = f[4:] if f.startswith("log_") else f
    pooled.append({"feature": rawcol, **flag_thresholds(df, rawcol, "flag_seg")})
    for s in SEGMENTS:
        g = df[df["segment"] == s]
        thr_feat_rows.append({"feature": rawcol, "segment": SEG_NAME[s], "segment_score_threshold_2pct": thr.loc[s, "threshold_2pct"],
                              **flag_thresholds(g, rawcol, "flag_seg")})
thr_feat = pd.DataFrame(thr_feat_rows).round(3)
pooled = pd.DataFrame(pooled).round(3)
savetab(thr_feat, "table_feature_flag_thresholds.csv", index=False)
savetab(pooled, "table_feature_flag_thresholds_pooled.csv", index=False)
log(pooled.to_string(index=False))
SUMMARY["feature_flag_thresholds_pooled"] = pooled.to_dict("records")

# ---------- SHAP for the supervised Random Forest ----------
ridx = RNG.choice(len(idx_te), size=min(1500, len(idx_te)), replace=False)
expl_rf = shap.TreeExplainer(rf)
sv_rf = to_2d(expl_rf.shap_values(X_te[ridx], check_additivity=False), class_index=1)
rf_imp = pd.Series(np.abs(sv_rf).mean(axis=0), index=AD_FEATURES).sort_values(ascending=False).round(4)
savetab(rf_imp.to_frame("mean_abs_shap"), "table_shap_rf_global_importance.csv")
SUMMARY["shap_rf_top10"] = rf_imp.head(10).to_dict()
rho_rf_if = spearmanr(rf_imp.reindex(AD_FEATURES), glob_imp["mean_abs_shap"].reindex(AD_FEATURES)).statistic
SUMMARY["spearman_rf_vs_if_importance"] = round(rho_rf_if, 4)
log(f"Spearman between RF and Isolation Forest SHAP importance rankings: {rho_rf_if:.4f}")
disp_te = display_values(df.index[idx_te[ridx]])
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plt.sca(axes[0])
top_rf = rf_imp.index[:15].tolist()
shap.summary_plot(sv_rf[:, [AD_FEATURES.index(f) for f in top_rf]], disp_te[top_rf], feature_names=top_rf, show=False, max_display=15, plot_size=None)
axes[0].set_xlabel("SHAP value (towards P(injected anomaly))"); axes[0].set_title("Beeswarm (top 15)")
rf_imp.head(15)[::-1].plot(kind="barh", ax=axes[1], color=PALETTE[2]); axes[1].set_xlabel("mean |SHAP|"); axes[1].set_title("Global importance, Random Forest")
fig.suptitle("Fig. 11  SHAP global explanation of the supervised Random Forest")
savefig("fig11_shap_rf_global.png")

# ============================================================================
# 21. INDIVIDUAL TAXPAYER EXPLANATIONS
# ============================================================================
section("SECTION 21 - INDIVIDUAL TAXPAYER EXPLANATIONS (local SHAP)")


def explain_one(row_idx):
    s = int(df.loc[row_idx, "segment"])
    expl = explainers[s]
    x = X_ad.loc[[row_idx]].values
    sv = to_2d(expl.shap_values(x, check_additivity=False))[0]
    ev = expl.expected_value if np.ndim(expl.expected_value) == 0 else np.asarray(expl.expected_value).ravel()[0]
    return -sv, -float(ev), s


flagged = df[df["flag_seg"] == 1]
case_ids = {
    "highest-score injected rental-deduction anomaly (A3)": flagged[flagged["injected_type"] == "A3_rental_deduction"].sort_values("anomaly_pct_rank_seg", ascending=False).index[0],
    "highest-score natural (non-injected) profile": flagged[flagged["injected"] == 0].sort_values("anomaly_pct_rank_seg", ascending=False).index[0],
    "borderline flagged profile": flagged.sort_values("anomaly_pct_rank_seg").index[0],
}
local_rows, panel_files = [], []
for n_, (label, rid) in enumerate(case_ids.items()):
    vals, base, s = explain_one(rid)
    dv = display_values([rid]).iloc[0]
    order = np.argsort(-np.abs(vals))[:10]
    plt.figure(figsize=(7.5, 6.5))
    ex = shap.Explanation(values=vals, base_values=base, data=dv.values.astype(float), feature_names=AD_FEATURES)
    shap.plots.waterfall(ex, max_display=10, show=False)
    plt.title(f"({chr(97 + n_)}) {label}\n{SEG_NAME[s]} | Ind={int(df.loc[rid, 'Ind'])} | type={df.loc[rid, 'injected_type']} | within-segment pct-rank={df.loc[rid, 'anomaly_pct_rank_seg']:.4f}", fontsize=8.5)
    fn = f"fig10{chr(97 + n_)}_waterfall.png"; panel_files.append(fn)
    savefig(fn)
    for j in order[:5]:
        local_rows.append({"case": label, "Ind": int(df.loc[rid, "Ind"]), "segment": SEG_NAME[s], "injected_type": df.loc[rid, "injected_type"],
                           "anomaly_score": round(df.loc[rid, "anomaly_score_seg"], 4), "feature": AD_FEATURES[j],
                           "feature_value": round(float(dv.iloc[j]), 3), "contribution_towards_anomaly": round(float(vals[j]), 4)})
    log(f"  {label}: Ind={int(df.loc[rid, 'Ind'])} {SEG_NAME[s]} -> top features: " +
        ", ".join(f"{AD_FEATURES[j]}={dv.iloc[j]:,.2f} ({vals[j]:+.3f})" for j in order[:3]))
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, fn in zip(axes, panel_files):
    ax.imshow(plt.imread(FIG / fn)); ax.axis("off")
fig.suptitle("Fig. 10  Local SHAP explanations for three flagged taxpayers (contribution towards 'anomalous'; feature values shown in AUD or as ratios)")
savefig("fig10_shap_local_waterfalls.png")
local_tab = pd.DataFrame(local_rows)
savetab(local_tab, "table_local_explanations.csv", index=False)
SUMMARY["local_cases"] = local_tab.to_dict("records")

# a ranked review list: the 15 highest-ranked flagged taxpayers with their top-3 drivers
rank_rows = []
for rid in flagged.sort_values("anomaly_pct_rank_seg", ascending=False).index[:15]:
    vals, base, s = explain_one(rid); dv = display_values([rid]).iloc[0]
    order = np.argsort(-vals)[:3]
    rank_rows.append({"Ind": int(df.loc[rid, "Ind"]), "segment": SEG_NAME[s], "injected_type": df.loc[rid, "injected_type"],
                      "anomaly_score": round(df.loc[rid, "anomaly_score_seg"], 4), "total_income": df.loc[rid, "Tot_IncLoss_amt"],
                      "total_deductions": df.loc[rid, "Tot_ded_amt"],
                      "top_drivers": "; ".join(f"{AD_FEATURES[j]}={dv.iloc[j]:,.2f}" for j in order)})
savetab(pd.DataFrame(rank_rows), "table_top15_review_list.csv", index=False)

# ============================================================================
# 22. VISUALIZATIONS - framework diagram (Fig. 1) and summary
# ============================================================================
section("SECTION 22 - FRAMEWORK DIAGRAM")
fig, ax = plt.subplots(figsize=(7.5, 9.5)); ax.axis("off")
steps = ["Raw taxpayer data\n(ATO 2022-23 individual sample file)", "Data preprocessing\n(sparsity + redundancy screens, signed-log, scaling)",
         "Feature engineering\n(income-composition shares, behavioural ratios)", "Stage 1: Taxpayer segmentation\n(K-Means on income composition and scale)",
         "Peer-group formation\n(one segment = one peer group)", "Stage 2: Isolation Forest within each segment\n(compared with a population-wide forest)",
         "Stage 3: Supervised classification\n(Random Forest / gradient boosting on injected labels)", "Stage 4: SHAP explainability\n(surrogate for segments; TreeExplainer for forests)",
         "Risk interpretation and visualisation\n(review lists, thresholds, local explanations)"]
for i, s_ in enumerate(steps):
    y = 1 - (i + 0.5) / len(steps)
    ax.add_patch(plt.Rectangle((0.08, y - 0.042), 0.84, 0.084, fc=PALETTE[0] if "Stage" in s_ else "#e8eef7", ec="black", lw=0.8))
    ax.text(0.5, y, s_, ha="center", va="center", fontsize=9, color="white" if "Stage" in s_ else "black")
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 0.045), xytext=(0.5, y - 0.062), arrowprops=dict(arrowstyle="-|>", lw=1))
ax.set_title("Fig. 1  Four-stage taxpayer analytics framework")
savefig("fig01_framework.png")

# ============================================================================
# 23. EXPORT FINAL RESULTS
# ============================================================================
section("SECTION 23 - EXPORT")
out_cols = ["Ind", "segment", "segment_name", "occupation", "age_range", "lodged_via_agent", "Tot_IncLoss_amt", "Taxable_Income", "Tot_ded_amt",
            "injected", "injected_type", "anomaly_score_seg", "anomaly_pct_rank_seg", "anomaly_score_glob"] + \
           [f"flag_seg_{int(c * 100)}pct" for c in CONTAMINATION_GRID] + [f"flag_glob_{int(c * 100)}pct" for c in CONTAMINATION_GRID] +            ["flag_ratio_seg", "flag_ratio_glob", "anomaly_score_ratio_seg", "anomaly_score_ratio_glob"] + RATIO_NAMES
scores = df[out_cols].copy()
scores["rf_probability"] = np.nan
scores.iloc[idx_te, scores.columns.get_loc("rf_probability")] = p_rf
scores["holdout_row"] = 0
scores.iloc[idx_te, scores.columns.get_loc("holdout_row")] = 1
scores.to_csv(RES / "taxpayer_scores.csv", index=False)
SUMMARY["runtime_seconds"] = round(time.time() - T0, 1)
SUMMARY["natural_flags_seg"] = int(((df["flag_seg"] == 1) & (df["injected"] == 0)).sum())
SUMMARY["natural_flags_glob"] = int(((df["flag_glob"] == 1) & (df["injected"] == 0)).sum())
with open(RES / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(SUMMARY, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log(f"Saved results/taxpayer_scores.csv ({len(scores):,} rows) and results/summary.json")
log(f"Total runtime: {SUMMARY['runtime_seconds']} s")
log("DONE.  Reminder: an anomaly flag is a statistical signal relative to a peer group, not evidence of non-compliance.")
