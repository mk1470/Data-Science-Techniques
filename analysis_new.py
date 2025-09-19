# ============================================
# Mental Health @ Work — Dual-Dataset EDA (Low-Stress focus)
# - Dataset A: Employees Attrition & Leadership Impact HR Data (Kaggle)
# - Dataset B: Remote Work & Mental Health (Kaggle)
#

# ============================================

import os
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- File paths ----------
ATTRITION_CSV = "hr analytics data - employees attrition and leadership impact.csv"
REMOTE_CSV    = "Impact_of_Remote_Work_on_Mental_Health.csv"

# ---------- Output folder ----------
OUT_DIR = "output_new"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def to_numeric_safe(s, downcast_float=False):
    x = pd.to_numeric(s, errors="coerce")
    if downcast_float:
        x = pd.to_numeric(x, downcast="float")
    return x

def normalize_yesno(series):
    if series.dtype.name == "bool":
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "available": 1, "provided": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "not available": 0, "none": 0
    }
    return s.map(mapping)

def save_table(df, name):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path

def wilson_ci(k, n, z=1.96):
    """Wilson interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2/(2*n)) / denom
    half = (z / denom) * np.sqrt((p_hat*(1-p_hat)/n) + (z**2/(4*n**2)))
    return (center - half, center + half)

# ============================================================
# PART 1 — DATASET A: Stress vs Job Satisfaction (Resignation only)
# ============================================================
print("\n=== Loading Dataset A (Attrition) ===")
dfA = pd.read_csv(ATTRITION_CSV)

# Keep only terminated with a reason
dfA = dfA[dfA["ReasonForLeaving"].notna()].copy()

# Keep needed columns
keepA = ["ReasonForLeaving", "JobSatisfactionScore", "StressLevelScore"]
dfA = dfA[keepA].copy()

# Numeric + clamp plausible ranges
dfA["JobSatisfactionScore"] = to_numeric_safe(dfA["JobSatisfactionScore"], downcast_float=True).clip(0, 10)
dfA["StressLevelScore"]     = to_numeric_safe(dfA["StressLevelScore"],     downcast_float=True).clip(0, 10)
dfA = dfA.dropna(subset=["JobSatisfactionScore", "StressLevelScore"])

# Summary by reason (optional table for write-up)
summaryA = (
    dfA.groupby("ReasonForLeaving")
       .agg(avg_stress=("StressLevelScore", "mean"),
            avg_satisfaction=("JobSatisfactionScore", "mean"),
            count=("ReasonForLeaving", "count"))
       .sort_values("count", ascending=False)
       .reset_index()
)
save_table(summaryA, "A_summary_by_reason")

# Resignation-only scatter with quadrants and trend
dfA["ReasonForLeaving"] = dfA["ReasonForLeaving"].astype(str).str.strip()
dfA_resignation = dfA[dfA["ReasonForLeaving"].str.casefold() == "resignation"].copy()

stress_threshold = 5.0
satisfaction_threshold = 3.0

def categorize_quadrant(stress, satisfaction, s_thr=5.0, j_thr=3.0):
    if stress < s_thr and satisfaction < j_thr:
        return "Low Stress, Low Satisfaction"
    elif stress >= s_thr and satisfaction < j_thr:
        return "High Stress, Low Satisfaction"
    elif stress < s_thr and satisfaction >= j_thr:
        return "Low Stress, High Satisfaction"
    else:
        return "High Stress, High Satisfaction"

if not dfA_resignation.empty:
    dfA_resignation["Quadrant"] = dfA_resignation.apply(
        lambda r: categorize_quadrant(r["StressLevelScore"], r["JobSatisfactionScore"],
                                      stress_threshold, satisfaction_threshold), axis=1
    )
    quadrant_counts = dfA_resignation["Quadrant"].value_counts()

    plt.figure(figsize=(12, 8))
    colors = {
        "Low Stress, Low Satisfaction": "lightcoral",
        "High Stress, Low Satisfaction": "darkred",
        "Low Stress, High Satisfaction": "lightgreen",
        "High Stress, High Satisfaction": "darkgreen"
    }
    for q in quadrant_counts.index:
        g = dfA_resignation[dfA_resignation["Quadrant"] == q]
        plt.scatter(g["StressLevelScore"], g["JobSatisfactionScore"],
                    s=20, alpha=0.6, c=colors[q], edgecolors="none", label=f"{q} (n={len(g)})")

    # Quadrant lines
    plt.axvline(x=stress_threshold, color='black', linestyle='--', alpha=0.7, linewidth=1)
    plt.axhline(y=satisfaction_threshold, color='black', linestyle='--', alpha=0.7, linewidth=1)

    # Quadrant labels
    plt.text(2.5, 1.5, "Low Stress\nLow Satisfaction", ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))
    plt.text(7.5, 1.5, "High Stress\nLow Satisfaction", ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', alpha=0.3))
    plt.text(2.5, 4.5, "Low Stress\nHigh Satisfaction", ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    plt.text(7.5, 4.5, "High Stress\nHigh Satisfaction", ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='darkgreen', alpha=0.3))

    # Trend line
    x = dfA_resignation["StressLevelScore"].values
    y = dfA_resignation["JobSatisfactionScore"].values
    if len(x) > 1:
        coeff = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = coeff[0]*xs + coeff[1]
        plt.plot(xs, ys, 'r-', linewidth=2, alpha=0.8, label='Trend line')

    plt.title("Dataset A: Stress vs Job Satisfaction — Resignation Only (Quadrant Analysis)")
    plt.xlabel("StressLevelScore")
    plt.ylabel("JobSatisfactionScore")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    # N badge
    plt.text(0.02, 0.98, f"Total n = {len(dfA_resignation)}", transform=plt.gca().transAxes,
             fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "A_scatter_stress_vs_satisfaction_by_reason.png")
    plt.savefig(out_scatter, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_scatter}")

    # Quadrant summary table
    quadrant_summary = pd.DataFrame({
        'Quadrant': quadrant_counts.index,
        'Count': quadrant_counts.values,
        'Percentage': (quadrant_counts.values / len(dfA_resignation) * 100).round(1)
    })
    save_table(quadrant_summary, "A_quadrant_analysis_resignation")
else:
    print("[warn] No 'Resignation' rows found; skipping resignation scatter.")

# ============================================================
# PART 2 — DATASET B: Access to Resources vs LOW Stress
# ============================================================
print("\n=== Loading Dataset B (Remote Work & Mental Health) ===")
dfB = pd.read_csv(REMOTE_CSV)

# Keep relevant columns if present
keepB = [
    "Stress_Level",
    "Access_to_Mental_Health_Resources",
    "Company_Support_for_Remote_Work",
    "Work_Life_Balance_Rating"
]
keepB = [c for c in keepB if c in dfB.columns]
dfB = dfB[keepB].copy()

# Coerce numeric-ish columns if present
for col in ["Company_Support_for_Remote_Work", "Work_Life_Balance_Rating"]:
    if col in dfB.columns:
        dfB[col] = to_numeric_safe(dfB[col], downcast_float=True)

# Map Stress_Level if categorical (Low/Medium/High -> 1/2/3)
if "Stress_Level" in dfB.columns and dfB["Stress_Level"].dtype == object:
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    dfB["Stress_Level"] = dfB["Stress_Level"].map(mapping)

# Normalize access flag (Yes/No-like -> {1,0})
if "Access_to_Mental_Health_Resources" in dfB.columns:
    dfB["Access_to_Mental_Health_Resources_norm"] = normalize_yesno(dfB["Access_to_Mental_Health_Resources"])
else:
    dfB["Access_to_Mental_Health_Resources_norm"] = np.nan

# Drop rows missing Stress_Level and clamp sanity
dfB = dfB.dropna(subset=["Stress_Level"])
dfB["Stress_Level"] = pd.to_numeric(dfB["Stress_Level"], errors="coerce").clip(lower=0)

# ---------- Define LOW stress flag ----------
# Ordinal case (1=Low, 2=Medium, 3=High): LowStressFlag = 1 if Stress_Level == 1
if dfB["Stress_Level"].dropna().nunique() <= 5 and set(dfB["Stress_Level"].dropna().unique()).issubset({0,1,2,3,4,5}):
    dfB["LowStressFlag"] = (dfB["Stress_Level"] == 1).astype(int)
else:
    # Continuous fallback (if ever needed): define "low" as <= 3 on a 0–10 scale
    LOW_THRESHOLD = 3.0
    dfB["LowStressFlag"] = (dfB["Stress_Level"] <= LOW_THRESHOLD).astype(int)

dfB_low = dfB.dropna(subset=["Access_to_Mental_Health_Resources_norm", "LowStressFlag"]).copy()

# ---------- Contingency table: Access x LowStressFlag ----------
ct_low = pd.crosstab(dfB_low["Access_to_Mental_Health_Resources_norm"], dfB_low["LowStressFlag"])
for col in [0, 1]:
    if col not in ct_low.columns:
        ct_low[col] = 0
ct_low = ct_low[[0, 1]]  # ensure order: Not Low (0), Low (1)

# Relabel for readability
ct_low.index = ct_low.index.map({0: "No Access", 1: "Access"})
ct_low.columns = ["Not Low Stress", "Low Stress"]

# Counts & rates
totals_low = ct_low.sum(axis=1)
low_counts = ct_low["Low Stress"]
notlow_counts = ct_low["Not Low Stress"]
low_rates = (low_counts / totals_low).rename("Low Stress Rate")
notlow_rates = (notlow_counts / totals_low).rename("Not Low Stress Rate")

# Save a tidy summary
summary_low = pd.concat([ct_low, totals_low.rename("Total"), low_rates], axis=1).reset_index().rename(columns={"index": "Access Group"})
save_table(summary_low, "B_low_stress_by_access_counts_and_rates")

print("\n=== Observed counts (Access x Low Stress) ===")
print(ct_low.to_string())

# ---------- Chi-square test (2x2, LOW stress) ----------
try:
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(ct_low.values)  # Yates correction by default
except Exception:
    print("\n[info] scipy not available; computing test without Yates correction (no p-value).")
    observed = ct_low.values
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    chi2 = ((observed - expected) ** 2 / expected).sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p = np.nan

observed = ct_low.values.astype(float)
expected = np.array(expected, dtype=float)

std_resid = (observed - expected) / np.sqrt(expected)
contrib = std_resid ** 2

std_resid_df = pd.DataFrame(std_resid, index=ct_low.index, columns=ct_low.columns)
contrib_df   = pd.DataFrame(contrib,   index=ct_low.index, columns=ct_low.columns)
expected_df  = pd.DataFrame(expected,  index=ct_low.index, columns=ct_low.columns)

print(f"\n=== Chi-square Test (LOW stress) ===")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom : {dof}")
print(f"p-value            : {p:.6g}")
print("\nExpected counts if independent:")
print(expected_df.round(2).to_string())
print("\nStandardized residuals (|z|>2 noteworthy):")
print(std_resid_df.round(2).to_string())

# Effect size: Cramér's V (for 2x2 equals Phi)
n = observed.sum()
min_dim = min(observed.shape) - 1
cramers_v = sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else np.nan
print(f"\nEffect size (Cramér's V): {cramers_v:.4f}")
if not np.isnan(cramers_v):
    print("Rule of thumb (2x2): ~0.10 small, ~0.30 medium, ~0.50 large association")

# Save chi-square artifacts
save_table(ct_low.reset_index().rename(columns={"index":"Access Group"}), "chi2_low_observed_counts")
save_table(expected_df.reset_index().rename(columns={"index":"Access Group"}), "chi2_low_expected_counts")
save_table(std_resid_df.reset_index().rename(columns={"index":"Access Group"}), "chi2_low_standardized_residuals")
save_table(contrib_df.reset_index().rename(columns={"index":"Access Group"}), "chi2_low_cell_contributions")

# ---------- Visuals (LOW stress) ----------
# (1) Raw counts of LOW stress by Access
plt.figure(figsize=(7,5))
ax = plt.gca()
bars = ax.bar(["No Access", "Access"],
              [int(low_counts.get("No Access", 0)), int(low_counts.get("Access", 0))])
ax.set_title("Low Stress Counts by Access to Mental Health Resources")
ax.set_ylabel("Count of Low Stress Employees")
for b in bars:
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h + max(1, h*0.02), f"n={int(h)}",
            ha="center", va="bottom", fontsize=10)
# xtick details
new_xticks = []
for lbl in ["No Access", "Access"]:
    total = int(totals_low.get(lbl, 0))
    rate  = float(low_rates.get(lbl, np.nan)) if lbl in low_rates.index else np.nan
    new_xticks.append(f"{lbl}\nTotal={total}{', Low='+format(rate*100,'.1f')+'%' if pd.notna(rate) else ''}")
ax.set_xticklabels(new_xticks)
plt.tight_layout()
out_low_counts = os.path.join(OUT_DIR, "B_low_stress_counts_by_access.png")
plt.savefig(out_low_counts, dpi=200); plt.close(); print(f"[saved] {out_low_counts}")

# (2) Percent LOW stress with 95% Wilson CI
labels = ["No Access", "Access"]
rates = [float(low_rates.get(l, np.nan)) for l in labels]
ns    = [int(totals_low.get(l, 0)) for l in labels]
ks    = [int(low_counts.get(l, 0)) for l in labels]
cis   = [wilson_ci(k, n) for k, n in zip(ks, ns)]
ci_l  = [ci[0] for ci in cis]
ci_u  = [ci[1] for ci in cis]
yerr  = [np.array(rates) - np.array(ci_l), np.array(ci_u) - np.array(rates)]

plt.figure(figsize=(7,5))
ax = plt.gca()
bars = ax.bar(labels, [r*100 if pd.notna(r) else 0 for r in rates],
              yerr=[(np.array(yerr[0])*100), (np.array(yerr[1])*100)], capsize=6)
ax.set_title("Percent Low Stress by Access to Mental Health Resources (95% Wilson CI)")
ax.set_ylabel("% Low Stress")
for i, b in enumerate(bars):
    height = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, height + max(1, height*0.02),
            f"{height:.1f}%\n(n={ns[i]})", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
out_low_rates = os.path.join(OUT_DIR, "B_low_stress_rates_by_access_with_CI.png")
plt.savefig(out_low_rates, dpi=200); plt.close(); print(f"[saved] {out_low_rates}")

# (3) 100% stacked bar (Low vs Not Low)
perc_low     = (low_counts / totals_low * 100).reindex(labels)
perc_notlow  = (notlow_counts / totals_low * 100).reindex(labels)

plt.figure(figsize=(7,5))
ax = plt.gca()
ax.bar(labels, perc_notlow.values, label="Not Low Stress")
ax.bar(labels, perc_low.values, bottom=perc_notlow.values, label="Low Stress")
ax.set_title("Stress Composition by Access to Mental Health Resources (100% stacked)")
ax.set_ylabel("% of Employees")
ax.legend()
for i, lbl in enumerate(labels):
    ax.text(i, 50, f"{perc_low.values[i]:.1f}% Low", ha="center", va="center", fontsize=10)
plt.tight_layout()
out_100 = os.path.join(OUT_DIR, "B_low_stress_100pct_stacked.png")
plt.savefig(out_100, dpi=200); plt.close(); print(f"[saved] {out_100}")

print("\nDone. Tables & figures saved in:", os.path.abspath(OUT_DIR))
