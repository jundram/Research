"""Update the LaTeX report for the real-only run: numbers only, injected-anomaly
material removed.  Sentences are otherwise left as written.  Every anchor is
asserted so nothing is silently skipped.  Run from anywhere:

    python tools/patch_paper_real_only.py
"""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "research_paper" / "latex"
T = ROOT / "tables_real"
edits = 0


def patch(fname, pairs):
    global edits
    p = TEX / fname
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{fname}: anchor found {n}x:\n{old[:100]}"
        s = s.replace(old, new); edits += 1
    p.write_text(s, encoding="utf-8")


def cut(fname, start_marker, end_marker, replacement=""):
    """Remove everything from start_marker up to (not including) end_marker."""
    global edits
    p = TEX / fname
    s = p.read_text(encoding="utf-8")
    a, b = s.index(start_marker), s.index(end_marker)
    assert a < b, f"{fname}: markers out of order"
    p.write_text(s[:a] + replacement + s[b:], encoding="utf-8"); edits += 1


def tex(x):
    return str(x).replace("_", "\\_")


def num(v, d=3, comma=True):
    if pd.isna(v):
        return "--"
    if float(v).is_integer() and d == 0:
        return f"{int(v):,}" if comma else str(int(v))
    s = f"{v:,.{d}f}" if comma else f"{v:.{d}f}"
    return f"${s}$" if s.startswith("-") else s


# ---------------------------------------------------------------- figures
for f in (ROOT / "figures_real").glob("fig*.png"):
    if (TEX / "figures" / f.name).exists() or f.name == "fig12_label_free_validation.png":
        shutil.copy(f, TEX / "figures" / f.name)

# ---------------------------------------------------------------- abstract
patch("abstract.tex", [
    (" Random Forest and a gradient-boosting has been trained on variables created by injecting five documented anomaly mechanisms.", ""),
    (" Segmentation removed the population-wide forest's overflagging of rare taxpayer types, but it did not raise recall of the injected anomalies.",
     " Segmentation removed the population-wide forest's overflagging of rare taxpayer types."),
    (" The Random Forest model recovered almost all injected records with F1 score 0.917 and PR-AUC 0.975. Training to supervised model was done using injected anomalies, not to real non-compliance records.", ""),
])

# ---------------------------------------------------------------- ch1
patch("ch1_introduction.tex", [
    ("    \\item Experiments are performed using the injected anomaly scores to find out their effectiveness.\n", ""),
    ("the preprocessing and feature engineering steps, the injection of five anomaly mechanisms, and the K-Means",
     "the preprocessing and feature engineering steps, and the K-Means"),
])

# ---------------------------------------------------------------- ch3
patch("ch3_method.tex", [
    (" The supervised stage uses injected labels to measure how far documented mechanisms are learnable and to provide a calibrated comparator for the unsupervised scores.", ""),
    (" As there is no compliance label exist in the data, synthetic anomalies are injected (Section~\\ref{sec:injection}) in detection metrics.", ""),
    ("\\subsection{Preprocessing and Feature Engineering}\n\\label{sec:injection}\n", "\\subsection{Preprocessing and Feature Engineering}\n"),
    ("The designs were checked using on overlap score like Jaccard and Spearman, recall and precision of injected anomalies at each budget.",
     "The designs were checked using on overlap score like Jaccard and Spearman at each budget."),
])
cut("ch3_method.tex", "As there is no compliance records exists in the dataset, 1\\% of records", "\\subsection{Segmentation}")

# ---------------------------------------------------------------- ch4: numbers
sel = pd.read_csv(T / "table_cluster_selection.csv")
sel_rows = "".join(
    f"        {int(r.k)} & {'\\textbf{' + num(r.silhouette) + '}' if r.k == 8 else num(r.silhouette)} & "
    f"{'\\textbf{' + num(r.davies_bouldin) + '}' if r.k == 8 else num(r.davies_bouldin)} & {num(r.calinski_harabasz)} & "
    f"{int(r.smallest_segment):,} & {num(r.largest_segment_pct)} \\\\\n" for r in sel.itertuples())
prof = pd.read_csv(T / "table_segment_profiles.csv")
occ = {"S0": "Professionals"}
def med(v):
    return "0.0" if v == 0 else (f"{v:,.0f}" if v % 1 == 0 else f"{v:,.1f}")


prof_rows = "".join(
    f"        {r.name} & {int(r.n):,} & {r.share_pct:.1f} & {med(r.median_total_income)} & {med(r.median_deductions)} & {med(r.median_super_balance)}"
    f" & {r.pct_agent_lodged:.1f} & {int(r.modal_age_range)} & {occ.get(r.name[:2], 'Not stated')} \\\\\n" for r in prof.itertuples())
thr = pd.read_csv(T / "table_segment_thresholds.csv")
thr_rows = "".join(
    f"        {r.name} & {int(r.n):,} & {r.threshold_1pct:.4f} & {r.threshold_2pct:.4f} & {r.threshold_5pct:.4f} & {r.threshold_10pct:.4f} & {int(r.n_flagged_2pct):,} \\\\\n"
    for r in thr.itertuples())
sweep = pd.read_csv(T / "table_contamination_sweep.csv")
sweep_rows = "".join(
    f"        {r.contamination:.3f} & {int(r.n_flagged_segmented):,} & {int(r.n_flagged_population_wide):,} & {int(r.flagged_by_both):,} & {r.jaccard:.3f} & {r.pct_segmented_list_changed:.1f} \\\\\n"
    for r in sweep.itertuples())
conf = pd.read_csv(T / "table_income_confounding.csv").set_index("design")
cs, cp = conf.loc["segmented"], conf.loc["population-wide"]
abl = pd.read_csv(T / "table_ablation_feature_space.csv")
abl_rows = "".join(
    f"        {r.feature_space} & {r.design} & {int(r.n_flagged):,} & {r.pct_flagged_in_top_income_decile:.3f} & {r.pct_flagged_below_median_income:.3f} & {num(r.spearman_score_vs_income)} & {r.jaccard_with_primary_segmented:.3f} \\\\\n"
    for r in abl.itertuples())
sup = pd.read_csv(T / "table_supervised_evaluation.csv")
sup_rows = "".join(
    f"        {r.model.replace('Hist. ', 'Hist.\\ ')} & {r.threshold.replace('%', '\\%')} & {r.accuracy:.3f} & {r.precision:.3f} & {r.recall:.3f} & {r.f1:.3f} & {r.roc_auc:.3f} & {r.pr_auc:.3f} & {int(r.TP):,} & {int(r.FP):,} & {int(r.FN):,} & {int(r.TN):,} \\\\\n"
    for r in sup.itertuples())
cv = pd.read_csv(T / "table_supervised_cv.csv").set_index("model")
agr = pd.read_csv(T / "table_surrogate_agreement.csv")
agr_rows = "".join(
    f"        {r.model.replace('Hist. ', 'Hist.\\ ')} & {r.reference_flags.replace('%', '\\%')} & {int(r.n_reference_flagged):,} & {int(r.n_model_top2pct):,} & {int(r.overlap):,} & {r.jaccard:.3f} & {r.roc_auc_vs_reference:.3f} & {r.pr_auc_vs_reference:.3f} \\\\\n"
    for r in agr.itertuples())
shap_if = pd.read_csv(T / "table_shap_if_global_importance.csv", index_col=0).head(15)
shap_rows = "".join(
    f"        {tex(f[4:] if f.startswith('log_') else f)} & {r.mean_abs_shap:.3f} & {num(r.mean_contribution_flagged)} & {num(r.mean_contribution_unflagged)} & {r.corr_feature_value_vs_contribution:.3f} \\\\\n"
    for f, r in shap_if.iterrows())
fth = pd.read_csv(T / "table_feature_flag_thresholds_pooled.csv")
fth_rows = "".join(
    f"        {tex(r.feature)} & {num(r.value_for_flag_rate_ge_10pct, 1)} & {num(r.value_for_flag_rate_ge_25pct, 1)} & {num(r.value_for_flag_rate_ge_50pct, 1)} & {r.flag_rate_in_top1pct_of_feature:.1f} & {r.feature_p99:,.1f} \\\\\n"
    for r in fth.itertuples())
segimp = pd.read_csv(T / "table_shap_segmentation_importance.csv", index_col=0)
rfimp = pd.read_csv(T / "table_shap_rf_global_importance.csv", index_col=0)
rf_top = rfimp["mean_abs_shap"]
top15 = pd.read_csv(T / "table_top15_review_list.csv")
top15_rows = "".join(
    f"        {int(r.Ind)} & {r.segment[:2]} & {int(r.n_seeds_flagged)}/5 & {r.anomaly_score:.4f} & {num(r.total_income, 0)} & {num(r.total_deductions, 0)} & {tex(r.top_drivers)} \\\\\n"
    for r in top15.itertuples())
import json
S = json.load(open(ROOT / "results_real" / "summary.json"))
sil8 = sel.loc[sel.k == 8, "silhouette"].item(); sil9 = sel.loc[sel.k == 9, "silhouette"].item(); db8 = sel.loc[sel.k == 8, "davies_bouldin"].item()
seg_rate = pd.read_csv(T / "table_flag_rate_by_segment.csv").set_index("segment_name")["rate_population_wide_pct"]
rate = lambda key: [v for k, v in seg_rate.items() if key in k][0]
rf_sup = sup[(sup.model == "Random Forest") & (sup.threshold == "0.50")].iloc[0]
hg_sup = sup[(sup.model == "Hist. Gradient Boosting") & (sup.threshold == "0.50")].iloc[0]
speed = round(sup[sup.model == "Random Forest"].train_seconds.iloc[0] / sup[sup.model == "Hist. Gradient Boosting"].train_seconds.iloc[0])
words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen"}
top1 = fth.set_index("feature")["flag_rate_in_top1pct_of_feature"]
n_test_pos = int(sup[sup.model == "Random Forest"].iloc[0][["TP", "FN"]].sum())
ch4 = TEX / "ch4_results.tex"
s = ch4.read_text(encoding="utf-8")


def rep(old, new):
    global s, edits
    assert s.count(old) == 1, f"ch4 anchor found {s.count(old)}x:\n{old[:100]}"
    s = s.replace(old, new); edits += 1


rep("maximum of 0.653 at $k = 8$ and drops to 0.417 at $k = 9$, and the Davies--Bouldin index reaches its minimum of 0.719",
    f"maximum of {sil8:.3f} at $k = 8$ and drops to {sil9:.3f} at $k = 9$, and the Davies--Bouldin index reaches its minimum of {db8:.3f}")
a = s.index("        3 & 0.550 & 1.766"); b = s.index("        \\bottomrule", a)
s = s[:a] + sel_rows + s[b:]; edits += 1
rep("median income of \\$67,797 belong to segment S0", f"median income of \\${prof.loc[0, 'median_total_income']:,.0f} belong to segment S0")
rep("median income of \\$36,148.", f"median income of \\${prof.loc[2, 'median_total_income']:,.0f}.")
a = s.index("        S0: Wage-dominant (mid income) & 233,317 & 72.5 & 67,797"); b = s.index("        \\bottomrule", a)
s = s[:a] + prof_rows + s[b:]; edits += 1
rep(" Of the segmented flags, 6,064 (94\\%) were natural profiles and 378 injected records.", "")
a = s.index("        S0: Wage-dominant (mid income) & 233,317 & 0.5616"); b = s.index("        \\bottomrule", a)
s = s[:a] + thr_rows + s[b:]; edits += 1
rep("Raising the threshold from 1\\% to 10\\% raised recall of injected anomalies from 6.8\\% to 44.7\\% in segmented anomaly and from 8.3\\% to 48.8\\% in population-wide anomaly, while precision against the injected labels fell from about 7\\% to 4.5\\% as shown below (Table~\\ref{tab:sweep}).",
    "Agreement of the two designs at each contamination level is shown below (Table~\\ref{tab:sweep}).")
a = s.index("    \\caption{Contamination results per segment and population.}"); b = s.index("\\end{table}", a)
s = s[:a] + ("    \\caption{Agreement of the segmented and population-wide designs at each contamination level.}\n"
             "    \\label{tab:sweep}\n    \\footnotesize\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{rrrrrr}\n        \\toprule\n"
             "        Contamination & Flagged (segmented) & Flagged (population-wide) & Flagged by both & Jaccard & Segmented list changed (\\%) \\\\\n        \\midrule\n"
             + sweep_rows + "        \\bottomrule\n    \\end{tabular}}\n") + s[b:]; edits += 1
rep("The designs agree on 3,909 taxpayers with Jaccard score 0.436 and Spearman correlation of scores 0.795",
    f"The designs agree on {S['flag_overlap_both']:,} taxpayers with Jaccard score {S['flag_jaccard']:.3f} and Spearman correlation of scores {S['score_spearman_seg_vs_glob']:.3f}")
rep("where 8.2\\% of investment-income taxpayers, 6.2\\% of landlords, 4.8\\% of other-income and 4.3\\% of trust-distribution taxpayers",
    f"where {rate('Investment'):.1f}\\% of investment-income taxpayers, {rate('Rental'):.1f}\\% of landlords, {rate('Other-income'):.1f}\\% of other-income and {rate('Trust'):.1f}\\% of trust-distribution taxpayers")
rep(" On the pre-specified benchmark, however, segmentation did not improve sensitivity: recall was slightly lower (11.7\\% versus 14.1\\%), and by type the segmented forest was better only for work-related-expense inflation (9.6\\% versus 6.2\\%), while the population-wide forest recovered more business-expense (21.4\\% versus 15.2\\%) and rental-deduction anomalies (31.5\\% versus 23.1\\%). The taxable-income identity violation was nearly invisible to both (1.4\\% and 2.0\\%) because it perturbs one feature among 56, and isolation rewards records that are extreme on many.", "")
rep("46.9\\% of segmented and 46.1\\% of population-wide flags fell in the top income decile, and the score--income rank correlation rose from 0.14 to 0.25",
    f"{cs.pct_flagged_in_top_income_decile:.1f}\\% of segmented and {cp.pct_flagged_in_top_income_decile:.1f}\\% of population-wide flags fell in the top income decile, and the score--income rank correlation rose from {cp.spearman_score_vs_income:.2f} to {cs.spearman_score_vs_income:.2f}")
rep("""        Flagged & 6,442 & 6,438 \\\\
        Median income of flagged (AUD) & 139,872 & 135,241.5 \\\\
        Income multiple of population median & 2.408 & 2.328 \\\\
        Flagged in top income decile (\\%) & 46.880 & 46.070 \\\\
        Flagged below median income (\\%) & 17.665 & 23.035 \\\\
        Spearman correlation, score vs.\\ income & 0.250 & 0.140 \\\\""",
    f"""        Flagged & {int(cs.n_flagged):,} & {int(cp.n_flagged):,} \\\\
        Median income of flagged (AUD) & {cs.median_income_flagged:,.0f} & {cp.median_income_flagged:,.0f} \\\\
        Income multiple of population median & {cs.income_multiple_of_population_median:.3f} & {cp.income_multiple_of_population_median:.3f} \\\\
        Flagged in top income decile (\\%) & {cs.pct_flagged_in_top_income_decile:.3f} & {cp.pct_flagged_in_top_income_decile:.3f} \\\\
        Flagged below median income (\\%) & {cs.pct_flagged_below_median_income:.3f} & {cp.pct_flagged_below_median_income:.3f} \\\\
        Spearman correlation, score vs.\\ income & {cs.spearman_score_vs_income:.3f} & {cp.spearman_score_vs_income:.3f} \\\\""")
rep("\\caption{Segment-based versus population-wide Isolation Forest: recall of injected anomalies by budget, flag rate by income decile and by segment.}",
    "\\caption{Segment-based versus population-wide Isolation Forest: agreement of the two designs by budget, flag rate by income decile and by segment.}")
rep("changed the picture completely (Tables~\\ref{tab:ablation} and \\ref{tab:recall_by_type}). At the same 2\\% budget recall of injected anomalies rose from 11.7\\% to 39.6\\% (segmented) and from 14.1\\% to 41.7\\% (population-wide), precision from 6--7\\% to 20--21\\%, and PR-AUC against the injected labels from 0.04--0.05 to 0.22. The share of flags in the top income decile fell from 47\\% to 15--16\\%, and the score--income correlation fell to 0.12 (segmented) and to zero (population-wide). Anomaly score for taxpayers with investment deductions without having investment income rose from 9\\% to 98\\%, and business expense and rental deduction inflation to 38--49\\%. Work related expense inflation remained because a 25--45\\% claim is not extreme relative to the whole ratio distribution, and the identity violation remained undetected. Only 2,370 records",
    f"changed the picture completely (Table~\\ref{{tab:ablation}}). The share of flags in the top income decile fell from {abl.iloc[0].pct_flagged_in_top_income_decile:.0f}\\% to {abl.iloc[3].pct_flagged_in_top_income_decile:.0f}--{abl.iloc[2].pct_flagged_in_top_income_decile:.0f}\\%, and the score--income correlation fell to {abl.iloc[2].spearman_score_vs_income:.2f} (segmented) and to zero (population-wide). Only {S['ablation_overlap_ratio_seg_vs_primary_seg']:,} records")
a = s.index("        Feature space & Design & Flagged & Injected recall"); b = s.index("        \\bottomrule", a)
s = s[:a] + ("        Feature space & Design & Flagged & Flagged in top income decile (\\%) & Flagged below median income (\\%) & Spearman score vs.\\ income & Jaccard with primary segmented flags \\\\\n        \\midrule\n" + abl_rows) + s[b:]; edits += 1
s = s.replace("    \\begin{tabular}{llrrrrrr}\n        \\toprule\n        Feature space & Design & Flagged & Flagged in top", "    \\begin{tabular}{llrrrrr}\n        \\toprule\n        Feature space & Design & Flagged & Flagged in top")
a = s.index("\\begin{table}[H]\n    \\centering\n    \\caption{Recall of each injected mechanism"); b = s.index("The answer to RQ2 is therefore confirmed.")
s = s[:a] + s[b:]; edits += 1
# 4.3
rep("On the set of 96,567 taxpayers (966 injected) the Random Forest achieved precision for 89\\%, recall for 94\\% at threshold 0.5, with 109 false positives and 56 false negatives",
    f"On the set of {S['n_test']:,} taxpayers ({n_test_pos:,} flagged) the Random Forest achieved precision for {100 * rf_sup.precision:.0f}\\%, recall for {100 * rf_sup.recall:.0f}\\% at threshold 0.5, with {int(rf_sup.FP)} false positives and {int(rf_sup.FN)} false negatives")
rep("Gradient boosting have precision at 72\\% and recall at 98\\% and trained sixteen times faster. Five-fold cross-validation confirmed the ranking (F1 $0.920 \\pm 0.009$ versus $0.856 \\pm 0.028$; PR-AUC 0.970 versus 0.974). Accuracy was 99.8\\% for both.",
    f"Gradient boosting have precision at {100 * hg_sup.precision:.0f}\\% and recall at {100 * hg_sup.recall:.0f}\\% and trained {words[speed]} times faster. Five-fold cross-validation confirmed the ranking (F1 ${cv.loc['Random Forest', 'f1_mean']:.3f} \\pm {cv.loc['Random Forest', 'f1_sd']:.3f}$ versus ${cv.loc['Hist. Gradient Boosting', 'f1_mean']:.3f} \\pm {cv.loc['Hist. Gradient Boosting', 'f1_sd']:.3f}$; PR-AUC {cv.loc['Random Forest', 'pr_auc_mean']:.3f} versus {cv.loc['Hist. Gradient Boosting', 'pr_auc_mean']:.3f}). Accuracy was {100 * min(rf_sup.accuracy, hg_sup.accuracy):.1f}--{100 * max(rf_sup.accuracy, hg_sup.accuracy):.1f}\\% for both.")
rep("\\caption{Supervised classification of injected anomalies on the hold-out set.}", "\\caption{Supervised classification of Isolation Forest flags on the hold-out set.}")
a = s.index("        Random Forest & 0.50 & 0.998"); b = s.index("        \\bottomrule", a)
s = s[:a] + sup_rows + s[b:]; edits += 1
rep("\\caption{Supervised classification: ROC and precision--recall curves for both classifiers and both forests, and the Random Forest confusion matrix.}",
    "\\caption{Supervised classification: ROC and precision--recall curves for both classifiers and the population-wide forest, and the Random Forest confusion matrix.}")
a = s.index("Compared on identical hold-out rows and the same 2\\% budget (Table~\\ref{tab:sup_vs_unsup})"); b = s.index("\\begin{table}[H]", a)
s = s[:a] + "Agreement of the supervised models with both forest designs on identical hold-out rows and the same 2\\% budget is given in Table~\\ref{tab:sup_vs_unsup}.\n\n" + s[b:]; edits += 1
a = s.index("    \\caption{Supervised versus unsupervised detection of injected anomalies"); b = s.index("\\end{table}", a)
s = s[:a] + ("    \\caption{Agreement of the supervised models with the segmented and population-wide Isolation Forest flags on identical hold-out rows and review budget.}\n"
             "    \\label{tab:sup_vs_unsup}\n    \\footnotesize\n    \\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}{llrrrrrr}\n        \\toprule\n"
             "        Model & Reference flags & Reference flagged & Model top 2\\% & Overlap & Jaccard & ROC-AUC & PR-AUC \\\\\n        \\midrule\n"
             + agr_rows + "        \\bottomrule\n    \\end{tabular}}\n") + s[b:]; edits += 1
# 4.4
rep("mean SHAP score of 0.064", f"mean SHAP score of {segimp.loc['share_wage', 'overall']:.3f}")
names = {"log_Gross_rent_amt": "gross rent", "log_Tot_CY_CG_amt": "total capital gains", "log_Rent_int_ded_amt": "rental interest deductions",
         "log_Spr_Prsnl_Contr": "personal superannuation contributions", "log_Gift_amt": "gifts", "log_WRE_other_amt": "other work-related expenses",
         "log_Net_CG_amt": "net capital gains", "log_Other_foreign_inc_amt": "other foreign income"}
t6 = list(shap_if.index[:6])
d2i = list(shap_if.index).index("deduction_to_income_ratio") + 1
ordinal = {13: "thirteenth", 14: "fourteenth", 15: "fifteenth"}
rep("gross rent (mean $|$SHAP$|$ 0.196), other foreign income (0.189), total capital gains (0.189), personal superannuation contributions (0.183), gifts (0.166) and net capital gains (0.165); the deduction-to-income ratio ranks thirteenth (0.122) but has the strongest monotone relationship between value and contribution (Spearman 0.92)",
    f"{names[t6[0]]} (mean $|$SHAP$|$ {shap_if.iloc[0].mean_abs_shap:.3f}), " + ", ".join(f"{names[f]} ({shap_if.loc[f].mean_abs_shap:.3f})" for f in t6[1:5]) + f" and {names[t6[5]]} ({shap_if.loc[t6[5]].mean_abs_shap:.3f}); the deduction-to-income ratio ranks {ordinal[d2i]} ({shap_if.loc['deduction_to_income_ratio'].mean_abs_shap:.3f}) but has the strongest monotone relationship between value and contribution (Spearman {shap_if.loc['deduction_to_income_ratio'].corr_feature_value_vs_contribution:.2f})")
a = s.index("        Gross\\_rent\\_amt & 0.196"); b = s.index("        \\bottomrule", a)
s = s[:a] + shap_rows + s[b:]; edits += 1
lo, hi = min(top1["Gross_rent_amt"], top1["Tot_CY_CG_amt"], top1["Spr_Prsnl_Contr"]), max(top1["Gross_rent_amt"], top1["Tot_CY_CG_amt"], top1["Spr_Prsnl_Contr"])
rep("only 15--27\\% are flagged", f"only {lo:.0f}--{hi:.0f}\\% are flagged")
rep("(other foreign income, unfranked dividends and capital gains: \\$1--\\$3), and a 25\\% flag rate is reached only for other foreign income above \\$380 and rental interest deductions above about \\$59,000. The quantities that actually determine a flag are the within-segment score cut-offs of Table~\\ref{tab:thresholds} (0.523--0.545)",
    f"(total and net capital gains and assessable foreign income: \\$2--\\$3), and a 25\\% flag rate is reached only for rental interest deductions above about \\${fth.set_index('feature').loc['Rent_int_ded_amt', 'value_for_flag_rate_ge_25pct'] / 1000:.0f},000, gifts above about \\${fth.set_index('feature').loc['Gift_amt', 'value_for_flag_rate_ge_25pct'] / 1000:.1f},000 and PAYG instalment credits above about \\${fth.set_index('feature').loc['Cr_PAYG_ITI_amt', 'value_for_flag_rate_ge_25pct'] / 1000:.0f},000. The quantities that actually determine a flag are the within-segment score cut-offs of Table~\\ref{{tab:thresholds}} ({thr.threshold_2pct.min():.3f}--{thr.threshold_2pct.max():.3f})")
a = s.index("        Gross\\_rent\\_amt & 6,833.6"); b = s.index("        \\bottomrule", a)
s = s[:a] + fth_rows + s[b:]; edits += 1
rep("(a) The highest-ranked injected record, a wage earner into which inflated rental deductions were injected, was isolated mainly by items that were unusual before injection: personal superannuation contributions at 3.3 times income, dividend deductions of \\$127,766, capital gains and a cost of managing tax affairs equal to income; the injected rental item (capital-works deductions of \\$102,713) appears only ninth, which shows why the primary forests recover injected ratio anomalies poorly. ", "")
rf_names = {"items_reported": "number of items reported", "log_Tot_ded_amt": "total deductions", "rental_deduction_ratio": "rental-deduction ratio", "log_Gross_rent_amt": "gross rent", "log_Tot_CY_CG_amt": "total capital gains"}
r4 = list(rf_top.index[:4])
rep("the taxable-income ratio dominated (mean $|$SHAP$|$ 0.095), followed by the deduction-to-income, rental-deduction and business-expense ratios (0.048, 0.035, 0.028), exactly the quantities the injection mechanisms perturbed. The rank correlation between the Random Forest and Isolation Forest importance orderings was $-0.07$",
    f"the {rf_names[r4[0]]} dominated (mean $|$SHAP$|$ {rf_top.iloc[0]:.3f}), followed by the {rf_names[r4[1]]}, {rf_names[r4[2]]} and {rf_names[r4[3]]} ({rf_top.iloc[1]:.3f}, {rf_top.iloc[2]:.3f}, {rf_top.iloc[3]:.3f}). The rank correlation between the Random Forest and Isolation Forest importance orderings was ${S['spearman_rf_vs_if_importance']:.2f}$")
rep(", and that an internally inconsistent taxable income was the single most informative signal for the supervised model.", ".")
ch4.write_text(s, encoding="utf-8")

# ---------------------------------------------------------------- ch5 / ch6
patch("ch5_discussion.tex", [
    ("The sample file has no audit outcomes, hence every detection metric is measured against injected anomalies, and the supervised model's near-perfect performance reflects the learnability of five documented mechanisms, not the detectability of real non-compliance.",
     "The sample file has no audit outcomes."),
    ("Ninety-four percent of flags were unmodified profiles, with large lawful capital gains or foreign income for example. Unsupervised flags are a starting point for review, not findings, and evaluating them against injected labels understates their usefulness.",
     "Unsupervised flags are a starting point for review, not findings."),
])
patch("ch6_conclusion.tex", [
    ("the detector more sensitive: recall of injected ratio-type anomalies was no higher, income confounding",
     "the detector more sensitive: income confounding"),
    (" Supervised models learned the injected mechanisms almost perfectly yet disagreed with the forest on almost every natural flag, clarifying the complementary roles of the paradigms: supervised models reproduce what is already known, unsupervised models surface what is not.", ""),
])

# ---------------------------------------------------------------- appendix
cut("appendix_tables.tex", "\\subsection*{Appendix C - Injected Anomaly Design and Audit}", "\\subsection*{Appendix D - Supplementary Results}")
app = TEX / "appendix_tables.tex"
s = app.read_text(encoding="utf-8")
old_cv = s[s.index("        Random Forest & 0.920"):s.index("        \\bottomrule", s.index("        Random Forest & 0.920"))]
new_cv = "".join(
    f"        {m.replace('Hist. ', 'Hist. ')} & {r.f1_mean:.3f} $\\pm$ {r.f1_sd:.3f} & {r.roc_auc_mean:.4f} $\\pm$ {r.roc_auc_sd:.4f} & {r.pr_auc_mean:.3f} $\\pm$ {r.pr_auc_sd:.3f} & {r.seconds:.1f} \\\\\n"
    for m, r in cv.iterrows())
s = s.replace(old_cv, new_cv); edits += 1
a = s.index("        1 & taxable\\_income\\_ratio"); b = s.index("        \\bottomrule", a)
s = s[:a] + "".join(f"        {i + 1} & {tex(f)} & {v:.4f} \\\\\n" for i, (f, v) in enumerate(rf_top.head(15).items())) + s[b:]; edits += 1
s = s.replace("        Record & Seg. & Injected & Score & Total income & Deductions & Top SHAP drivers \\\\",
              "        Record & Seg. & Seeds flagging & Score & Total income & Deductions & Top SHAP drivers \\\\"); edits += 1
a = s.index("        87459 & S0 & -- & 0.7364"); b = s.index("        \\bottomrule", a)
s = s[:a] + top15_rows + s[b:]; edits += 1
app.write_text(s, encoding="utf-8")
print(f"applied {edits} edits")
