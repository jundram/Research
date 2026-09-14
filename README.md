# Taxpayer Classification and Tax Anomaly Detection Using Unsupervised Machine Learning with SHAP-Based Explainability

Master of Data Science capstone project. A four-stage taxpayer analytics
framework, implemented end-to-end in one runnable Python script, applied to the
ATO 2022–23 individual sample file with a documented set of injected synthetic
anomalies.

| Stage | What it does | Method |
|---|---|---|
| 1 | Taxpayer segmentation into peer groups | K-Means on income-composition shares + scale (compared with GMM and Ward) |
| 2 | Unsupervised anomaly detection within each peer group | Isolation Forest per segment, compared with a population-wide forest; contamination 1/2/5/10 % |
| 3 | Supervised anomaly classification | Random Forest (primary) and Histogram Gradient Boosting on injected labels |
| 4 | Explainability | SHAP: surrogate model for segments, TreeExplainer for the forests |

**Interpretive rule used throughout:** an anomaly flag is a statistical signal
that a return is unusual relative to its peer group. It is not evidence of
non-compliance or fraud. The injected labels are synthetic and are not audit
outcomes.

## Folder layout

```
Segmented_Anomaly_SHAP_Framework/
├── taxpayer_framework.py      the complete pipeline (sections 1–23)
├── requirements.txt
├── figures/                   fig01 … fig11 (PNG, 200 dpi)
├── tables/                    every table quoted in the paper (CSV)
├── results/                   run_log.txt, summary.json, taxpayer_scores.csv
└── research_paper/            Capstone_Report.pdf (the submitted report)
    └── latex/                 LaTeX/Overleaf source of the report (main.tex, chapters, bib.bib, figures)
```

## Data (not included)

The ATO 2022–23 individual sample file is confidential and is **not** in this
repository. The script expects it at `../data/2023_sample_file_SA4.csv`.
If it is absent, a schema-matched synthetic dataset is generated so the
pipeline still runs end-to-end (results then differ from the paper).
Row-level outputs (`results/taxpayer_scores.csv` and the two per-record
tables) are likewise excluded via `.gitignore`; the aggregate tables, figures
and `results/summary.json` that the report quotes are included.

The submitted report is `research_paper/Capstone_Report.pdf`; its LaTeX source is in
`research_paper/latex/` (compile `main.tex` with pdflatex + bibtex, or upload the folder to Overleaf).

## How to run

```bash
pip install -r requirements.txt
python taxpayer_framework.py     # ~15 min on an 8-core laptop; writes figures/, tables/, results/
```

`results/summary.json` contains every headline number quoted in the paper;
`results/taxpayer_scores.csv` contains one row per taxpayer with segment,
anomaly scores, flags at each contamination level, injected label and (for
hold-out rows) the Random Forest probability.
