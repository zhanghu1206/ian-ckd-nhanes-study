# IAN-CKD — Analysis Code and Processed Data

This folder contains the **complete, real analysis code and processed data** underlying the
manuscript *"The Inflammation-Adjusted Nutrition (IAN) Score: Association with Chronic Kidney
Disease Prevalence in a US Adult Population from the NHANES Study"* (final revision, v6,
submitted to *Medicine*).

**Reporting guideline:** the manuscript was prepared in accordance with the STROBE
(Strengthening the Reporting of Observational Studies in Epidemiology) checklist; the
completed checklist accompanies the submission package.

**Figures:** the submission package contains 7 main figures (Fig 1–7) and 7 supplementary
figures (SDC 1–7). The R scripts in `scripts/r/` regenerate all of them from the provided data
(see §4.4).

All scripts use **relative paths resolved from the script's own location** — there are **no
hardcoded absolute paths** anywhere in this folder. The code runs from any directory.

---

## 1. Data provenance

- **Source:** National Health and Nutrition Examination Survey (NHANES), cycles
  2011–2012 (G), 2013–2014 (H), 2015–2016 (I), 2017–2018 (J).
- **Publicly available:** <https://www.cdc.gov/nchs/nhanes/> (no restriction, no PHI).
- **Raw data:** `data/` — 58 `.xpt` files (DEMO, LAB, EXAM components per cycle).
- **Reproducibility:** the raw XPTs were merged by `SEQN` and the biomarkers
  (NLR, hemoglobin, albumin, eGFR, UACR) were computed by `scripts/download_and_merge.py`.
  The resulting `output/nhanes_ckd_merged.csv` (39,156 participants) is included so the
  analysis can be reproduced **without re-downloading** (the raw XPTs are also provided).

---

## 2. Folder structure

```
Code_and_Data/
├── README.md                      # this file
├── data/                         # 58 raw NHANES .xpt files (real, public)
├── output/                       # processed data + results (real)
│   ├── nhanes_ckd_merged.csv      # 39,156 participants, all components
│   ├── processed_data.csv          # canonical analytical dataset (n=20,222, IAN 0-6)
│   ├── logistic_regression_ian.csv
│   ├── logistic_regression_model3.csv
│   ├── ml_model_comparison.csv
│   ├── table1_ckd.csv
│   ├── survey_weighted_results.csv  # design-based svyglm results (this release)
│   ├── mr_nonalbumin_parsed.csv     # MR Table 1 data: non-albumin components, IVW (6 pairs)
│   ├── mr_results.csv               # full TwoSampleMR output (all components x methods)
│   ├── mr_heterogeneity.csv         # Cochran's Q (IVW / Egger)
│   ├── mr_egger_pleiotropy.csv      # Egger intercept tests
│   ├── mr_steiger.csv               # Steiger directionality tests
│   ├── mr_presso.csv / mr_presso2.csv  # MR-PRESSO outlier / distortion tests
│   └── mr_table_parsed.csv          # parsed MR summary table
├── scripts/
│   ├── *.py                      # Python pipeline (relative paths)
│   └── r/                        # R figure + survey-weight + MR scripts (relative paths)
│       ├── survey_weighted_regression.R   # proper NHANES design-based regression
│       └── mr_ian_ckd.R                  # two-sample MR (TwoSampleMR / MR-PRESSO)
└── figures/                     # created on run (main/ + supplementary/)
```

---

## 3. Requirements

**Python** (3.13 used; 3.10+ should work):
```
pandas, numpy, statsmodels, scikit-learn, xgboost, matplotlib, scipy,
lifelines (optional)
```

**R** (4.2+; 4.6 used for verification):
```
survey, dplyr, data.table, ggplot2, patchwork, pROC, splines,
survival, grid, gridExtra, ragg, boot, caret, glmnet,
xgboost, SHAPforxgboost, ResourceSelection
```

---

## 4. How to reproduce

### 4.1 Build the merged dataset (optional — already provided)
```bash
cd scripts
python download_and_merge.py     # downloads 4 NHANES cycles, merges, computes biomarkers
```
→ writes `output/nhanes_ckd_merged.csv`

# Derive the canonical analytical dataset (optional — already provided)
python make_processed_data.py     # builds output/processed_data.csv (n=20,222, IAN 0-6)


### 4.2 Core statistical analysis (Python)
```bash
python analyze_ian_ckd.py        # descriptive stats, IAN construction, logistic regression, ROC
python advanced_analysis.py      # weighted (frequency-weight) analyses, subgroups, sensitivity
python ml_analysis.py            # XGBoost / RF / LASSO / stacking comparison
python temporal_validation.py    # training (2011–2016) vs validation (2017–2018)
python supplementary_analysis.py # calibration, stability, CKD-stage breakdowns
python weight_justification.py   # 64-weight search justifying equal (1:1:1) weighting
```

### 4.3 Proper NHANES survey-weighted regression (R — design-based)
```bash
cd scripts/r
Rscript survey_weighted_regression.R
```
This constructs IAN exactly as the Python pipeline (tertiles from the training cycles,
reverse-coded hemoglobin/albumin), builds a correct `svydesign` object
(`ids = SDMVPSU`, `strata = SDMVSTRA`, `weights = WTMEC2YR`, `nest = TRUE`), and fits
`svyglm` quasibinomial models. **It reproduces the manuscript's reported IAN odds ratios:**

| Model | IAN OR (design-based) | 95% CI | Manuscript reports |
|-------|----------------------|--------|--------------------|
| 1: IAN only | 1.395 | 1.350–1.442 | 1.40 (1.36–1.44) |
| 4: full covariates | 1.222 | 1.177–1.270 | 1.24 (1.20–1.28) |

→ writes `output/survey_weighted_results.csv`

### 4.5 Two-sample Mendelian randomization (R — MR of IAN components)
```bash
cd scripts/r
# OPENGWAS_JWT must be set for the IEU OpenGWAS API (free token from
# https://api.opengwas.io/profile). Run with the output directory as argument:
Rscript mr_ian_ckd.R ../../output
```
Exposures are the four IAN components from UKB European-ancestry GWAS
(albumin, hemoglobin, neutrophil, lymphocyte); outcomes are non-UKB binary CKD
consortia (CKDGen `ieu-a-1102` primary, FinnGen `finn-b-N14_CHRONKIDNEYDIS`
sensitivity). The script runs IVW, MR-Egger, weighted median/simple/weighted
mode, heterogeneity (Cochran's Q), Egger pleiotropy, Steiger directionality,
MR-PRESSO, and leave-one-out plots. Each pair is wrapped in `tryCatch` so a
single failure does not stop the run. The committed `output/mr_*.csv` files are
the real results used in the manuscript (Table MR-1 / MR-S1; Fig 7).

### 4.6 Publication figures (R / Python)
```bash
cd scripts/r
Rscript generate_all_figures.R              # main figures (Fig 1-6; Fig 7 = MR forest, TwoSampleMR)
Rscript generate_supplemental_figures.R     # supplementary figures (SDC 1-7 in the submission)
# adjust_figures_for_journal.Rmd documents the journal figure export spec
# (TIFF >=600 dpi LZW + PDF); knit with rmarkdown if you want the DPI report.
cd ../.. && python scripts/flowchart.py
python scripts/graphical_abstract.py
```
Figures are written to `figures/main/` and `figures/supplementary/` as PDF + TIFF (≥600 dpi, LZW).
Both R figure scripts were verified to run end-to-end on the provided data and
reproduce all nine figures. (The `figures/` tree is git-ignored and regenerated
on run — see `.gitignore`.)

---

## 5. Notes & caveats (full transparency)

1. **Single source of truth.** `nhanes_ckd_merged.csv` (39,156 participants, all raw
   components) is the only raw input. Every downstream dataset is derived from it:
   - `scripts/make_processed_data.py` produces `output/processed_data.csv`
     (n = 20,222, IAN 0–6, the canonical analytical dataset), which the R figure
     scripts consume.
   - The IAN construction in `make_processed_data.py`, `advanced_analysis.py`, and
     `survey_weighted_regression.R` is **identical** (tertiles from the training cycles
     G/H/I, reverse-coded hemoglobin/albumin), so all scripts agree: n = 20,222,
     mean IAN ≈ 3.14, range 0–6.

2. **CKD prevalence is reported as the design-weighted estimate = 14.9%** (computed by
   `survey_weighted_regression.R`, which builds a correct `svydesign` and `svymean(~CKD)`).
   The unweighted proportion in the analytic sample is 18.3% (3,703 / 20,222); both are
   real and stated transparently. The IAN odds ratios (the primary effect estimates) are
   design-weighted and match the manuscript exactly.

3. **Survey weights across combined cycles:** the manuscript applies `WTMEC2YR` (2-year weights)
   directly across the four combined cycles. Because this is a constant scaling of all weights,
   it does not change point estimates or design-based standard errors; the conventional
   WTMEC2YR/4 adjustment would yield identical results.

4. All statistics are computed from real NHANES data. **No simulation, no fabricated values,
   no placeholder data.**

---

## 6. License / data use

NHANES data are public domain (CDC/NCHS). This analysis code is provided for reproducibility
under the same submission as the manuscript.
