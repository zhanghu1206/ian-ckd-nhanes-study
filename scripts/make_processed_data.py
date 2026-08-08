#!/usr/bin/env python3
# ==============================================================================
# IAN-CKD: derive the analytical (processed) dataset from nhanes_ckd_merged.csv
#
# IAN construction is IDENTICAL to the manuscript / advanced_analysis.py /
# survey_weighted_regression.R:
#   - tertile cut points derived from the TRAINING cycles (G/H/I, 2011-2016)
#   - NLR scored 0-2 (higher = worse); Hemoglobin & Albumin reverse-coded (2-0)
#   - IAN = NLR_T + HB_T + ALB_T   (range 0-6)
#
# Analytical sample = adults >=20 with complete NLR/Hemoglobin/Albumin/CKD
# (n = 20,222, matching the manuscript). No pregnancy subtraction in the
# reported arithmetic (22,617 eligible - 2,395 missing components = 20,222).
#
# Output: output/processed_data.csv  (consumed by scripts/r/*.R figure scripts)
# Relative paths only -- runs from any directory.
# ==============================================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
SRC = os.path.join(PROJ, "output", "nhanes_ckd_merged.csv")
OUT = os.path.join(PROJ, "output", "processed_data.csv")


def tertile_bins(series):
    """Return 4 break points (-Inf, q1, q2, Inf) like pandas qcut/q=3."""
    q = series.quantile([1 / 3, 2 / 3]).values
    return [-np.inf, float(q[0]), float(q[1]), np.inf]


def score(x, bins, labels):
    """Tertile score as numeric (labels -> float, never factor index)."""
    return pd.cut(x, bins=bins, include_lowest=True, labels=labels).astype(float)


def main():
    df = pd.read_csv(SRC, low_memory=False)
    adult = df[df["RIDAGEYR"] >= 20].copy()
    data = adult.dropna(subset=["NLR", "HEMOGLOBIN", "ALBUMIN", "CKD"]).copy()

    # Tertile cut points from the TRAINING cycles only (G/H/I)
    train = data[data["CYCLE"].isin(["G", "H", "I"])]
    nlr_b = tertile_bins(train["NLR"])
    hb_b = tertile_bins(train["HEMOGLOBIN"])
    alb_b = tertile_bins(train["ALBUMIN"])

    data["NLR_T"] = score(data["NLR"], nlr_b, [0, 1, 2])
    data["HB_T"] = score(data["HEMOGLOBIN"], hb_b, [2, 1, 0])   # reverse-coded
    data["ALB_T"] = score(data["ALBUMIN"], alb_b, [2, 1, 0])    # reverse-coded
    data["ian"] = data["NLR_T"] + data["HB_T"] + data["ALB_T"]

    data["ckd"] = data["CKD"].astype(int)
    data["age"] = data["RIDAGEYR"].astype(float)
    data["sex"] = np.where(data["RIAGENDR"] == 1, "Male", "Female")
    data["bmi"] = data["BMXBMI"]
    data["nlr"] = data["NLR"]
    data["diabetes"] = data["DIABETES"].astype(int)
    data["hypertension"] = (data["BPQ020"] == 1).astype(int)

    cols = ["SEQN", "CYCLE", "ian", "ckd", "age", "sex", "eGFR", "UACR", "bmi",
            "HEMOGLOBIN", "ALBUMIN", "nlr", "diabetes", "hypertension"]
    out_df = data[cols].dropna(subset=["ian", "ckd", "age", "sex"]).copy()

    out_df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(f"  n = {len(out_df):,}")
    print(f"  IAN mean = {out_df['ian'].mean():.3f}, range [{out_df['ian'].min():.0f}, {out_df['ian'].max():.0f}]")
    print(f"  CKD prevalence = {out_df['ckd'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
