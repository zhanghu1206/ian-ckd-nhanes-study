#!/usr/bin/env python3
"""
NHANES CKDstudy - data download and merge script
includeCBCdata used to computeNLR,constructIAN(inflammation-nutrition score)
cycle: 2011-2012 (G), 2013-2014 (H), 2015-2016 (I), 2017-2018 (J)
"""

import os
import sys
import pandas as pd
import urllib.request
import time

# ============ configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJ_DIR, "data")
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NHANES cycle
CYCLES = {
    "G": ("2011-2012", 2011),
    "H": ("2013-2014", 2013),
    "I": ("2015-2016", 2015),
    "J": ("2017-2018", 2017),
}

# required files
REQUIRED_FILES = {
    "DEMO":  ("Demographics", "demographics"),
    "DIQ":   ("Questionnaire", "diabetes questionnaire"),
    "GHB":   ("Laboratory", "glycated hemoglobin (HbA1c)"),
    "ALB_CR":("Laboratory", "urine albumin/creatinine"),
    "BIOPRO":("Laboratory", "biochemistry panel"),
    "BPQ":   ("Questionnaire", "blood pressure"),
    "BMX":   ("Examination", "body measurements"),
    "MCQ":   ("Questionnaire", "medical conditions"),
    "KIQ_U": ("Questionnaire", "kidney disease"),
    "HDL":   ("Laboratory", "HDLcholesterol"),
    "TCHOL": ("Laboratory", "total cholesterol"),
    "SMQ":   ("Questionnaire", "smoking"),
    "CBC":   ("Laboratory", "complete blood count"),  # new:forNLRandHb
}

# optional files
OPTIONAL_FILES = {
    "GLU":   ("Laboratory", "fasting glucose"),
    "HSCRP": ("Laboratory", "hsCC-reactive protein"),
}


def download_xpt(cycle_letter, file_prefix, retries=3):
    """fromCDC NHANESwebsite downloadXPTfile"""
    cycle_info = CYCLES[cycle_letter]
    label, pub_year = cycle_info
    filename = f"{file_prefix}_{cycle_letter}.xpt"
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{pub_year}/DataFiles/{filename}"
    local_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(local_path):
        print(f"  [SKIP] {filename} already exists")
        return local_path

    for attempt in range(retries):
        try:
            print(f"  [download] {filename} ({file_prefix} - {label}) ...")
            urllib.request.urlretrieve(url, local_path)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  [OK]   {filename} ({size_kb:.1f} KB)")
            time.sleep(0.5)
            return local_path
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [retry] {filename} th{attempt+2}retry attempt...")
                time.sleep(2)
            else:
                print(f"  [failed] {filename}: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                return None


def load_xpt(file_prefix, cycle_letter):
    """loadXPTfile isDataFrame"""
    filename = f"{file_prefix}_{cycle_letter}.xpt"
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        return None
    try:
        df = pd.read_sas(local_path, format='xport', encoding='utf-8')
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception as e:
        print(f"  [error] read {filename} failed: {e}")
        return None


def calculate_egfr(scr, gender, age, is_black, scr_col_name="LBXSCR"):
    """CKD-EPI 2009equation calculationeGFR"""
    kappa = pd.Series(index=scr.index, dtype=float)
    alpha = pd.Series(index=scr.index, dtype=float)
    kappa[gender == 2] = 0.7
    kappa[gender == 1] = 0.9
    alpha[gender == 2] = -0.329
    alpha[gender == 1] = -0.411

    scr_k = scr / kappa
    egfr = 141 * scr_k.pow(-1.209) * 0.993 ** age
    egfr[gender == 2] *= 1.018
    egfr[is_black] *= 1.159
    egfr_low = 141 * scr_k.pow(alpha) * 0.993 ** age
    egfr_low[gender == 2] *= 1.018
    egfr_low[is_black] *= 1.159

    result = egfr.where(scr_k > 1, egfr_low)
    result.loc[result > 200] = 200
    return result


def process_cycle(cycle_letter):
    """process a singleNHANEScycle"""
    cycle_label = CYCLES[cycle_letter][0]
    print(f"\n{'='*60}")
    print(f"process cycle: {cycle_label} ({cycle_letter})")
    print('='*60)

    # ---- 1. download files ----
    print("\n--- download required files ---")
    downloaded = {}
    for prefix, (_, desc) in REQUIRED_FILES.items():
        path = download_xpt(cycle_letter, prefix)
        downloaded[prefix] = path is not None

    missing_required = [p for p, ok in downloaded.items() if not ok]
    if missing_required:
        print(f"\n[warning] missing required files: {missing_required}")

    print("\n--- download optional files ---")
    for prefix, (_, desc) in OPTIONAL_FILES.items():
        download_xpt(cycle_letter, prefix)

    # ---- 2. Load demographic data ----
    print("\n--- merge data ---")
    demo = load_xpt("DEMO", cycle_letter)
    if demo is None:
        print("[error] demographic data missing,skip this cycle")
        return None

    print(f"  DEMO: {demo.shape[0]} rows, {demo.shape[1]} cols")
    master = demo.copy()
    master['CYCLE'] = cycle_letter

    # ---- 3. merge one by one ----
    merge_files = {
        'DIQ': 'diabetes questionnaire',
        'GHB': 'glycated hemoglobin (HbA1c)',
        'ALB_CR': 'urine albumin/creatinine',
        'BIOPRO': 'biochemistry panel',
        'BPQ': 'blood-pressure questionnaire',
        'BMX': 'body measurements',
        'MCQ': 'medical conditions',
        'KIQ_U': 'kidney disease',
        'HDL': 'HDLcholesterol',
        'TCHOL': 'total cholesterol',
        'SMQ': 'smoking',
        'CBC': 'complete blood count',
        'GLU': 'fasting glucose',
        'HSCRP': 'hsCRP',
    }

    for prefix, desc in merge_files.items():
        df = load_xpt(prefix, cycle_letter)
        if df is None:
            print(f"  {desc}: skip (file does not exist)")
            continue
        cols = [c for c in df.columns]
        keep_cols = ['SEQN'] + [c for c in cols if c != 'SEQN']
        dup_cols = [c for c in keep_cols if c in master.columns and c != 'SEQN']
        if dup_cols:
            df = df.rename(columns={c: f"{prefix}_{c}" for c in dup_cols})
            keep_cols = ['SEQN'] + [f"{prefix}_{c}" for c in cols if c != 'SEQN' and c in dup_cols]
        df_keep = df[keep_cols].copy()
        master = pd.merge(master, df_keep, on='SEQN', how='left')

    print(f"\n  cycle {cycle_label} after merging: {master.shape[0]} rows, {master.shape[1]} cols")
    return master


def main():
    """main function:Download all cycle data and merge"""
    all_data = []

    for cycle_letter in CYCLES:
        result = process_cycle(cycle_letter)
        if result is not None:
            all_data.append(result)

    if not all_data:
        print("\n[error] No cycle data was successfully downloaded!")
        sys.exit(1)

    # ---- merge all cycles ----
    print(f"\n{'='*60}")
    print("Merge all cycle data...")
    print('='*60)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"total sample size: {combined.shape[0]} rows, {combined.shape[1]} cols")

    # ---- compute derived variables ----
    print("\n--- compute derived variables ---")

    # 1. eGFR (CKD-EPI 2009)
    scr_col = None
    for col in combined.columns:
        if col.upper() == 'LBXSCR':
            scr_col = col
            break
    # Check for BIOPRO_ prefixed versions
    if scr_col is None:
        for col in combined.columns:
            if 'LBXSCR' in col.upper():
                scr_col = col
                break

    if scr_col:
        print(f"  serum creatinine column: {scr_col}")
        combined['eGFR'] = calculate_egfr(
            combined[scr_col],
            combined['RIAGENDR'],
            combined['RIDAGEYR'],
            combined['RIDRETH1'] == 4
        )
        print(f"  eGFR range: {combined['eGFR'].min():.1f} - {combined['eGFR'].max():.1f}")
    else:
        print("  [warning] serum creatinine not found")
        combined['eGFR'] = None

    # 2. UACR
    uma_col = ucr_col = None
    for col in combined.columns:
        if 'URXUMA' in col.upper(): uma_col = col
        if 'URXUCR' in col.upper(): ucr_col = col
    if uma_col and ucr_col:
        combined['UACR'] = combined[uma_col] * 100 / combined[ucr_col]
        combined.loc[combined['UACR'] > 5000, 'UACR'] = 5000
        print(f"  UACR range: {combined['UACR'].min():.1f} - {combined['UACR'].max():.1f}")
    else:
        print("  [warning] urine albumin not found/creatinine")
        combined['UACR'] = None

    # 3. NLR (Neutrophil-to-Lymphocyte Ratio)
    neut_col = None
    lymph_col = None
    for col in combined.columns:
        if 'LBDNENO' in col.upper():
            neut_col = col
        if 'LBDLYMNO' in col.upper():
            lymph_col = col

    if neut_col and lymph_col:
        combined['NEUTROPHIL'] = combined[neut_col]  # 1000 cells/uL
        combined['LYMPHOCYTE'] = combined[lymph_col]  # 1000 cells/uL
        combined['NLR'] = combined['NEUTROPHIL'] / combined['LYMPHOCYTE']
        # limit outliers
        combined.loc[combined['NLR'] > 20, 'NLR'] = 20
        print(f"  NLR range: {combined['NLR'].min():.2f} - {combined['NLR'].max():.2f}")
    else:
        print("  [warning] Neutrophil or lymphocyte counts not found")
        combined['NLR'] = None

    # 4. Hemoglobin
    hgb_col = None
    for col in combined.columns:
        if 'LBXHGB' in col.upper():
            hgb_col = col
            break
    if hgb_col:
        combined['HEMOGLOBIN'] = combined[hgb_col]
        print(f"  Hb range: {combined['HEMOGLOBIN'].min():.1f} - {combined['HEMOGLOBIN'].max():.1f}")
    else:
        print("  [warning] hemoglobin not found")
        combined['HEMOGLOBIN'] = None

    # 5. Albumin
    alb_col = None
    for col in combined.columns:
        if col.upper() == 'LBXSAL':
            alb_col = col
            break
    if alb_col is None:
        for col in combined.columns:
            if 'LBXSAL' in col.upper():
                alb_col = col
                break
    if alb_col:
        combined['ALBUMIN'] = combined[alb_col]
        print(f"  albumin range: {combined['ALBUMIN'].min():.1f} - {combined['ALBUMIN'].max():.1f}")
    else:
        print("  [warning] albumin not found")
        combined['ALBUMIN'] = None

    # 6. diabetes definition
    has_diabetes = pd.Series(False, index=combined.index)
    # physician diagnosis
    diq_col = None
    for col in combined.columns:
        if col.upper() == 'DIQ010':
            diq_col = col
            break
    if diq_col is not None:
        has_diabetes = has_diabetes | (combined[diq_col] == 1)
    # HbA1c >= 6.5%
    for col in combined.columns:
        if 'LBXGH' in col.upper():
            hba1c_dm = combined[col] >= 6.5
            has_diabetes = has_diabetes | hba1c_dm
            break
    # Fasting glucose >= 126
    for col in combined.columns:
        if 'LBXGLU' in col.upper():
            fg_dm = combined[col] >= 126
            has_diabetes = has_diabetes | fg_dm
            break
    combined['DIABETES'] = has_diabetes.astype(int)
    print(f"  diabetes: {has_diabetes.sum()} persons")

    # 7. CKDdefinition (eGFR < 60 OR UACR >= 30) - regardless of diabetes status
    if 'eGFR' in combined.columns and 'UACR' in combined.columns:
        ckd_condition = (
            combined['eGFR'].notna() | combined['UACR'].notna()
        ) & (
            (combined['eGFR'].fillna(999) < 60) | (combined['UACR'].fillna(0) >= 30)
        )
        combined['CKD'] = ckd_condition.astype(int)
        n_ckd = ckd_condition.sum()
        print(f"  CKD: {n_ckd} persons ({n_ckd/combined.shape[0]*100:.1f}%)")
    else:
        combined['CKD'] = None
        print("  [warning] missingeGFRorUACR,cannot be definedCKD")

    # ---- save ----
    output_path = os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Merged data saved: {output_path}")
    print(f"final dataset: {combined.shape[0]} rows, {combined.shape[1]} cols")
    print('='*60)

    return combined


if __name__ == "__main__":
    main()
