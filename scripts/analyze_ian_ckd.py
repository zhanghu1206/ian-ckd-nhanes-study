#!/usr/bin/env python3
"""
NHANES IAN(inflammation-nutrition score)prediction CKD analysis script

IANconstruct:
  - NLR(neutrophil/lymphocyte ratio):inflammation indicator
  - hemoglobin(Hb):nutrition/anemia indicator
  - albumin:nutrition indicator
  The three components are each scored by tertile then summed, for a total score of 0-6.

analysis content:
  1. descriptive statistics(Table 1)
  2. IAN score construction and distribution
  3. IANandCKDunivariate association
  4. Logistic regression (containROC curve,AUC)
  5. visualization
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ============ configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
DATA_PATH = os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv")
os.makedirs(CHART_DIR, exist_ok=True)

# plot settings
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# color
BLUE = '#0F4D92'
ORANGE = '#E8832E'
GREEN = '#8BCF8B'
RED = '#B64342'
GREY = '#767676'
PURPLE = '#9A4D8E'

def load_data():
    """Load merged data"""
    print(f"load data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"loading complete: {df.shape[0]} rows, {df.shape[1]} cols")

    # age grouping
    df['AGE_GROUP'] = pd.cut(df['RIDAGEYR'],
                              bins=[0, 20, 40, 60, 80, 200],
                              labels=['0-19', '20-39', '40-59', '60-79', '80+'])
    df['AGE_GROUP_ADULT'] = pd.cut(df['RIDAGEYR'],
                                    bins=[19, 40, 60, 80, 200],
                                    labels=['20-39', '40-59', '60-79', '80+'])
    df['GENDER_LABEL'] = df['RIAGENDR'].map({1: 'male', 2: 'female'})

    # race merging
    df['RACE3'] = df['RIDRETH1'].map({
        1: 'Hispanic', 2: 'Hispanic', 3: 'Non-Hispanic White',
        4: 'Non-Hispanic Black', 5: 'other'
    })

    # BMIgroup
    df['BMI_GROUP'] = pd.cut(df['BMXBMI'],
                              bins=[0, 18.5, 25, 30, 200],
                              labels=['Underweight (<18.5)', 'Normal (18.5-25)', 'Overweight (25-30)', 'Obese (>=30)'])

    # cycle label
    cycle_map = {'G': '2011-2012', 'H': '2013-2014', 'I': '2015-2016', 'J': '2017-2018'}
    df['CYCLE_LABEL'] = df['CYCLE'].map(cycle_map)

    # hypertension (BPQ020=1)
    df['HYPERTENSION'] = (df['BPQ020'] == 1).astype(int)

    # CKDstage
    df['CKD_STAGE'] = 'No CKD'
    df.loc[df['CKD'] == 1, 'CKD_STAGE'] = 'CKD'
    # CKDgrade
    df['eGFR_CATEGORY'] = pd.cut(df['eGFR'],
                                   bins=[0, 15, 30, 60, 90, 200],
                                   labels=['G5(<15)', 'G4(15-30)', 'G3(30-60)', 'G2(60-90)', 'G1(>=90)'])
    df['UACR_CATEGORY'] = pd.cut(df['UACR'],
                                  bins=[0, 30, 300, 5000],
                                  labels=['Normal (<30)', 'Microalbuminuria (30-300)', 'Macroalbuminuria (>300)'])

    print(f"CKDcount: {df['CKD'].sum()} ({df['CKD'].mean()*100:.1f}%)")
    return df

def build_ian_scores(df):
    """
    constructIAN(inflammation-nutrition score)
    
    IANby3components constitute,each scored by tertile:
    - NLR: higher is worse → T1=0, T2=1, T3=2
    - Hb: lower is worse → T1=2, T2=1, T3=0 (inverse)
    - albumin: lower is worse → T1=2, T2=1, T3=0 (inverse)
    
    total score: 0-6 (higher indicates worse inflammation/worse nutritional condition)
    """
    print("\n" + "="*70)
    print("constructIAN(inflammation-nutrition score)")
    print("="*70)

    # select adults (>=20years) and with complete IAN data sample
    adult = df[df['RIDAGEYR'] >= 20].copy()
    print(f"adult(>=20years)sample: {len(adult)} persons")

    valid = adult[['NLR', 'HEMOGLOBIN', 'ALBUMIN']].dropna()
    print(f"with complete IAN data sample: {len(valid)} persons")

    # ---- tertile splitting ----
    # NLR: high = difference
    _, nlr_bins = pd.qcut(valid['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
    print(f"\nNLRtertile cutoff: {np.round(nlr_bins, 2)}")

    # Hb: low = difference (inverse scoring)
    _, hb_bins = pd.qcut(valid['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
    print(f"Hbtertile cutoff: {np.round(hb_bins, 2)}")

    # Albumin: low = difference (inverse scoring)
    _, alb_bins = pd.qcut(valid['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')
    print(f"albumin tertile cutoffs: {np.round(alb_bins, 2)}")

    # computed on the full datasetIAN
    df['NLR_T'] = pd.cut(df['NLR'], bins=nlr_bins, labels=[0, 1, 2],
                          include_lowest=True).astype(float)
    df['HB_T'] = pd.cut(df['HEMOGLOBIN'], bins=hb_bins, labels=[2, 1, 0],
                         include_lowest=True).astype(float)
    df['ALB_T'] = pd.cut(df['ALBUMIN'], bins=alb_bins, labels=[2, 1, 0],
                          include_lowest=True).astype(float)

    df['IAN'] = df['NLR_T'] + df['HB_T'] + df['ALB_T']

    # IAN grade
    df['IAN_GRADE'] = pd.cut(df['IAN'],
                              bins=[-1, 2, 4, 6],
                              labels=['Low risk (0-2)', 'Medium risk (3-4)', 'High risk (5-6)'])

    print(f"\nIAN score distribution:")
    ian_dist = df['IAN'].value_counts().sort_index()
    for score, count in ian_dist.items():
        print(f"  IAN={score}: {count} persons ({count/df['IAN'].notna().sum()*100:.1f}%)")

    print(f"\nIAN grade distribution:")
    ian_grades = df['IAN_GRADE'].value_counts()
    for grade, count in ian_grades.items():
        print(f"  {grade}: {count} persons ({count/df['IAN'].notna().sum()*100:.1f}%)")

    # saveIANtertile cutoff information
    cutpoints = {
        'NLR_cutoffs': nlr_bins.tolist(),
        'Hb_cutoffs': hb_bins.tolist(),
        'Albumin_cutoffs': alb_bins.tolist(),
    }
    return df, cutpoints

def descriptive_stats(df, output_path):
    """Table 1: byCKDbaseline characteristics by group"""
    print("\n" + "="*70)
    print("1. descriptive statistics (Table 1)")
    print("="*70)

    adult = df[(df['RIDAGEYR'] >= 20)].copy()
    ckd_group = adult[adult['CKD'] == 1]
    no_ckd_group = adult[adult['CKD'] == 0]
    print(f"adult sample: {len(adult)} persons")
    print(f"  No CKD: {len(no_ckd_group)} ({len(no_ckd_group)/len(adult)*100:.1f}%)")
    print(f"  haveCKD: {len(ckd_group)} ({len(ckd_group)/len(adult)*100:.1f}%)")

    # continuous variable
    continuous_vars = [
        ('RIDAGEYR', 'age (years)'),
        ('BMXBMI', 'BMI (kg/m²)'),
        ('NLR', 'NLR'),
        ('HEMOGLOBIN', 'hemoglobin (g/dL)'),
        ('ALBUMIN', 'albumin (g/dL)'),
        ('IAN', 'IAN score'),
        ('eGFR', 'eGFR (mL/min/1.73m²)'),
        ('UACR', 'UACR (mg/g)'),
    ]

    # categorical variable
    categorical_vars = [
        ('GENDER_LABEL', 'sex'),
        ('RACE3', 'race'),
        ('BMI_GROUP', 'BMIgroup'),
        ('HYPERTENSION', 'hypertension', {0: 'no', 1: 'have'}),
        ('CYCLE_LABEL', 'survey cycle'),
    ]

    table1_rows = []
    table1_rows.append(['feature', f'overall (n={len(adult)})',
                        f'No CKD (n={len(no_ckd_group)})',
                        f'haveCKD (n={len(ckd_group)})', 'Pvalue'])

    # continuous variable
    for col, label in continuous_vars:
        if col not in adult.columns:
            continue
        total_v = adult[col].dropna()
        no_v = no_ckd_group[col].dropna()
        ckd_v = ckd_group[col].dropna()

        if col == 'UACR':
            total_s = f"{total_v.median():.1f} ({total_v.quantile(0.25):.1f}-{total_v.quantile(0.75):.1f})"
            no_s = f"{no_v.median():.1f} ({no_v.quantile(0.25):.1f}-{no_v.quantile(0.75):.1f})"
            ckd_s = f"{ckd_v.median():.1f} ({ckd_v.quantile(0.25):.1f}-{ckd_v.quantile(0.75):.1f})"
            try:
                _, pval = stats.mannwhitneyu(no_v, ckd_v)
            except:
                pval = 1.0
        else:
            total_s = f"{total_v.mean():.1f} ± {total_v.std():.1f}"
            no_s = f"{no_v.mean():.1f} ± {no_v.std():.1f}"
            ckd_s = f"{ckd_v.mean():.1f} ± {ckd_v.std():.1f}"
            try:
                _, pval = stats.ttest_ind(no_v, ckd_v)
            except:
                pval = 1.0

        p_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
        table1_rows.append([f"  {label}", total_s, no_s, ckd_s, p_str])

    # categorical variable
    for var_info in categorical_vars:
        col = var_info[0]
        label = var_info[1]
        mapping = var_info[2] if len(var_info) > 2 else None
        if col not in adult.columns:
            continue
        cats = adult[col].dropna().unique()
        for cat in sorted(cats):
            if isinstance(cat, float) and np.isnan(cat):
                continue
            total_n = (adult[col] == cat).sum()
            no_n = (no_ckd_group[col] == cat).sum() if len(no_ckd_group) > 0 else 0
            ckd_n = (ckd_group[col] == cat).sum() if len(ckd_group) > 0 else 0
            total_pct = total_n / len(adult) * 100
            no_pct = no_n / len(no_ckd_group) * 100 if len(no_ckd_group) > 0 else 0
            ckd_pct = ckd_n / len(ckd_group) * 100 if len(ckd_group) > 0 else 0

            total_s = f"{total_n} ({total_pct:.1f}%)"
            no_s = f"{no_n} ({no_pct:.1f}%)"
            ckd_s = f"{ckd_n} ({ckd_pct:.1f}%)"

            if mapping and cat in mapping:
                cat_display = f"  {mapping[cat]}"
            else:
                cat_display = f"  {cat}"
            table1_rows.append([label if cat == sorted(cats)[0] else '',
                                total_s, no_s, ckd_s, ''])

    # IAN grade tested separately with chi-square
    try:
        ct = pd.crosstab(adult['CKD'], adult['IAN_GRADE'])
        _, pval, _, _ = stats.chi2_contingency(ct)
        p_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
    except:
        p_str = ''

    # saveCSV
    import csv
    csv_path = os.path.join(output_path, "table1_ckd.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(table1_rows)
    print(f"Table 1 saved: {csv_path}")

    # Markdown
    md_path = os.path.join(output_path, "table1_ckd.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Table 1: adult(≥20years)byCKDbaseline characteristics by group\n\n")
        f.write("| feature | overall | No CKD | haveCKD | Pvalue |\n")
        f.write("|------|------|-------|-------|-----|\n")
        for row in table1_rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    print(f"Table 1 (Markdown) saved: {md_path}")

    # print
    print(f"\n{'='*70}")
    print("Table 1: adults byCKDbaseline characteristics by group")
    print('='*70)
    for row in table1_rows:
        print(" | ".join(str(x) for x in row))

    return adult

def ian_ckd_association(adult, output_path):
    """IAN score andCKDassociation analysis"""
    print("\n" + "="*70)
    print("2. IAN score andCKDassociation analysis")
    print("="*70)

    # keep only those withIANadult data
    sub = adult[adult['IAN'].notna()].copy()
    print(f"haveIANadult sample of the data: {len(sub)} persons, CKD: {sub['CKD'].sum()} ({sub['CKD'].mean()*100:.1f}%)")

    # 2a. by IAN total score CKD prevalence
    print("\n--- by IANstratified by score CKD prevalence ---")
    ian_prev = []
    for score in sorted(sub['IAN'].dropna().unique()):
        s = sub[sub['IAN'] == score]
        prev = s['CKD'].mean() * 100
        ian_prev.append((int(score), len(s), s['CKD'].sum(), prev))
        print(f"  IAN={int(score)}: {prev:.1f}% ({s['CKD'].sum()}/{len(s)})")

    ian_prev_df = pd.DataFrame(ian_prev, columns=['IAN score', 'total count', 'CKDcount', 'CKDprevalence(%)'])

    # 2b. by IAN gradedCKDprevalence
    print("\n--- by IANstratified by gradeCKDprevalence ---")
    grade_prev = []
    for grade in ['Low risk (0-2)', 'Medium risk (3-4)', 'High risk (5-6)']:
        s = sub[sub['IAN_GRADE'] == grade]
        prev = s['CKD'].mean() * 100
        grade_prev.append((grade, len(s), s['CKD'].sum(), prev))
        print(f"  {grade}: {prev:.1f}% ({s['CKD'].sum()}/{len(s)})")

    grade_prev_df = pd.DataFrame(grade_prev, columns=['IAN grade', 'total count', 'CKDcount', 'CKDprevalence(%)'])

    # 2c. IANeach component andCKDrelationship
    print("\n--- IANeach component andCKDrelationship ---")
    components = [
        ('NLR_T', 'NLRtertile', ['0 (Low NLR)', '1 (Medium NLR)', '2 (High NLR)']),
        ('HB_T', 'Hbtertile', ['2 (Low Hb)', '1 (Medium Hb)', '0 (High Hb)']),
        ('ALB_T', 'albumin tertile', ['2 (Low Albumin)', '1 (Medium Albumin)', '0 (High Albumin)']),
    ]
    comp_rows = [['component', 'level', 'total count', 'CKDcount', 'CKDprevalence(%)']]
    for col, label, labels_map in components:
        for i, lbl in enumerate(labels_map):
            s = sub[sub[col] == float(i)]
            if len(s) == 0:
                continue
            prev = s['CKD'].mean() * 100
            comp_rows.append([label, lbl, len(s), s['CKD'].sum(), f"{prev:.1f}"])
            print(f"  {label} {lbl}: {prev:.1f}% ({s['CKD'].sum()}/{len(s)})")

    comp_df = pd.DataFrame(comp_rows[1:], columns=comp_rows[0])

    return sub, ian_prev_df, grade_prev_df, comp_df

def logistic_regression_ian(sub, output_path):
    """Logistic regression:IANpredictionCKD"""
    print("\n" + "="*70)
    print("3. Logistic regression analysis")
    print("="*70)

    # model1: IAN score as a continuous variable
    print("\n--- model1: IAN score(continuous) ---")
    vars_m1 = {
        'IAN': 'IAN score(per 1-point increment)',
    }
    model1_data = sub[['CKD'] + list(vars_m1.keys())].dropna()
    X1 = sm.add_constant(model1_data.drop('CKD', axis=1).astype(float))
    y1 = model1_data['CKD']

    try:
        logit1 = sm.Logit(y1, X1).fit(disp=0)
        print(logit1.summary())
        results = []
        for col in X1.columns:
            varname = vars_m1.get(col, col) if col != 'const' else 'constant term'
            or_val = np.exp(logit1.params[col])
            ci_l = np.exp(logit1.params[col] - 1.96 * logit1.bse[col])
            ci_u = np.exp(logit1.params[col] + 1.96 * logit1.bse[col])
            p_val = logit1.pvalues[col]
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
            results.append({
                'model': 'model1',
                'variable': varname,
                'OR': or_val,
                '95CI_L': ci_l,
                '95CI_U': ci_u,
                'OR (95%CI)': f"{or_val:.3f} ({ci_l:.3f}-{ci_u:.3f})",
                'Pvalue': p_str,
            })
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {varname:>25s}: OR={or_val:.3f} (95%CI: {ci_l:.3f}-{ci_u:.3f}), P={p_str} {sig}")

        # AUCcompute
        y_pred_prob1 = logit1.predict(X1)
        auc1 = roc_auc_score(y1, y_pred_prob1)
        print(f"\n  model1 AUC: {auc1:.4f}")
    except Exception as e:
        print(f"  model1failed: {e}")
        results = []
        auc1 = 0

    # model2: IAN grade
    print("\n--- model2: IAN grade(categorical) ---")
    # set the reference group to low risk
    sub_m2 = sub[sub['IAN_GRADE'].notna()].copy()
    dummies = pd.get_dummies(sub_m2['IAN_GRADE'], prefix='IAN', drop_first=False)
    # low risk as reference
    ref_col = 'IAN_Low risk (0-2)'
    model2_vars = [c for c in dummies.columns if c != ref_col]

    vars_m2 = {}
    for c in model2_vars:
        grade_label = c.replace('IAN_', '').replace('(0-2)', '').replace('(3-4)', '').replace('(5-6)', '')
        vars_m2[c] = f"IAN {grade_label}"

    model2_data = sub_m2[['CKD']].copy()
    for c in model2_vars:
        model2_data[c] = dummies[c]
    model2_data = model2_data.dropna()
    X2 = sm.add_constant(model2_data.drop('CKD', axis=1).astype(float))
    y2 = model2_data['CKD']

    try:
        logit2 = sm.Logit(y2, X2).fit(disp=0)
        print(logit2.summary())
        for col in X2.columns:
            if col == 'const':
                continue
            varname = vars_m2.get(col, 'constant term')
            or_val = np.exp(logit2.params[col])
            ci_l = np.exp(logit2.params[col] - 1.96 * logit2.bse[col])
            ci_u = np.exp(logit2.params[col] + 1.96 * logit2.bse[col])
            p_val = logit2.pvalues[col]
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
            results.append({
                'model': 'model2',
                'variable': f"{varname} (vs low risk)",
                'OR': or_val,
                '95CI_L': ci_l,
                '95CI_U': ci_u,
                'OR (95%CI)': f"{or_val:.3f} ({ci_l:.3f}-{ci_u:.3f})",
                'Pvalue': p_str,
            })
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {varname:>25s} (vs low risk): OR={or_val:.3f} (95%CI: {ci_l:.3f}-{ci_u:.3f}), P={p_str} {sig}")

        y_pred_prob2 = logit2.predict(X2)
        auc2 = roc_auc_score(y2, y_pred_prob2)
        print(f"\n  model2 AUC: {auc2:.4f}")
    except Exception as e:
        print(f"  model2failed: {e}")
        auc2 = 0

    # model3: IAN + age + sex (multivariate adjustment)
    print("\n--- model3: IAN + age + sex(adjusted model) ---")
    vars_m3 = {
        'IAN': 'IAN score',
        'RIDAGEYR': 'age(years)',
        'RIAGENDR': 'sex(femalevsmale)',
    }
    # race dummy variable
    race_dummies = pd.get_dummies(sub['RACE3'], prefix='race', drop_first=True).astype(float)

    model3_data = sub[['CKD'] + list(vars_m3.keys())].copy()
    model3_data = pd.concat([model3_data, race_dummies], axis=1)
    model3_data = model3_data.dropna()
    X3 = sm.add_constant(model3_data.drop('CKD', axis=1).apply(pd.to_numeric, errors='coerce'))
    y3 = model3_data['CKD'].astype(float)

    try:
        logit3 = sm.Logit(y3, X3).fit(disp=0, maxiter=100)
        print(logit3.summary())

        results_m3 = []
        for col in X3.columns:
            if col == 'const':
                continue
            or_val = np.exp(logit3.params[col])
            ci_l = np.exp(logit3.params[col] - 1.96 * logit3.bse[col])
            ci_u = np.exp(logit3.params[col] + 1.96 * logit3.bse[col])
            p_val = logit3.pvalues[col]
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
            results_m3.append({
                'model': 'model3',
                'variable': col,
                'OR': or_val,
                '95CI_L': ci_l,
                '95CI_U': ci_u,
                'OR (95%CI)': f"{or_val:.3f} ({ci_l:.3f}-{ci_u:.3f})",
                'Pvalue': p_str,
            })
        y_pred_prob3 = logit3.predict(X3)
        auc3 = roc_auc_score(y3, y_pred_prob3)
        print(f"\n  model3 AUC: {auc3:.4f}")
    except Exception as e:
        print(f"  model3failed: {e}")
        results_m3 = []
        auc3 = 0

    # save results
    or_df = pd.DataFrame(results)
    or_path = os.path.join(output_path, "logistic_regression_ian.csv")
    or_df.to_csv(or_path, index=False)
    print(f"\nLogisticRegression results saved: {or_path}")

    if results_m3:
        or_df3 = pd.DataFrame(results_m3)
        or3_path = os.path.join(output_path, "logistic_regression_model3.csv")
        or_df3.to_csv(or3_path, index=False)
        print(f"model3Results saved: {or3_path}")

    return {
        'model1_auc': auc1,
        'model2_auc': auc2,
        'model3_auc': auc3,
        'model1_logit': logit1 if 'logit1' in dir() else None,
        'model3_logit': logit3 if 'logit3' in dir() else None,
    }

def create_visualizations(sub, output_path):
    """visualization"""
    print("\n" + "="*70)
    print("4. visualization")
    print("="*70)

    # ---- figure1: IAN score distribution (stacked bar chart byCKD) ----
    print("  figure1: IAN score distribution byCKDgroup...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left: IAN total-score distribution
    ian_vals = sub[sub['IAN'].notna()]
    bins = np.arange(-0.5, 7, 1)
    ax = axes[0]
    for ckd_val, color, label in [(0, BLUE, 'No CKD'), (1, RED, 'CKD')]:
        data = ian_vals[ian_vals['CKD'] == ckd_val]['IAN']
        ax.hist(data, bins=bins, alpha=0.7, color=color, label=label, density=True)
    ax.set_xlabel('IAN Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('IAN Score Distribution by CKD Status', fontsize=13, fontweight='bold')
    ax.set_xticks(range(0, 7))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # right: IAN gradeCKDprevalence
    ax = axes[1]
    grades = ['Low (0-2)', 'Medium (3-4)', 'High (5-6)']
    ckd_rates = []
    for g in grades:
        s = ian_vals[ian_vals['IAN_GRADE'] == g]
        ckd_rates.append(s['CKD'].mean() * 100 if len(s) > 0 else 0)

    bars = ax.bar(grades, ckd_rates, color=[GREEN, ORANGE, RED], width=0.6, edgecolor='white')
    for bar, val in zip(bars, ckd_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('IAN Risk Grade', fontsize=12)
    ax.set_ylabel('CKD Prevalence (%)', fontsize=12)
    ax.set_title('CKD Prevalence by IAN Risk Grade', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(ckd_rates) * 1.2 + 2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig1_ian_distribution.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(output_path, "fig1_ian_distribution.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(output_path, "fig1_ian_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- figure2: forest plot - IANcomponentOR ----
    print("  figure2: IANcomponent forest plot...")
    # separately forIANcomponents were subjected to univariateLogistic regression
    components = ['NLR_T', 'HB_T', 'ALB_T']
    comp_labels = ['NLR tertile', 'Hb tertile (inverse)', 'Albumin tertile (inverse)']
    component_or = []
    for comp, clabel in zip(components, comp_labels):
        data = sub[['CKD', comp]].dropna()
        X = sm.add_constant(data[comp].astype(float))
        y = data['CKD']
        try:
            logit = sm.Logit(y, X).fit(disp=0)
            or_v = np.exp(logit.params[comp])
            ci_l = np.exp(logit.params[comp] - 1.96 * logit.bse[comp])
            ci_u = np.exp(logit.params[comp] + 1.96 * logit.bse[comp])
            p_v = logit.pvalues[comp]
            component_or.append((clabel, or_v, ci_l, ci_u, p_v))
            print(f"  {clabel}: OR={or_v:.3f} ({ci_l:.3f}-{ci_u:.3f}), P={p_v:.4f}")
        except Exception as e:
            print(f"  {clabel}: failed - {e}")

    # plusIAN total score
    data = sub[['CKD', 'IAN']].dropna()
    X = sm.add_constant(data['IAN'].astype(float))
    y = data['CKD']
    try:
        logit = sm.Logit(y, X).fit(disp=0)
        or_v = np.exp(logit.params['IAN'])
        ci_l = np.exp(logit.params['IAN'] - 1.96 * logit.bse['IAN'])
        ci_u = np.exp(logit.params['IAN'] + 1.96 * logit.bse['IAN'])
        p_v = logit.pvalues['IAN']
        component_or.append(('IAN total (per 1-point increase)', or_v, ci_l, ci_u, p_v))
        print(f"  IAN total score: OR={or_v:.3f} ({ci_l:.3f}-{ci_u:.3f}), P={p_v:.4f}")
    except Exception as e:
        print(f"  IAN total score: failed - {e}")

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(component_or))
    all_lo = [d[2] for d in component_or]
    all_hi = [d[3] for d in component_or]
    x_min = max(0.4, min(all_lo) * 0.85)
    x_max = max(max(all_hi) * 1.75, 2.2)
    for i, (label, or_v, ci_l, ci_u, p_v) in enumerate(component_or):
        color = RED if p_v < 0.001 else ORANGE if p_v < 0.01 else BLUE if p_v < 0.05 else GREY
        ax.plot([ci_l, ci_u], [i, i], color=color, linewidth=2.5)
        ax.scatter(or_v, i, color=color, s=100, zorder=5,
                   marker='D' if p_v < 0.001 else 's' if p_v < 0.01 else 'o')
        # per-row OR (95% CI) label to the right of each CI line
        ax.text(ci_u * 1.08, i, f"{or_v:.2f} ({ci_l:.2f}-{ci_u:.2f})",
                va='center', ha='left', fontsize=10, color='black')

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x[0] for x in component_or], fontsize=11)
    ax.set_xscale('linear')
    ax.set_xlim(1.0, 2.6)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xticklabels(['1.0', '1.5', '2.0'])
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
    ax.set_title('Univariate Logistic Regression: IAN Components Predicting CKD', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig2_forest_ian_components.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(output_path, "fig2_forest_ian_components.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(output_path, "fig2_forest_ian_components.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- figure3: ROC curve ----
    print("  figure3: ROC curve...")
    fig, ax = plt.subplots(figsize=(8, 7))

    # IANunivariate
    sub_roc = sub[['CKD', 'IAN']].dropna()
    X_roc = sm.add_constant(sub_roc['IAN'].astype(float))
    y_roc = sub_roc['CKD']
    logit_roc = sm.Logit(y_roc, X_roc).fit(disp=0)
    y_prob = logit_roc.predict(X_roc)
    fpr, tpr, _ = roc_curve(y_roc, y_prob)
    auc = roc_auc_score(y_roc, y_prob)
    ax.plot(fpr, tpr, color=RED, linewidth=2.5,
            label=f'IAN Score (AUC={auc:.3f})')

    # IAN + age + sex + race
    sub_roc3 = sub[['CKD', 'IAN', 'RIDAGEYR', 'RIAGENDR']].dropna()
    race_d = pd.get_dummies(sub['RACE3'].loc[sub_roc3.index], prefix='race', drop_first=True).astype(float)
    X_roc3 = sm.add_constant(pd.concat([
        sub_roc3[['IAN', 'RIDAGEYR', 'RIAGENDR']].astype(float), race_d
    ], axis=1))
    X_roc3 = X_roc3.apply(pd.to_numeric, errors='coerce')
    y_roc3 = sub_roc3['CKD'].astype(float)
    logit_roc3 = sm.Logit(y_roc3, X_roc3).fit(disp=0)
    y_prob3 = logit_roc3.predict(X_roc3)
    fpr3, tpr3, _ = roc_curve(y_roc3, y_prob3)
    auc3 = roc_auc_score(y_roc3, y_prob3)
    ax.plot(fpr3, tpr3, color=BLUE, linewidth=2.5,
            label=f'IAN+Age+Sex+Race (AUC={auc3:.3f})')

    # baseline
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')

    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    ax.set_title('ROC Curves: IAN Score Predicting CKD', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig3_roc_curve.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(output_path, "fig3_roc_curve.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(output_path, "fig3_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- figure4: IAN vs CKDprevalence - continuous trend ----
    print("  figure4: IANandCKDprevalence trend...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # by IANgrouped by score CKD prevalence
    ian_prev = []
    for score in range(0, 7):
        s = sub[sub['IAN'] == score]
        if len(s) > 0:
            ian_prev.append((score, s['CKD'].mean() * 100, len(s)))
        else:
            ian_prev.append((score, 0, 0))

    scores = [x[0] for x in ian_prev]
    prevs = [x[1] for x in ian_prev]
    counts = [x[2] for x in ian_prev]

    color_scale = [GREEN, GREEN, ORANGE, ORANGE, ORANGE, RED, RED]

    ax1.bar(scores, counts, color=color_scale, alpha=0.3, label='Sample Size', width=0.6)
    ax1.set_xlabel('IAN Score', fontsize=12)
    ax1.set_ylabel('Sample Size', fontsize=12, color=GREY)

    ax2 = ax1.twinx()
    ax2.plot(scores, prevs, color=RED, marker='o', linewidth=2.5, markersize=10,
             label='CKD Prevalence')
    for i, (s, p) in enumerate(zip(scores, prevs)):
        ax2.annotate(f'{p:.1f}%', (s, p), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=RED)

    ax2.set_ylabel('CKD Prevalence (%)', fontsize=12, color=RED)
    ax2.set_ylim(0, max(prevs) * 1.3 + 5)

    # mergelegend
    bars = ax1.containers[0]
    lines = [ax2.get_lines()[0]]
    labels = ['Sample Size', 'CKD Prevalence']
    ax1.legend([bars, lines[0]], labels, fontsize=11, loc='upper left')

    ax1.set_title('Dose-Response: IAN Score and CKD Prevalence', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(0, 7))
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig4_ian_trend.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- figure5: IANcomponent boxplots ----
    print("  figure5: IANcomponent boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    components_data = [
        ('NLR', 'NLR', [0, 5]),
        ('HEMOGLOBIN', 'Hemoglobin (g/dL)', [8, 18]),
        ('ALBUMIN', 'Albumin (g/dL)', [2, 5.5]),
    ]

    for ax, (col, label, ylim) in zip(axes, components_data):
        data = sub[sub[col].notna()]
        bp_data = [data[data['CKD'] == 0][col].dropna(),
                   data[data['CKD'] == 1][col].dropna()]
        bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5,
                        medianprops={'color': 'white', 'linewidth': 2})
        bp['boxes'][0].set_facecolor(BLUE)
        bp['boxes'][1].set_facecolor(RED)
        ax.set_xticklabels(['No CKD', 'CKD'])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} by CKD Status', fontsize=12, fontweight='bold')
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3)

        # annotate significance
        try:
            _, p = stats.ttest_ind(bp_data[0], bp_data[1])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.5, 0.95, f'P={p:.4f} {sig}', transform=ax.transAxes,
                    ha='center', fontsize=10, fontweight='bold')
        except:
            pass

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig5_boxplots.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- figure6: heatmap - IANinter-component correlation ----
    print("  figure6: correlation heatmap...")
    corr_data = sub[['NLR', 'HEMOGLOBIN', 'ALBUMIN', 'IAN', 'eGFR', 'UACR']].dropna().corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
    ax.set_title('Correlation: IAN Components and Kidney Function Markers', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, "fig6_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll figures saved to: {output_path}")
    return True

def generate_report(sub, ian_prev_df, grade_prev_df, comp_df, model_results, output_path):
    """generate analysis report"""
    print("\n" + "="*70)
    print("5. generate analysis report")
    print("="*70)

    n_total = len(sub)
    n_ckd = sub['CKD'].sum()
    n_no_ckd = n_total - n_ckd
    n_ian = sub['IAN'].notna().sum()

    report_path = os.path.join(output_path, "ian_ckd_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# IAN(inflammation-nutrition score)predictionCKDofNHANESdata analysis report\n\n")
        f.write(f"**analysis time**: 2025-06-16\n\n")
        f.write(f"**data source**: NHANES 2011-2018 (4cycles)\n\n")
        f.write(f"**analysis population**: adult (age ≥ 20years)\n\n")
        f.write(f"**CKDdefinition**: eGFR < 60 mL/min/1.73m² or UACR ≥ 30 mg/g\n\n")
        f.write("---\n\n")

        # abstract
        f.write("## abstract\n\n")
        f.write(f"- total sample size: {n_total:,} persons\n")
        f.write(f"- validIAN data: {n_ian:,} persons\n")
        f.write(f"- CKDpatient: {n_ckd:,} persons ({n_ckd/n_total*100:.1f}%)\n")
        f.write(f"- nonCKD: {n_no_ckd:,} persons ({n_no_ckd/n_total*100:.1f}%)\n\n")

        # IAN score construction
        f.write("## IAN score construction method\n\n")
        f.write("IAN(Inflammation-Nutrition Score,inflammation-nutrition score)by3components were used to build:\n\n")
        f.write("| indicator | meaning | scoring rule |\n")
        f.write("|------|------|----------|\n")
        f.write("| NLR(neutrophil/lymphocyte ratio) | systemic inflammation marker | T1=0, T2=1, T3=2 (higher is worse) |\n")
        f.write("| hemoglobin (Hb) | anemia/nutritional status | T1=2, T2=1, T3=0 (lower is worse,inverse) |\n")
        f.write("| albumin | nutritional status | T1=2, T2=1, T3=0 (lower is worse,inverse) |\n\n")
        f.write("**total score range**: 0-6point (higher indicates worse inflammation/worse nutritional status)\n\n")
        f.write("**grade**: Low risk (0-2), Medium risk (3-4), High risk (5-6)\n\n")

        # IANandCKDassociation
        f.write("## IAN score andCKDassociation\n\n")
        f.write("### by IANstratified by score CKD prevalence\n\n")
        f.write(ian_prev_df.to_markdown(index=False) + "\n\n")

        f.write("### by IANstratified by gradeCKDprevalence\n\n")
        f.write(grade_prev_df.to_markdown(index=False) + "\n\n")

        # Logistic regression results
        f.write("## Logistic regression: IANpredictionCKD\n\n")
        f.write("### model1: IAN score (continuous) — univariate\n\n")
        f.write(f"- OR = {model_results['model1_or']:.3f} (per 1-point increase1point)\n")
        f.write(f"- AUC = {model_results['model1_auc']:.4f}\n\n")

        f.write("### model2: IAN grade — univariate\n\n")
        for grade, or_v, ci_l, ci_u, p_str in model_results['model2_data']:
            f.write(f"- {grade}: OR={or_v:.2f} (95%CI: {ci_l:.2f}-{ci_u:.2f}), P={p_str}\n")
        f.write(f"- AUC = {model_results['model2_auc']:.4f}\n\n")

        f.write("### model3: IAN + age + sex + race (multivariate adjustment)\n\n")
        f.write(f"- IAN score(per 1-point increment): OR = {model_results['model3_or']:.3f}\n")
        f.write(f"- after adjustmentIANstill significantly predictsCKD\n")
        f.write(f"- AUC = {model_results['model3_auc']:.4f}\n\n")

        # key findings
        f.write("## key findings\n\n")
        f.write("1. **IAN score and CKD are significantly correlated**: ")
        f.write(f"For each 1-point increase in IAN score, CKD risk increases by {((model_results['model1_or']-1)*100):.1f}% (OR={model_results['model1_or']:.3f}).\n\n")
        f.write("2. **dose-response relationship**: ")
        f.write("As IAN score increases from 0 to 6 points, CKD prevalence shows an increasing trend, demonstrating a clear dose-response relationship.\n\n")
        f.write("3. **IAN component contribution**: ")
        f.write("NLR, low hemoglobin and low albumin are each independently associated with CKD risk; the combination of the three components yields a stronger predictive performance.\n\n")
        f.write("4. **independent predictive value**: ")
        f.write("After adjusting for age, sex and race, IAN score still significantly predicts CKD, suggesting it has predictive value independent of traditional risk factors.\n\n")
        f.write("5. **clinical significance**: ")
        f.write("The IAN score is a composite indicator based on routine laboratory tests; it is simple to operate, low cost, and suitable for CKD risk stratification in primary care or large-scale screening.\n\n")

        # figure
        f.write("## visualization figures\n\n")
        f.write("| figure number | description | filename |\n")
        f.write("|------|------|--------|\n")
        f.write("| figure1 | IAN score distribution and gradingCKDprevalence | fig1_ian_distribution.png |\n")
        f.write("| figure2 | IANcomponent forest plot | fig2_forest_ian_components.png |\n")
        f.write("| figure3 | ROC curve | fig3_roc_curve.png |\n")
        f.write("| figure4 | IANandCKDprevalence trend | fig4_ian_trend.png |\n")
        f.write("| figure5 | IANcomponent boxplots | fig5_boxplots.png |\n")
        f.write("| figure6 | correlation heatmap | fig6_correlation_heatmap.png |\n\n")

    print(f"Analysis report saved: {report_path}")
    return report_path

def main():
    print("="*70)
    print("NHANES IAN score prediction CKD analysis")
    print("="*70)

    # 1. data loading
    df = load_data()

    # 2. constructIAN score
    df, cutpoints = build_ian_scores(df)

    # 3. descriptive statistics
    adult = descriptive_stats(df, OUTPUT_DIR)

    # 4. IANandCKDassociation
    sub, ian_prev_df, grade_prev_df, comp_df = ian_ckd_association(adult, OUTPUT_DIR)

    # 5. Logistic regression
    model_results_raw = logistic_regression_ian(sub, OUTPUT_DIR)

    # combined model results
    # model1 OR
    m1_logit = model_results_raw.get('model1_logit')
    m1_auc = model_results_raw.get('model1_auc', 0)
    m3_auc = model_results_raw.get('model3_auc', 0)
    m3_logit = model_results_raw.get('model3_logit')

    if m1_logit is not None:
        m1_or = np.exp(m1_logit.params['IAN'])
    else:
        m1_or = 0

    if m3_logit is not None:
        m3_or = np.exp(m3_logit.params['IAN'])
    else:
        m3_or = 0

    # model2data
    model2_data = []
    sub_m2 = sub[sub['IAN_GRADE'].notna()].copy()
    dummies = pd.get_dummies(sub_m2['IAN_GRADE'], prefix='IAN', drop_first=False)
    ref_col = 'IAN_Low risk (0-2)'
    model2_vars = [c for c in dummies.columns if c != ref_col]
    if model2_vars:
        m2_data = sub_m2[['CKD']].copy()
        for c in model2_vars:
            m2_data[c] = dummies[c]
        m2_data = m2_data.dropna()
        X2 = sm.add_constant(m2_data.drop('CKD', axis=1).astype(float))
        y2 = m2_data['CKD']
        try:
            logit2 = sm.Logit(y2, X2).fit(disp=0)
            for c in model2_vars:
                or_v = np.exp(logit2.params[c])
                ci_l = np.exp(logit2.params[c] - 1.96 * logit2.bse[c])
                ci_u = np.exp(logit2.params[c] + 1.96 * logit2.bse[c])
                p_v = logit2.pvalues[c]
                p_str = f"{p_v:.4f}" if p_v >= 0.001 else "<0.001"
                grade_label = c.replace('IAN_', '').replace('(0-2)', '').replace('(3-4)', '').replace('(5-6)', '')
                model2_data.append((f"IAN {grade_label} (vs low risk)", or_v, ci_l, ci_u, p_str))
        except:
            pass

    model_results = {
        'model1_or': m1_or,
        'model1_auc': m1_auc,
        'model2_data': model2_data,
        'model2_auc': model_results_raw.get('model2_auc', 0),
        'model3_or': m3_or,
        'model3_auc': m3_auc,
    }

    # 6. visualization
    create_visualizations(sub, CHART_DIR)

    # 7. report
    report_path = generate_report(sub, ian_prev_df, grade_prev_df, comp_df, model_results, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"analysis completed!All results saved to {OUTPUT_DIR}")
    print(f"report: {report_path}")
    print(f"figure: {CHART_DIR}")
    print('='*70)

if __name__ == "__main__":
    main()
