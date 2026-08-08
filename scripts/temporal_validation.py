#!/usr/bin/env python3
"""
IAN(inflammation-nutrition score)predictionCKD — temporal split validation analysis
- training set: G(2011-2012) + H(2013-2014) + I(2015-2016)
- validation set: J(2017-2018) — temporal external validation

analysis content:
1. IAN score construction and baseline characteristics
2. training set: Logistic regression + ROC
3. validation set: external validationROC
4. comparison with other indices (PNI, ALI, CONUT)
5. two-threshold strategy
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import statsmodels.api as sm

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

from nature_config import *  # BLUE

rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 14})

# ============ 1. Data loading and preprocessing ============
print("="*70)
print("IAN score predictionCKD - temporal split validation analysis")
print("="*70)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv"), low_memory=False)
print(f"\ntotal sample: {len(df):,} persons")
print(f"cycle distribution: {df['CYCLE'].value_counts().sort_index().to_dict()}")

# adult(>=20years)
adult = df[df['RIDAGEYR'] >= 20].copy()
print(f"adult(>=20years): {len(adult):,} persons")

# extractIANrequired variables(includeSEQNused for merging,andLYMPHOCYTEforPNI)
ian_cols = ['NLR', 'HEMOGLOBIN', 'ALBUMIN', 'CYCLE', 'CKD', 'SEQN',
            'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'BMXBMI', 'eGFR', 'UACR',
            'LYMPHOCYTE', 'NEUTROPHIL', 'DIABETES', 'BPQ020']
# keep only columns that actually exist
ian_cols = [c for c in ian_cols if c in adult.columns]
valid_ian = adult[ian_cols].dropna(subset=['NLR', 'HEMOGLOBIN', 'ALBUMIN']).copy()
print(f"with complete IAN data(adult): {len(valid_ian):,} persons")
print(f"columns included: {list(valid_ian.columns[:18])}...")

# sex label
valid_ian['GENDER_LABEL'] = valid_ian['RIAGENDR'].map({1: 'Male', 2: 'Female'})
# race merging
valid_ian['RACE3'] = valid_ian['RIDRETH1'].map({1:'Hispanic',2:'Hispanic',3:'Non-Hisp White',4:'Non-Hisp Black',5:'Other'})

# ============ 2. data splitting ============
# training set: G(2011-2012) + H(2013-2014) + I(2015-2016)
# validation set: J(2017-2018)
train = valid_ian[valid_ian['CYCLE'].isin(['G','H','I'])].copy()
valid = valid_ian[valid_ian['CYCLE'] == 'J'].copy()

print(f"\n{'='*70}")
print("temporal split validation design")
print(f"  training set (2011-2016): {len(train):,} persons, CKD: {train['CKD'].sum()} ({train['CKD'].mean()*100:.1f}%)")
print(f"  validation set (2017-2018): {len(valid):,} persons, CKD: {valid['CKD'].sum()} ({valid['CKD'].mean()*100:.1f}%)")

# ============ 3. constructed on the training setIANtertile ============
print(f"\n{'='*70}")
print("IAN score construction (Tertile cutoffs based on the training set)")
print('='*70)

# training-set tertiles
_, nlr_bins = pd.qcut(train['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
_, hb_bins = pd.qcut(train['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
_, alb_bins = pd.qcut(train['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')

print(f"\ntraining-set tertile cutoffs:")
print(f"  NLR: {np.round(nlr_bins, 2)}")
print(f"  Hb:  {np.round(hb_bins, 2)}")
print(f"  Alb: {np.round(alb_bins, 2)}")

def assign_ian(data, nlr_b, hb_b, alb_b):
    """assign to dataIAN score"""
    d = data.copy()
    d['NLR_T'] = pd.cut(d['NLR'], bins=nlr_b, labels=[0,1,2], include_lowest=True).astype(float)
    d['HB_T'] = pd.cut(d['HEMOGLOBIN'], bins=hb_b, labels=[2,1,0], include_lowest=True).astype(float)
    d['ALB_T'] = pd.cut(d['ALBUMIN'], bins=alb_b, labels=[2,1,0], include_lowest=True).astype(float)
    d['IAN'] = d['NLR_T'] + d['HB_T'] + d['ALB_T']
    d['IAN_GRADE'] = pd.cut(d['IAN'], bins=[-1,2,4,6], labels=['Low(0-2)','Medium(3-4)','High(5-6)'])
    return d

train = assign_ian(train, nlr_bins, hb_bins, alb_bins)
valid = assign_ian(valid, nlr_bins, hb_bins, alb_bins)

print(f"\ntraining setIANdistribution:")
for s in sorted(train['IAN'].dropna().unique()):
    ckd_r = train[train['IAN']==s]['CKD'].mean()*100
    print(f"  IAN={int(s)}: {len(train[train['IAN']==s])}persons, CKD={ckd_r:.1f}%")

print(f"\nvalidation setIANdistribution:")
for s in sorted(valid['IAN'].dropna().unique()):
    ckd_r = valid[valid['IAN']==s]['CKD'].mean()*100
    print(f"  IAN={int(s)}: {len(valid[valid['IAN']==s])}persons, CKD={ckd_r:.1f}%")

# ============ 4. construct comparison indices ============
print(f"\n{'='*70}")
print("construct comparison composite indices (PNI, ALI, CONUT)")
print('='*70)

# PNI = albumin(g/L) + 5 × lymphocyte(10^9/L)
def calc_pni(alb_gdl, lymph):
    """PNI = albumin(g/L) + 5×lymphocyte(10^9/L)"""
    alb_gl = alb_gdl * 10  # g/dL -> g/L
    return alb_gl + 5 * lymph

# ALI = BMI × albumin(g/dL) / NLR
def calc_ali(bmi, alb, nlr):
    return bmi * alb / nlr

# CONUT (simplified version, use albumin only+lymphocyte, for lack of cholesterol)
def calc_conut_simple(alb_gdl, lymph):
    """simplified versionCONUTscore"""
    score = 0
    # albumin
    if alb_gdl >= 3.5: score += 0
    elif alb_gdl >= 3.0: score += 2
    elif alb_gdl >= 2.5: score += 4
    else: score += 6
    # lymphocyte (1000 cells/uL = 10^3/uL)
    lymph_per_ul = lymph * 1000  # convert to/uL
    if lymph_per_ul >= 1600: score += 0
    elif lymph_per_ul >= 1200: score += 1
    elif lymph_per_ul >= 800: score += 2
    else: score += 3
    return score

for dset, name in [(train, 'Train'), (valid, 'Valid')]:
    lymph_col = None
    for c in ['LYMPHOCYTE', 'LBDLYMNO']:
        if c in dset.columns:
            lymph_col = c
            break
    has_bmi = 'BMXBMI' in dset.columns
    
    if lymph_col and dset['ALBUMIN'].notna().any():
        dset['PNI'] = calc_pni(dset['ALBUMIN'], dset[lymph_col])
        # CONUT simplified
        dset['CONUT_simple'] = dset.apply(
            lambda r: calc_conut_simple(r['ALBUMIN'], r[lymph_col])
            if pd.notna(r['ALBUMIN']) and pd.notna(r[lymph_col]) else np.nan, axis=1
        )
    if has_bmi and dset['NLR'].notna().any():
        dset['ALI'] = calc_ali(dset['BMXBMI'], dset['ALBUMIN'], dset['NLR'])
    
    pni_mean = dset['PNI'].mean() if 'PNI' in dset.columns and dset['PNI'].notna().any() else np.nan
    ali_mean = dset['ALI'].mean() if 'ALI' in dset.columns and dset['ALI'].notna().any() else np.nan
    conut_mean = dset['CONUT_simple'].mean() if 'CONUT_simple' in dset.columns and dset['CONUT_simple'].notna().any() else np.nan
    print(f"\n  {name}: PNI mean={pni_mean:.1f}, ALI mean={ali_mean:.1f}, CONUT_s mean={conut_mean:.1f}")

# ============ 5. baseline characteristics table (Table 1) ============
print(f"\n{'='*70}")
print("Table 1: baseline characteristics")
print('='*70)

def make_table1(data, label):
    """generateTable 1"""
    ckd = data[data['CKD']==1]
    noc = data[data['CKD']==0]
    print(f"\n--- {label} ---")
    print(f"  N: {len(data)} (CKD: {len(ckd)} [{len(ckd)/len(data)*100:.1f}%])")
    
    vars_check = [
        ('RIDAGEYR', 'Age (years)'),
        ('BMXBMI', 'BMI (kg/m²)'),
        ('NLR', 'NLR'),
        ('HEMOGLOBIN', 'Hb (g/dL)'),
        ('ALBUMIN', 'Albumin (g/dL)'),
        ('IAN', 'IAN Score'),
        ('eGFR', 'eGFR (mL/min/1.73m²)'),
        ('UACR', 'UACR (mg/g)'),
        ('PNI', 'PNI'),
        ('ALI', 'ALI'),
    ]
    
    rows = []
    for col, label_v in vars_check:
        if col not in data.columns or data[col].isna().all():
            continue
        tot_v = data[col].dropna()
        noc_v = noc[col].dropna()
        ckd_v = ckd[col].dropna()
        
        t_m = tot_v.mean(); t_s = tot_v.std()
        n_m = noc_v.mean(); n_s = noc_v.std()
        c_m = ckd_v.mean(); c_s = ckd_v.std()
        
        if col == 'UACR':
            fmt = lambda v: f"{v.median():.1f} ({v.quantile(0.25):.1f}-{v.quantile(0.75):.1f})"
            t_s = fmt(tot_v); n_s = fmt(noc_v); c_s = fmt(ckd_v)
            try: _, p = stats.mannwhitneyu(noc_v, ckd_v)
            except: p = 1.0
        else:
            t_s = f"{t_m:.1f}±{t_s:.1f}"
            n_s = f"{n_m:.1f}±{n_s:.1f}"
            c_s = f"{c_m:.1f}±{c_s:.1f}"
            try: _, p = stats.ttest_ind(noc_v, ckd_v)
            except: p = 1.0
        
        p_str = f"{p:.4f}" if p >= 0.001 else "<0.001"
        rows.append([f"  {label_v}", t_s, n_s, c_s, p_str])
        print(f"  {label_v:>20s}: {t_s} | NoCKD: {n_s} | CKD: {c_s} | P={p_str}")
    
    # categorical variable
    cat_vars = [('GENDER_LABEL','Gender'), ('RACE3','Race'),
                ('IAN_GRADE','IAN Grade'), ('DIABETES','Diabetes')]
                
    for col_name, cat_label in cat_vars:
        if col_name not in data.columns:
            continue
        vals = data[col_name].dropna()
        if col_name == 'DIABETES':
            vals = vals.map({1:'Yes',0:'No'})
            ckd_vals = ckd[col_name].map({1:'Yes',0:'No'})
            noc_vals = noc[col_name].map({1:'Yes',0:'No'})
        else:
            ckd_vals = ckd[col_name]; noc_vals = noc[col_name]
        
        uniq = sorted(vals.unique())
        for i, v in enumerate(uniq):
            n_t = (vals==v).sum(); n_n = (noc_vals==v).sum(); n_c = (ckd_vals==v).sum()
            p_t = n_t/len(data)*100; p_n = n_n/len(noc)*100; p_c = n_c/len(ckd)*100
            l = cat_label if i==0 else ''
            rows.append([l, f"{n_t}({p_t:.1f}%)", f"{n_n}({p_n:.1f}%)", f"{n_c}({p_c:.1f}%)", ''])
            print(f"  {v:>15s}: {n_t}({p_t:.1f}%) | {n_n}({p_n:.1f}%) | {n_c}({p_c:.1f}%)")

make_table1(train, "Training Set (2011-2016)")
make_table1(valid, "Validation Set (2017-2018)")

# ============ 6. Logistic regression (training set) ============
print(f"\n{'='*70}")
print("Logistic regression - training set")
print('='*70)

def run_logit(data, formula_name, x_vars, y='CKD'):
    d = data[[y] + x_vars].dropna()
    X = sm.add_constant(d[x_vars].astype(float))
    yv = d[y].astype(float)
    logit = sm.Logit(yv, X).fit(disp=0)
    y_pred = logit.predict(X)
    auc = roc_auc_score(yv, y_pred)
    
    results = []
    for col in X.columns:
        if col == 'const': continue
        or_v = np.exp(logit.params[col])
        ci_l = np.exp(logit.params[col] - 1.96*logit.bse[col])
        ci_u = np.exp(logit.params[col] + 1.96*logit.bse[col])
        p_v = logit.pvalues[col]
        results.append({'Variable': col, 'OR': or_v, '95CI_L': ci_l, '95CI_U': ci_u, 'P': p_v})
        sig = '***' if p_v<0.001 else '**' if p_v<0.01 else '*' if p_v<0.05 else ''
        print(f"  {col:>20s}: OR={or_v:.4f} (95%CI: {ci_l:.4f}-{ci_u:.4f}), P={p_v:.4f} {sig}")
    print(f"  {'Model AUC':>20s}: {auc:.4f}")
    
    return {'logit': logit, 'auc': auc, 'y_pred': y_pred, 'results': results}

# model1: IANunivariate
print("\n--- model1: IAN (univariate) ---")
m1 = run_logit(train, 'IAN', ['IAN'])

# model2: IAN + age + sex
print("\n--- model2: IAN + Age + Sex ---")
m2 = run_logit(train, 'IAN+Age+Sex', ['IAN', 'RIDAGEYR', 'RIAGENDR'])

# model3: IAN + age + sex + hypertension + diabetes + BMI
print("\n--- model3: IAN + Age + Sex + HTN + DM + BMI ---")
m3_vars = ['IAN', 'RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'DIABETES']
train_tmp = train.copy()
# hypertension (if anyBPQ020)
if 'BPQ020' in train_tmp.columns:
    train_tmp['HYPERTENSION'] = (train_tmp['BPQ020'] == 1).astype(int)
    m3_vars.append('HYPERTENSION')
else:
    print("  [WARN] BPQ020 not available, skipping HTN adjustment")
m3 = run_logit(train_tmp, 'IAN+Age+Sex+HTN+DM+BMI', m3_vars)

# model4: compare with other indices
print("\n--- model4: univariate comparison of each index ---")
index_scores = [
    ('IAN', 'IAN'),
    ('PNI', 'PNI'),
    ('ALI', 'ALI'),
    ('NLR', 'NLR'),
    ('HEMOGLOBIN', 'Hemoglobin'),
    ('ALBUMIN', 'Albumin'),
]
index_comparison = []
for col, label in index_scores:
    if col not in train.columns or train[col].isna().all():
        continue
    d = train[['CKD', col]].dropna()
    X = sm.add_constant(d[col].astype(float))
    yv = d['CKD'].astype(float)
    logit = sm.Logit(yv, X).fit(disp=0)
    yp = logit.predict(X)
    auc = roc_auc_score(yv, yp)
    or_v = np.exp(logit.params[col])
    ci_l = np.exp(logit.params[col]-1.96*logit.bse[col])
    ci_u = np.exp(logit.params[col]+1.96*logit.bse[col])
    p_v = logit.pvalues[col]
    index_comparison.append({'Index': label, 'OR': or_v, '95CI_L': ci_l, '95CI_U': ci_u, 'P': p_v, 'AUC': auc})
    print(f"  {label:>35s}: OR={or_v:.3f}, AUC={auc:.4f}")

# ============ 7. validation-set performance ============
print(f"\n{'='*70}")
print("validation-set external validation")
print('='*70)

def validate_model(data, logit_model, x_vars, label=''):
    d = data[['CKD'] + x_vars].dropna()
    X = sm.add_constant(d[x_vars].astype(float))
    # ensure column names match the model
    X_cols = list(logit_model.params.index)
    X = X.reindex(columns=X_cols, fill_value=0)
    y_true = data.loc[X.index, 'CKD'].astype(float)
    y_pred = logit_model.predict(X)
    
    try:
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        # Youden index
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_thr = thresholds[best_idx]
        best_sens = tpr[best_idx]
        best_spec = 1 - fpr[best_idx]
        
        print(f"  {label} — AUC={auc:.4f}")
        print(f"    optimal threshold: {best_thr:.4f}, Sens={best_sens:.3f}, Spec={best_spec:.3f}")
        
        # predicted label
        y_pred_binary = (y_pred >= best_thr).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        ppv = tp/(tp+fp) if (tp+fp)>0 else 0
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        print(f"    PPV={ppv:.3f}, NPV={npv:.3f}")
        
        return {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'threshold': thresholds,
                'best_thr': best_thr, 'sens': best_sens, 'spec': best_spec}
    except Exception as e:
        print(f"  {label} — FAILED: {e}")
        return {'auc': 0, 'fpr': [], 'tpr': []}

# validation model1
v1_train = validate_model(train, m1['logit'], ['IAN'], 'IAN (Train)')
v1_valid = validate_model(valid, m1['logit'], ['IAN'], 'IAN (Valid)')

# validation model3
v3_train = validate_model(train_tmp, m3['logit'], m3_vars, 'Full Model (Train)')
valid_tmp = valid.copy()
if 'BPQ020' in valid_tmp.columns:
    valid_tmp['HYPERTENSION'] = (valid_tmp['BPQ020'] == 1).astype(int)
# For the validation, we need matching columns
v3_valid_x = [v for v in m3_vars if v in valid_tmp.columns]
if len(v3_valid_x) == len(m3_vars):
    v3_valid = validate_model(valid_tmp, m3['logit'], m3_vars, 'Full Model (Valid)')
else:
    print("  Full model validation skipped: missing columns")
    v3_valid = {'auc': 0, 'fpr': [], 'tpr': []}

# ============ 8. visualization ============
print(f"\n{'='*70}")
print("Generate visualization figures")
print('='*70)

# figure1: ROC curve (training setvsvalidation set)
print("  figure1: ROC curve...")
fig, ax = plt.subplots(figsize=(9, 8))

# IANtraining set
ax.plot(v1_train['fpr'], v1_train['tpr'], color=RED, linewidth=2.5,
        label=f"IAN Train (AUC={v1_train['auc']:.3f})")
# IANvalidation set
ax.plot(v1_valid['fpr'], v1_valid['tpr'], color=RED, linewidth=2.5, linestyle='--',
        label=f"IAN Valid (AUC={v1_valid['auc']:.3f})")

# comparison indices on the training set
colors_comp = [BLUE, GREEN, PURPLE, ORANGE]
for i, idx in enumerate(index_comparison):
    if idx['AUC'] > 0:
        col_name = idx['Index'].split(' ')[0].split('(')[0].strip()
        # obtainROCdata
        if col_name in ['PNI', 'ALI', 'NLR'] or col_name in ['Hemoglobin', 'Albumin']:
            vn = col_name if col_name in train.columns else None
            if vn is None:
                # try other column names
                for tc in ['PNI','ALI','NLR','HEMOGLOBIN','ALBUMIN']:
                    if tc in train.columns:
                        vn = tc
                        break
            if vn:
                d = train[['CKD', vn]].dropna()
                X = sm.add_constant(d[vn].astype(float))
                yv = d['CKD'].astype(float)
                lr = sm.Logit(yv, X).fit(disp=0)
                yp = lr.predict(X)
                fpr, tpr, _ = roc_curve(yv, yp)
                auc_v = roc_auc_score(yv, yp)
                ax.plot(fpr, tpr, color=colors_comp[i % len(colors_comp)], linewidth=1.8, alpha=0.7,
                        label=f"{idx['Index'].split('(')[0].strip()} (AUC={auc_v:.3f})")

ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random (AUC=0.5)')
ax.set_xlabel('1 - Specificity (FPR)', fontsize=14)
ax.set_ylabel('Sensitivity (TPR)', fontsize=14)
ax.set_title('IAN Score Predicting CKD - Temporal Validation', fontsize=16, fontweight='bold')
ax.legend(fontsize=13, loc='lower right')
ax.grid(alpha=0.3); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_roc.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_roc.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_roc.png"), dpi=300, bbox_inches="tight")
plt.close()

# figure2: training set vs validation set IAN-CKDprevalence trend comparison
print("  figure2: training setvsvalidation-set trend comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, dset, title, color in [(axes[0], train, 'Training (2011-2016)', RED),
                                 (axes[1], valid, 'Validation (2017-2018)', BLUE)]:
    ian_vals = dset['IAN'].dropna()
    scores = sorted(ian_vals.unique())
    prevs = [dset[dset['IAN']==s]['CKD'].mean()*100 for s in scores]
    counts = [(dset['IAN']==s).sum() for s in scores]
    
    ax2 = ax.twinx()
    ax.bar(scores, counts, color=color, alpha=0.25, width=0.6, label='N')
    ax2.plot(scores, prevs, color=color, marker='o', linewidth=2.5, markersize=8, label='CKD %')
    for s,p in zip(scores, prevs):
        ax2.annotate(f'{p:.1f}%', (s,p), textcoords='offset points',
                     xytext=(0,10), ha='center', fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('IAN Score', fontsize=13); ax.set_ylabel('N', color=GREY)
    ax2.set_ylabel('CKD Prevalence (%)', color=color); ax2.set_ylim(0, max(prevs)*1.4+5)
    ax.set_title(title, fontsize=16, fontweight='bold'); ax.set_xticks(scores)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_trend.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_trend.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_trend.png"), dpi=300, bbox_inches="tight")
plt.close()

# figure3: forest plot - comparison of each index
print("  figure3: index-comparison forest plot...")
fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(index_comparison))
for i, idx in enumerate(index_comparison):
    color = RED if idx['P']<0.001 else ORANGE if idx['P']<0.01 else BLUE if idx['P']<0.05 else GREY
    ax.plot([idx['95CI_L'], idx['95CI_U']], [i,i], color=color, linewidth=3)
    ax.scatter(idx['OR'], i, color=color, s=120, zorder=5)
    # per-row OR (95% CI) label to the right of each CI line
    ax.text(idx['95CI_U'] * 1.05, i, f"{idx['OR']:.2f} ({idx['95CI_L']:.2f}-{idx['95CI_U']:.2f})",
            va='center', ha='left', fontsize=13, color=color)
    # AUC at far right
    ax.text(2.55, i, f"AUC={idx['AUC']:.3f}", fontsize=13, va='center', ha='right', color=color)

ax.axvline(x=1, color='gray', linestyle='--', alpha=0.6, linewidth=2)
ax.set_yticks(y_pos)
ax.set_yticklabels([idx['Index'] for idx in index_comparison], fontsize=14)
ax.set_xlabel('Odds Ratio (95% CI) - Training Set', fontsize=16)
ax.set_title('Univariate Logistic Regression: IAN vs Other Indices', fontsize=18, fontweight='bold')
ax.set_xscale('linear')
ax.set_xlim(0.2, 2.7)
ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
ax.set_xticklabels(['0.5', '1.0', '1.5', '2.0', '2.5'])
ax.tick_params(axis='x', labelsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_forest.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_forest.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_temporal_forest.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============ 9. two-threshold strategy ============
print(f"\n{'='*70}")
print("two-threshold strategy (Two-threshold approach)")
print('='*70)

# Determine thresholds based on the training set
y_pred_train = m1['y_pred']
y_true_train = train.loc[m1['y_pred'].index, 'CKD'].astype(float)
fpr_tr, tpr_tr, thr_tr = roc_curve(y_true_train, y_pred_train)

# screening threshold: sensitivity>=80%
sens80_idx = np.where(tpr_tr >= 0.80)[0]
if len(sens80_idx) > 0:
    screening_thr = thr_tr[sens80_idx[-1]]
    screening_sens = tpr_tr[sens80_idx[-1]]
    screening_spec = 1 - fpr_tr[sens80_idx[-1]]
    print(f"\n  screening threshold (Sens>=80%): {screening_thr:.4f}")
    print(f"    Sens={screening_sens:.3f}, Spec={screening_spec:.3f}")
else:
    screening_thr = np.percentile(y_pred_train, 50)
    screening_sens = 0.8
    screening_spec = 0.5

# diagnostic threshold: Youdenindex
youden = tpr_tr - fpr_tr
diag_idx = np.argmax(youden)
diagnostic_thr = thr_tr[diag_idx]
diagnostic_sens = tpr_tr[diag_idx]
diagnostic_spec = 1 - fpr_tr[diag_idx]
print(f"\n  diagnostic threshold (Youden): {diagnostic_thr:.4f}")
print(f"    Sens={diagnostic_sens:.3f}, Spec={diagnostic_spec:.3f}")

# Validate the two-threshold strategy on the validation set
for dset, name in [(train, 'Train'), (valid, 'Valid')]:
    print(f"\n--- {name} ---")
    # IANcorresponding threshold
    dset_sub = dset.loc[y_true_train.index if name == 'Train' else dset.index]
    
    # screening
    yp = m1['logit'].predict(sm.add_constant(dset['IAN'].astype(float)))
    dset = dset.copy()
    dset['IAN_PROB'] = yp
    
    # screening threshold
    screen_pred = (yp >= screening_thr).astype(int)
    screen_cm = confusion_matrix(dset['CKD'].astype(float), screen_pred)
    screen_tn, screen_fp, screen_fn, screen_tp = screen_cm.ravel()
    print(f"  screening threshold ({screening_thr:.3f}):")
    print(f"    Sens={screen_tp/(screen_tp+screen_fn):.3f}, Spec={screen_tn/(screen_tn+screen_fp):.3f}")
    print(f"    needs further examination: {screen_pred.sum()}persons ({screen_pred.mean()*100:.1f}%)")
    
    # diagnostic threshold
    diag_pred = (yp >= diagnostic_thr).astype(int)
    diag_cm = confusion_matrix(dset['CKD'].astype(float), diag_pred)
    diag_tn, diag_fp, diag_fn, diag_tp = diag_cm.ravel()
    print(f"  diagnostic threshold ({diagnostic_thr:.3f}):")
    print(f"    Sens={diag_tp/(diag_tp+diag_fn):.3f}, Spec={diag_tn/(diag_tn+diag_fp):.3f}")
    print(f"    diagnosed asCKD: {diag_pred.sum()}persons ({diag_pred.mean()*100:.1f}%)")

# ============ 10. generalization-gap analysis ============
print(f"\n{'='*70}")
print("generalization gap (Generalization Gap)")
print('='*70)

gap = v1_train['auc'] - v1_valid['auc']
print(f"  training setAUC: {v1_train['auc']:.4f}")
print(f"  validation setAUC: {v1_valid['auc']:.4f}")
print(f"  generalization gap: {gap:.4f}")
print(f"  degree of overfitting: {'minimal' if gap<0.02 else 'smaller' if gap<0.05 else 'medium' if gap<0.1 else 'severe'}")

# ============ 11. save results ============
print(f"\n{'='*70}")
print("save results")
print('='*70)

# save model results
results_summary = {
    'model1_auc_train': v1_train['auc'],
    'model1_auc_valid': v1_valid['auc'],
    'generalization_gap': gap,
    'screening_threshold': screening_thr,
    'diagnostic_threshold': diagnostic_thr,
    'screening_thr_sens_train': screening_sens,
    'screening_thr_spec_train': screening_spec,
    'diagnostic_thr_sens_train': diagnostic_sens,
    'diagnostic_thr_spec_train': diagnostic_spec,
}

print(f"  training setIAN AUC: {v1_train['auc']:.4f}")
print(f"  validation setIAN AUC: {v1_valid['auc']:.4f}")
print(f"  generalization gap: {gap:.4f}")
print(f"\n  screening threshold: {screening_thr:.4f} (Sens={screening_sens:.3f}, Spec={screening_spec:.3f})")
print(f"  diagnostic threshold: {diagnostic_thr:.4f} (Sens={diagnostic_sens:.3f}, Spec={diagnostic_spec:.3f})")
print(f"\n{'='*70}")
print("analysis completed!")
print('='*70)
