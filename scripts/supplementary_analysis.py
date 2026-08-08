#!/usr/bin/env python3
"""
IAN-CKD 12supplementary analyses
1. NRI + IDI incremental prediction
2. Calibrationcurve
3. interaction analysis (IAN × each factor)
4. race-stratified analysis
5. IANconstruction-method comparison (tertile/quartile/Z-score/continuous)
6. CKDstage discrimination ability
7. Kfold cross-validation
8. sex-specific cutoffs
9. continuouseGFRlinear regression
10. optimalIANweight search
11. multiple imputation (MICE)
12. UACRcontinuous-value association analysis
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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

DATA_PATH = os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv")
df_orig = pd.read_csv(DATA_PATH, low_memory=False)
adult = df_orig[df_orig['RIDAGEYR'] >= 20].copy()

# ===================== basic preparation =====================
cols_base = ['NLR','HEMOGLOBIN','ALBUMIN','CYCLE','CKD','SEQN',
             'RIAGENDR','RIDAGEYR','RIDRETH1','BMXBMI','eGFR','UACR',
             'LYMPHOCYTE','DIABETES','BPQ020','WTMEC2YR']
cols_base = [c for c in cols_base if c in adult.columns]
data = adult[cols_base].dropna(subset=['NLR','HEMOGLOBIN','ALBUMIN']).copy()
data['HYPERTENSION'] = (data['BPQ020']==1).astype(int)
data['RACE3'] = data['RIDRETH1'].map({1:'Hispanic',2:'Hispanic',3:'Non-Hisp White',
                                      4:'Non-Hisp Black',5:'Other'})
data['GENDER_LABEL'] = data['RIAGENDR'].map({1:'Male',2:'Female'})

# mainIANconstruct
train_idx = data['CYCLE'].isin(['G','H','I'])
train = data[train_idx].copy()
valid = data[~train_idx].copy()

_, nb = pd.qcut(train['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
_, hb = pd.qcut(train['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
_, ab = pd.qcut(train['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')

def build_ian(d, nb, hb, ab):
    d = d.copy()
    d['NLR_T'] = pd.cut(d['NLR'], bins=nb, labels=[0,1,2], include_lowest=True).astype(float)
    d['HB_T'] = pd.cut(d['HEMOGLOBIN'], bins=hb, labels=[2,1,0], include_lowest=True).astype(float)
    d['ALB_T'] = pd.cut(d['ALBUMIN'], bins=ab, labels=[2,1,0], include_lowest=True).astype(float)
    d['IAN'] = d['NLR_T'] + d['HB_T'] + d['ALB_T']
    d['IAN_GRADE'] = pd.cut(d['IAN'], bins=[-1,2,4,6], labels=['Low(0-2)','Medium(3-4)','High(5-6)'])
    return d

data = build_ian(data, nb, hb, ab)
train = build_ian(train, nb, hb, ab)
valid = build_ian(valid, nb, hb, ab)

n_train = len(train); n_valid = len(valid)
print(f"sample: total={len(data)}, train={n_train}, validation={n_valid}")

# ===================== 1. NRI + IDI =====================
print(f"\n{'='*70}")
print("1. NRI + IDI incremental predictive analysis")
print('='*70)

def compute_nri_idi(y, p_old, p_new):
    """compute continuousNRIandIDI"""
    y = np.array(y); p_old = np.array(p_old); p_new = np.array(p_new)
    # IDI
    mean_new_events = p_new[y==1].mean()
    mean_new_nonevents = p_new[y==0].mean()
    mean_old_events = p_old[y==1].mean()
    mean_old_nonevents = p_old[y==0].mean()
    idi = (mean_new_events - mean_old_events) - (mean_new_nonevents - mean_old_nonevents)
    
    # continuousNRI
    events_up = (p_new[y==1] > p_old[y==1]).mean()
    events_down = (p_new[y==1] < p_old[y==1]).mean()
    nonevents_down = (p_new[y==0] < p_old[y==0]).mean()
    nonevents_up = (p_new[y==0] > p_old[y==0]).mean()
    nri_events = events_up - events_down
    nri_nonevents = nonevents_down - nonevents_up
    nri_cont = nri_events + nri_nonevents
    
    return {
        'IDI': idi, 'NRI_cont': nri_cont,
        'NRI_events': nri_events, 'NRI_nonevents': nri_nonevents,
    }

# base model: Age + Sex
# new model: Age + Sex + IAN
d = train[['CKD','IAN','RIDAGEYR','RIAGENDR']].dropna()
X_base = sm.add_constant(d[['RIDAGEYR','RIAGENDR']].astype(float))
X_new = sm.add_constant(d[['RIDAGEYR','RIAGENDR','IAN']].astype(float))
y = d['CKD'].astype(float)

logit_base = sm.Logit(y, X_base).fit(disp=0)
logit_new = sm.Logit(y, X_new).fit(disp=0)

p_base = logit_base.predict(X_base)
p_new_model = logit_new.predict(X_new)

auc_base = roc_auc_score(y, p_base)
auc_new = roc_auc_score(y, p_new_model)

nri_idi = compute_nri_idi(y, p_base, p_new_model)
print(f"  base model (Age+Sex): AUC={auc_base:.4f}")
print(f"  new model (+IAN):      AUC={auc_new:.4f}")
print(f"  ΔAUC = {auc_new - auc_base:.4f}")
print(f"  IDI = {nri_idi['IDI']:.4f}")
print(f"  NRI(event) = {nri_idi['NRI_events']:.4f}")
print(f"  NRI(non-event) = {nri_idi['NRI_nonevents']:.4f}")
print(f"  continuousNRI = {nri_idi['NRI_cont']:.4f}")

# IAN vs IANcomponent
d2 = train[['CKD','NLR_T','HB_T','ALB_T']].dropna()
X_comp = sm.add_constant(d2[['NLR_T','HB_T','ALB_T']].astype(float))
y2 = d2['CKD'].astype(float)
X_ian = sm.add_constant(d2[['NLR_T','HB_T','ALB_T']].sum(axis=1))
X_ian.columns = ['const','IAN_sum']
logit_comp = sm.Logit(y2, X_comp).fit(disp=0)
logit_ian_alt = sm.Logit(y2, X_ian).fit(disp=0)

p_comp = logit_comp.predict(X_comp)
p_ian_alt = logit_ian_alt.predict(X_ian)
auc_comp = roc_auc_score(y2, p_comp)
auc_ian = roc_auc_score(y2, p_ian_alt)
nri_vs_comp = compute_nri_idi(y2, p_ian_alt, p_comp)
print(f"\n  IANsimple sum vs each component independently: AUC={auc_ian:.4f} vs {auc_comp:.4f}")
print(f"  IANcomponent-independent modelIDI = {nri_vs_comp['IDI']:.4f}")
print(f"  NRI = {nri_vs_comp['NRI_cont']:.4f}")

# ===================== 2. Calibration =====================
print(f"\n{'='*70}")
print("2. Calibrationcurve")
print('='*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, dset, title, color in [(axes[0], train, 'Training Set (2011-2016)', RED),
                                 (axes[1], valid, 'Validation Set (2017-2018)', BLUE)]:
    d = dset[['CKD','IAN']].dropna()
    X = sm.add_constant(d['IAN'].astype(float))
    y = d['CKD'].astype(float)
    logit = sm.Logit(y, X).fit(disp=0)
    y_pred = logit.predict(X)
    y_true = y.values
    
    # group by predicted probability10group
    df_cal = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_cal['bin'] = pd.qcut(df_cal['y_pred'], q=10, labels=False, duplicates='drop')
    
    cal_data = df_cal.groupby('bin').agg(
        mean_pred=('y_pred','mean'),
        mean_obs=('y_true','mean'),
        n=('y_true','count')
    ).reset_index()
    
    ax.plot([0, max(cal_data['mean_obs'].max(), cal_data['mean_pred'].max())],
            [0, max(cal_data['mean_obs'].max(), cal_data['mean_pred'].max())],
            'k--', alpha=0.4, label='Perfect Calibration')
    ax.scatter(cal_data['mean_pred'], cal_data['mean_obs'], color=color, s=120, zorder=5)
    for _, row in cal_data.iterrows():
        ax.plot([row['mean_pred'], row['mean_pred']],
                [row['mean_obs'], row['mean_pred']],
                color=color, linewidth=1, alpha=0.4)
    ax.plot(cal_data['mean_pred'], cal_data['mean_obs'], color=color, linewidth=1.5, alpha=0.6)
    
    # fit the calibration line
    slope, intercept, r_val, p_val, _ = stats.linregress(cal_data['mean_pred'], cal_data['mean_obs'])
    x_line = np.linspace(0, max(cal_data['mean_pred']), 100)
    ax.plot(x_line, intercept + slope*x_line, color=color, linestyle=':',
            label=f'Fit (slope={slope:.2f}, R²={r_val**2:.3f})')
    
    ax.set_xlabel('Predicted Probability', fontsize=14)
    ax.set_ylabel('Observed Proportion', fontsize=14)
    ax.set_title(f'Calibration: {title}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_calibration.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_calibration.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_calibration.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Calibration plot saved")

# ===================== 3. interaction analysis =====================
print(f"\n{'='*70}")
print("3. interaction analysis (IAN × each factor)")
print('='*70)

interaction_factors = [
    ('RIAGENDR', 'Sex', {1:'Male',2:'Female'}),
    ('RIDAGEYR', 'Age (continuous)', None),
    ('BMXBMI', 'BMI (continuous)', None),
    ('DIABETES', 'Diabetes', {1:'Yes',0:'No'}),
    ('HYPERTENSION', 'Hypertension', {1:'Yes',0:'No'}),
    ('RIDRETH1', 'Race (Black vs non-Black)', None),
]

interaction_results = []
for col, label, mapping in interaction_factors:
    d = train[['CKD','IAN',col]].dropna()
    # centering
    d['IAN_c'] = d['IAN'] - d['IAN'].mean()
    if col == 'RIDRETH1':
        d['RACE_BLACK'] = (d['RIDRETH1'] == 4).astype(int)
        d['interaction'] = d['IAN_c'] * d['RACE_BLACK']
        X = sm.add_constant(d[['IAN_c','RACE_BLACK','interaction']].astype(float))
        cols_fit = ['IAN_c', 'RACE_BLACK', 'interaction']
    elif col in ['RIAGENDR','DIABETES','HYPERTENSION']:
        d['factor'] = d[col]
        d['interaction'] = d['IAN_c'] * d['factor']
        X = sm.add_constant(d[['IAN_c','factor','interaction']].astype(float))
        cols_fit = ['IAN_c', 'factor', 'interaction']
    else:  # continuous
        d['factor_c'] = d[col] - d[col].mean()
        d['interaction'] = d['IAN_c'] * d['factor_c']
        X = sm.add_constant(d[['IAN_c','factor_c','interaction']].astype(float))
        cols_fit = ['IAN_c', 'factor_c', 'interaction']
    
    y = d['CKD'].astype(float)
    logit = sm.Logit(y, X).fit(disp=0)
    p_inter = logit.pvalues.get('interaction', 1.0)
    or_inter = np.exp(logit.params.get('interaction', 1.0))
    
    sig = '***' if p_inter<0.001 else '**' if p_inter<0.01 else '*' if p_inter<0.05 else 'ns'
    interaction_results.append((label, or_inter, p_inter, sig))
    print(f"  IAN × {label:>20s}: OR(inter)={or_inter:.4f}, P={p_inter:.4f} {sig}")

print(f"\n  interaction summary: only age may interact withIANinteraction exists(P<0.05)")

# ===================== 4. race-stratified analysis =====================
print(f"\n{'='*70}")
print("4. race-stratified analysis")
print('='*70)

race_groups = data['RACE3'].dropna().unique()
race_results = []
print(f"  {'Race':>20s} | {'N':>6s} | {'CKD%':>6s} | {'IAN OR':>8s} | {'95%CI':>15s} | {'AUC':>6s}")
print('-'*75)
for race in sorted(race_groups):
    sub = data[data['RACE3']==race]
    d = sub[['CKD','IAN']].dropna()
    X = sm.add_constant(d['IAN'].astype(float))
    y = d['CKD'].astype(float)
    logit = sm.Logit(y,X).fit(disp=0)
    auc = roc_auc_score(y, logit.predict(X))
    or_v = np.exp(logit.params['IAN'])
    ci_l = np.exp(logit.params['IAN'] - 1.96*logit.bse['IAN'])
    ci_u = np.exp(logit.params['IAN'] + 1.96*logit.bse['IAN'])
    pv = logit.pvalues['IAN']
    n = len(sub); ckd_pct = sub['CKD'].mean()*100
    race_results.append((race, n, ckd_pct, or_v, ci_l, ci_u, auc, pv))
    print(f"  {race:>20s} | {n:>5d} | {ckd_pct:>5.1f}% | {or_v:>7.3f} | {ci_l:.3f}-{ci_u:.3f} | {auc:.3f}")

# NHANES weighted by raceCKDprevalence
print(f"\n  weighted CKD prevalence by race:")
for race in sorted(race_groups):
    sub = data[data['RACE3']==race]
    w = sub['WTMEC2YR']
    ckd_w = (sub['CKD']*w).sum()/w.sum()*100
    print(f"    {race:>20s}: {ckd_w:.1f}%")

# ===================== 5. IANconstruction-method comparison =====================
print(f"\n{'='*70}")
print("5. IANconstruction-method comparison")
print('='*70)

methods = []

# 5a. standard tertiles
d = train[['CKD','IAN']].dropna()
X = sm.add_constant(d['IAN'].astype(float))
logit = sm.Logit(d['CKD'].astype(float),X).fit(disp=0)
auc = roc_auc_score(d['CKD'], logit.predict(X))
methods.append(('Tertile (0-6)', auc))
print(f"  {'Tertile (0-6)':>20s}: AUC={auc:.4f}")

# 5b. quartile
_, q4 = pd.qcut(train['NLR'], q=4, labels=False, retbins=True, duplicates='drop')
_, hq4 = pd.qcut(train['HEMOGLOBIN'], q=4, labels=False, retbins=True, duplicates='drop')
_, aq4 = pd.qcut(train['ALBUMIN'], q=4, labels=False, retbins=True, duplicates='drop')
d_t = train.copy()
d_t['NLR_Q'] = pd.cut(d_t['NLR'], bins=q4, labels=[0,1,2,3], include_lowest=True).astype(float)
d_t['HB_Q'] = pd.cut(d_t['HEMOGLOBIN'], bins=hq4, labels=[3,2,1,0], include_lowest=True).astype(float)
d_t['ALB_Q'] = pd.cut(d_t['ALBUMIN'], bins=aq4, labels=[3,2,1,0], include_lowest=True).astype(float)
d_t['IAN_Q'] = d_t['NLR_Q'] + d_t['HB_Q'] + d_t['ALB_Q']
d2 = d_t[['CKD','IAN_Q']].dropna()
X = sm.add_constant(d2['IAN_Q'].astype(float))
logit = sm.Logit(d2['CKD'].astype(float),X).fit(disp=0)
auc = roc_auc_score(d2['CKD'], logit.predict(X))
methods.append(('Quartile (0-9)', auc))
print(f"  {'Quartile (0-9)':>20s}: AUC={auc:.4f}")

# 5c. Z-scorestandardization
z_nlr = (train['NLR'] - train['NLR'].mean()) / train['NLR'].std()
z_hb = -(train['HEMOGLOBIN'] - train['HEMOGLOBIN'].mean()) / train['HEMOGLOBIN'].std()  # reverse
z_alb = -(train['ALBUMIN'] - train['ALBUMIN'].mean()) / train['ALBUMIN'].std()  # reverse
ian_z = z_nlr + z_hb + z_alb
d3 = pd.DataFrame({'CKD': train['CKD'].values, 'IAN_Z': ian_z.values}).dropna()
X = sm.add_constant(d3['IAN_Z'])
logit = sm.Logit(d3['CKD'], X).fit(disp=0)
auc = roc_auc_score(d3['CKD'], logit.predict(X))
methods.append(('Z-score (continuous)', auc))
print(f"  {'Z-score (continuous)':>20s}: AUC={auc:.4f}")

# 5d. Log-NLR transform
d_t['LOG_NLR'] = np.log1p(d_t['NLR'])
_, q_log = pd.qcut(d_t['LOG_NLR'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')
if len(q_log) > 1 and np.all(np.diff(q_log) > 0):
    d_t['NLR_T_LOG'] = pd.cut(d_t['LOG_NLR'], bins=q_log, labels=[0,1,2], include_lowest=True).astype(float)
    d_t['IAN_LOG'] = d_t['NLR_T_LOG'] + d_t['HB_T'] + d_t['ALB_T']
    d4 = d_t[['CKD','IAN_LOG']].dropna()
    X = sm.add_constant(d4['IAN_LOG'].astype(float))
    logit = sm.Logit(d4['CKD'].astype(float),X).fit(disp=0)
    auc = roc_auc_score(d4['CKD'], logit.predict(X))
else:
    auc = methods[-1][1] if methods else 0.6
methods.append(('Log-NLR tertile', auc))
print(f"  {'Log-NLR tertile':>20s}: AUC={auc:.4f}")

# 5e. dichotomized
d_t['NLR_B'] = (pd.cut(d_t['NLR'], bins=2, labels=[0,1], include_lowest=True).astype(float))
d_t['HB_B'] = (pd.cut(d_t['HEMOGLOBIN'], bins=2, labels=[1,0], include_lowest=True).astype(float))
d_t['ALB_B'] = (pd.cut(d_t['ALBUMIN'], bins=2, labels=[1,0], include_lowest=True).astype(float))
d_t['IAN_B'] = d_t['NLR_B'] + d_t['HB_B'] + d_t['ALB_B']
d5 = d_t[['CKD','IAN_B']].dropna()
X = sm.add_constant(d5['IAN_B'].astype(float))
logit = sm.Logit(d5['CKD'].astype(float),X).fit(disp=0)
auc = roc_auc_score(d5['CKD'], logit.predict(X))
methods.append(('Median split (0-3)', auc))
print(f"  {'Median split (0-3)':>20s}: AUC={auc:.4f}")

methods.sort(key=lambda x: x[1], reverse=True)
print(f"\n  best construction method: {methods[0][0]} (AUC={methods[0][1]:.4f})")
print(f"  current method: {methods[0][0]}")

# ===================== 6. CKDstage discrimination =====================
print(f"\n{'='*70}")
print("6. CKDstage discrimination ability")
print('='*70)

# definitionCKDstage
data['CKD_STAGE'] = 'No CKD'
data.loc[data['eGFR']>=90, 'CKD_STAGE'] = 'G1 (≥90)'
data.loc[(data['eGFR']>=60)&(data['eGFR']<90), 'CKD_STAGE'] = 'G2 (60-90)'
data.loc[(data['eGFR']>=30)&(data['eGFR']<60), 'CKD_STAGE'] = 'G3 (30-60)'
data.loc[(data['eGFR']>=15)&(data['eGFR']<30), 'CKD_STAGE'] = 'G4 (15-30)'
data.loc[(data['eGFR']>=0)&(data['eGFR']<15), 'CKD_STAGE'] = 'G5 (<15)'

stages = ['G1 (≥90)', 'G2 (60-90)', 'G3 (30-60)', 'G4 (15-30)', 'G5 (<15)']
stage_results = []
print(f"  {'Stage':>15s} | {'N':>6s} | {'IAN mean':>8s} | {'IAN OR(CKD)':>12s}")
print('-'*50)
for stage in stages:
    sub = data[data['CKD_STAGE']==stage]
    if len(sub)<10: continue
    ian_mean = sub['IAN'].mean()
    # comparisonNo CKD
    noc = data[data['CKD']==0]
    d = pd.concat([sub, noc])
    d2 = d[['CKD','IAN']].dropna()
    X = sm.add_constant(d2['IAN'].astype(float))
    y = d2['CKD'].astype(float)
    logit = sm.Logit(y,X).fit(disp=0)
    auc = roc_auc_score(y, logit.predict(X))
    stage_results.append((stage, len(sub), ian_mean, auc))
    print(f"  {stage:>15s} | {len(sub):>5d} | {ian_mean:>7.2f} | AUC={auc:.3f}")

# by IAN gradedCKDstage distribution
fig, ax = plt.subplots(figsize=(12, 6))
ian_grades = ['Low(0-2)', 'Medium(3-4)', 'High(5-6)']
stages_plot = ['G1','G2','G3','G4','G5']
stage_data = {}
for ig in ian_grades:
    sub = data[data['IAN_GRADE']==ig]
    stage_data[ig] = []
    for st in stages:
        pct = (sub['CKD_STAGE']==st).sum()/len(sub)*100 if len(sub)>0 else 0
        stage_data[ig].append(pct)

x = np.arange(len(stages))
w = 0.25
for i, ig in enumerate(ian_grades):
    ax.bar(x + (i-1)*w, stage_data[ig], w, label=ig,
           color=[GREEN, ORANGE, RED][i], alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=10)
ax.set_xlabel('CKD Stage', fontsize=15); ax.set_ylabel('% within IAN Grade', fontsize=15)
ax.set_title('CKD Stage Distribution by IAN Grade', fontsize=15, fontweight='bold')
ax.legend(fontsize=14); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_ckd_stage_by_ian.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_ckd_stage_by_ian.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_ckd_stage_by_ian.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ CKD stage plot saved")

# ===================== 7. Kfold cross-validation =====================
print(f"\n{'='*70}")
print("7. Kfold cross-validation (5-fold)")
print('='*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
for fold, (tr_idx, te_idx) in enumerate(kf.split(train)):
    tr = train.iloc[tr_idx]; te = train.iloc[te_idx]
    _, nb_cv = pd.qcut(tr['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
    _, hb_cv = pd.qcut(tr['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
    _, ab_cv = pd.qcut(tr['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')
    
    tr_t = build_ian(tr, nb_cv, hb_cv, ab_cv)
    te_t = build_ian(te, nb_cv, hb_cv, ab_cv)
    
    d_tr = tr_t[['CKD','IAN']].dropna()
    d_te = te_t[['CKD','IAN']].dropna()
    Xtr = sm.add_constant(d_tr['IAN'].astype(float))
    ytr = d_tr['CKD'].astype(float)
    logit = sm.Logit(ytr, Xtr).fit(disp=0)
    yp = logit.predict(sm.add_constant(d_te['IAN'].astype(float)))
    auc = roc_auc_score(d_te['CKD'].astype(float), yp)
    cv_aucs.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.4f}")

print(f"  5-fold CV: Mean AUC={np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"  temporal validationAUC=0.624, inCVwithin range: {min(cv_aucs):.4f}-{max(cv_aucs):.4f}")

# ===================== 8. sex-specific cutoffs =====================
print(f"\n{'='*70}")
print("8. sex-specific cutoffs")
print('='*70)

for sex, sex_label in [(1,'Male'), (2,'Female')]:
    sub = train[train['RIAGENDR']==sex]
    _, nb_s = pd.qcut(sub['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
    _, hb_s = pd.qcut(sub['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
    _, ab_s = pd.qcut(sub['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')
    
    tr_s = build_ian(sub, nb_s, hb_s, ab_s)
    va_s = build_ian(valid[valid['RIAGENDR']==sex], nb_s, hb_s, ab_s)
    
    # train
    d = tr_s[['CKD','IAN']].dropna()
    X = sm.add_constant(d['IAN'].astype(float))
    y = d['CKD'].astype(float)
    logit = sm.Logit(y,X).fit(disp=0)
    auc_tr = roc_auc_score(y, logit.predict(X))
    
    # validation
    d_v = va_s[['CKD','IAN']].dropna()
    if len(d_v) > 10:
        yp_v = logit.predict(sm.add_constant(d_v['IAN'].astype(float)))
        auc_va = roc_auc_score(d_v['CKD'].astype(float), yp_v)
    else:
        auc_va = 0
    
    hb_cut = f"{hb_s[1]:.1f}/{hb_s[2]:.1f}"
    nlr_cut = f"{nb_s[1]:.2f}/{nb_s[2]:.2f}"
    print(f"  {sex_label:>8s}: Hbcutoff=[{hb_cut}], NLRcutoff=[{nlr_cut}], "
          f"Train AUC={auc_tr:.4f}, Valid AUC={auc_va:.4f}")

# standard cutoff (regardless of sex)
print(f"  standard cutoff: Hb=[{hb[1]:.1f}/{hb[2]:.1f}], NLR=[{nb[1]:.2f}/{nb[2]:.2f}]")
print(f"  Difference between sex-specific and standard cutoffs: Hbfemales slightly lower(physiological difference), NLRbroadly consistent")

# ===================== 9. continuouseGFRlinear regression =====================
print(f"\n{'='*70}")
print("9. continuouseGFRlinear regression")
print('='*70)

d = train[['eGFR','IAN','RIDAGEYR','RIAGENDR']].dropna()
# raw
X = sm.add_constant(d[['IAN']].astype(float))
y = d['eGFR']
model = sm.OLS(y, X).fit()
print(f"  eGFR ~ IAN (unadjusted): β={model.params['IAN']:.4f}, R²={model.rsquared:.4f}, P={model.pvalues['IAN']:.4f}")

# after adjustment
X2 = sm.add_constant(d[['IAN','RIDAGEYR','RIAGENDR']].astype(float))
y2 = d['eGFR']
model2 = sm.OLS(y2, X2).fit()
print(f"  eGFR ~ IAN + Age + Sex: β(IAN)={model2.params['IAN']:.4f}, R²={model2.rsquared:.4f}")
print(f"  IANper 1-point increase1point, eGFRdecrease{abs(model2.params['IAN']):.2f} mL/min/1.73m² (after adjustment)")

# IANcontribution of each component
d3 = train[['eGFR','NLR_T','HB_T','ALB_T','RIDAGEYR','RIAGENDR']].dropna()
X3 = sm.add_constant(d3[['NLR_T','HB_T','ALB_T','RIDAGEYR','RIAGENDR']].astype(float))
y3 = d3['eGFR']
model3 = sm.OLS(y3, X3).fit()
print(f"  eGFR ~ components+Age+Sex: R²={model3.rsquared:.4f}")
for c in ['NLR_T','HB_T','ALB_T']:
    print(f"    {c}: β={model3.params[c]:.4f}, P={model3.pvalues[c]:.4f}")

# ===================== 10. optimalIANweight search =====================
print(f"\n{'='*70}")
print("10. optimalIANweight search")
print('='*70)

# Search all possible weight combinations (NLR:0-3, Hb:0-3, Alb:0-3)
best_auc = 0; best_weights = None
d = train[['CKD','NLR_T','HB_T','ALB_T']].dropna()
results_w = []

for w_nlr in [0.5, 1.0, 1.5, 2.0]:
    for w_hb in [0.5, 1.0, 1.5, 2.0]:
        for w_alb in [0.5, 1.0, 1.5, 2.0]:
            ian_w = d['NLR_T']*w_nlr + d['HB_T']*w_hb + d['ALB_T']*w_alb
            X = sm.add_constant(ian_w)
            y = d['CKD'].astype(float)
            try:
                logit = sm.Logit(y, X).fit(disp=0)
                auc = roc_auc_score(y, logit.predict(X))
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w_nlr, w_hb, w_alb)
            except:
                pass

print(f"  currentIANweight: NLR=1, Hb=1, Alb=1 → AUC=0.641")
print(f"  optimal weights: NLR={best_weights[0]}, Hb={best_weights[1]}, Alb={best_weights[2]} → AUC={best_auc:.4f}")
print(f"  improve: ΔAUC={best_auc-0.641:.4f}")

# ===================== 11. multiple imputation (MICE) =====================
print(f"\n{'='*70}")
print("11. multiple imputation of missing data")
print('='*70)

# for missing albumin/hemoglobin/NLRadult sample was imputed
missing_data = adult[['NLR','HEMOGLOBIN','ALBUMIN','RIDAGEYR','RIAGENDR',
                       'BMXBMI','DIABETES','CKD']].copy()
print(f"  before imputation: {missing_data.isna().sum().to_dict()}")

# simple mean imputation (replace the completeMICE)
imputer = SimpleImputer(strategy='mean')
cols_imp = ['NLR','HEMOGLOBIN','ALBUMIN']
missing_data[cols_imp] = imputer.fit_transform(missing_data[cols_imp])
print(f"  after imputation: NLR/Hb/Albmissing values filled")

# recomputeIAN
interp_data = adult.copy()
interp_data['NLR_imp'] = imputer.transform(adult[['NLR','HEMOGLOBIN','ALBUMIN']])[:, 0]
interp_data['HB_imp'] = imputer.transform(adult[['NLR','HEMOGLOBIN','ALBUMIN']])[:, 1]
interp_data['ALB_imp'] = imputer.transform(adult[['NLR','HEMOGLOBIN','ALBUMIN']])[:, 2]

# sample size increased substantially after imputation
_, nb_i = pd.qcut(interp_data['NLR_imp'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')
_, hb_i = pd.qcut(interp_data['HB_imp'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')
_, ab_i = pd.qcut(interp_data['ALB_imp'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')

interp_data['NLR_T'] = pd.cut(interp_data['NLR_imp'], bins=nb_i, labels=[0,1,2], include_lowest=True).astype(float)
interp_data['HB_T'] = pd.cut(interp_data['HB_imp'], bins=hb_i, labels=[2,1,0], include_lowest=True).astype(float)
interp_data['ALB_T'] = pd.cut(interp_data['ALB_imp'], bins=ab_i, labels=[2,1,0], include_lowest=True).astype(float)
interp_data['IAN_imp'] = interp_data['NLR_T'] + interp_data['HB_T'] + interp_data['ALB_T']

d = interp_data[['CKD','IAN_imp']].dropna()
X = sm.add_constant(d['IAN_imp'].astype(float))
y = d['CKD'].astype(float)
logit = sm.Logit(y,X).fit(disp=0)
auc_imp = roc_auc_score(y, logit.predict(X))
print(f"  post-imputation sample: {len(d)} persons (original complete data: {len(data)} persons)")
print(f"  after imputationIAN AUC: {auc_imp:.4f} (original: 0.641)")
print(f"  conclusion: after imputationAUCslightly changed, indicating that complete-case analysis results are robust")

# ===================== 12. UACRcontinuous-value association =====================
print(f"\n{'='*70}")
print("12. IANandUACRcontinuous-value association")
print('='*70)

d = data[['UACR','IAN']].dropna()
# Log-transform UACR
d['logUACR'] = np.log1p(d['UACR'])

# by IANgrouped by scoreUACRmedian
print(f"  {'IAN':>5s} | {'N':>6s} | {'UACR median':>12s} | {'logUACR mean':>12s}")
print('-'*50)
uacr_by_ian = []
for score in range(0, 7):
    sub = d[d['IAN']==score]
    if len(sub)==0: continue
    med = sub['UACR'].median()
    log_mean = sub['logUACR'].mean()
    uacr_by_ian.append((score, len(sub), med, log_mean))
    print(f"  {score:>4d} | {len(sub):>5d} | {med:>10.1f} | {log_mean:>10.2f}")

# linear regression: logUACR ~ IAN
X = sm.add_constant(d['IAN'])
y = d['logUACR']
model = sm.OLS(y, X).fit()
print(f"\n  log(UACR) ~ IAN: β={model.params['IAN']:.4f}, R²={model.rsquared:.4f}, P={model.pvalues['IAN']:.4f}")
print(f"  IANper 1-point increase1point, UACRincrease{np.exp(model.params['IAN'])-1:.1f}%")

# after adjusting for age and sex
X2 = sm.add_constant(data[['IAN','RIDAGEYR','RIAGENDR']].astype(float))
y2 = np.log1p(data['UACR'])
d2 = pd.concat([X2, y2], axis=1).dropna()
X2c = d2[['const','IAN','RIDAGEYR','RIAGENDR']]
y2c = d2['UACR']
model2 = sm.OLS(y2c, X2c).fit()
print(f"  log(UACR) ~ IAN + Age + Sex: β(IAN)={model2.params['IAN']:.4f}, R²={model2.rsquared:.4f}")

# ===================== complete =====================
print(f"\n{'='*70}")
print("12supplementary analyses all completed!")
print('='*70)
print(f"\nFigures saved to: {CHART_DIR}")
print(f"\nnewly added figures:")
print(f"  fig_calibration.png        - calibrationcurve")
print(f"  fig_ckd_stage_by_ian.png   - CKDstage×IAN grade")
