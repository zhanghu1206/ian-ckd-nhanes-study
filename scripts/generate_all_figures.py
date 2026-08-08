#!/usr/bin/env python3
"""
Generate all missing figures for IAN-CKD analysis
1. Sensitivity analysis forest plot
2. NRI/IDI bar chart
3. IAN construction comparison 
4. Two-threshold strategy figure
5. Complete index comparison bar chart
6. eGFR ~ IAN linear regression scatter
7. IAN grade × CKD stage heatmap
8. Bootstrap distribution histogram
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
CHART = os.path.join(PROJ, "output", "charts")
os.makedirs(CHART, exist_ok=True)

from nature_config import *  # BLUE

rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 14})

df = pd.read_csv(os.path.join(PROJ, "output", "nhanes_ckd_merged.csv"), low_memory=False)
adult = df[df['RIDAGEYR']>=20].copy()
cols = ['NLR','HEMOGLOBIN','ALBUMIN','CYCLE','CKD','RIAGENDR','RIDAGEYR',
        'RIDRETH1','BMXBMI','eGFR','UACR','DIABETES','BPQ020','WTMEC2YR','LYMPHOCYTE']
cols = [c for c in cols if c in adult.columns]
data = adult[cols].dropna(subset=['NLR','HEMOGLOBIN','ALBUMIN']).copy()
data['HYPERTENSION'] = (data['BPQ020']==1).astype(int)
data['RACE3'] = data['RIDRETH1'].map({1:'Hispanic',2:'Hispanic',3:'Non-Hisp White',4:'Non-Hisp Black',5:'Other'})

train_idx = data['CYCLE'].isin(['G','H','I'])
train = data[train_idx].copy()
valid = data[~train_idx].copy()

_, nb = pd.qcut(train['NLR'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')
_, hb = pd.qcut(train['HEMOGLOBIN'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')
_, ab = pd.qcut(train['ALBUMIN'].dropna(), q=3, labels=False, retbins=True, duplicates='drop')

def build_ian(d):
    d=d.copy()
    d['NLR_T']=pd.cut(d['NLR'],bins=nb,labels=[0,1,2],include_lowest=True).astype(float)
    d['HB_T']=pd.cut(d['HEMOGLOBIN'],bins=hb,labels=[2,1,0],include_lowest=True).astype(float)
    d['ALB_T']=pd.cut(d['ALBUMIN'],bins=ab,labels=[2,1,0],include_lowest=True).astype(float)
    d['IAN']=d['NLR_T']+d['HB_T']+d['ALB_T']
    d['IAN_GRADE']=pd.cut(d['IAN'],bins=[-1,2,4,6],labels=['Low(0-2)','Medium(3-4)','High(5-6)'])
    return d

data=build_ian(data); train=build_ian(train); valid=build_ian(valid)
print(f"Samples: total={len(data)}, train={len(train)}, valid={len(valid)}")

# ============================================================
# FIGURE 1: Sensitivity Analysis Forest Plot
# ============================================================
print("\nFig 1: Sensitivity Analysis Forest...")

def run_or(data, x_vars, label):
    d = data[['CKD']+x_vars].dropna()
    X = sm.add_constant(d[x_vars].astype(float))
    y = d['CKD'].astype(float)
    m = sm.Logit(y,X).fit(disp=0)
    auc = roc_auc_score(y, m.predict(X))
    or_v = np.exp(m.params['IAN'])
    ci_l = np.exp(m.params['IAN']-1.96*m.bse['IAN'])
    ci_u = np.exp(m.params['IAN']+1.96*m.bse['IAN'])
    return label, len(d), or_v, ci_l, ci_u, auc

sens_analyses = [
    run_or(data, ['IAN'], 'Full sample (n=20,222)'),
    run_or(data[data['DIABETES']==0], ['IAN'], 'Exclude diabetes'),
    run_or(data[data['HYPERTENSION']==0], ['IAN'], 'Exclude hypertension'),
    run_or(data[data['RIDAGEYR']>=40], ['IAN'], 'Age >= 40'),
    run_or(data[data['RIDAGEYR']>=60], ['IAN'], 'Age >= 60'),
    run_or(data[data['BMXBMI']>=18.5], ['IAN'], 'BMI >= 18.5'),
]

# eGFR-only and UACR-only
d_egfr = data[data['eGFR'].notna()].copy()
d_egfr['CKD_e'] = (d_egfr['eGFR']<60).astype(int)
d2 = d_egfr[['CKD_e','IAN']].dropna()
X = sm.add_constant(d2['IAN'].astype(float)); y = d2['CKD_e'].astype(float)
m = sm.Logit(y,X).fit(disp=0); auc = roc_auc_score(y,m.predict(X))
sens_analyses.append(('CKD by eGFR<60', len(d2), np.exp(m.params['IAN']),
    np.exp(m.params['IAN']-1.96*m.bse['IAN']), np.exp(m.params['IAN']+1.96*m.bse['IAN']), auc))

d_uacr = data[data['UACR'].notna()].copy()
d_uacr['CKD_u'] = (d_uacr['UACR']>=30).astype(int)
d3 = d_uacr[['CKD_u','IAN']].dropna()
X = sm.add_constant(d3['IAN'].astype(float)); y = d3['CKD_u'].astype(float)
m = sm.Logit(y,X).fit(disp=0); auc = roc_auc_score(y,m.predict(X))
sens_analyses.append(('CKD by UACR>=30', len(d3), np.exp(m.params['IAN']),
    np.exp(m.params['IAN']-1.96*m.bse['IAN']), np.exp(m.params['IAN']+1.96*m.bse['IAN']), auc))

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(sens_analyses))
for i, (label, n, or_v, ci_l, ci_u, auc) in enumerate(sens_analyses):
    color = RED if ci_l>1 else GREY
    ax.plot([ci_l, ci_u], [i,i], color=color, linewidth=3)
    ax.scatter(or_v, i, color=color, s=150, zorder=5, marker='s')
    # per-row OR (95% CI) label to the right of each CI line
    ax.text(ci_u * 1.05, i, f"{or_v:.2f} ({ci_l:.2f}-{ci_u:.2f})",
            va='center', ha='left', fontsize=13, color=color)
    # AUC text at far right, n annotation on left
    ax.text(2.65, i, f'AUC={auc:.3f}', fontsize=14, va='center', ha='right', color=color)
    ax.text(0.65, i, f'n={n}', fontsize=14, va='center', ha='right', color=GREEN)

ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5, linewidth=2)
ax.set_yticks(y_pos)
ax.set_yticklabels([x[0] for x in sens_analyses], fontsize=16)
ax.set_xlabel('Odds Ratio (95% CI) per IAN point', fontsize=18)
ax.set_title('Sensitivity Analysis: IAN-CKD Association', fontsize=20, fontweight='bold')
ax.set_xscale('linear')
ax.set_xlim(0, 2.7)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5'])
ax.tick_params(axis='x', labelsize=16)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s1_sensitivity_forest.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s1_sensitivity_forest.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s1_sensitivity_forest.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 2: Index Comparison Bar Chart
# ============================================================
print("Fig 2: Index Comparison Bar Chart...")

# Compute actual index AUCs from training data (not hardcoded)
def calc_pni(alb, lymph): return alb*10 + 5*lymph
def calc_ali(bmi, alb, nlr): return bmi * alb / nlr
def calc_conut(alb, lymph):
    score = 0
    if alb >= 3.5: score += 0
    elif alb >= 3.0: score += 2
    elif alb >= 2.5: score += 4
    else: score += 6
    l = lymph * 1000
    if l >= 1600: score += 0
    elif l >= 1200: score += 1
    elif l >= 800: score += 2
    else: score += 3
    return score

# Compute on training set
t = train[['IAN','NLR','HEMOGLOBIN','ALBUMIN','CKD','BMXBMI']].copy()
lymph_c = 'LYMPHOCYTE' if 'LYMPHOCYTE' in train.columns else 'LBDLYMNO'
t['LYMPH'] = train[lymph_c].copy()
t = t.dropna(subset=['CKD','LYMPH'])
t['PNI'] = calc_pni(t['ALBUMIN'], t['LYMPH'])
t['CONUT'] = t.apply(lambda r: calc_conut(r['ALBUMIN'], r['LYMPH']), axis=1)
t['ALI'] = calc_ali(t['BMXBMI'], t['ALBUMIN'], t['NLR'])

from sklearn.metrics import roc_auc_score as _auc
def idx_auc(pred_series, y):
    mask = pred_series.notna() & y.notna()
    if mask.sum() < 2: return 0.5
    pred = pred_series[mask]; y_true = y[mask]
    a = _auc(y_true, pred); return max(a, 1-a)

indices = [
    ('IAN Score',  idx_auc(t['IAN'], t['CKD']), RED),
    ('PNI',        idx_auc(-t['PNI'], t['CKD']), BLUE),
    ('Albumin',    idx_auc(-t['ALBUMIN'], t['CKD']), ORANGE),
    ('NLR',        idx_auc(t['NLR'], t['CKD']), GREEN),
    ('Hemoglobin', idx_auc(-t['HEMOGLOBIN'], t['CKD']), '#9A4D8E'),
    ('CONUT',      idx_auc(t['CONUT'], t['CKD']), GREY),
    ('ALI',        _auc(t['CKD'][t['ALI'].notna()], t['ALI'].dropna()), '#C55A11'),  # no negation: raw AUC < 0.5 matches Table 3/ms
]
print(f"  Computed AUCs: {[(n, round(a,4)) for n,a,_ in indices]}")

fig, ax = plt.subplots(figsize=(10, 6))
idx_names = [x[0] for x in indices]
idx_aucs = [x[1] for x in indices]
idx_colors = [x[2] for x in indices]

bars = ax.barh(range(len(indices)), idx_aucs, color=idx_colors, height=0.6, edgecolor='white')
for i, (bar, auc, name) in enumerate(zip(bars, idx_aucs, idx_names)):
    ax.text(auc+0.01, i, f'AUC={auc:.3f}', va='center', fontsize=13, fontweight='bold',
            color=idx_colors[i])
    # Highlight IAN
    if name == 'IAN Score':
        bar.set_linewidth(3)
        bar.set_edgecolor('black')

ax.set_yticks(range(len(indices)))
ax.set_yticklabels(idx_names, fontsize=13)
ax.set_xlabel('Area Under the Curve (AUC)', fontsize=12)
ax.set_title('Univariate Predictive Performance: IAN vs Existing Indices', fontsize=16, fontweight='bold')
ax.set_xlim(0.35, 0.75)
ax.axvline(x=0.5, color='grey', linestyle=':', alpha=0.4, label='Random (AUC=0.5)')
ax.legend(fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s2_index_comparison_bar.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s2_index_comparison_bar.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s2_index_comparison_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 3: NRI/IDI Bar Chart
# ============================================================
print("Fig 3: NRI/IDI Bar Chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: AUC improvement
ax = axes[0]
models = ['Age+Sex', '+ IAN', '+ Full*']
aucs_mod = [0.755, 0.767, 0.788]
colors_mod = [GREY, BLUE, RED]
bars = ax.bar(models, aucs_mod, color=colors_mod, width=0.5, edgecolor='white')
for bar, auc in zip(bars, aucs_mod):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{auc:.3f}',
            ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('AUC', fontsize=14)
ax.set_title('AUC Improvement with IAN', fontsize=17, fontweight='bold')
ax.set_ylim(0.7, 0.85)
ax.tick_params(axis='x', labelsize=13)
ax.grid(axis='y', alpha=0.3)
# Red arrows removed per user request (2026-06-27)

# Right: NRI components
ax = axes[1]
nri_labels = ['NRI\n(Events)', 'NRI\n(Non-events)', 'NRI\n(Continuous)', 'IDI']
nri_vals = [0.119, 0.153, 0.272, 0.020]
nri_colors = [RED, BLUE, GREEN, ORANGE]
bars = ax.bar(nri_labels, nri_vals, color=nri_colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, nri_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Value', fontsize=14)
ax.set_title('Incremental Predictive Value\n(Age+Sex → +IAN)', fontsize=17, fontweight='bold')
ax.tick_params(axis='x', labelsize=13)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s3_nri_idi.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s3_nri_idi.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s3_nri_idi.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 4: Two-Threshold Strategy
# ============================================================
print("Fig 4: Two-Threshold Strategy...")

# Fit model
d = train[['CKD','IAN']].dropna()
m = sm.Logit(d['CKD'].astype(float), sm.add_constant(d['IAN'].astype(float))).fit(disp=0)
yp = m.predict(sm.add_constant(d['IAN'].astype(float)))
fpr, tpr, thr = roc_curve(d['CKD'].astype(float), yp)

# Find screening threshold (Sens>=0.80)
sens80_idx = np.where(tpr>=0.80)[0]
screen_thr = thr[sens80_idx[-1]] if len(sens80_idx)>0 else 0.067
screen_sens = tpr[sens80_idx[-1]] if len(sens80_idx)>0 else 1.0
screen_spec = 1-fpr[sens80_idx[-1]] if len(sens80_idx)>0 else 0

# Diagnostic (Youden)
youden = tpr-fpr
diag_idx = np.argmax(youden)
diag_thr = thr[diag_idx]
diag_sens = tpr[diag_idx]
diag_spec = 1-fpr[diag_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: ROC with thresholds
ax = axes[0]
ax.plot(fpr, tpr, color=RED, linewidth=2.5, label='IAN ROC')
ax.scatter(1-screen_spec, screen_sens, color=GREEN, s=200, zorder=5, marker='D',
           label=f'Screening: Sens={screen_sens:.2f}, Spec={screen_spec:.2f}')
ax.scatter(1-diag_spec, diag_sens, color=BLUE, s=200, zorder=5, marker='s',
           label=f'Diagnostic: Sens={diag_sens:.2f}, Spec={diag_spec:.2f}')
ax.plot([0,1],[0,1],'k--',alpha=0.4)
ax.set_xlabel('1 - Specificity', fontsize=13); ax.set_ylabel('Sensitivity', fontsize=13)
ax.set_title('Two-Threshold Strategy on ROC', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, loc='lower right'); ax.grid(alpha=0.3)

# Right: clinical decision diagram
ax = axes[1]
ax.axis('off')
ax.text(0.5, 0.96, 'IAN-Based CKD Screening Protocol', ha='center', fontsize=16, fontweight='bold')

# ── STEP 1 box (enlarged: height 0.15 → 0.22) ──
BOX_H = 0.22
BOX_BOTTOM = 0.68  # top = 0.90
ax.add_patch(plt.Rectangle((0.05, BOX_BOTTOM), 0.4, BOX_H,
                           facecolor='#E2EFDA', ec='green', lw=2, transform=ax.transAxes))
ax.text(0.25, 0.86, 'STEP 1: SCREENING', ha='center', fontsize=14,
        fontweight='bold', transform=ax.transAxes)
ax.text(0.25, 0.78, f'IAN Probability ≥ {screen_thr:.2f}', ha='center',
        fontsize=12, transform=ax.transAxes)
ax.text(0.25, 0.71, f'Sensitivity = {screen_sens:.1%}', ha='center',
        fontsize=12, color='green', transform=ax.transAxes)

# Arrow
ax.annotate('', xy=(0.5, 0.78), xytext=(0.45, 0.78),
            arrowprops=dict(arrowstyle='->', lw=2), transform=ax.transAxes)

# ── STEP 2 box (enlarged) ──
ax.add_patch(plt.Rectangle((0.55, BOX_BOTTOM), 0.4, BOX_H,
                           facecolor='#FDE9D9', ec='red', lw=2, transform=ax.transAxes))
ax.text(0.75, 0.86, 'STEP 2: DIAGNOSTIC', ha='center', fontsize=14,
        fontweight='bold', transform=ax.transAxes)
ax.text(0.75, 0.78, f'IAN Probability ≥ {diag_thr:.2f}', ha='center',
        fontsize=12, transform=ax.transAxes)
ax.text(0.75, 0.71, f'Specificity = {diag_spec:.1%}', ha='center',
        fontsize=12, color='red', transform=ax.transAxes)

# Risk zones
ax.add_patch(plt.Rectangle((0.05,0.35), 0.28, 0.25, facecolor=GREEN, alpha=0.15, ec=GREEN, lw=1.5, transform=ax.transAxes))
ax.text(0.19, 0.52, 'LOW RISK', ha='center', fontsize=13, fontweight='bold', color=GREEN, transform=ax.transAxes)
ax.text(0.19, 0.46, f'IAN ≤ {screen_thr:.2f}', ha='center', fontsize=12, transform=ax.transAxes)
ax.text(0.19, 0.40, 'Reassure & monitor', ha='center', fontsize=12, color=GREEN, transform=ax.transAxes)

ax.add_patch(plt.Rectangle((0.36,0.35), 0.28, 0.25, facecolor=ORANGE, alpha=0.15, ec=ORANGE, lw=1.5, transform=ax.transAxes))
ax.text(0.50, 0.52, 'MEDIUM RISK', ha='center', fontsize=13, fontweight='bold', color=ORANGE, transform=ax.transAxes)
ax.text(0.50, 0.46, f'{screen_thr:.2f} ≤ IAN < {diag_thr:.2f}', ha='center', fontsize=12, transform=ax.transAxes)
ax.text(0.50, 0.40, 'Check eGFR & UACR', ha='center', fontsize=12, color=ORANGE, transform=ax.transAxes)

ax.add_patch(plt.Rectangle((0.67,0.35), 0.28, 0.25, facecolor=RED, alpha=0.1, ec=RED, lw=1.5, transform=ax.transAxes))
ax.text(0.81, 0.52, 'HIGH RISK', ha='center', fontsize=13, fontweight='bold', color=RED, transform=ax.transAxes)
ax.text(0.81, 0.46, f'IAN ≥ {diag_thr:.2f}', ha='center', fontsize=12, transform=ax.transAxes)
ax.text(0.81, 0.40, 'Refer to nephrology', ha='center', fontsize=12, color=RED, transform=ax.transAxes)

# Prevalence
ax.text(0.5, 0.15, f'Population CKD Prevalence: {data["CKD"].mean()*100:.1f}%', ha='center', fontsize=13,
        fontweight='bold', transform=ax.transAxes)

plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s4_two_threshold.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s4_two_threshold.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s4_two_threshold.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 5: eGFR ~ IAN Linear Regression Scatter
# ============================================================
print("Fig 5: eGFR ~ IAN Scatter...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, dset, title, color in [(axes[0], train, 'Training Set', RED),
                                 (axes[1], valid, 'Validation Set', BLUE)]:
    d = dset[['eGFR','IAN']].dropna()
    # Add jitter
    x_jitter = d['IAN'] + np.random.normal(0, 0.08, len(d))
    ax.scatter(x_jitter, d['eGFR'], alpha=0.15, color=color, s=5)
    
    # Regression line
    X = sm.add_constant(d['IAN'].astype(float))
    y = d['eGFR']
    m = sm.OLS(y,X).fit()
    x_line = np.linspace(0, 6, 100)
    y_line = m.params['const'] + m.params['IAN'] * x_line
    ax.plot(x_line, y_line, color='black', linewidth=2.5, label=f'eGFR = {m.params["const"]:.1f} {m.params["IAN"]:+.1f}×IAN')
    
    # CI band
    pred = m.get_prediction(sm.add_constant(x_line))
    ci = pred.conf_int()
    ax.fill_between(x_line, ci[:,0], ci[:,1], alpha=0.15, color='black')
    
    ax.set_xlabel('IAN Score', fontsize=12); ax.set_ylabel('eGFR (mL/min/1.73m²)', fontsize=12)
    ax.set_title(f'{title}: eGFR vs IAN', fontsize=16, fontweight='bold')
    ax.set_xticks(range(0,7)); ax.set_xlim(-0.3, 6.3); ax.set_ylim(0, 140)
    ax.legend(fontsize=14); ax.grid(alpha=0.3)
    
    # Annotate stats
    ax.text(0.05, 0.95, f'β={m.params["IAN"]:.2f}\nR²={m.rsquared:.3f}\nP<{m.pvalues["IAN"]:.4f}',
            transform=ax.transAxes, fontsize=14, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s5_egfr_ian_scatter.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s5_egfr_ian_scatter.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s5_egfr_ian_scatter.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 6: IAN Construction Comparison
# ============================================================
print("Fig 6: IAN Construction Comparison...")

methods_data = [
    ('Median split\n(0-3)', 0.583, GREY),
    ('Tertile\n(0-6)', 0.641, RED),
    ('Quartile\n(0-9)', 0.644, ORANGE),
    ('Log-NLR\ntertile', 0.641, BLUE),
    ('Z-score\n(continuous)', 0.654, GREEN),
]

fig, ax = plt.subplots(figsize=(11, 6))
names = [x[0] for x in methods_data]
aucs_m = [x[1] for x in methods_data]
colors_m = [x[2] for x in methods_data]

bars = ax.bar(range(len(methods_data)), aucs_m, color=colors_m, width=0.5, edgecolor='white')
for i, (bar, auc, name) in enumerate(zip(bars, aucs_m, names)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'AUC={auc:.3f}',
            ha='center', fontsize=14, fontweight='bold')
    # Highlight chosen method
    if 'Tertile' in name:
        bar.set_linewidth(3)
        bar.set_edgecolor('black')

ax.set_xticks(range(len(methods_data)))
ax.set_xticklabels(names, fontsize=14, rotation=15, ha='center')
ax.set_ylabel('AUC', fontsize=14)
ax.set_title('IAN Construction Method Comparison', fontsize=17, fontweight='bold')
ax.set_ylim(0.5, 0.7)
ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.4)
ax.text(4.5, 0.505, 'Random', fontsize=12, color='grey')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s6_construction_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s6_construction_comparison.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s6_construction_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 7: IAN Grade × CKD Stage Heatmap (Already exists: fig_ckd_stage_by_ian.png)
# Just create an alternative version as a heatmap
# ============================================================
print("Fig 7: IAN by CKD Stage Heatmap...")

data['CKD_STAGE'] = 'No CKD'
data.loc[data['eGFR']>=90, 'CKD_STAGE'] = 'G1'
data.loc[(data['eGFR']>=60)&(data['eGFR']<90), 'CKD_STAGE'] = 'G2'
data.loc[(data['eGFR']>=30)&(data['eGFR']<60), 'CKD_STAGE'] = 'G3'
data.loc[(data['eGFR']>=15)&(data['eGFR']<30), 'CKD_STAGE'] = 'G4'
data.loc[(data['eGFR']>=0)&(data['eGFR']<15), 'CKD_STAGE'] = 'G5'

stages = ['G1','G2','G3','G4','G5']
grades = ['Low(0-2)','Medium(3-4)','High(5-6)']

# Build heatmap matrix: CKD prevalence within each IAN grade × CKD stage cell
heat_data = np.zeros((len(grades), len(stages)))
counts_data = np.zeros((len(grades), len(stages)))
for i, ig in enumerate(grades):
    for j, st in enumerate(stages):
        cell = data[(data['IAN_GRADE']==ig)&(data['CKD_STAGE']==st)]
        if len(cell)>0:
            heat_data[i,j] = cell['CKD'].mean()*100
            counts_data[i,j] = len(cell)

fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(heat_data, cmap='Reds', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=13)
ax.set_yticks(range(len(grades)))
ax.set_yticklabels(grades, fontsize=13)

for i in range(len(grades)):
    for j in range(len(stages)):
        val = heat_data[i,j]
        n = int(counts_data[i,j])
        ax.text(j, i, f'{val:.0f}%\n(n={n})', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white' if val>60 else 'black')

ax.set_xlabel('CKD Stage (by eGFR)', fontsize=12)
ax.set_ylabel('IAN Grade', fontsize=12)
ax.set_title('CKD Prevalence by IAN Grade and CKD Stage', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, label='CKD Prevalence (%)', shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s7_ian_stage_heatmap.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s7_ian_stage_heatmap.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s7_ian_stage_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 8: Bootstrap AUC Distribution
# ============================================================
print("Fig 8: Bootstrap Distribution...")

# Correct bootstrap: fit on bootstrap sample, evaluate on OUT-OF-BAG sample (.632 bootstrap)
# This gives a realistic AUC distribution reflecting sampling uncertainty
np.random.seed(42)
n_boot = 1000
boot_aucs = []
d = train[['CKD','IAN']].dropna()
n = len(d)

for b in range(n_boot):
    in_idx = np.random.choice(n, n, replace=True)
    in_mask = np.zeros(n, dtype=bool)
    in_mask[in_idx] = True
    oob_mask = ~in_mask
    boot_in  = d.iloc[in_idx]
    boot_oob = d[oob_mask]
    if boot_oob['CKD'].nunique() < 2 or len(boot_oob) < 10:
        continue
    try:
        Xb  = sm.add_constant(boot_in['IAN'].astype(float))
        yb  = boot_in['CKD'].astype(float)
        mb  = sm.Logit(yb, Xb).fit(disp=0)
        Xoob = sm.add_constant(boot_oob['IAN'].astype(float))
        yp  = mb.predict(Xoob)
        boot_aucs.append(roc_auc_score(boot_oob['CKD'].astype(float), yp))
    except:
        pass

boot_aucs = np.array(boot_aucs)
mean_auc  = boot_aucs.mean()
ci_lo     = np.percentile(boot_aucs, 2.5)
ci_hi     = np.percentile(boot_aucs, 97.5)

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(boot_aucs, bins=40, color=BLUE, alpha=0.7, edgecolor='white')
ax.axvline(x=mean_auc, color=RED, linewidth=2.5, linestyle='-',
           label=f'Mean AUC = {mean_auc:.4f}')
ax.axvline(x=ci_lo, color=RED, linewidth=2, linestyle='--',
           label=f'95% CI: ({ci_lo:.4f}, {ci_hi:.4f})')
ax.axvline(x=ci_hi, color=RED, linewidth=2, linestyle='--')

# Explicitly format x-axis to show 4-decimal values correctly
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
x_margin = (boot_aucs.max() - boot_aucs.min()) * 0.05
ax.set_xlim(boot_aucs.min() - x_margin, boot_aucs.max() + x_margin)
plt.xticks(rotation=30, ha='right')

ax.set_xlabel('AUC', fontsize=12)
ax.set_ylabel('Frequency (out of 1000)', fontsize=12)
ax.set_title('Bootstrap Validation: IAN AUC Distribution (n=1000)', fontsize=16, fontweight='bold')
ax.legend(fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s8_bootstrap_dist.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 9: IAN vs UACR Boxplot by Score
# ============================================================
print("Fig 9: UACR by IAN Boxplot...")

fig, ax = plt.subplots(figsize=(10, 6))
d = data[['UACR','IAN']].dropna()
bp_data = [d[d['IAN']==s]['UACR'].values for s in range(0,7)]
bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6)
colors_bp = [GREEN, GREEN, ORANGE, ORANGE, ORANGE, RED, RED]
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)

# Add means
means = [d[d['IAN']==s]['UACR'].mean() for s in range(0,7)]
ax.plot(range(1,8), means, 'D', color='black', markersize=8, label='Mean')

ax.set_xticklabels([str(s) for s in range(0,7)], fontsize=13)
ax.set_xlabel('IAN Score', fontsize=12)
ax.set_ylabel('UACR (mg/g)', fontsize=12)
ax.set_title('UACR Distribution by IAN Score', fontsize=16, fontweight='bold')
ax.legend(fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s9_uacr_by_ian.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s9_uacr_by_ian.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s9_uacr_by_ian.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 10: Weighted CKD Prevalence by IAN (with NHANES weights)
# ============================================================
print("Fig 10: Weighted CKD Prevalence...")

fig, ax = plt.subplots(figsize=(12, 6))
scores = list(range(0,7))
# Weighted
weighted_prev = []
for s in scores:
    sub = data[data['IAN']==s]
    w = sub['WTMEC2YR']
    weighted_prev.append((sub['CKD']*w).sum()/w.sum()*100)
# Unweighted
unweighted_prev = [data[data['IAN']==s]['CKD'].mean()*100 for s in scores]

x = np.arange(len(scores))
w = 0.25
ax.bar(x-w/2, unweighted_prev, w, color=RED, alpha=0.6, label='Unweighted')
ax.bar(x+w/2, weighted_prev, w, color=BLUE, alpha=0.6, label='Survey-Weighted')
max_val = max(max(unweighted_prev), max(weighted_prev))
ax.set_ylim(0, max_val * 1.25)
for i in range(len(scores)):
    ax.text(i-w/2, unweighted_prev[i] + max_val * 0.02, f'{unweighted_prev[i]:.1f}%', ha='center', fontsize=11, color=RED, rotation=90)
    ax.text(i+w/2, weighted_prev[i] + max_val * 0.02, f'{weighted_prev[i]:.1f}%', ha='center', fontsize=11, color=BLUE, rotation=90)

ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scores], fontsize=13)
ax.set_xlabel('IAN Score', fontsize=14); ax.set_ylabel('CKD Prevalence (%)', fontsize=14)
ax.set_title('CKD Prevalence by IAN Score: Unweighted vs Survey-Weighted', fontsize=17, fontweight='bold')
ax.legend(fontsize=13); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART, "fig_s10_weighted_prevalence.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART, "fig_s10_weighted_prevalence.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART, "fig_s10_weighted_prevalence.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
print(f"\n{'='*50}")
print(f"All 10 additional figures saved to: {CHART}")
print(f"{'='*50}")
print(f"\nTotal figures now: Check with ls")
