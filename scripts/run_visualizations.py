#!/usr/bin/env python3
"""Nature-style figures for IAN-CKD analysis"""
import os, sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJ_DIR, "output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
DATA_PATH = os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv")

# Import Nature-style unified config
sys.path.insert(0, BASE_DIR)
from nature_config import *

df = pd.read_csv(DATA_PATH, low_memory=False)

# Recompute derived vars and IAN
df['AGE_GROUP'] = pd.cut(df['RIDAGEYR'], bins=[0,20,40,60,80,200], labels=['0-19','20-39','40-59','60-79','80+'])
df['GENDER_LABEL'] = df['RIAGENDR'].map({1:'Male',2:'Female'})
df['RACE3'] = df['RIDRETH1'].map({1:'Hispanic',2:'Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',5:'Other'})

adult = df[df['RIDAGEYR'] >= 20].copy()

# Compute tertiles
valid = adult[['NLR','HEMOGLOBIN','ALBUMIN']].dropna()
_, nlr_b = pd.qcut(valid['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
_, hb_b = pd.qcut(valid['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
_, alb_b = pd.qcut(valid['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')

df['NLR_T'] = pd.cut(df['NLR'], bins=nlr_b, labels=[0,1,2], include_lowest=True).astype(float)
df['HB_T'] = pd.cut(df['HEMOGLOBIN'], bins=hb_b, labels=[2,1,0], include_lowest=True).astype(float)
df['ALB_T'] = pd.cut(df['ALBUMIN'], bins=alb_b, labels=[2,1,0], include_lowest=True).astype(float)
df['IAN'] = df['NLR_T'] + df['HB_T'] + df['ALB_T']
df['IAN_GRADE'] = pd.cut(df['IAN'], bins=[-1,2,4,6], labels=['Low (0-2)','Medium (3-4)','High (5-6)'])

sub = df[(df['RIDAGEYR']>=20) & df['IAN'].notna()].copy()
sub['RACE3'] = pd.Categorical(sub['RACE3'])

print(f"visualization data: {len(sub)} rows")

# ==== Fig1: Distribution ====
print("Fig1: IAN Score Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
bins = np.arange(-0.5, 7, 1)
ax = axes[0]
for ckd_val, color, label in [(0, BLUE, 'No CKD'), (1, RED, 'CKD')]:
    data = sub[sub['CKD']==ckd_val]['IAN']
    ax.hist(data, bins=bins, alpha=0.7, color=color, label=label, density=True)
ax.set_xlabel('IAN Score', fontsize=14); ax.set_ylabel('Density', fontsize=14)
ax.set_title('IAN Score Distribution by CKD Status', fontsize=15, fontweight='bold')
ax.set_xticks(range(0,7)); ax.legend(fontsize=13); ax.grid(alpha=0.3)

ax = axes[1]
grades = ['Low (0-2)', 'Medium (3-4)', 'High (5-6)']
ckd_rates = [sub[sub['IAN_GRADE']==g]['CKD'].mean()*100 for g in grades]
bars = ax.bar(grades, ckd_rates, color=[GREEN,ORANGE,RED], width=0.6, edgecolor='white')
for bar, val in zip(bars, ckd_rates):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_xlabel('IAN Risk Grade', fontsize=14); ax.set_ylabel('CKD Prevalence (%)', fontsize=14)
ax.set_title('CKD Prevalence by IAN Risk Grade', fontsize=15, fontweight='bold')
ax.set_ylim(0, max(ckd_rates)*1.2+2); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig1_ian_distribution.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig1_ian_distribution.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig1_ian_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Fig2: Forest plot (Supp Figure S21) ====
print("Fig2: Forest plot (IAN components)...")
components = ['NLR_T','HB_T','ALB_T']
clabels = ['NLR Tertile','Hb Tertile (reverse)','Albumin Tertile (reverse)']
comp_or = []
for comp, cl in zip(components, clabels):
    d = sub[['CKD',comp]].dropna()
    X = sm.add_constant(d[comp].astype(float))
    y = d['CKD'].astype(float)
    logit = sm.Logit(y,X).fit(disp=0)
    or_v = np.exp(logit.params[comp])
    ci_l = np.exp(logit.params[comp]-1.96*logit.bse[comp])
    ci_u = np.exp(logit.params[comp]+1.96*logit.bse[comp])
    p_v = logit.pvalues[comp]
    comp_or.append((cl, or_v, ci_l, ci_u, p_v))
    print(f"  {cl}: OR={or_v:.3f} ({ci_l:.3f}-{ci_u:.3f}), P={p_v:.4f}")

d = sub[['CKD','IAN']].dropna()
X = sm.add_constant(d['IAN'].astype(float))
y = d['CKD'].astype(float)
logit = sm.Logit(y,X).fit(disp=0)
or_v = np.exp(logit.params['IAN'])
ci_l = np.exp(logit.params['IAN']-1.96*logit.bse['IAN'])
ci_u = np.exp(logit.params['IAN']+1.96*logit.bse['IAN'])
p_v = logit.pvalues['IAN']
comp_or.append(('IAN Score (per point)', or_v, ci_l, ci_u, p_v))

fig, ax = plt.subplots(figsize=(12,6))
y_pos = np.arange(len(comp_or))
for i,(label,or_v,ci_l,ci_u,p_v) in enumerate(comp_or):
    color = RED if p_v<0.001 else ORANGE if p_v<0.01 else BLUE if p_v<0.05 else GREY
    ax.plot([ci_l,ci_u],[i,i], color=color, linewidth=2.5)
    ax.scatter(or_v, i, color=color, s=100, zorder=5,
               marker='D' if p_v<0.001 else 's' if p_v<0.01 else 'o')
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
ax.set_yticks(y_pos); ax.set_yticklabels([x[0] for x in comp_or], fontsize=13)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=14)
ax.set_title('Univariate Logistic Regression: IAN Components Predicting CKD', fontsize=15, fontweight='bold')
ax.set_xlim(0, 2.0)
ax.grid(axis='x', alpha=0.3)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'.rstrip('0').rstrip('.')))
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig2_forest_ian_components.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig2_forest_ian_components.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig2_forest_ian_components.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Fig3: ROC curve ====
print("Fig3: ROC Curve...")
fig, ax = plt.subplots(figsize=(10,9))

# IAN only
sub_roc = sub[['CKD','IAN']].dropna()
X_roc = sm.add_constant(sub_roc['IAN'].astype(float))
y_roc = sub_roc['CKD'].astype(float)
logit_roc = sm.Logit(y_roc, X_roc).fit(disp=0)
y_prob = logit_roc.predict(X_roc)
fpr, tpr, _ = roc_curve(y_roc, y_prob)
auc = roc_auc_score(y_roc, y_prob)
ax.plot(fpr, tpr, color=RED, linewidth=2.5, label=f'IAN Score (AUC={auc:.3f})')
print(f"  IAN-only AUC: {auc:.4f}")

# IAN + age + sex + race
sub_roc3 = sub[['CKD','IAN','RIDAGEYR','RIAGENDR']].dropna()
race_d = pd.get_dummies(sub['RACE3'].loc[sub_roc3.index], prefix='Race', drop_first=True).astype(float)
X_roc3 = sm.add_constant(pd.concat([
    sub_roc3[['IAN','RIDAGEYR','RIAGENDR']].astype(float), race_d
], axis=1).apply(pd.to_numeric, errors='coerce'))
y_roc3 = sub_roc3['CKD'].astype(float)
logit_roc3 = sm.Logit(y_roc3, X_roc3).fit(disp=0)
y_prob3 = logit_roc3.predict(X_roc3)
fpr3, tpr3, _ = roc_curve(y_roc3, y_prob3)
auc3 = roc_auc_score(y_roc3, y_prob3)
ax.plot(fpr3, tpr3, color=BLUE, linewidth=2.5, label=f'IAN + Age + Sex + Race (AUC={auc3:.3f})')
print(f"  IAN+Age+Sex+Race AUC: {auc3:.4f}")

ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random (AUC=0.5)')
ax.set_xlabel('1 - Specificity', fontsize=14)
ax.set_ylabel('Sensitivity', fontsize=14)
ax.set_title('ROC Curves: IAN Score Predicting CKD', fontsize=15, fontweight='bold')
ax.legend(fontsize=13, loc='lower right'); ax.grid(alpha=0.3)
ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig3_roc_curve.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig3_roc_curve.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig3_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Fig4: Trend ====
print("Fig4: IAN Score vs. CKD Prevalence...")
fig, ax1 = plt.subplots(figsize=(12,7))
scores = list(range(0,7))
prevs = []; counts = []
for s in scores:
    ss = sub[sub['IAN']==s]
    counts.append(len(ss))
    prevs.append(ss['CKD'].mean()*100 if len(ss)>0 else 0)
color_scale = [GREEN,GREEN,ORANGE,ORANGE,ORANGE,RED,RED]
ax1.bar(scores, counts, color=color_scale, alpha=0.3, label='Sample Size', width=0.6)
ax1.set_xlabel('IAN Score', fontsize=14); ax1.set_ylabel('Sample Size', fontsize=14, color=GREY)
ax2 = ax1.twinx()
ax2.plot(scores, prevs, color=RED, marker='o', linewidth=2.5, markersize=10, label='CKD Prevalence')
for s,p in zip(scores, prevs):
    ax2.annotate(f'{p:.1f}%', (s,p), textcoords="offset points",
                 xytext=(0,12), ha='center', fontsize=12, fontweight='bold', color=RED)
ax2.set_ylabel('CKD Prevalence (%)', fontsize=14, color=RED)
ax2.set_ylim(0, max(prevs)*1.3+5)
bars = ax1.containers[0]
lines = [ax2.get_lines()[0]]
handles = list(bars) + lines
ax1.legend(handles, ['Sample Size', 'CKD Prevalence'], fontsize=13, loc='upper left')
ax1.set_title('Dose-Response: IAN Score and CKD Prevalence', fontsize=15, fontweight='bold')
ax1.set_xticks(range(0,7)); ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig4_ian_trend.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig4_ian_trend.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig4_ian_trend.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Fig5: Boxplots ====
print("Fig5: IAN Component Boxplots...")
fig, axes = plt.subplots(1,3,figsize=(18,6))
comp_data = [('NLR','NLR',[0,5]), ('HEMOGLOBIN','Hemoglobin (g/dL)',[8,18]), ('ALBUMIN','Albumin (g/dL)',[2,5.5])]
for ax,(col,label,ylim) in zip(axes,comp_data):
    d = sub[sub[col].notna()]
    bp_data = [d[d['CKD']==0][col].dropna(), d[d['CKD']==1][col].dropna()]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5, medianprops={'color':'white','linewidth':2})
    bp['boxes'][0].set_facecolor(BLUE); bp['boxes'][1].set_facecolor(RED)
    ax.set_xticklabels(['No CKD','CKD']); ax.set_ylabel(label, fontsize=13)
    ax.set_title(f'{label} by CKD Status', fontsize=14, fontweight='bold')
    if ylim: ax.set_ylim(ylim); ax.grid(axis='y', alpha=0.3)
    try:
        _, p = stats.ttest_ind(bp_data[0], bp_data[1])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        ax.text(0.5,0.95,f'P<{p:.4f} {sig}', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
    except: pass
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig5_boxplots.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig5_boxplots.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig5_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Fig6: Correlation heatmap ====
print("Fig6: Correlation Heatmap...")
corr_data = sub[['NLR','HEMOGLOBIN','ALBUMIN','IAN','eGFR','UACR']].dropna().corr()
fig, ax = plt.subplots(figsize=(10,9))
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label':'Correlation','shrink':0.8})
ax.set_title('Correlation: IAN Components and Kidney Function Markers', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig6_correlation_heatmap.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig6_correlation_heatmap.tiff"), dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig6_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==== Screening performance (IAN threshold) ====
print("\n--- Screening Performance (full analytic sample) ---")
for thresh, lbl in [(3, 'IAN>=3 (screening)'), (5, 'IAN>=5 (Youden)')]:
    pred = (sub['IAN'] >= thresh).astype(int)
    tp = ((pred==1)&(sub['CKD']==1)).sum()
    tn = ((pred==0)&(sub['CKD']==0)).sum()
    fp = ((pred==1)&(sub['CKD']==0)).sum()
    fn = ((pred==0)&(sub['CKD']==1)).sum()
    sens = tp/(tp+fn)*100; spec = tn/(tn+fp)*100
    ppv = tp/(tp+fp)*100 if (tp+fp)>0 else 0; npv = tn/(tn+fn)*100 if (tn+fn)>0 else 0
    print(f"  {lbl}: Sens={sens:.1f}%, Spec={spec:.1f}%, PPV={ppv:.1f}%, NPV={npv:.1f}%")

print(f"\nAll figures saved to: {CHART_DIR}")
print("complete!")
