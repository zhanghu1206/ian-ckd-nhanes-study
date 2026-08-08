#!/usr/bin/env python3
"""
Quickly regenerate main manuscript figures (Figures 1-6) with English labels only.
Loads pre-processed data from output/nhanes_ckd_merged.csv
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 14})

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "output", "nhanes_ckd_merged.csv")
CHARTS = os.path.join(PROJ, "output", "charts")
os.makedirs(CHARTS, exist_ok=True)

BLUE = '#0F4D92'; ORANGE = '#E8832E'; GREEN = '#8BCF8B'; RED = '#B64342'; GREY = '#767676'

print("Loading pre-processed data...")
df = pd.read_csv(DATA, low_memory=False)
print(f"  {df.shape[0]} rows, {df.shape[1]} cols")

# Filter adults
sub = df[df['RIDAGEYR'] >= 20].copy()

# Add derived columns
sub['HYPERTENSION'] = (sub['BPQ020'] == 1).astype(int)

# Rebuild IAN scores using stored tertile cutoffs
valid = sub[['NLR', 'HEMOGLOBIN', 'ALBUMIN']].dropna()
_, nlr_bins = pd.qcut(valid['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
_, hb_bins = pd.qcut(valid['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
_, alb_bins = pd.qcut(valid['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')

sub['NLR_T'] = pd.cut(sub['NLR'], bins=nlr_bins, labels=[0,1,2], include_lowest=True).astype(float)
sub['HB_T'] = pd.cut(sub['HEMOGLOBIN'], bins=hb_bins, labels=[2,1,0], include_lowest=True).astype(float)
sub['ALB_T'] = pd.cut(sub['ALBUMIN'], bins=alb_bins, labels=[2,1,0], include_lowest=True).astype(float)
sub['IAN'] = sub['NLR_T'] + sub['HB_T'] + sub['ALB_T']
sub['IAN_GRADE'] = pd.cut(sub['IAN'], bins=[-1,2,4,6], labels=['Low (0-2)', 'Medium (3-4)', 'High (5-6)'])

# ── Figure 2: IAN Distribution ──
print("Figure 2: IAN Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ian_vals = sub[sub['IAN'].notna()]
bins = np.arange(-0.5, 7, 1)
ax = axes[0]
for ckd_val, color, label in [(0, BLUE, 'No CKD'), (1, RED, 'CKD')]:
    data = ian_vals[ian_vals['CKD'] == ckd_val]['IAN']
    ax.hist(data, bins=bins, alpha=0.7, color=color, label=label, density=True)
ax.set_xlabel('IAN Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('IAN Score Distribution by CKD Status', fontsize=16, fontweight='bold')
ax.set_xticks(range(0, 7))
ax.legend(fontsize=13)
ax.grid(alpha=0.3)

ax = axes[1]
grades = ['Low (0-2)', 'Medium (3-4)', 'High (5-6)']
ckd_rates = []
for g in grades:
    s = ian_vals[ian_vals['IAN_GRADE'] == g]
    ckd_rates.append(s['CKD'].mean() * 100 if len(s) > 0 else 0)
bars = ax.bar(grades, ckd_rates, color=[GREEN, ORANGE, RED], width=0.6, edgecolor='white')
for bar, val in zip(bars, ckd_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_xlabel('IAN Risk Grade', fontsize=12)
ax.set_ylabel('CKD Prevalence (%)', fontsize=12)
ax.set_title('CKD Prevalence by IAN Risk Grade', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(ckd_rates) * 1.2 + 2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "fig1_ian_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── Figure 3: ROC Curves ──
from sklearn.metrics import roc_curve, roc_auc_score
print("Figure 3: ROC Curves...")
fig, ax = plt.subplots(figsize=(7, 6))
# Model 1: IAN only
d = sub[['IAN', 'CKD']].dropna()
fpr1, tpr1, _ = roc_curve(d['CKD'], d['IAN'])
auc1 = roc_auc_score(d['CKD'], d['IAN'])
ax.plot(fpr1, tpr1, label=f'IAN alone (AUC = {auc1:.3f})', color=BLUE, linewidth=2)
# Model 3: IAN + Age + Sex + HTN + DM + BMI
feat = ['IAN', 'RIDAGEYR', 'RIAGENDR', 'HYPERTENSION', 'DIABETES', 'BMXBMI']
d3 = sub[feat + ['CKD']].dropna()
from sklearn.linear_model import LogisticRegression
m3 = LogisticRegression(max_iter=1000)
m3.fit(d3[feat], d3['CKD'])
prob3 = m3.predict_proba(d3[feat])[:, 1]
fpr3, tpr3, _ = roc_curve(d3['CKD'], prob3)
auc3 = roc_auc_score(d3['CKD'], prob3)
ax.plot(fpr3, tpr3, label=f'Full model (AUC = {auc3:.3f})', color=RED, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
ax.set_title('ROC Curves: IAN Score Predicting CKD', fontsize=16, fontweight='bold')
ax.legend(fontsize=13, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "fig3_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── Figure 4: Dose-Response ──
print("Figure 4: Dose-Response...")
ian_prev = [(s, sub[sub['IAN'] == s]['CKD'].mean() * 100, len(sub[sub['IAN'] == s]))
            for s in range(0, 7) if s in sub['IAN'].values]
scores = [x[0] for x in ian_prev]
prevs = [x[1] for x in ian_prev]
counts = [x[2] for x in ian_prev]
color_scale = [GREEN, GREEN, ORANGE, ORANGE, ORANGE, RED, RED]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(scores, counts, color=color_scale, alpha=0.3, label='Sample Size', width=0.6)
ax1.set_xlabel('IAN Score', fontsize=12)
ax1.set_ylabel('Sample Size', fontsize=12, color=GREY)
ax2 = ax1.twinx()
ax2.plot(scores, prevs, color=RED, marker='o', linewidth=2.5, markersize=10, label='CKD Prevalence')
for s, p in zip(scores, prevs):
    ax2.annotate(f'{p:.1f}%', (s, p), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=12, fontweight='bold', color=RED)
ax2.set_ylabel('CKD Prevalence (%)', fontsize=12, color=RED)
ax2.set_ylim(0, max(prevs) * 1.3 + 5)
bars = [ax1.containers[0]]
lines = [ax2.get_lines()[0]]
labels = ['Sample Size', 'CKD Prevalence']
ax1.legend(bars + lines, labels, fontsize=13, loc='upper left')
ax1.set_title('Dose-Response: IAN Score and CKD Prevalence', fontsize=16, fontweight='bold')
ax1.set_xticks(range(0, 7))
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "fig4_ian_trend.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── Figure 5: Boxplots ──
print("Figure 5: Boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
components_data = [
    ('NLR', 'NLR', [0, 5]),
    ('HEMOGLOBIN', 'Hemoglobin (g/dL)', [8, 18]),
    ('ALBUMIN', 'Albumin (g/dL)', [2, 5.5]),
]
for ax, (col, label, ylim) in zip(axes, components_data):
    data = sub[sub[col].notna()]
    bp_data = [data[data['CKD'] == 0][col].dropna(), data[data['CKD'] == 1][col].dropna()]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'white', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(BLUE)
    bp['boxes'][1].set_facecolor(RED)
    ax.set_xticklabels(['No CKD', 'CKD'])
    ax.set_ylabel(label, fontsize=13)
    ax.set_title(f'{label} by CKD Status', fontsize=12, fontweight='bold')
    if ylim: ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3)
    try:
        _, p = stats.ttest_ind(bp_data[0], bp_data[1])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.5, 0.95, f'P={p:.4f} {sig}', transform=ax.transAxes,
                ha='center', fontsize=12, fontweight='bold')
    except: pass
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "fig5_boxplots.png"), dpi=300, bbox_inches='tight')
plt.close()

# ── Figure 6: Correlation Heatmap ──
print("Figure 6: Correlation Heatmap...")
corr_data = sub[['NLR', 'HEMOGLOBIN', 'ALBUMIN', 'IAN', 'eGFR', 'UACR']].dropna().corr()
fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
ax.set_title('Correlation: IAN Components and Kidney Function Markers', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(CHARTS, "fig6_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n√ All figures regenerated successfully with English labels!")
