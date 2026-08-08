#!/usr/bin/env python3
"""
IAN-CKD machine learning comparison analysis
compare models: Logistic Regression vs Random Forest vs XGBoost vs Stacking
temporal split validation: train=G+H+I(2011-2016), validation=J(2017-2018)
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap

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

print("="*70)
print("IAN-CKD machine learning comparison analysis")
print("="*70)

# ============ 0. data preparation ============
DATA_PATH = os.path.join(OUTPUT_DIR, "nhanes_ckd_merged.csv")
df = pd.read_csv(DATA_PATH, low_memory=False)
adult = df[df['RIDAGEYR'] >= 20].copy()

cols = ['NLR','HEMOGLOBIN','ALBUMIN','CYCLE','CKD','RIAGENDR','RIDAGEYR',
        'RIDRETH1','BMXBMI','eGFR','UACR','DIABETES','BPQ020','WTMEC2YR']
cols = [c for c in cols if c in adult.columns]
data = adult[cols].dropna(subset=['NLR','HEMOGLOBIN','ALBUMIN']).copy()

# IANtertile (based on the training set)
train_idx = data['CYCLE'].isin(['G','H','I'])
train_data = data[train_idx].copy()
valid_data = data[~train_idx].copy()

_, nlr_b = pd.qcut(train_data['NLR'], q=3, labels=False, retbins=True, duplicates='drop')
_, hb_b = pd.qcut(train_data['HEMOGLOBIN'], q=3, labels=False, retbins=True, duplicates='drop')
_, alb_b = pd.qcut(train_data['ALBUMIN'], q=3, labels=False, retbins=True, duplicates='drop')

def assign_ian(d):
    d = d.copy()
    d['NLR_T'] = pd.cut(d['NLR'], bins=nlr_b, labels=[0,1,2], include_lowest=True).astype(float)
    d['HB_T'] = pd.cut(d['HEMOGLOBIN'], bins=hb_b, labels=[2,1,0], include_lowest=True).astype(float)
    d['ALB_T'] = pd.cut(d['ALBUMIN'], bins=alb_b, labels=[2,1,0], include_lowest=True).astype(float)
    d['IAN'] = d['NLR_T'] + d['HB_T'] + d['ALB_T']
    d['HYPERTENSION'] = (d['BPQ020'] == 1).astype(int)
    return d

data = assign_ian(data)
train_data = assign_ian(train_data)
valid_data = assign_ian(valid_data)

print(f"training set: {len(train_data):,}, CKD={train_data['CKD'].mean()*100:.1f}%")
print(f"validation set: {len(valid_data):,}, CKD={valid_data['CKD'].mean()*100:.1f}%")

# ============ 1. feature-set definition ============
feature_sets = {
    'IAN_only': ['IAN'],
    'IAN_comp': ['NLR_T','HB_T','ALB_T'],
    'IAN_raw': ['NLR','HEMOGLOBIN','ALBUMIN'],
    'demographics': ['RIDAGEYR','RIAGENDR'],
}
feature_sets['IAN_demo'] = feature_sets['IAN_only'] + feature_sets['demographics']
feature_sets['full_model'] = (feature_sets['IAN_only'] + feature_sets['demographics'] +
                               ['BMXBMI','DIABETES','HYPERTENSION'])
feature_sets['ml_full'] = (['NLR','HEMOGLOBIN','ALBUMIN','RIDAGEYR','RIAGENDR',
                             'BMXBMI','DIABETES','HYPERTENSION'])

# ============ 2. model training function ============
def train_evaluate(X_train, y_train, X_valid, y_valid, model, model_name,
                   feature_names=None):
    """train and evaluate the model"""
    model.fit(X_train, y_train)
    
    # prediction
    if hasattr(model, 'predict_proba'):
        yp_train = model.predict_proba(X_train)[:, 1]
        yp_valid = model.predict_proba(X_valid)[:, 1]
    else:
        yp_train = model.predict(X_train)
        yp_valid = model.predict(X_valid)
    
    # AUC
    auc_train = roc_auc_score(y_train, yp_train)
    auc_valid = roc_auc_score(y_valid, yp_valid)
    
    # FPR/TPR for ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, yp_train)
    fpr_valid, tpr_valid, _ = roc_curve(y_valid, yp_valid)
    
    # PR-AUC
    pr_auc_train = average_precision_score(y_train, yp_train)
    pr_auc_valid = average_precision_score(y_valid, yp_valid)
    
    gap = auc_train - auc_valid
    
    # optimal threshold (Youden)
    youden = tpr_valid - fpr_valid
    best_idx = np.argmax(youden)
    best_thr = _[best_idx]
    sens = tpr_valid[best_idx]
    spec = 1 - fpr_valid[best_idx]
    
    print(f"  {model_name:>25s}: Train AUC={auc_train:.4f}, Valid AUC={auc_valid:.4f}, "
          f"Gap={gap:.4f}, PR-AUC={pr_auc_valid:.4f}, "
          f"Sens={sens:.3f}, Spec={spec:.3f}")
    
    return {
        'model': model,
        'name': model_name,
        'auc_train': auc_train,
        'auc_valid': auc_valid,
        'gap': gap,
        'pr_auc_train': pr_auc_train,
        'pr_auc_valid': pr_auc_valid,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train,
        'fpr_valid': fpr_valid,
        'tpr_valid': tpr_valid,
        'best_thr': best_thr,
        'sens': sens,
        'spec': spec,
        'yp_valid': yp_valid,
    }

# ============ 3. prepare feature matrix ============
def prep_data(data, features):
    d = data[['CKD'] + features].dropna()
    X = d[features].astype(float).values
    y = d['CKD'].astype(float).values
    return X, y, d.index

# train/verify feature matrix
X_train_ml, y_train_ml, _ = prep_data(train_data, feature_sets['ml_full'])
X_valid_ml, y_valid_ml, _ = prep_data(valid_data, feature_sets['ml_full'])

# standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ml)
X_valid_scaled = scaler.transform(X_valid_ml)

print(f"\nMLfeature matrix: train={X_train_ml.shape}, validation={X_valid_ml.shape}")
print(f"feature: {feature_sets['ml_full']}")

# ============ 4. model definition and training ============
print(f"\n{'='*70}")
print("4. model training and comparison")
print('='*70)

results = []

# --- 4a. baseline: IANsimpleLogistic ---
X_tr, y_tr, _ = prep_data(train_data, ['IAN'])
X_va, y_va, _ = prep_data(valid_data, ['IAN'])
lr_ian = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
res = train_evaluate(X_tr, y_tr, X_va, y_va, lr_ian, 'IAN (Logistic)')
results.append(res)

# --- 4b. IANcomponent (NLR_T, HB_T, ALB_T) Logistic ---
X_tr, y_tr, _ = prep_data(train_data, ['NLR_T','HB_T','ALB_T'])
X_va, y_va, _ = prep_data(valid_data, ['NLR_T','HB_T','ALB_T'])
lr_comp = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
res = train_evaluate(X_tr, y_tr, X_va, y_va, lr_comp, 'IAN comp (Logistic)')
results.append(res)

# --- 4c. IAN + age + sex Logistic ---
X_tr, y_tr, _ = prep_data(train_data, ['IAN','RIDAGEYR','RIAGENDR'])
X_va, y_va, _ = prep_data(valid_data, ['IAN','RIDAGEYR','RIAGENDR'])
lr_ian_demo = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
res = train_evaluate(X_tr, y_tr, X_va, y_va, lr_ian_demo, 'IAN+Demo (Logistic)')
results.append(res)

# --- 4d. raw variables Logistic (noIAN) ---
X_tr_scaled = scaler.fit_transform(X_tr)
# Use the ml_full features for raw logistic
X_tr_log, y_tr_log, _ = prep_data(train_data, feature_sets['ml_full'])
X_va_log, y_va_log, _ = prep_data(valid_data, feature_sets['ml_full'])
scaler_log = StandardScaler()
X_tr_log_s = scaler_log.fit_transform(X_tr_log)
X_va_log_s = scaler_log.transform(X_va_log)
lr_full = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced')
res = train_evaluate(X_tr_log_s, y_tr_log, X_va_log_s, y_va_log, lr_full, 'Full LR (balanced)')
results.append(res)

# --- 4e. raw variables L1-regularization Logistic ---
lr_l1 = LogisticRegression(C=1.0, penalty='l1', solver='saga', max_iter=1000)
res = train_evaluate(X_tr_log_s, y_tr_log, X_va_log_s, y_va_log, lr_l1, 'Full LR (L1)')
# L1coefficient
coefs = dict(zip(feature_sets['ml_full'], lr_l1.coef_[0]))
print(f"  L1coefficient: {coefs}")
results.append(res)

# --- 4f. Random Forest ---
print("\n  Random Foresthyperparameter tuning...")
rf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [10, 30, 50],
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=2)
rf_gs = GridSearchCV(rf, rf_grid, cv=3, scoring='roc_auc', n_jobs=2, verbose=0)
rf_gs.fit(X_train_ml, y_train_ml)
print(f"  RFbest parameters: {rf_gs.best_params_}, CV AUC={rf_gs.best_score_:.4f}")
res = train_evaluate(X_train_ml, y_train_ml, X_valid_ml, y_valid_ml,
                     rf_gs.best_estimator_, 'Random Forest')
results.append(res)

# RFfeature importance
rf_imp = pd.DataFrame({
    'feature': feature_sets['ml_full'],
    'importance': rf_gs.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)
print(f"  RFfeature importance:\n{rf_imp.to_string(index=False)}")

# --- 4g. XGBoost ---
print("\n  XGBoosthyperparameter tuning...")
xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=2)
xgb_gs = GridSearchCV(xgb_model, xgb_grid, cv=3, scoring='roc_auc', n_jobs=2, verbose=0)

# XGBneeds to handle missing values
X_train_xgb = X_train_ml.copy()
y_train_xgb = y_train_ml.copy()
X_valid_xgb = X_valid_ml.copy()
y_valid_xgb = y_valid_ml.copy()

xgb_gs.fit(X_train_xgb, y_train_xgb)
print(f"  XGBbest parameters: {xgb_gs.best_params_}, CV AUC={xgb_gs.best_score_:.4f}")
res = train_evaluate(X_train_xgb, y_train_xgb, X_valid_xgb, y_valid_xgb,
                     xgb_gs.best_estimator_, 'XGBoost')
results.append(res)

# XGBfeature importance
xgb_imp = pd.DataFrame({
    'feature': feature_sets['ml_full'],
    'importance': xgb_gs.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)
print(f"  XGBfeature importance:\n{xgb_imp.to_string(index=False)}")

# --- 4h. Stacking (LR + RF + XGB) ---
print("\n  Stacking (LR+RF+XGB)...")
base_models = [
    ('lr', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=30, random_state=42,
                                   class_weight='balanced', n_jobs=2)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5,
                               learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42,
                               eval_metric='logloss', n_jobs=2)),
]
# note: Stackingfeed standardized data toLRuse,raw forRF/XGBuse
# simple approach:use raw data(RFandXGBinsensitive)
stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=1.0, class_weight='balanced'),
    cv=3,
    n_jobs=2
)
res = train_evaluate(X_train_ml, y_train_ml, X_valid_ml, y_valid_ml,
                     stack, 'Stacking (LR+RF+XGB)')
results.append(res)

# ============ 5. ROC curve comparison ============
print(f"\n{'='*70}")
print("5. ROC curve comparison")
print('='*70)

fig, ax = plt.subplots(figsize=(10, 8))
colors_ml = [RED, ORANGE, PURPLE, GREEN, BROWN := '#8B4513', BLUE, '#FF69B4', '#00CED1']

# by validation setAUCsort
results_sorted = sorted(results, key=lambda r: r['auc_valid'], reverse=True)

for i, res in enumerate(results_sorted):
    c = colors_ml[i % len(colors_ml)]
    ls = '-' if 'Train' not in res['name'] else '--'
    ax.plot(res['fpr_valid'], res['tpr_valid'], color=c, linewidth=2,
            label=f"{res['name']} (AUC={res['auc_valid']:.3f})")
    # training set shown with dashed line
    ax.plot(res['fpr_train'], res['tpr_train'], color=c, linewidth=1.2, linestyle=':',
            alpha=0.4)

ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random (AUC=0.5)')
ax.set_xlabel('1 - Specificity (FPR)', fontsize=12)
ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
ax.set_title('ML Models: IAN Components Predicting CKD\n(Temporal Validation)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_ml_roc_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_ml_roc_comparison.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_ml_roc_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ ML ROC curve saved")

# ============ 6. AUCcomparison bar chart ============
print("\n6. AUCcomparison bar chart")
fig, ax = plt.subplots(figsize=(16, 8))

names = [r['name'] for r in results]
auc_t = [r['auc_train'] for r in results]
auc_v = [r['auc_valid'] for r in results]
gaps = [r['gap'] for r in results]

x = np.arange(len(names))
w = 0.3
bars_t = ax.bar(x - w/2, auc_t, w, color=BLUE, alpha=0.7, label='Train AUC')
bars_v = ax.bar(x + w/2, auc_v, w, color=RED, alpha=0.7, label='Valid AUC')

# annotate values
for i, (t, v) in enumerate(zip(auc_t, auc_v)):
    ax.text(i - w/2, t + 0.025, f'{t:.3f}', ha='center', va='bottom', fontsize=11,
            color=BLUE, fontweight='bold')
    ax.text(i + w/2, v + 0.025, f'{v:.3f}', ha='center', va='bottom', fontsize=11,
            color=RED, fontweight='bold')
    # gap
    gap_y = max(t, v) + 0.055
    ax.annotate(f'Δ={gaps[i]:+.3f}', xy=(i, max(t,v)), xytext=(i, gap_y),
                ha='center', fontsize=9, color='dimgrey',
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.8))

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=13, rotation=25, ha='right')
ax.set_ylabel('AUC', fontsize=16)
ax.set_title('Model Performance Comparison: Train vs Temporal Validation', fontsize=18, fontweight='bold')
ax.legend(fontsize=13); ax.set_ylim(0.3, 0.98); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_ml_auc_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_ml_auc_comparison.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_ml_auc_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ ML AUC comparison bar chart saved")

# ============ 7. SHAPanalysis (best model) ============
print(f"\n{'='*70}")
print("7. SHAPinterpretability analysis (XGBoost)")
print('='*70)

best_model = xgb_gs.best_estimator_

# SHAPanalysis
X_shap = pd.DataFrame(X_train_ml, columns=feature_sets['ml_full'])
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)

# SHAPsummary plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, show=False, max_display=10)
plt.title('SHAP Feature Importance: XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_shap_summary.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_shap_summary.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ SHAP summary plot saved")

# SHAPbar chart
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False, max_display=10)
plt.title('SHAP Feature Importance (Bar)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "fig_shap_bar.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(CHART_DIR, "fig_shap_bar.tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(CHART_DIR, "fig_shap_bar.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ SHAP bar plot saved")

# SHAPdependence plot - most critical2features
top_features = np.argsort(np.abs(shap_values).mean(0))[-3:]
top_feat_names = [feature_sets['ml_full'][i] for i in top_features]
print(f"  Top 3 features by SHAP: {top_feat_names}")

# ============ 8. results summary table ============
print(f"\n{'='*70}")
print("8. results summary")
print('='*70)

print(f"\n{'Model':>25s} | {'Train AUC':>10s} | {'Valid AUC':>10s} | {'Gap':>6s} | {'PR-AUC':>7s}")
print('-'*70)

summary = []
for res in results_sorted:
    print(f"{res['name']:>25s} | {res['auc_train']:.4f}    | {res['auc_valid']:.4f}    | "
          f"{res['gap']:.3f}  | {res['pr_auc_valid']:.3f}")
    summary.append({
        'Model': res['name'],
        'Train_AUC': f"{res['auc_train']:.4f}",
        'Valid_AUC': f"{res['auc_valid']:.4f}",
        'Gap': f"{res['gap']:.4f}",
        'PR_AUC_Valid': f"{res['pr_auc_valid']:.3f}",
        'Sens': f"{res['sens']:.3f}",
        'Spec': f"{res['spec']:.3f}",
    })

# save summary
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTPUT_DIR, "ml_model_comparison.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nResults saved: {summary_path}")

# ============ 9. IAN scoreVS MLmodel comparison conclusions ============
print(f"\n{'='*70}")
print("9. key conclusions")
print('='*70)

# bestMLmodel
best_ml = results_sorted[0]
# IANsimple model
ian_simple = [r for r in results if r['name'] == 'IAN (Logistic)'][0]

print(f"\n1. best model: {best_ml['name']} (Valid AUC={best_ml['auc_valid']:.4f})")
print(f"2. IANsimple model: Valid AUC={ian_simple['auc_valid']:.4f}")
print(f"3. MLimprove: ΔAUC={best_ml['auc_valid'] - ian_simple['auc_valid']:.4f}")
print(f"4. IAN+Demo: Valid AUC={[r for r in results if 'IAN+Demo' in r['name']][0]['auc_valid']:.4f}")

print(f"\n{'='*70}")
print("MLanalysis completed!")
print('='*70)
