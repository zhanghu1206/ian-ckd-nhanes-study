"""Weight justification for IAN components"""
import os
import pandas as pd, numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

_MERGED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "nhanes_ckd_merged.csv")
df = pd.read_csv(_MERGED, low_memory=False)
adult = df[df["RIDAGEYR"]>=20].copy()
train = adult[adult["CYCLE"].isin(["G","H","I"])].copy()

_, nb = pd.qcut(train["NLR"].dropna(), q=3, labels=False, retbins=True, duplicates="drop")
_, hb = pd.qcut(train["HEMOGLOBIN"].dropna(), q=3, labels=False, retbins=True, duplicates="drop")
_, ab = pd.qcut(train["ALBUMIN"].dropna(), q=3, labels=False, retbins=True, duplicates="drop")

print("="*70)
print("RATIONALE FOR IAN COMPONENT WEIGHTING")
print("="*70)

print("\n--- Step 1: Literature Basis ---")
print("""
Existing composite scores use various weighting strategies:
- CONUT (Controlling Nutritional Status): Albumin(0-6)+Lymphocyte(0-3)+Cholesterol(0-3)
  → Unequal max scores but conceptually each component contributes one domain
- PNI (Prognostic Nutritional Index): Albumin(g/L) + 5 × Lymphocyte(10^9/L)
  → Puts both on similar numerical scale (Alb ~35-55, 5×Lym ~5-15)
- GNRI (Geriatric Nutritional Risk Index): 14.89×Alb + 41.7×(weight/ideal)
  → Weighted by regression-derived coefficients
- ALI (Advanced Lung Cancer Inflammation Index): BMI×Albumin/NLR
  → Multiplicative, no explicit component weights

For IAN, we chose equal weighting (1:1:1) based on:
  1. Each component represents a distinct biological pathway
  2. Tertile encoding (0/1/2) normalizes each to the same scale
  3. Simplicity and clinical interpretability
  4. Equal weighting is the standard approach in the literature for tertile-based
     composite scores (e.g., the Dietary Inflammatory Index, HEI components)
""")

print("--- Step 2: Empirical Validation ---")
data = adult.copy()

# Assign tertiles
data["NLR_T"] = pd.cut(data["NLR"], bins=nb, labels=[0,1,2], include_lowest=True).astype(float)
data["HB_T"] = pd.cut(data["HEMOGLOBIN"], bins=hb, labels=[2,1,0], include_lowest=True).astype(float)
data["ALB_T"] = pd.cut(data["ALBUMIN"], bins=ab, labels=[2,1,0], include_lowest=True).astype(float)
data = data.dropna(subset=["NLR_T","HB_T","ALB_T"])

results = []
for wn in [0.5, 1.0, 1.5, 2.0]:
    for wh in [0.5, 1.0, 1.5, 2.0]:
        for wa in [0.5, 1.0, 1.5, 2.0]:
            ian = data["NLR_T"]*wn + data["HB_T"]*wh + data["ALB_T"]*wa
            try:
                X = sm.add_constant(ian)
                y = data["CKD"].astype(float)
                m = sm.Logit(y,X).fit(disp=0)
                auc = roc_auc_score(y, m.predict(X))
                results.append((wn, wh, wa, auc))
            except:
                pass

results_df = pd.DataFrame(results, columns=["w_NLR","w_Hb","w_Alb","AUC"])
best_eq = results_df[(results_df.w_NLR==1)&(results_df.w_Hb==1)&(results_df.w_Alb==1)]
best_all = results_df.loc[results_df.AUC.idxmax()]

print(f"Equal-weight (1:1:1) AUC  = {best_eq.AUC.values[0]:.4f}")
print(f"Optimal weight ({best_all.w_NLR:.1f}:{best_all.w_Hb:.1f}:{best_all.w_Alb:.1f}) AUC = {best_all.AUC:.4f}")
print(f"ΔAUC = {best_all.AUC - best_eq.AUC.values[0]:.4f}")

print("\nTop 10 weighting schemes:")
for _, r in results_df.sort_values("AUC", ascending=False).head(10).iterrows():
    print(f"  NLR*{r.w_NLR:.1f} + Hb*{r.w_Hb:.1f} + Alb*{r.w_Alb:.1f} = AUC={r.AUC:.4f}")

print("\n--- Step 3: Rationale for Tertile Encoding ---")
print("""
Why tertiles rather than raw values?
  1. Tertiles handle non-linear relationships automatically
  2. Each component's contribution is bounded (0-2), preventing
     extreme values from dominating
  3. Ordinal scoring (0,1,2) is more robust to outliers than continuous
  4. Reverse encoding of Hb and Albumin (higher tertile = lower score)
     ensures all three point in the same direction (higher = worse)

Why reverse Hb and Albumin?
  - For NLR: higher values indicate more inflammation (= worse)
  - For Hb: lower values indicate anemia (= worse) → reverse encoding
  - For Albumin: lower values indicate malnutrition/ inflammation (= worse) → reverse encoding
""")

print("--- Step 4: Final Definition ---")
print("""
IAN = NLR_T + HB_T + ALB_T

where:
  NLR_T ∈ {0,1,2} (tertile 1→0, tertile 2→1, tertile 3→2)
  HB_T  ∈ {2,1,0} (tertile 1→2, tertile 2→1, tertile 3→0) [reverse]
  ALB_T ∈ {2,1,0} (tertile 1→2, tertile 2→1, tertile 3→0) [reverse]

IAN range: 0-6
IAN grades: Low (0-2), Medium (3-4), High (5-6)
""")
print("="*70)
print("Done.")
