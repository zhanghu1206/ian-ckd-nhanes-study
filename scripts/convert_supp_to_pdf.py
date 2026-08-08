#!/usr/bin/env python3
"""
Convert all supplementary figures to PDF format.
- Individual PDF per supplementary figure
- Compiled single PDF "supplementary_figures.pdf" with all figures
"""

import os
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(PROJ, "output", "charts")
OUT_DIR = os.path.join(PROJ, "ian_ckd_v2")

# All supplementary figures (fig_s*.png)
SUPP_FIGS = [
    "fig_s1_sensitivity_forest.png",
    "fig_s2_index_comparison_bar.png",
    "fig_s3_nri_idi.png",
    "fig_s4_two_threshold.png",
    "fig_s5_egfr_ian_scatter.png",
    "fig_s6_construction_comparison.png",
    "fig_s7_ian_stage_heatmap.png",
    "fig_s8_bootstrap_dist.png",
    "fig_s9_uacr_by_ian.png",
    "fig_s10_weighted_prevalence.png",
]

# Additional supplementary charts from other scripts
# ORDER MUST MATCH manuscript numbering (S11-S21)
EXTRA_SUPP = [
    "fig_ml_roc_comparison.png",      # S11
    "fig_ml_auc_comparison.png",      # S12
    "fig_shap_summary.png",           # S13
    "fig_shap_bar.png",               # S14
    "fig_temporal_roc.png",           # S15
    "fig_temporal_trend.png",         # S16
    "fig_temporal_forest.png",        # S17
    "fig_calibration.png",            # S18
    "fig_ckd_stage_by_ian.png",       # S19
    "fig_rcs_dose_response.png",      # S20
    "fig2_forest_ian_components.png", # S21
]

# Legends for supplementary figures (ORDER must match EXTRA_SUPP)
SUPP_LEGENDS = {
    "fig_s1_sensitivity_forest.png": "Supplemental Figure S1 — Sensitivity Analysis Forest Plot. IAN-CKD association across sensitivity analyses.",
    "fig_s2_index_comparison_bar.png": "Supplemental Figure S2 — Comparison of IAN with Existing Indices. Bar chart comparing AUC values.",
    "fig_s3_nri_idi.png": "Supplemental Figure S3 — Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI).",
    "fig_s4_two_threshold.png": "Supplemental Figure S4 — Two-Threshold Strategy. Screening vs. diagnostic thresholds for IAN score.",
    "fig_s5_egfr_ian_scatter.png": "Supplemental Figure S5 — eGFR vs. IAN Score. Scatter plot with regression line.",
    "fig_s6_construction_comparison.png": "Supplemental Figure S6 — IAN Construction Method Comparison. Different encoding strategies.",
    "fig_s7_ian_stage_heatmap.png": "Supplemental Figure S7 — IAN Score by CKD Stage. Heatmap of IAN distribution across CKD stages.",
    "fig_s8_bootstrap_dist.png": "Supplemental Figure S8 — Bootstrap Validation Distribution. AUC distribution from 1,000 bootstrap replicates.",
    "fig_s9_uacr_by_ian.png": "Supplemental Figure S9 — UACR Levels by IAN Score. Boxplot across IAN categories.",
    "fig_s10_weighted_prevalence.png": "Supplemental Figure S10 — Survey-Weighted CKD Prevalence by IAN Score.",
    "fig_ml_roc_comparison.png": "Supplemental Figure S11 — Machine Learning ROC Curves. Model discrimination comparison.",
    "fig_ml_auc_comparison.png": "Supplemental Figure S12 — Machine Learning AUC Comparison. Bar chart of AUC values.",
    "fig_shap_summary.png": "Supplemental Figure S13 — SHAP Summary Plot. Feature impact on XGBoost model output.",
    "fig_shap_bar.png": "Supplemental Figure S14 — SHAP Feature Importance. Top predictors from XGBoost model.",
    "fig_temporal_roc.png": "Supplemental Figure S15 — Temporal Validation ROC Curves. Training vs. validation discrimination.",
    "fig_temporal_trend.png": "Supplemental Figure S16 — Temporal Validation Trend. IAN-CKD association across cycles.",
    "fig_temporal_forest.png": "Supplemental Figure S17 — Temporal Validation Forest Plot. Odds ratios across sets.",
    "fig_calibration.png": "Supplemental Figure S18 — Calibration Curves. Predicted vs. observed CKD probability.",
    "fig_ckd_stage_by_ian.png": "Supplemental Figure S19 — CKD Stage Distribution by IAN Risk Grade.",
    "fig_rcs_dose_response.png": "Supplemental Figure S20 — B-spline Dose-Response Curve. Non-linear dose-response relationship.",
    "fig2_forest_ian_components.png": "Supplemental Figure S21 — Forest Plot of IAN Components. Univariate associations with CKD.",
}


def convert_to_pdf(img_path, pdf_path, legend=None):
    """Convert single PNG to PDF, optionally adding a legend page."""
    if not os.path.exists(img_path):
        print(f"  ⚠  File not found: {img_path}")
        return False

    img = Image.open(img_path)
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img.save(pdf_path, "PDF", resolution=300)
    print(f"  ✓ {os.path.basename(pdf_path)}")
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Convert each supplementary figure to individual PDF
    pdf_dir = os.path.join(OUT_DIR, "supplementary_pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    individual_pdfs = []
    all_images_rgb = []

    print("Converting supplementary figures to PDF:")
    print("─" * 50)

    # Process supp figs (S1-S10)
    for fname in SUPP_FIGS:
        img_path = os.path.join(CHARTS, fname)
        pdf_name = fname.replace('.png', '.pdf')
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if convert_to_pdf(img_path, pdf_path):
            individual_pdfs.append(pdf_path)
            # Collect for compiled PDF
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            all_images_rgb.append(img)

    # Process extra supp figs (S11-S21)
    for fname in EXTRA_SUPP:
        img_path = os.path.join(CHARTS, fname)
        pdf_name = fname.replace('.png', '.pdf')
        pdf_path = os.path.join(pdf_dir, pdf_name)
        if convert_to_pdf(img_path, pdf_path):
            individual_pdfs.append(pdf_path)
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            all_images_rgb.append(img)

    # 2. Create compiled PDF with all supplementary figures
    if all_images_rgb:
        compiled_path = os.path.join(OUT_DIR, "supplementary_figures.pdf")
        # Save first image and append the rest
        all_images_rgb[0].save(
            compiled_path,
            "PDF",
            resolution=300,
            save_all=True,
            append_images=all_images_rgb[1:]
        )
        print(f"\n  ✓ Compiled PDF: {os.path.relpath(compiled_path, OUT_DIR)}")

    print(f"\nDone! {len(individual_pdfs)} individual PDFs in: {pdf_dir}")


if __name__ == "__main__":
    main()
