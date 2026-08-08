#!/usr/bin/env python3
"""Generate Graphical Abstract for IAN-CKD paper (BMC Nephrology)"""

import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nature_config import BLUE, RED, GREEN, ORANGE, PURPLE, GREY

rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 7.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

# Colors
C1 = BLUE
C2 = ORANGE
C3 = GREEN
C4 = RED
C5 = PURPLE
C6 = GREY

def box(x, y, w, h, text, fc="white", ec=C1, fs=10, bold=True, ha="center"):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                                   facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color="black")

def arrow(x1, y1, x2, y2, color="black", lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

# ===== Title =====
ax.text(8, 6.6, "IAN Score: A Novel Inflammation-Adjusted Nutrition Score for CKD Risk Stratification",
        ha="center", va="center", fontsize=16, fontweight="bold", color=C1)

# ===== Section 1: DATA =====
box(3, 5.8, 5, 0.7, "NHANES 2011-2018\nN = 20,222 Adults (≥20 yrs)", fc="#D6E4F0", ec=C1, fs=9)
arrow(5.5, 5.8, 7, 5.8)

# ===== Section 2: IAN SCORE =====
box(9.5, 5.8, 5, 0.7, "Training: 2011-2016 (n=15,311)\nTemporal Validation: 2017-2018 (n=4,911)", fc="#D6E4F0", ec=C1, fs=9)

arrow(9.5, 5.45, 9.5, 4.8)

# IAN components
box(3.5, 4.5, 2.5, 0.6, "NLR\n(Neutrophil/\nLymphocyte)", fc="white", ec=C4, fs=8)
box(6.5, 4.5, 2.5, 0.6, "Hemoglobin\n(Hb, g/dL)", fc="white", ec=C5, fs=8)
box(9.5, 4.5, 2.5, 0.6, "Albumin\n(g/dL)", fc="white", ec=C2, fs=8)
box(12.5, 4.5, 2.5, 0.6, "IAN Score\n= NLR_T+Hb_T+Alb_T\nRange: 0-6", fc="#E2EFDA", ec=C3, fs=8)

arrow(4.75, 4.5, 5.5, 4.5)
arrow(7.75, 4.5, 8.5, 4.5)
arrow(10.75, 4.5, 11.5, 4.5)

# Encoding
box(3.5, 3.4, 3, 0.5, "Tertile: 0/1/2", fc="white", ec=C6, fs=8, bold=False)
box(6.5, 3.4, 3, 0.5, "Reverse Tertile: 2/1/0", fc="white", ec=C6, fs=8, bold=False)
box(9.5, 3.4, 3, 0.5, "Reverse Tertile: 2/1/0", fc="white", ec=C6, fs=8, bold=False)

arrow(3.5, 3.65, 3.5, 4.2, lw=1)
arrow(6.5, 3.65, 6.5, 4.2, lw=1)
arrow(9.5, 3.65, 9.5, 4.2, lw=1)

# ===== Arrow down to results =====
arrow(9.5, 3.15, 9.5, 2.45)

# ===== Section 3: KEY RESULTS =====
box(9.5, 2.2, 10, 0.5, "Key Results", fc="#FDE9D9", ec=C2, fs=10)

# Result boxes — V37 style, title above box, detail inside
results_y = 1.0
res_w = 2.3
res_h = 0.55

# Adjusted x positions with more rightward spacing for AUC-containing boxes
result_data = [
    (1.6, "Dose-Response", "CKD: 7.7% (IAN=0)\n→ 38.7% (IAN=6)", C4),
    (4.6, "Logistic Regression", "OR = 1.40 per point\nAdjusted OR = 1.24", C1),
    (7.6, "Temporal Validation", "Train AUC=0.641\nValid AUC=0.624", C5),
    (10.7, "vs Other Indices", "IAN 0.641 > PNI 0.618\n> NLR 0.601 > Hb 0.598", C3),
    (13.7, "Clinical Impact", "NRI = 0.27\nE-value = 1.79", C2),
]

for x, title, detail, color in result_data:
    box(x, results_y, res_w, res_h, detail, fc="white", ec=color, fs=7.5, bold=False)
    # Title above box — offset proportional to box size to stay visible
    ax.text(x, results_y + 0.38, title, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color)

# ===== Bottom: Risk Grades =====
ax.text(1.5, 0.15, "Clinical Risk Grades:", ha="center", va="center", fontsize=9, fontweight="bold")

for i, (grade, color, prev) in enumerate([("Low (0-2)", C3, "11.0%"), 
                                           ("Medium (3-4)", C2, "18.7%"), 
                                           ("High (5-6)", C4, "30.0%")]):
    x_pos = 5 + i * 3.5
    rect = patches.FancyBboxPatch((x_pos-1.5, 0.08), 3, 0.25, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor=color, linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x_pos, 0.205, f"{grade}: CKD {prev}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

# ===== Conclusion =====
ax.text(8, -0.15, "Conclusion: The IAN score independently predicts CKD, is stable in temporal validation, and uses low-cost routine labs.",
        ha="center", va="center", fontsize=10, fontweight="bold", color=C1,
        style="italic")

plt.tight_layout()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHART_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "output", "charts")
os.makedirs(_CHART_DIR, exist_ok=True)
fig.savefig(os.path.join(_CHART_DIR, "graphical_abstract.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(_CHART_DIR, "graphical_abstract.tiff"), dpi=600, bbox_inches="tight")
plt.close()
print("Graphical Abstract saved!")
