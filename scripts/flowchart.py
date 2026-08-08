"""Generate study flow chart — SCI-ready (English only, no Chinese)"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nature_config import BLUE, RED, GREEN, ORANGE

rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams.update({"font.size": 11})

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")

COL1 = BLUE
COL2 = RED
COL3 = ORANGE
COL4 = GREEN

def draw_box(x, y, w, h, text, fc="white", ec="black", fs=13):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                                   facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="black")

def draw_arrow(x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Title
ax.text(5, 12.2, "Study Flow Diagram", ha="center", va="center", fontsize=15, fontweight="bold")

# Box 1
draw_box(5, 11.3, 6, 0.8, "NHANES 2011-2018\n4 cycles (G, H, I, J)", fc="#D6E4F0", ec=COL1, fs=13)
draw_arrow(5, 10.9, 5, 9.9)

# Box 2
draw_box(5, 9.5, 6, 0.8, "Total participants\nN = 39,156", fc="#D6E4F0", ec=COL1, fs=13)
draw_arrow(5, 9.1, 5, 8.1)

# Exclusion 1
draw_box(5, 7.7, 6, 0.8, "Age < 20 years\n(n = 16,539 excluded)", fc="#FDE9D9", ec=COL2, fs=12)
ax.text(7.2, 7.3, "Excluded", ha="left", va="center", fontsize=10, color=COL2, fontstyle="italic")
draw_arrow(5, 7.3, 5, 6.3)

# Box 3
draw_box(5, 5.9, 6, 0.8, "Adults aged >= 20 years\nN = 22,617", fc="#D6E4F0", ec=COL1, fs=13)
draw_arrow(5, 5.5, 5, 4.5)

# Exclusion 2
draw_box(5, 4.1, 6, 0.8, "Missing NLR, Hb, or Albumin\n(n = 2,395 excluded)", fc="#FDE9D9", ec=COL2, fs=12)
ax.text(7.2, 3.7, "Excluded", ha="left", va="center", fontsize=10, color=COL2, fontstyle="italic")
draw_arrow(5, 3.7, 5, 2.7)

# Box 4
draw_box(5, 2.3, 6, 0.8, "Final analytic sample\nN = 20,222 (CKD = 3,703; 14.9% weighted)", fc="#E2EFDA", ec=COL4, fs=13)

# Split arrows
draw_arrow(3.5, 1.9, 2.5, 0.95)
draw_arrow(6.5, 1.9, 7.5, 0.95)

# Training
draw_box(2.5, 0.6, 4.5, 0.7, "Training set 2011-2016 (G/H/I)\nn = 15,311 (CKD = 17.8%)", fc="white", ec=COL1, fs=12)
# Validation
draw_box(7.5, 0.6, 4.5, 0.7, "Temporal validation 2017-2018 (J)\nn = 4,911 (CKD = 19.8%)", fc="white", ec=COL3, fs=12)

plt.tight_layout()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHART_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "output", "charts")
os.makedirs(_CHART_DIR, exist_ok=True)
fig.savefig(os.path.join(_CHART_DIR, "fig_flowchart.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(_CHART_DIR, "fig_flowchart.tiff"), dpi=600, bbox_inches="tight")
plt.close()
print("Flowchart saved!")
