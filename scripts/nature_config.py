# -*- coding: utf-8 -*-
"""Nature-style unified figure configuration for IAN-CKD manuscript."""
import matplotlib as mpl

# ===== Typography =====
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 600,
    "savefig.dpi": 600,
})

# ===== Axes & Spines =====
mpl.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
    "legend.frameon": False,
})

# ===== Unified Nature-style Color Palette =====
BLUE      = "#0F4D92"   # Main blue (No CKD, primary)
RED       = "#B64342"   # Strong red (CKD, significant)
GREEN     = "#8BCF8B"   # Soft green (low risk)
ORANGE    = "#E8832E"   # Orange (medium risk)
TEAL      = "#42949E"   # Teal (supplementary)
PURPLE    = "#9A4D8E"   # Violet (temporal validation)
GREY      = "#767676"   # Neutral mid (non-significant)
LIGHT_BLUE  = "#D6E4F0" # Light blue (background boxes)
LIGHT_GREEN = "#DDF3DE" # Light green (background)
LIGHT_ORANGE= "#FDE9D9" # Light orange (background)
NEUTRAL_LT  = "#CFCECE" # Neutral light
NEUTRAL_DK  = "#4D4D4D" # Neutral dark

# ===== Recommended Figure Sizes (inches) =====
# Single column: 3.5" wide
# 1.5 column: 5" wide  
# Double column: 7" wide

SIZE_SINGLE  = (4.0, 3.5)   # Single column
SIZE_15COL   = (5.5, 4.0)   # 1.5 column
SIZE_DOUBLE  = (7.5, 4.5)   # Double column
SIZE_WIDE    = (7.5, 3.0)   # Wide & short (forest plot)
SIZE_TALL    = (4.0, 5.0)   # Tall (ROC curve)
SIZE_LANDSCAPE = (11, 5.5)  # Landscape (flowchart)
SIZE_FOREST  = (11, 5.5)    # Forest plot
SIZE_BAR     = (13, 6.5)    # Bar chart comparisons

# ===== Export Helper =====
def save_nature_figure(fig, filename, dpi=600):
    """Save figure in Nature-ready formats."""
    import os
    base = os.path.splitext(filename)[0]
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=dpi, bbox_inches="tight", pil_kwargs={'compression': 'tiff_lzw'})
    print(f"  ✓ Saved: {os.path.basename(base)}")
