#!/usr/bin/env python3
"""
Export true 600dpi TIFF for ALL figures.
Uses subprocess with matplotlib monkey-patch wrapper.
"""
import os, sys, subprocess, glob, textwrap

BASE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(os.path.dirname(BASE), "output", "charts")

SCRIPTS = [
    "run_visualizations.py",
    "flowchart.py",
    "graphical_abstract.py",
    "supplementary_analysis.py",
    "temporal_validation.py",
    "ml_analysis.py",
    "advanced_analysis.py",
    "generate_all_figures.py",
]

# Write the wrapper as a separate .py file so __file__ works
WRAPPER_PATH = os.path.join(BASE, "_tiff_wrapper_header.py")
with open(WRAPPER_PATH, 'w') as f:
    f.write(textwrap.dedent("""\
import matplotlib.pyplot as _plt
_orig_savefig = _plt.Figure.savefig
def _tiff_savefig(self, fname, **kwargs):
    if isinstance(fname, str) and '.png' in fname:
        tiff_path = fname.replace('.png', '.tiff')
        _orig_savefig(self, tiff_path, dpi=600, bbox_inches='tight')
    return _orig_savefig(self, fname, **kwargs)
_plt.Figure.savefig = _tiff_savefig
"""))

# Clean old TIFFs
for f in glob.glob(os.path.join(CHARTS, "*.tiff")):
    os.remove(f)

print("=" * 65)
print("Exporting true 600dpi TIFF for all figures")
print("=" * 65)

for script_name in SCRIPTS:
    path = os.path.join(BASE, script_name)
    if not os.path.exists(path):
        continue
    
    # Create temp run script that imports the wrapper header first
    run_path = os.path.join(BASE, f"_{script_name}_tiff_run.py")
    with open(run_path, 'w') as f:
        f.write(f"import _tiff_wrapper_header\n")
        f.write(f"import sys; sys.path.insert(0, {repr(BASE)})\n")
        with open(path) as src:
            f.write(src.read())
    
    print(f"  ▶ {script_name}...", end=' ', flush=True)
    r = subprocess.run(
        [sys.executable, run_path],
        capture_output=True, text=True, timeout=600
    )
    
    if r.returncode == 0:
        print("✅")
    else:
        # Show full error
        error = (r.stderr or r.stdout)[:300]
        print(f"❌\n{error}")
    
    if os.path.exists(run_path):
        os.remove(run_path)

# Cleanup wrapper header
if os.path.exists(WRAPPER_PATH):
    os.remove(WRAPPER_PATH)

# Verify DPI metadata
print(f"\n{'='*65}")
print("VERIFICATION")
print("="*65)
from PIL import Image
tiffs = sorted(glob.glob(os.path.join(CHARTS, "*.tiff")))
print(f"  TIFF files: {len(tiffs)}")
for t in tiffs:
    img = Image.open(t)
    dpi = img.info.get('dpi', (0,0))
    print(f"    {os.path.basename(t):45s} {img.size[0]:>5d}x{img.size[1]:<5d}px  dpi={dpi[0]:.0f}")
