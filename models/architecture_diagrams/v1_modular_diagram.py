"""
Figure 3.X — V1 Modular Community GNN Architecture (professional styling)
============================================================================
Usage:
    python3 v1_modular_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diagram_style import box, arrow, title_block, small_caption, save_figure, style_axes

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, 7.1))
ax.set_xlim(0, 11)
ax.set_ylim(3.0, 10.8)
style_axes(ax)

title_block(ax, "V1  —  Modular Community GNN", 5.5, 10.55,
            subtitle="K = 4 independent processing modules with sparse inter-module gating")

# Input
box(ax, 4.1, 9.15, 2.8, 0.6, "Node input $h_v$", style="gray", title_size=10.5)
arrow(ax, 5.5, 9.15, 5.5, 8.72)
small_caption(ax, 5.5, 8.58, "split into K = 4 modules")

# 4 module boxes
module_x = [0.5, 3.15, 5.8, 8.45]
for i, x in enumerate(module_x, start=1):
    box(ax, x, 7.15, 2.05, 0.85, f"Module {i}",
        f"dims {(i-1)*16+1}\u2013{i*16}", style="teal")
    arrow(ax, x + 1.025, 7.15, x + 1.025, 6.35)

# Concatenate
box(ax, 1.0, 5.45, 9.0, 0.85, "Concatenate module outputs",
    "restores full hidden dimension = 64", style="purple")
arrow(ax, 5.5, 5.45, 5.5, 4.85)

# Sparse gate
box(ax, 1.0, 3.65, 9.0, 1.15,
    "Sparse inter-module gate",
    "70% of cross-module connections zeroed  \u00b7  fixed mask M  \u00b7  \u03b1 = 0.1",
    style="amber", title_size=10.5, subtitle_size=8.8)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v1_modular_community_gnn")
plt.close()