"""
Figure 3.X — V2 Hierarchical Multi-Scale GNN Architecture (professional styling)
====================================================================================
Usage:
    python3 v2_hierarchical_diagram.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from diagram_style import (box, arrow, dashed_skip, title_block,
                            small_caption, save_figure, style_axes, SKIP_COLOR)

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, 10.3))
ax.set_xlim(0, 11)
ax.set_ylim(3.0, 15.0)
style_axes(ax)

title_block(ax, "V2  —  Hierarchical Multi-Scale GNN", 5.5, 14.7,
            subtitle="Local and global processing streams unified via skip connections")

# Input
box(ax, 4.0, 13.55, 3.0, 0.62, "$h_v^{(0)}$", style="gray", title_size=11)
arrow(ax, 5.5, 13.55, 5.5, 13.1)

# Layer 1 (low-level)
box(ax, 2.3, 12.15, 6.4, 0.85, "Layer 1  \u2014  sum aggregation",
    "1-hop local detail", style="teal")
arrow(ax, 5.5, 12.15, 5.5, 11.35)

# Layer 2 (low-level)
box(ax, 2.3, 10.5, 6.4, 0.85, "Layer 2  \u2014  sum aggregation",
    "2-hop local detail", style="teal")
arrow(ax, 5.5, 10.5, 5.5, 9.7)

# Layer 3 (high-level)
box(ax, 2.3, 8.85, 6.4, 0.85, "Layer 3  \u2014  mean aggregation",
    "regional neighbourhood context", style="purple")
arrow(ax, 5.5, 8.85, 5.5, 8.05)

# Layer 4 (high-level)
box(ax, 2.3, 7.2, 6.4, 0.85, "Layer 4  \u2014  mean aggregation",
    "global neighbourhood context", style="purple")

# Skip connections — routed cleanly on the left margin, staggered to avoid overlap
dashed_skip(ax, 2.3, 12.575, 1.55, 9.275, 2.28)   # layer1 -> layer3
dashed_skip(ax, 1.85, 12.575, 1.1,  7.625, 2.28)  # layer1 -> layer4
dashed_skip(ax, 2.3, 10.925, 1.75, 9.275, 2.28)   # layer2 -> layer3
dashed_skip(ax, 1.95, 10.925, 1.3,  7.625, 2.28)  # layer2 -> layer4

small_caption(ax, 0.55, 10.1, "skip connections", color=SKIP_COLOR,
              size=9, rotation=90, style='normal')

arrow(ax, 5.5, 7.2, 5.5, 6.55)

# Fusion
box(ax, 1.8, 5.7, 7.4, 0.85,
    "Fusion  \u2014  concatenate all five scales",
    "$h^{(0)}$ through $h^{(4)}$, followed by projection", style="coral")
arrow(ax, 5.5, 5.7, 5.5, 5.05)

# Output
box(ax, 3.9, 4.35, 3.2, 0.6, "$h_v^{\\mathrm{final}}$", style="gray", title_size=11)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v2_hierarchical_multiscale_gnn")
plt.close()