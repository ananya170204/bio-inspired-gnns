"""
Figure 3.X — V7 Unified Brain-Inspired GNN Architecture (professional styling)
====================================================================================
Usage:
    python3 v7_unified_diagram.py
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

fig, ax = plt.subplots(figsize=(10, 10.6))
ax.set_xlim(0, 11.6)
ax.set_ylim(3.0, 14.6)
style_axes(ax)

title_block(ax, "V7  —  Unified Brain-Inspired GNN", 5.8, 14.3,
            subtitle="All six principles combined within a single forward pass per layer")

# Input
box(ax, 4.3, 13.15, 3.0, 0.6, "$h_v^{(t)}$", style="gray", title_size=11)
arrow(ax, 5.8, 13.15, 5.8, 12.72)

# Three streams
box(ax, 0.4, 11.55, 3.5, 0.95, "Modular stream",
    "V1 modules + V4 hub scaling", style="teal", subtitle_size=8.7)
box(ax, 4.05, 11.55, 3.5, 0.95, "Functional stream",
    "V6 dynamic edges", style="coral", subtitle_size=8.7)
box(ax, 7.7, 11.55, 3.5, 0.95, "Shortcut stream",
    "V5 learned shortcuts", style="purple", subtitle_size=8.7)

arrow(ax, 4.6, 13.15, 2.15, 12.5, curve=0.15, color="#9A9995", lw=0.9)
arrow(ax, 5.8, 13.15, 5.8, 12.5, lw=0.9)
arrow(ax, 7.0, 13.15, 9.45, 12.5, curve=-0.15, color="#9A9995", lw=0.9)

# Converge to attention fusion
arrow(ax, 2.15, 11.55, 4.4, 10.75, curve=-0.15, color="#2F7A63")
arrow(ax, 5.8, 11.55, 5.8, 10.75)
arrow(ax, 9.45, 11.55, 7.2, 10.75, curve=0.15, color="#5A50A8")

# Attention fusion (V3-style)
box(ax, 3.5, 9.9, 4.6, 0.85, "Attention fusion",
    "combines the three streams (as in V3)", style="amber")
arrow(ax, 5.8, 9.9, 5.8, 9.25)

# Layer output — repeated across layers
box(ax, 3.9, 8.35, 3.8, 0.9, "Layer output  $h_v^{(t+1)}$",
    "repeated for $t = 0, \\ldots, L-1$", style="gray", subtitle_size=8.7)

# Collected representations feeding hierarchical fusion
dashed_skip(ax, 3.9, 8.575, 1.6, 6.55, 3.15)
small_caption(ax, 1.1, 7.7, "collected at\nevery layer", color=SKIP_COLOR,
              size=8, style='normal')

arrow(ax, 5.8, 8.35, 5.8, 7.55, lw=0.9)
small_caption(ax, 6.35, 7.95, "next layer\n($t \\to t+1$)", size=8, style='normal')

# Hierarchical fusion (V2-style)
box(ax, 3.05, 5.75, 5.6, 0.95,
    "Hierarchical fusion  (as in V2)",
    r"$\mathrm{ReLU}(W_{\mathrm{fuse}} \cdot \mathrm{CONCAT}(h^{(0)}, \ldots, h^{(L)}))$",
    style="coral", subtitle_size=9.5)
arrow(ax, 5.8, 5.75, 5.8, 5.1)

# Final output
box(ax, 4.1, 4.35, 3.4, 0.6, "$h_v^{\\mathrm{final}}$", style="gray", title_size=11)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v7_unified_brain_inspired_gnn")
plt.close()