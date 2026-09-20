"""
Figure 3.X — V4 Hub-Aware GNN Architecture (professional styling)
======================================================================
Usage:
    python3 v4_hub_aware_diagram.py
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

fig, ax = plt.subplots(figsize=(9.5, 8.4))
ax.set_xlim(0, 11)
ax.set_ylim(3.0, 11.2)
style_axes(ax)

title_block(ax, "V4  —  Hub-Aware GNN", 5.5, 10.9,
            subtitle="Messages scaled by a learned function of sender degree")

# Neighbour nodes (varying degree, represented as small circles)
node_specs = [(1.6, 9.2, 0.32, "deg 2", "gray"),
              (1.6, 8.15, 0.42, "deg 40", "amber"),
              (1.6, 7.1, 0.32, "deg 3", "gray")]
for x, y, r, label, style in node_specs:
    from diagram_style import PALETTE
    colors = PALETTE[style]
    circ = plt.Circle((x, y), r, facecolor=colors["face"], edgecolor=colors["edge"],
                       linewidth=1.15, zorder=2)
    ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center', fontsize=8.3,
            color=colors["text"], fontweight='bold', zorder=3)

# Degree scaling box
box(ax, 3.1, 7.85, 4.6, 0.95, "Degree scaling",
    r"$\sigma(\mathrm{MLP}_{\mathrm{hub}}(\log(d_u+1)))$", style="coral",
    title_size=10.5, subtitle_size=10)

arrow(ax, 1.92, 9.2, 3.05, 8.55, curve=-0.15, color="#9A9995", lw=0.9)
arrow(ax, 2.02, 8.15, 3.05, 8.3, color="#A8791F", lw=2.4)
arrow(ax, 1.92, 7.1, 3.05, 8.05, curve=0.15, color="#9A9995", lw=0.9)

small_caption(ax, 6.9, 9.35, "hub node \u2192 stronger message weight", size=8.5)

# Sum aggregation
box(ax, 3.1, 6.55, 4.6, 0.85, "Sum aggregation",
    "weighted by degree-scaled messages", style="teal")
arrow(ax, 5.4, 7.85, 5.4, 7.4)

# Update
box(ax, 3.1, 5.25, 4.6, 0.85, "Linear update + BatchNorm",
    "ReLU activation", style="purple")
arrow(ax, 5.4, 6.55, 5.4, 6.1)
arrow(ax, 5.4, 5.25, 5.4, 4.8)

# Output
box(ax, 3.6, 4.05, 3.6, 0.6, "Updated node embedding", style="gray", title_size=10)



plt.tight_layout()
save_figure(fig, OUT_DIR, "v4_hub_aware_gnn")
plt.close()