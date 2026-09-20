"""
Figure 3.X — V3 Parallel Duplication GNN Architecture (professional styling)
=================================================================================
Usage:
    python3 v3_parallel_diagram.py
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

fig, ax = plt.subplots(figsize=(9.5, 7.6))
ax.set_xlim(0, 11.6)
ax.set_ylim(3.2, 10.5)
style_axes(ax)

title_block(ax, "V3  —  Parallel Duplication GNN", 5.8, 10.2,
            subtitle="Three independent aggregation channels combined via learned attention")

# Input
box(ax, 4.3, 9.05, 3.0, 0.6, "Node input  $h_v$", style="gray", title_size=10.5)

# Fan-out arrows
arrow(ax, 4.6, 9.05, 2.35, 8.35, curve=0.18, color="#9A9995")
arrow(ax, 5.8, 9.05, 5.8, 8.35)
arrow(ax, 7.0, 9.05, 9.25, 8.35, curve=-0.18, color="#9A9995")

# 3 channels
box(ax, 0.5, 7.35, 3.3, 0.85, "Channel: sum", "MLP + BatchNorm + ReLU", style="teal")
box(ax, 4.15, 7.35, 3.3, 0.85, "Channel: mean", "MLP + BatchNorm + ReLU", style="purple")
box(ax, 7.8, 7.35, 3.3, 0.85, "Channel: max", "MLP + BatchNorm + ReLU", style="coral")

# Converge into fusion
arrow(ax, 2.15, 7.35, 4.3, 6.6, curve=-0.18, color="#2F7A63")
arrow(ax, 5.8, 7.35, 5.8, 6.6)
arrow(ax, 9.45, 7.35, 7.3, 6.6, curve=0.18, color="#B85A34")

# Attention fusion
box(ax, 3.0, 5.75, 5.6, 0.85, "Attention fusion",
    "softmax-weighted combination of channel outputs", style="amber")
arrow(ax, 5.8, 5.75, 5.8, 5.1)

# Output
box(ax, 3.6, 4.35, 4.4, 0.62, "$h_v^{\\mathrm{final}} = \\sum_c \\alpha_c \\, h_v^{(L),c}$",
    style="gray", title_size=10.5)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v3_parallel_duplication_gnn")
plt.close()