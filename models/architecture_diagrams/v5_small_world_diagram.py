"""
Figure 3.X — V5 Small-World Shortcut GNN Architecture (professional styling)
=================================================================================
Usage:
    python3 v5_small_world_diagram.py
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

fig, ax = plt.subplots(figsize=(9.5, 8.1))
ax.set_xlim(0, 11)
ax.set_ylim(3.0, 10.9)
style_axes(ax)

title_block(ax, "V5  —  Small-World Shortcut GNN", 5.5, 10.6,
            subtitle="Local clustering combined with sparse learned long-range shortcuts")

# Input
box(ax, 4.0, 9.45, 3.0, 0.6, "Node representation  $h_v$", style="gray", title_size=10)

arrow(ax, 4.7, 9.45, 2.7, 8.75, curve=0.15, color="#9A9995")
arrow(ax, 6.3, 9.45, 8.3, 8.75, curve=-0.15, color="#9A9995")

# Local stream
box(ax, 0.5, 7.9, 4.4, 0.85, "Local stream",
    "mean aggregation, original edges", style="teal")

# Shortcut stream
box(ax, 6.1, 7.9, 4.4, 0.85, "Shortcut stream",
    "learned top-k long-range edges", style="coral")

small_caption(ax, 8.3, 7.55, r"$s_{uv} = W_{\mathrm{attn}} \cdot [\,h_u \,\Vert\, h_v\,]$", size=9)

# Gate
box(ax, 2.6, 6.05, 5.8, 0.85, "Learned linear fusion",
    "weighted combination of local and shortcut streams", style="purple")

arrow(ax, 2.7, 7.9, 4.6, 6.9, curve=-0.15, color="#2F7A63")
arrow(ax, 8.3, 7.9, 6.4, 6.9, curve=0.15, color="#B85A34")

arrow(ax, 5.5, 6.05, 5.5, 5.4)

# Output
box(ax, 3.9, 4.7, 3.2, 0.6, "Updated  $h_v$", style="gray", title_size=10)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v5_small_world_shortcut_gnn")
plt.close()