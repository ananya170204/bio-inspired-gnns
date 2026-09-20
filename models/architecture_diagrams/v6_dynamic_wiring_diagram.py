"""
Figure 3.X — V6 Dynamic Rewiring GNN Architecture (professional styling)
==============================================================================
Usage:
    python3 v6_dynamic_wiring_diagram.py
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

title_block(ax, "V6  —  Dynamic Rewiring GNN", 5.5, 10.6,
            subtitle="Fixed structural edges combined with input-dependent functional edges")

# Input
box(ax, 4.0, 9.45, 3.0, 0.6, "Node representation  $h_v$", style="gray", title_size=10)

arrow(ax, 4.7, 9.45, 2.7, 8.75, curve=0.15, color="#9A9995")
arrow(ax, 6.3, 9.45, 8.3, 8.75, curve=-0.15, color="#9A9995")

# Structural stream
box(ax, 0.5, 7.9, 4.4, 0.85, "Structural stream",
    "sum aggregation, fixed edges", style="teal")

# Functional stream
box(ax, 6.1, 7.9, 4.4, 0.85, "Functional stream",
    "dynamic edges, recomputed each pass", style="coral")

small_caption(ax, 8.3, 7.5, r"$f_{uv} = \sigma(\mathrm{MLP}_{\mathrm{func}}([h_u \Vert h_v])) > \tau$", size=8.7)

# Integration
box(ax, 2.6, 6.05, 5.8, 0.85, "Linear integration",
    "concatenation followed by projection", style="purple")

arrow(ax, 2.7, 7.9, 4.6, 6.9, curve=-0.15, color="#2F7A63")
arrow(ax, 8.3, 7.9, 6.4, 6.9, curve=0.15, color="#B85A34")

arrow(ax, 5.5, 6.05, 5.5, 5.4)

# Output
box(ax, 3.9, 4.7, 3.2, 0.6, "Updated  $h_v$", style="gray", title_size=10)

plt.tight_layout()
save_figure(fig, OUT_DIR, "v6_dynamic_rewiring_gnn")
plt.close()