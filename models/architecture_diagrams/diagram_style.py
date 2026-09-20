"""
Shared styling utilities for thesis architecture diagrams.
Import from this module in each variant's diagram script.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────────────────────────────────────
# Professional, muted, print-safe palette (desaturated, consistent value range)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "gray":   {"face": "#F4F3F1", "edge": "#6B6B68", "text": "#2B2B29"},
    "teal":   {"face": "#DCEEE7", "edge": "#2F7A63", "text": "#1B4A3C"},
    "purple": {"face": "#E4E2F5", "edge": "#5A50A8", "text": "#332C63"},
    "coral":  {"face": "#F7E4DC", "edge": "#B85A34", "text": "#6B341E"},
    "amber":  {"face": "#F9EDD6", "edge": "#A8791F", "text": "#5F4212"},
}

SKIP_COLOR = "#A8791F"
ARROW_COLOR = "#6B6B68"
FONT_FAMILY = "DejaVu Sans"   # clean, widely-available sans-serif


def style_axes(ax):
    ax.set_facecolor("white")
    ax.axis('off')


def box(ax, x, y, w, h, title, subtitle=None, style="gray", title_size=10.5,
        subtitle_size=9, shadow=True):
    """Draw a rounded box with a bold title and optional lighter subtitle line."""
    colors = PALETTE[style]

    if shadow:
        shadow_patch = FancyBboxPatch(
            (x + 0.045, y - 0.045), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor="#000000", edgecolor="none", alpha=0.06, zorder=1)
        ax.add_patch(shadow_patch)

    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.09",
        facecolor=colors["face"], edgecolor=colors["edge"],
        linewidth=1.15, zorder=2)
    ax.add_patch(b)

    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=colors["text"],
                family=FONT_FAMILY, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha='center', va='center',
                fontsize=subtitle_size, color=colors["text"],
                family=FONT_FAMILY, alpha=0.85, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=colors["text"],
                family=FONT_FAMILY, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.3, curve=None, mutation=11):
    connectionstyle = f"arc3,rad={curve}" if curve else None
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                         color=color, linewidth=lw, mutation_scale=mutation,
                         connectionstyle=connectionstyle,
                         capstyle='round', joinstyle='round', zorder=4)
    ax.add_patch(a)


def dashed_skip(ax, x1, y1, xmid, y2, x3, color=SKIP_COLOR, lw=1.1):
    """L-shaped dashed skip-connection arrow."""
    path = MplPath(
        [(x1, y1), (xmid, y1), (xmid, y2), (x3, y2)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                                linewidth=lw, linestyle=(0, (4, 2.5)),
                                capstyle='round', zorder=3)
    ax.add_patch(patch)
    a = FancyArrowPatch((x3 - 0.12, y2), (x3, y2), arrowstyle='-|>',
                         color=color, linewidth=lw, mutation_scale=9, zorder=4)
    ax.add_patch(a)


def title_block(ax, text, x_center, y, subtitle=None):
    ax.text(x_center, y, text, ha='center', va='center',
            fontsize=15, fontweight='bold', color="#1A1A18",
            family=FONT_FAMILY)
    if subtitle:
        ax.text(x_center, y - 0.42, subtitle, ha='center', va='center',
                fontsize=10, color="#6B6B68", family=FONT_FAMILY, style='italic')


def small_caption(ax, x, y, text, color="#6B6B68", size=8.5, rotation=0, style='italic'):
    ax.text(x, y, text, ha='center', va='center', fontsize=size,
            color=color, family=FONT_FAMILY, style=style, rotation=rotation)


def save_figure(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches='tight', dpi=200,
                facecolor='white')
    fig.savefig(out_dir / f"{name}.png", bbox_inches='tight', dpi=200,
                facecolor='white')
    print(f"Saved -> {out_dir}/{name}.pdf")
    print(f"Saved -> {out_dir}/{name}.png")