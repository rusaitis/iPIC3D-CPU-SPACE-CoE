#!/usr/bin/env python3
"""
plot_theme.py — Shared dark/light theme for iPIC3D test plotting scripts.

Provides a Catppuccin-inspired dark theme (default) and a modernised light
theme, plus solver-specific color/marker/linestyle mappings that adapt to the
active mode.

Usage from any plotting script:

    from plot_theme import add_theme_arg, apply_theme, get_solver_style

    add_theme_arg(parser)              # adds --light flag
    args = parser.parse_args()
    theme = apply_theme(args)          # sets rcParams, returns Theme

    style = get_solver_style("PETSc_gmres")  # {"color", "marker", "ls"}
"""

import dataclasses
from typing import Dict, Optional

import matplotlib.pyplot as plt


# ── Theme dataclass ──────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Theme:
    mode: str                  # "dark" | "light"
    fig_face: str
    axes_face: str
    text_color: str
    text_secondary: str
    text_tertiary: str
    grid_color: str
    grid_alpha: float
    spine_color: str
    annot_face: str
    annot_edge: str
    annot_alpha: float
    box_face: str
    box_edge: str
    box_alpha: float
    marker_edge: str
    bar_edge: str
    refline_color: str
    zero_line_color: str
    legend_face: str
    legend_edge: str
    legend_alpha: float
    diverging_cmap: str


# ── Palettes ─────────────────────────────────────────────────────────────

_DARK = Theme(
    mode="dark",
    fig_face="#1e1e2e",
    axes_face="#252536",
    text_color="#cdd6f4",
    text_secondary="#9399b2",
    text_tertiary="#6c7086",
    grid_color="#45475a",
    grid_alpha=0.5,
    spine_color="#45475a",
    annot_face="#313244",
    annot_edge="#585b70",
    annot_alpha=0.92,
    box_face="#313244",
    box_edge="#585b70",
    box_alpha=0.92,
    marker_edge="#1e1e2e",
    bar_edge="#1e1e2e",
    refline_color="#6c7086",
    zero_line_color="#9399b2",
    legend_face="#313244",
    legend_edge="#585b70",
    legend_alpha=0.92,
    diverging_cmap="RdBu_r",
)

_LIGHT = Theme(
    mode="light",
    fig_face="#ffffff",
    axes_face="#f8f9fa",
    text_color="#1e1e2e",
    text_secondary="#585b70",
    text_tertiary="#7c7f93",
    grid_color="#dce0e8",
    grid_alpha=0.6,
    spine_color="#ccd0da",
    annot_face="#ffffff",
    annot_edge="#ccd0da",
    annot_alpha=0.9,
    box_face="#eff1f5",
    box_edge="#ccd0da",
    box_alpha=0.9,
    marker_edge="#ffffff",
    bar_edge="#ffffff",
    refline_color="#9ca0b0",
    zero_line_color="#4c4f69",
    legend_face="#ffffff",
    legend_edge="#ccd0da",
    legend_alpha=0.9,
    diverging_cmap="RdBu_r",
)

# ── Solver styles (mode-dependent) ───────────────────────────────────────

_SOLVER_STYLES_LIGHT: Dict[str, dict] = {
    "GMRES":        {"color": "#4477AA", "marker": "o", "ls": "-"},
    "PETSc":        {"color": "#EE6677", "marker": "s", "ls": "-"},
    "PETSc_gmres":  {"color": "#EE6677", "marker": "s", "ls": "-"},
    "PETSc_bcgs":   {"color": "#228833", "marker": "^", "ls": "--"},
    "PETSc_fgmres": {"color": "#AA3377", "marker": "D", "ls": "-."},
    "PETSc_tfqmr":  {"color": "#CCBB44", "marker": "v", "ls": ":"},
    "GMRES_2":      {"color": "#66CCEE", "marker": "d", "ls": "--"},
    "CG":           {"color": "#117733", "marker": "P", "ls": "-"},
    "BiCGStab":     {"color": "#882255", "marker": "X", "ls": "-."},
}

_SOLVER_STYLES_DARK: Dict[str, dict] = {
    "GMRES":        {"color": "#7aafdf", "marker": "o", "ls": "-"},
    "PETSc":        {"color": "#f38ba8", "marker": "s", "ls": "-"},
    "PETSc_gmres":  {"color": "#f38ba8", "marker": "s", "ls": "-"},
    "PETSc_bcgs":   {"color": "#a6e3a1", "marker": "^", "ls": "--"},
    "PETSc_fgmres": {"color": "#cba6f7", "marker": "D", "ls": "-."},
    "PETSc_tfqmr":  {"color": "#f9e2af", "marker": "v", "ls": ":"},
    "GMRES_2":      {"color": "#89dceb", "marker": "d", "ls": "--"},
    "CG":           {"color": "#94e2d5", "marker": "P", "ls": "-"},
    "BiCGStab":     {"color": "#f0a6ca", "marker": "X", "ls": "-."},
}

_DEFAULT_STYLE_LIGHT = {"color": "#607D8B", "marker": "x", "ls": ":"}
_DEFAULT_STYLE_DARK  = {"color": "#94a3b8", "marker": "x", "ls": ":"}

_EXTRA_COLORS_LIGHT = ["#882255", "#332288", "#117733", "#999933", "#CC6677"]
_EXTRA_COLORS_DARK  = ["#f0a6ca", "#b4befe", "#94e2d5", "#f9e2af", "#f38ba8"]
_EXTRA_MARKERS = ["s", "^", "D", "v", "P", "X", "d", "h", "<", ">"]
_EXTRA_LINESTYLES = ["-", "--", "-.", ":", "-"]

# User-registered styles (populated via register_solver_style())
_CUSTOM_STYLES: Dict[str, dict] = {}

# Component breakdown colors (field, moments, mover) keyed by mode
COMPONENT_COLORS = {
    "dark":  {"field": "#7aafdf", "moments": "#a6e3a1", "mover": "#f38ba8",
              "field_overlay": "#89b4fa"},
    "light": {"field": "#4477AA", "moments": "#228833", "mover": "#EE6677",
              "field_overlay": "#004488"},
}

# Profile line styles cycle through these for different solvers
PROFILE_LINESTYLES = ["-", "--", "-.", ":"]


# ── Module state ─────────────────────────────────────────────────────────

active: Optional[Theme] = None


# ── Public API ───────────────────────────────────────────────────────────

def add_theme_arg(parser):
    """Add ``--light`` flag to an argparse parser."""
    parser.add_argument(
        "--light", action="store_true",
        help="Use light theme instead of the default dark theme",
    )


def apply_theme(args=None) -> Theme:
    """Apply matplotlib rcParams and return the active ``Theme``.

    Parameters
    ----------
    args : argparse.Namespace or None
        If *args* has a truthy ``.light`` attribute the light palette is
        used; otherwise dark (default).
    """
    global active

    light = getattr(args, "light", False) if args is not None else False
    t = _LIGHT if light else _DARK
    active = t

    plt.rcParams.update({
        "figure.facecolor":     t.fig_face,
        "figure.edgecolor":     t.fig_face,
        "savefig.facecolor":    t.fig_face,
        "savefig.edgecolor":    "none",
        "savefig.dpi":          150,
        "axes.facecolor":       t.axes_face,
        "axes.edgecolor":       t.spine_color,
        "axes.labelcolor":      t.text_color,
        "axes.titlecolor":      t.text_color,
        "axes.linewidth":       0.8,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           t.grid_color,
        "grid.alpha":           t.grid_alpha,
        "grid.linewidth":       0.5,
        "xtick.color":          t.text_color,
        "ytick.color":          t.text_color,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "text.color":           t.text_color,
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica",
                                 "DejaVu Sans", "Arial", "sans-serif"],
        "font.size":            11,
        "legend.facecolor":     t.legend_face,
        "legend.edgecolor":     t.legend_edge,
        "legend.framealpha":    t.legend_alpha,
        "legend.fontsize":      10,
        "lines.linewidth":      1.8,
    })

    return t


def register_solver_style(key: str, color: str, marker: str = "o",
                          ls: str = "-"):
    """Register a custom solver style for use in plots.

    Registered styles take priority over the hash-based fallback but
    not over the built-in style tables.
    """
    _CUSTOM_STYLES[key] = {"color": color, "marker": marker, "ls": ls}


def get_solver_style(label_or_path: str) -> dict:
    """Return ``{"color": ..., "marker": ..., "ls": ...}`` for a solver.

    *label_or_path* can be a directory path (basename is extracted) or a
    plain solver key like ``"PETSc_gmres"``.  Grid-size suffixes of the
    form ``_NxNxN`` are stripped before lookup.

    Lookup order: built-in styles → prefix match → custom styles →
    hash-based fallback (cycles through distinct markers and linestyles).
    """
    import os
    import re

    basename = os.path.basename(label_or_path)
    key = re.sub(r"_\d+x\d+x\d+$", "", basename)

    dark = active is not None and active.mode == "dark"
    styles = _SOLVER_STYLES_DARK if dark else _SOLVER_STYLES_LIGHT
    extras = _EXTRA_COLORS_DARK if dark else _EXTRA_COLORS_LIGHT

    if key in styles:
        return dict(styles[key])
    for k, v in styles.items():
        if basename.startswith(k):
            return dict(v)
    if key in _CUSTOM_STYLES:
        return dict(_CUSTOM_STYLES[key])

    idx = hash(basename) % len(extras)
    marker = _EXTRA_MARKERS[idx % len(_EXTRA_MARKERS)]
    ls = _EXTRA_LINESTYLES[idx % len(_EXTRA_LINESTYLES)]
    return {"color": extras[idx], "marker": marker, "ls": ls}
