#!/usr/bin/env python3
"""
plot_scaling.py — Scaling sweep figure for iPIC3D solver comparison.

Generates a multi-panel figure comparing solver performance across multiple
grid sizes: line plot of solver times, improvement bars, and optional
component breakdown stacked bars.

Usage (standalone):
    python3 plot_scaling.py [results.csv] [--loglog] [--ref-slope N] [--sort] [--output PATH]

Usually invoked via the dispatcher plot_timing.py.
"""

import argparse
import math
import os
import sys
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter
except ImportError:
    print("  WARNING: matplotlib not installed, skipping plot generation.")
    print("  Install with: pip3 install matplotlib")
    raise SystemExit(2)

from plot_theme import (add_theme_arg, apply_theme, get_solver_style,
                        COMPONENT_COLORS)
from plot_timing_common import (
    ResultsData, load_results_csv, resolve_csv_path, resolve_plot_path,
    darken_color, auto_time_unit, compute_improvements,
    render_improvement_bars, render_summary_annotation, apply_tick_rotation,
)


def main(data, args, theme, plot_path):
    """Render the scaling sweep figure and save to plot_path."""
    unit, scale = auto_time_unit(data.all_times)
    solver_pcts, avg_improvements = compute_improvements(data)

    has_breakdown = data.has_breakdown

    # ── Layout ────────────────────────────────────────────────────────
    n_rows = 1
    height_ratios = [3]
    if has_breakdown:
        n_rows = 2
        height_ratios = [3, 2]

    fig_height = {1: 6, 2: 10}[n_rows]
    fig = plt.figure(figsize=(15, fig_height))
    gs  = GridSpec(n_rows, 2, figure=fig,
                   height_ratios=height_ratios,
                   hspace=0.45, wspace=0.35,
                   top=0.93, bottom=0.06, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    date_str = datetime.now().strftime("%Y-%m-%d")
    fig.suptitle(
        f'iPIC3D Field Solver Scaling  \u2014  '
        f'{data.np_} procs \u00b7 {data.cycles} cycles \u00b7 {date_str}',
        fontsize=12, color=theme.text_secondary, y=0.97)

    grids = data.grids
    nzc = data.nzc
    tick_labels = data.tick_labels
    tick_rotation = 45 if len(grids) > 6 else 0
    tick_ha = 'right' if tick_rotation else 'center'

    x_label_map = dict(zip(grids, tick_labels))
    x_fmt = FuncFormatter(
        lambda x, _: x_label_map.get(round(x), f'{x:.10g}'))
    plain_fmt = FuncFormatter(lambda x, _: f'{x:.10g}')

    # ── Left panel: line plot of solver times vs grid size ────────────
    for label in data.solver_labels:
        times = data.solver_times[label]
        stds  = data.solver_stds[label]
        valid = [(g, t, e) for g, t, e in zip(grids, times, stds)
                 if t is not None]
        if not valid:
            continue
        grid_vals, time_vals_raw, std_vals_raw = zip(*valid)
        vt = [t * scale for t in time_vals_raw]
        s  = get_solver_style(label)
        ax1.plot(grid_vals, vt, marker=s["marker"], linestyle=s["ls"],
                 color=s["color"], linewidth=2, markersize=7, label=label,
                 markeredgecolor=theme.marker_edge, markeredgewidth=0.8)
        has_errs = any(e is not None for e in std_vals_raw)
        if has_errs:
            ve = [e * scale if e is not None else 0 for e in std_vals_raw]
            lo = [t - e for t, e in zip(vt, ve)]
            hi = [t + e for t, e in zip(vt, ve)]
            ax1.fill_between(grid_vals, lo, hi, color=s["color"], alpha=0.12)

    # Optional reference slope (--ref-slope N)
    ref_slope = getattr(args, 'ref_slope', None)
    if ref_slope is not None:
        exp = ref_slope
        all_positive = [t * scale for t in data.all_times if t > 0]
        if len(grids) >= 2 and all_positive:
            x0, x1 = grids[0], grids[-1]
            x_mid = math.exp((math.log(x0) + math.log(x1)) / 2)
            y_mid = math.exp(
                sum(math.log(t) for t in all_positive) / len(all_positive)
            ) * 0.3
            C = y_mid / x_mid ** exp
            ax1.plot([x0, x1], [C * x0 ** exp, C * x1 ** exp],
                     color=theme.refline_color, linestyle=':', linewidth=1.5,
                     alpha=0.5, label=f'$\\propto N^{{{exp:g}}}$')

    ax1.set_xlabel(data.grid_label, fontsize=12)
    ax1.set_ylabel(f'Field solver time ({unit})', fontsize=14)
    ax1.set_title('Field Solver Time vs Grid Size',
                  fontsize=13, fontweight='bold')
    loglog = getattr(args, 'loglog', False)
    if loglog:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(plain_fmt)
    ax1.set_xticks(grids)
    ax1.xaxis.set_major_formatter(x_fmt)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right panel: improvement bars ─────────────────────────────────
    if data.ref_times and data.comparison_labels:
        render_improvement_bars(ax2, data, solver_pcts, avg_improvements,
                                args.sort, theme)

    # ── Bottom panel: component breakdown (PROFILING builds) ──────────
    ax3 = None

    if has_breakdown:
        ax3 = fig.add_subplot(gs[1, :])
        x_pos = list(range(len(data.bd_grids)))
        bar_w = 0.55

        fracs_field, fracs_moments, fracs_mover = [], [], []
        for ft, mt, pt in zip(data.bd_field, data.bd_moments, data.bd_mover):
            total = sum(v for v in [ft, mt, pt] if v is not None)
            if total > 0:
                fracs_field.append((ft or 0) / total)
                fracs_moments.append((mt or 0) / total)
                fracs_mover.append((pt or 0) / total)
            else:
                fracs_field.append(0)
                fracs_moments.append(0)
                fracs_mover.append(0)

        bottom_f = fracs_moments
        bottom_p = [m + f for m, f in zip(fracs_moments, fracs_field)]

        comp_colors = COMPONENT_COLORS[theme.mode]
        ax3.bar(x_pos, fracs_moments, bar_w,
                label='Moments (CalculateMoments)',
                color=comp_colors['moments'], alpha=0.85,
                edgecolor=theme.bar_edge)
        ax3.bar(x_pos, fracs_field, bar_w, bottom=bottom_f,
                label='Field solver (FieldSolver)',
                color=comp_colors['field'], alpha=0.85,
                edgecolor=theme.bar_edge)
        ax3.bar(x_pos, fracs_mover, bar_w, bottom=bottom_p,
                label='Particle mover (ParticlesMover)',
                color=comp_colors['mover'], alpha=0.85,
                edgecolor=theme.bar_edge)
        ax3.plot(x_pos, fracs_field, 'o--',
                 color=comp_colors['field_overlay'],
                 linewidth=1.5, markersize=5, zorder=5,
                 label='Field solver fraction')

        bd_tick_labels = [f'{g}\u00d7{nzc}' if nzc > 1 else str(g)
                          for g in data.bd_grids]
        ax3.set_xlabel(data.grid_label, fontsize=12)
        ax3.set_ylabel('Fraction of component time', fontsize=14)
        ax3.set_title(f'Time Breakdown by Component  ({data.ref_label} baseline)',
                      fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bd_tick_labels, fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    # ── Summary annotation ────────────────────────────────────────────
    render_summary_annotation(fig, gs, ax2, n_rows, data,
                              avg_improvements, args.sort, theme)

    # ── Re-apply formatters & tick rotation, then save ────────────────
    ax1.xaxis.set_major_formatter(x_fmt)
    if loglog:
        ax1.yaxis.set_major_formatter(plain_fmt)
    all_axes = [ax1, ax2] + ([ax3] if ax3 else [])
    apply_tick_rotation(all_axes, grids)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {plot_path}")


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='?', default=None, metavar='CSV_FILE',
                        help='Path to CSV (default: results.csv next to this script)')
    parser.add_argument('--loglog', action='store_true',
                        help='Use log-log axes')
    parser.add_argument('--ref-slope', type=float, default=None, metavar='N',
                        help='Show reference slope N^exp')
    parser.add_argument('--sort', action='store_true',
                        help='Group improvement bars by solver')
    parser.add_argument('--output', default=None, metavar='PATH',
                        help='Override output PNG path')
    add_theme_arg(parser)

    args = parser.parse_args()
    theme = apply_theme(args)

    csv_path = resolve_csv_path(args)
    plot_path = resolve_plot_path(args, csv_path)
    data = load_results_csv(csv_path)

    if data.single_row:
        print("  ERROR: CSV has only one grid size. Use plot_single_grid.py instead.")
        sys.exit(1)

    main(data, args, theme, plot_path)
