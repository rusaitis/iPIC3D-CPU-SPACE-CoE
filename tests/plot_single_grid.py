#!/usr/bin/env python3
"""
plot_single_grid.py — Single-grid solver comparison figure for iPIC3D.

Generates a multi-panel figure comparing solver performance at a single
grid size: bar chart of solver times, improvement bars, per-cycle timing
profile, and energy drift.

Usage (standalone):
    python3 plot_single_grid.py [results.csv] [--sort] [--profile-smooth N] [--output PATH]

Usually invoked via the dispatcher plot_petsc_test_results.py.
"""

import argparse
import csv
import os
import re
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
                        COMPONENT_COLORS, PROFILE_LINESTYLES)
from plot_results_common import (
    ResultsData, load_results_csv, resolve_csv_path, resolve_plot_path,
    darken_color, auto_time_unit, compute_improvements,
    render_improvement_bars, render_summary_annotation, apply_tick_rotation,
)


def main(data, args, theme, plot_path):
    """Render the single-grid comparison figure and save to plot_path."""
    unit, scale = auto_time_unit(data.all_times)
    solver_pcts, avg_improvements = compute_improvements(data)

    has_bottom = data.has_profile
    has_energy = data.has_energy

    # ── Layout ────────────────────────────────────────────────────────
    n_rows = 1
    height_ratios = [3]
    if has_bottom:
        n_rows = 2
        height_ratios = [3, 2]
    if has_energy:
        n_rows = 3
        height_ratios = [3, 2, 2]

    fig_height = {1: 6, 2: 10, 3: 14}[n_rows]
    fig = plt.figure(figsize=(15, fig_height))
    gs  = GridSpec(n_rows, 2, figure=fig,
                   height_ratios=height_ratios,
                   hspace=0.45, wspace=0.35,
                   top=0.93, bottom=0.06, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    date_str = datetime.now().strftime("%Y-%m-%d")
    suptitle_y = 0.98 if n_rows >= 3 else 0.97
    fig.suptitle(
        f'iPIC3D Solver Comparison  \u2014  {data.grid_strings[0]}  \u00b7  '
        f'{data.np_} procs \u00b7 {data.cycles} cycles \u00b7 {date_str}',
        fontsize=12, color=theme.text_secondary, y=suptitle_y)

    # ── Left panel: bar chart of solver times ─────────────────────────
    bar_x = list(range(len(data.solver_labels)))
    bar_vals = []
    bar_errs = []
    bar_colors = []
    for label in data.solver_labels:
        t = data.solver_times[label][0]
        e = data.solver_stds[label][0]
        bar_vals.append(t * scale if t is not None else 0)
        bar_errs.append(e * scale if e is not None else 0)
        s = get_solver_style(label)
        bar_colors.append(s["color"])

    bars = ax1.bar(bar_x, bar_vals, color=bar_colors, alpha=0.85,
                   edgecolor=theme.bar_edge, linewidth=0.5,
                   yerr=bar_errs if any(e > 0 for e in bar_errs) else None,
                   capsize=5)
    for bar, val, col in zip(bars, bar_vals, bar_colors):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val / 2,
                     f'{val:.1f}', ha='center', va='center',
                     fontsize=20, fontweight='heavy', color=darken_color(col))

    ax1.set_xticks(bar_x)
    ax1.set_xticklabels(data.solver_labels, fontsize=10)
    ax1.set_ylabel(f'Field solver time ({unit})', fontsize=14)
    ax1.set_title(f'Field Solver Time  ({data.grid_strings[0]})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Right panel: improvement bars ─────────────────────────────────
    if data.gmres_times and data.petsc_labels:
        render_improvement_bars(ax2, data, solver_pcts, avg_improvements,
                                args.sort, theme)

    # ── Profile panel (row 2) ─────────────────────────────────────────
    ax3 = None
    profile_iterations = {}

    if has_bottom:
        ax3 = fig.add_subplot(gs[1, :])
        comp_colors = COMPONENT_COLORS[theme.mode]
        component_styles = [
            ('field_solver_s',     'Field solver',     comp_colors['field']),
            ('particle_mover_s',   'Particle mover',   comp_colors['mover']),
            ('moment_gatherer_s',  'Moment gatherer',  comp_colors['moments']),
        ]
        solver_linestyles = PROFILE_LINESTYLES
        plot_cycles = []

        for si, pf in enumerate(data.profile_files):
            base = os.path.splitext(os.path.basename(pf))[0]
            parts = base[len("profile_"):]
            label = re.sub(r'_\d+x\d+x\d+$', '', parts)
            ls = solver_linestyles[si % len(solver_linestyles)]

            with open(pf, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                continue

            cycles_col = [int(r['cycle']) for r in rows]

            # Collect per-cycle iteration counts if available
            if ('iterations' in rows[0]
                    and any(r.get('iterations', '') for r in rows)):
                iter_cycles = []
                iter_vals = []
                for r in rows:
                    v = r.get('iterations', '')
                    if v:
                        iter_cycles.append(int(r['cycle']))
                        iter_vals.append(int(v))
                if iter_vals:
                    profile_iterations[label] = (iter_cycles, iter_vals)

            smooth = getattr(args, 'profile_smooth', 5)
            for col, comp_name, color in component_styles:
                if col not in rows[0]:
                    continue
                vals = [float(r[col]) for r in rows]
                plot_cycles, plot_vals = cycles_col, vals
                if smooth and smooth > 1:
                    half = smooth // 2
                    ax3.plot(cycles_col, vals, color=color, linestyle=ls,
                             linewidth=0.5, alpha=0.3)
                    plot_vals = [
                        sum(vals[max(0, i - half):i + half + 1])
                        / len(vals[max(0, i - half):i + half + 1])
                        for i in range(len(vals))
                    ]
                ax3.plot(plot_cycles, plot_vals, color=color, linestyle=ls,
                         linewidth=1.5, alpha=0.85,
                         label=f'{label} \u2014 {comp_name}')

        ax3.set_xlabel('Cycle', fontsize=12)
        ax3.set_ylabel('Time per cycle (s)', fontsize=14)
        grid_tag = (f'{data.max_grid}\u00d7{data.max_grid}'
                    if data.profile_files else '')
        ax3.set_title(f'Per-Cycle Timing Profile  ({grid_tag} grid)',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, ncol=2, loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        if plot_cycles:
            ax3.set_xlim(plot_cycles[0], plot_cycles[-1])

    # ── Energy drift + iterations panel (row 3) ──────────────────────
    ax4 = None

    if has_energy:
        ax4 = fig.add_subplot(gs[2, :], sharex=ax3)

        has_delta = any('delta_energy' in cq
                        for cq in data.energy_data.values())
        if has_delta:
            for label, cq in data.energy_data.items():
                if 'delta_energy' not in cq:
                    continue
                e0 = cq['total_energy'][0]
                delta_pct = [d / e0 * 100 for d in cq['delta_energy']]
                s = get_solver_style(label)
                ax4.plot(cq['cycle'], delta_pct,
                         color=s['color'], linestyle='-', linewidth=1.5,
                         alpha=0.85, label=f'{label} \u0394E/E\u2080')
            ax4.set_ylabel('\u0394E / E\u2080 (%)', fontsize=14)
            ax4.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f'{v:.4g}%'))

        if profile_iterations:
            ax4_iters = ax4.twinx()
            for label, (iter_cyc, iter_vals) in profile_iterations.items():
                s = get_solver_style(label)
                ax4_iters.plot(iter_cyc, iter_vals, color=s['color'],
                               linestyle='--', linewidth=1.2,
                               alpha=0.4, label=f'{label} iters')
            ax4_iters.set_ylabel('Iterations per solve', fontsize=12)
            ax4_iters.yaxis.set_major_locator(
                plt.MaxNLocator(integer=True))
            ax4_iters.spines['top'].set_visible(False)

        ax4.set_xlabel('Cycle', fontsize=12)
        grid_tag = (f'{data.max_grid}\u00d7{data.max_grid}'
                    if data.profile_files else '')
        ax4.set_title(
            r'Energy Drift ($\Delta E / E_0$) & Iterations'
            + f'  ({grid_tag} grid)',
            fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        for cq in data.energy_data.values():
            if cq.get('cycle'):
                ax4.set_xlim(cq['cycle'][0], cq['cycle'][-1])
                break

        # Combined legend
        handles, labels_leg = ax4.get_legend_handles_labels()
        if profile_iterations:
            h2, l2 = ax4_iters.get_legend_handles_labels()
            handles += h2
            labels_leg += l2
        ax4.legend(handles, labels_leg, fontsize=10, ncol=2,
                   loc='upper left')

    # ── Summary annotation ────────────────────────────────────────────
    render_summary_annotation(fig, gs, ax2, n_rows, data,
                              avg_improvements, args.sort, theme)

    # ── Tick rotation & save ──────────────────────────────────────────
    all_axes = [ax1, ax2] + ([ax3] if ax3 else []) + ([ax4] if ax4 else [])
    apply_tick_rotation(all_axes, data.grids)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {plot_path}")


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='?', default=None, metavar='CSV_FILE',
                        help='Path to CSV (default: results.csv next to this script)')
    parser.add_argument('--sort', action='store_true',
                        help='Group improvement bars by solver')
    parser.add_argument('--profile-smooth', type=int, default=5, metavar='N',
                        help='Smooth profile plot with rolling average over N cycles')
    parser.add_argument('--output', default=None, metavar='PATH',
                        help='Override output PNG path')
    add_theme_arg(parser)

    args = parser.parse_args()
    theme = apply_theme(args)

    csv_path = resolve_csv_path(args)
    plot_path = resolve_plot_path(args, csv_path)
    data = load_results_csv(csv_path)

    if not data.single_row:
        print("  ERROR: CSV has multiple grid sizes. Use plot_scaling.py instead.")
        sys.exit(1)

    main(data, args, theme, plot_path)
