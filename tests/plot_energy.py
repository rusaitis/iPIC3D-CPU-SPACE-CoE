#!/usr/bin/env python3
"""
plot_energy.py — Detailed energy conservation figure for iPIC3D solver comparison.

Generates results_energy.png with two panels:
  1. Energy components (total, B, E, kinetic) vs cycle
  2. Energy drift ΔE = E(t) − E(0) vs cycle

Usage:
    python3 plot_energy.py [path/to/results.csv] [--light]

The CSV path defaults to results.csv next to this script.
Output is written alongside the CSV as results_energy.png.
"""

import argparse
import csv
import glob
import os
import re
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("  WARNING: matplotlib not installed, skipping energy plot.")
    raise SystemExit(2)

from plot_theme import (add_theme_arg, apply_theme, get_solver_style,
                        COMPONENT_COLORS, PROFILE_LINESTYLES)
from plot_utils import parse_conserved_quantities, discover_profile_files


def main(argv=None):
    # ── CLI args ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='?', default=None, metavar='CSV_FILE',
                        help='Path to results CSV (default: results.csv next to this script)')
    parser.add_argument('--output', default=None,
                        help='Override output PNG path')
    add_theme_arg(parser)
    args = parser.parse_args(argv)
    theme = apply_theme(args)

    # ── I/O paths ─────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.csv is not None:
        csv_path = args.csv
    elif "CSV_FILE" in os.environ:
        csv_path = os.environ["CSV_FILE"]
    else:
        csv_path = os.path.join(script_dir, "results.csv")

    if args.output:
        plot_path = args.output
    else:
        plot_path = os.path.splitext(csv_path)[0] + "_energy.png"

    # ── Profile file discovery ────────────────────────────────────────────
    profile_dir = os.environ.get("PROFILE_DIR", os.path.dirname(csv_path))
    profile_files, max_grid = discover_profile_files(profile_dir)

    # ── Energy data discovery ─────────────────────────────────────────────
    energy_data = {}   # label → dict from parse_conserved_quantities

    for pf in profile_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        parts = base[len("profile_"):]
        label = re.sub(r'_\d+x\d+x\d+$', '', parts)
        grid_suffix = re.search(r'_(\d+x\d+x\d+)$', parts)
        run_dir_name = f"{label}_{grid_suffix.group(1)}" if grid_suffix else parts
        run_dir = os.path.join(os.path.dirname(pf), run_dir_name)
        cq = parse_conserved_quantities(run_dir)
        if cq:
            energy_data[label] = cq

    if not energy_data:
        print("  No energy data found (ConservedQuantities.txt missing from run dirs).")
        sys.exit(0)

    # ── Figure: 2 rows, full width ───────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                  hspace=0.35, top=0.93, bottom=0.07, left=0.08, right=0.95)
    ax_comp = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1], sharex=ax_comp)

    grid_tag = f'{max_grid}\u00d7{max_grid}' if max_grid else ''
    fig.suptitle(f'iPIC3D Energy Conservation  ({grid_tag} grid)',
                 fontsize=13, color=theme.text_secondary, y=1.0)

    # ── Panel 1: Energy components ────────────────────────────────────────
    energy_colors = {
        'total_energy': theme.text_color,
        'B_energy':     COMPONENT_COLORS[theme.mode]['field'],
        'E_energy':     COMPONENT_COLORS[theme.mode]['mover'],
        'KE':           COMPONENT_COLORS[theme.mode]['moments'],
    }
    energy_labels = {
        'total_energy': '$E_{\\mathrm{tot}}$',
        'B_energy':     '$E_B$',
        'E_energy':     '$E_E$',
        'KE':           '$E_K$',
    }

    for si, (label, cq) in enumerate(energy_data.items()):
        ls = PROFILE_LINESTYLES[si % len(PROFILE_LINESTYLES)]
        cyc = cq['cycle']
        for key in ('total_energy', 'B_energy', 'E_energy', 'KE'):
            ax_comp.plot(cyc, cq[key], color=energy_colors[key], linestyle=ls,
                         linewidth=1.5, alpha=0.85,
                         label=f'{label} \u2014 {energy_labels[key]}')

    ax_comp.set_ylabel('Energy', fontsize=12)
    ax_comp.set_title(r'Energy Components  ($E_{\mathrm{tot}},\; E_B,\; E_E,\; E_K$)',
                      fontsize=13, fontweight='bold')
    ax_comp.set_ylim(bottom=0)
    ax_comp.legend(fontsize=11, ncol=2, loc='center left')
    ax_comp.grid(True, alpha=0.3)
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)

    # ── Panel 2: ΔE (energy drift) ───────────────────────────────────────
    has_delta = any('delta_energy' in cq for cq in energy_data.values())
    if has_delta:
        for label, cq in energy_data.items():
            if 'delta_energy' not in cq:
                continue
            s = get_solver_style(label)
            ax_delta.plot(cq['cycle'], cq['delta_energy'],
                          color=s['color'], linestyle='-', linewidth=1.5,
                          alpha=0.85, label=label)

    ax_delta.set_xlabel('Cycle', fontsize=12)
    ax_delta.set_ylabel(r'$\Delta E = E(t) - E(0)$', fontsize=12)
    ax_delta.set_title(r'Energy Drift  $\Delta E$', fontsize=13, fontweight='bold')
    ax_delta.set_ylim(bottom=0)
    ax_delta.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    ax_delta.legend(fontsize=11, loc='center left')
    ax_delta.grid(True, alpha=0.3)
    ax_delta.spines['top'].set_visible(False)
    ax_delta.spines['right'].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Energy plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
