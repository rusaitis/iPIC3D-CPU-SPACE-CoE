#!/usr/bin/env python3
"""
plot_timing.py — Generate solver comparison plots from results.csv.

Dispatcher: reads the CSV, detects single-grid vs scaling mode, and routes
to plot_single_grid or plot_scaling accordingly.

Usage:
    python3 plot_timing.py [path/to/results.csv] [--loglog] [--ref-slope N] [--sort]

The CSV path defaults to results.csv next to this script.
The output PNG is written alongside the CSV with the same base name.

All configuration (solver labels, np, cycles, nzc, breakdown data) is read
directly from the CSV. Environment variables are still accepted as overrides
for backward compatibility when called from the test script.
"""

import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("  WARNING: matplotlib not installed, skipping plot generation.")
    print("  Install with: pip3 install matplotlib")
    raise SystemExit(2)

from plot_theme import add_theme_arg, apply_theme
from plot_timing_common import load_results_csv, resolve_csv_path, resolve_plot_path


def main(argv=None):
    # ── CLI args (superset of both modes) ────────────────────────────────
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='?', default=None, metavar='CSV_FILE',
                        help='Path to CSV (default: results.csv next to this script)')
    parser.add_argument('--output', default=None, metavar='PATH',
                        help='Override output PNG path (priority over PLOT_FILE env var)')
    parser.add_argument('--sort', action='store_true',
                        help='Group improvement bars by solver (sorted by avg improvement)')
    # Single-grid mode args
    parser.add_argument('--profile-smooth', type=int, default=5, metavar='N',
                        help='Smooth profile plot with rolling average over N cycles (0 to disable)')
    # Scaling mode args
    parser.add_argument('--loglog', action='store_true',
                        help='Use log-log axes for the time plot (default: linear)')
    parser.add_argument('--ref-slope', type=float, default=None, metavar='N',
                        help='Show reference slope N^exp (e.g. --ref-slope 2)')
    add_theme_arg(parser)

    args = parser.parse_args(argv)
    theme = apply_theme(args)

    csv_path = resolve_csv_path(args)

    # Auto-aggregate profile CSVs if results.csv doesn't exist yet
    if not os.path.isfile(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "test_output")
        if os.path.isdir(output_dir):
            from plot_utils import aggregate_profiles_to_csv
            try:
                csv_path = aggregate_profiles_to_csv(output_dir)
                print(f"  Auto-aggregated profiles → {csv_path}")
            except FileNotFoundError:
                pass  # no profiles either — let load_results_csv report the error

    plot_path = resolve_plot_path(args, csv_path)
    data = load_results_csv(csv_path)

    if data.single_row:
        from plot_single_grid import main as render
    else:
        from plot_scaling import main as render

    render(data, args, theme, plot_path)


if __name__ == "__main__":
    main()
