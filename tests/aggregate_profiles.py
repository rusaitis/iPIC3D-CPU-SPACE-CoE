#!/usr/bin/env python3
"""
aggregate_profiles.py — Build results.csv from profile_*.csv files.

Scans a directory for profile CSVs (profile_{solver}_{NxNxN}.csv), sums
per-cycle timings to get totals, and writes a results.csv compatible with
the plotting scripts.

Usage:
    python3 aggregate_profiles.py [OUTPUT_DIR]        # default: tests/test_output
    python3 aggregate_profiles.py tests/test_output --ref-solver GMRES
"""

import argparse
import os
import sys

from plot_utils import aggregate_profiles_to_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', nargs='?',
                        default=os.path.join(os.path.dirname(__file__),
                                             'test_output'),
                        help='Directory containing profile_*.csv files')
    parser.add_argument('--ref-solver', default='',
                        help='Reference solver label (default: auto-detect '
                             'from ref_solver.txt or GMRES)')
    parser.add_argument('--np', default='8',
                        help='Number of MPI processes (for CSV metadata)')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: {args.output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    try:
        csv_path = aggregate_profiles_to_csv(
            args.output_dir, ref_solver=args.ref_solver, np_str=args.np)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  CSV saved to: {csv_path}")
    print()
    print(f"  Now run the plots:")
    print(f"    pixi run plot-timing -- {csv_path}")
    print(f"    pixi run plot-energy -- {csv_path}")


if __name__ == '__main__':
    main()
