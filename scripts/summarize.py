#!/usr/bin/env python3
"""
summarize.py — CLI summary of an iPIC3D simulation run.

Reads output from a sim.sh run directory and prints a text report with:
  - Per-cycle timing breakdown (field solver, mover, moments, I/O)
  - Energy conservation (total energy, drift)
  - Solver convergence (iterations, convergence status)

Usage:
    python scripts/summarize.py <output_dir>
    pixi run summarize -- <output_dir>
"""

import argparse
import csv
import os
import sys

from log_patterns import CONV_SUCCESS, CONV_FAIL
from plot_utils import parse_conserved_quantities


# ── Timing from profile CSV ─────────────────────────────────────────────

def read_profile(profile_csv):
    """Read per-cycle timing from a profile CSV.

    Returns dict with lists: field, mover, moment, write, iterations.
    """
    result = {'field': [], 'mover': [], 'moment': [], 'write': [], 'iters': []}
    with open(profile_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result['field'].append(float(row['field_solver_s']))
            result['mover'].append(float(row['particle_mover_s']))
            result['moment'].append(float(row['moment_gatherer_s']))
            result['write'].append(float(row['write_data_s']))
            iters = row.get('iterations', '')
            result['iters'].append(int(iters) if iters else None)
    return result


# ── Convergence from log ────────────────────────────────────────────────

def check_convergence(log_path):
    """Parse log for convergence. Returns (n_ok, n_fail)."""
    n_ok = n_fail = 0
    with open(log_path) as f:
        for line in f:
            for p in CONV_SUCCESS:
                if p.search(line):
                    n_ok += 1
                    break
            else:
                for p in CONV_FAIL:
                    if p.search(line):
                        n_fail += 1
                        break
    return n_ok, n_fail


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', help='Simulation output directory')
    args = parser.parse_args()

    d = args.output_dir
    if not os.path.isdir(d):
        print(f'ERROR: {d} is not a directory', file=sys.stderr)
        sys.exit(2)

    print()
    print('━━━ iPIC3D Run Summary ━━━━━━━━━━━━━━━━━━━━━━━')

    # ── Timing ───────────────────────────────────────────────────────
    profile_csv = os.path.join(d, 'profile_run.csv')
    if os.path.isfile(profile_csv):
        timing = read_profile(profile_csv)
        n = len(timing['field'])
        print(f'  Cycles:    {n}')
        print()

        mean_f = sum(timing['field']) / n
        mean_m = sum(timing['mover']) / n
        mean_mo = sum(timing['moment']) / n
        mean_w = sum(timing['write']) / n
        total = mean_f + mean_m + mean_mo + mean_w

        print('  Timing (mean per cycle):')
        if total > 0:
            print(f'    Field solver:     {mean_f:.4f} s  ({mean_f/total*100:5.1f}%)')
            print(f'    Particle mover:   {mean_m:.4f} s  ({mean_m/total*100:5.1f}%)')
            print(f'    Moment gatherer:  {mean_mo:.4f} s  ({mean_mo/total*100:5.1f}%)')
            print(f'    Write data:       {mean_w:.4f} s  ({mean_w/total*100:5.1f}%)')
            print(f'    Total:            {total:.4f} s')
        else:
            print('    (no timing data)')

        # Iterations
        valid_iters = [i for i in timing['iters'] if i is not None]
        if valid_iters:
            mean_iters = sum(valid_iters) / len(valid_iters)
            print(f'    Mean iterations:  {mean_iters:.1f}')
    else:
        print('  Timing:    not available (no profile_run.csv)')
        print('             Run with logging enabled or use pixi run test')

    print()

    # ── Energy ───────────────────────────────────────────────────────
    cq = parse_conserved_quantities(d)
    if cq:
        e0 = cq['total_energy'][0]
        de = cq['delta_energy'][-1]
        n_cycles = cq['cycle'][-1]
        ratio = abs(de / e0) if abs(e0) > 1e-30 else 0
        print('  Energy conservation:')
        print(f'    E₀ = {e0:.4e}')
        print(f'    ΔE = {de:.2e}  (|ΔE/E₀| = {ratio:.2e})')
        print(f'    Cycles: {n_cycles}')
    else:
        print('  Energy:    not available (no ConservedQuantities.txt)')

    print()

    # ── Convergence ──────────────────────────────────────────────────
    log_path = os.path.join(d, 'run.log')
    if os.path.isfile(log_path):
        n_ok, n_fail = check_convergence(log_path)
        total = n_ok + n_fail
        if total > 0:
            print('  Solver convergence:')
            if n_fail == 0:
                print(f'    {n_ok}/{total} cycles converged')
            else:
                print(f'    {n_ok}/{total} converged, {n_fail} FAILED')
        else:
            print('  Convergence: no solver output found in log')
    else:
        print('  Convergence: not available (no run.log)')

    print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
