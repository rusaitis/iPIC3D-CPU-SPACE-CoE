#!/usr/bin/env python3
"""
validate.py — Automated pass/fail validation for iPIC3D test output.

Checks:
  1. Energy conservation: |ΔE/E₀| at final cycle < tolerance
  2. Solver convergence:  all cycles converged (from persisted log files)
  3. Cross-solver L2:     field agreement between reference and test solvers

Usage:
  python scripts/validate.py <output_dir> [options]
  python scripts/validate.py tests/test_output/ --energy-tol 1e-6 --l2-tol 1e-10

Exit code 0 = all checks passed, 1 = any check failed.
"""

import argparse
import os
import sys

from log_patterns import CONV_SUCCESS, CONV_FAIL
from plot_utils import (parse_conserved_quantities, load_field,
                        discover_run_groups,
                        parse_field_specs, find_common_cycles)


def check_convergence_log(log_path):
    """Parse a log file for convergence status.

    Returns (n_converged, n_failed, total) tuple.
    """
    n_ok = 0
    n_fail = 0
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
    return n_ok, n_fail, n_ok + n_fail


# ── Validation checks ───────────────────────────────────────────────────

def validate_energy(output_dir, tol, results):
    """Check energy conservation for all solver run dirs."""
    import glob
    # Find all directories with ConservedQuantities.txt
    for d in sorted(glob.glob(os.path.join(output_dir, '*'))):
        if not os.path.isdir(d):
            continue
        cq = parse_conserved_quantities(d)
        if not cq:
            continue
        label = os.path.basename(d)
        e0 = cq['total_energy'][0]
        de_final = cq['delta_energy'][-1]
        if abs(e0) < 1e-30:
            results.append(('SKIP', f'Energy: {label}', 'E₀ ≈ 0'))
            continue
        ratio = abs(de_final / e0)
        passed = ratio < tol
        status = 'PASS' if passed else 'FAIL'
        results.append((status, f'Energy: {label}', f'|ΔE/E₀| = {ratio:.2e}'))


def validate_convergence(output_dir, results):
    """Check solver convergence from persisted log files."""
    import glob
    log_files = sorted(glob.glob(os.path.join(output_dir, '*.log')))
    if not log_files:
        results.append(('SKIP', 'Convergence', 'No log files found'))
        return
    for log_path in log_files:
        label = os.path.splitext(os.path.basename(log_path))[0]
        n_ok, n_fail, total = check_convergence_log(log_path)
        if total == 0:
            results.append(('SKIP', f'Convergence: {label}', 'No solver output in log'))
            continue
        if n_fail > 0:
            results.append(('FAIL', f'Convergence: {label}',
                            f'{n_fail}/{total} cycles did not converge'))
        else:
            results.append(('PASS', f'Convergence: {label}',
                            f'all {n_ok} cycles converged'))


def validate_l2(output_dir, tol, field_specs, results):
    """Check cross-solver L2 error for field components."""
    try:
        import numpy as np
        import h5py  # noqa: F401 — needed by load_field
    except ImportError:
        results.append(('SKIP', 'L2 cross-solver', 'numpy/h5py not available'))
        return

    groups = discover_run_groups(output_dir, require_fields=True)
    if not groups:
        results.append(('SKIP', 'L2 cross-solver', 'No field output found'))
        return

    for grid_tag, ref_dir, test_dirs in groups:
        ref_label = os.path.basename(ref_dir)
        for test_dir in test_dirs:
            test_label = os.path.basename(test_dir)
            common = find_common_cycles(ref_dir, test_dir)
            if not common:
                results.append(('SKIP', f'L2: {test_label} vs {ref_label}',
                                'No common cycles'))
                continue
            last_cycle = common[-1]
            for ftype, comp, _ in field_specs:
                fname = f'{ftype}{comp}'
                try:
                    ref_data = load_field(ref_dir, last_cycle, ftype, comp)
                    test_data = load_field(test_dir, last_cycle, ftype, comp)
                except (FileNotFoundError, KeyError) as e:
                    results.append(('SKIP', f'L2: {test_label} {fname}', str(e)))
                    continue
                ref_norm = np.linalg.norm(ref_data.ravel())
                if ref_norm < 1e-30:
                    results.append(('SKIP', f'L2: {test_label} {fname}',
                                    '||ref|| ≈ 0'))
                    continue
                l2 = np.linalg.norm((test_data - ref_data).ravel()) / ref_norm
                passed = l2 < tol
                status = 'PASS' if passed else 'FAIL'
                results.append((status, f'L2: {test_label} {fname} @ cycle {last_cycle}',
                                f'{l2:.2e}'))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate iPIC3D test output (energy, convergence, L2).')
    parser.add_argument('output_dir', help='Test output directory')
    parser.add_argument('--energy-tol', type=float, default=1e-6,
                        help='Max |ΔE/E₀| at final cycle (default: 1e-6)')
    parser.add_argument('--l2-tol', type=float, default=1e-10,
                        help='Max L2 relative error between solvers (default: 1e-10)')
    parser.add_argument('--fields', default='Bx,Ez',
                        help='Field components for L2 check (default: Bx,Ez)')
    parser.add_argument('--strict', action='store_true',
                        help='Exit non-zero on any failure (default: only for energy/convergence)')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f'ERROR: {args.output_dir} is not a directory', file=sys.stderr)
        sys.exit(2)

    field_specs = parse_field_specs(args.fields)
    results = []  # list of (status, check_name, detail)

    # Run checks
    validate_energy(args.output_dir, args.energy_tol, results)
    validate_convergence(args.output_dir, results)
    validate_l2(args.output_dir, args.l2_tol, field_specs, results)

    # Print results
    use_color = sys.stdout.isatty()
    print()
    print('Validation Results')
    print('━' * 50)
    n_pass = n_fail = n_skip = 0
    for status, name, detail in results:
        if status == 'PASS':
            tag = '\033[32m[PASS]\033[0m' if use_color else '[PASS]'
            n_pass += 1
        elif status == 'FAIL':
            tag = '\033[31m[FAIL]\033[0m' if use_color else '[FAIL]'
            n_fail += 1
        else:
            tag = '\033[33m[SKIP]\033[0m' if use_color else '[SKIP]'
            n_skip += 1
        print(f'  {tag} {name}  {detail}')
    print('━' * 50)

    total_checked = n_pass + n_fail
    summary = f'  {n_pass}/{total_checked} checks passed'
    if n_skip:
        summary += f', {n_skip} skipped'
    if n_fail:
        summary += f', {n_fail} FAILED'
    print(summary)
    print()

    if n_fail > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
