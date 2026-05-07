#!/usr/bin/env python3
"""analyze_heating.py — measure numerical heating in a quiet PIC run.

Reads ConservedQuantities.txt from a NumericalHeating-style simulation
(cold uniform Maxwellian, no perturbation, no B0) and reports:

    KE drift          : (KE_final - KE_initial) / KE_initial
    peak E-field      : max(E-field-energy) / KE_initial
    peak B-field      : max(B-field-energy) / KE_initial
    relative ΔE_total : |max(total_energy) - min(total_energy)| / KE_initial

In a perfectly resolved code total energy is bit-exact; KE and B-field stay
flat; E-field stays at machine zero. In a real PIC code at coarse grid (deep
Debye violation) particle-grid aliasing pumps KE into spurious E-field modes.

The CIC-vs-TSC contrast: TSC's quadratic B-spline shape function has a
transfer function with much faster k-roll-off than CIC's piecewise-linear,
so high-k particle features alias less back into the resolved low-k grid.

Pass criterion: peak(E-field-energy)/KE_initial < tol.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dir', help='Run output directory (contains ConservedQuantities.txt)')
    ap.add_argument('--inp', default=None, help='(Unused — kept for harness compatibility)')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Pass tolerance: peak(E-energy)/KE_initial < tol. Default 0.05.')
    ap.add_argument('--label', default=None, help='Test label for the printout')
    ap.add_argument('--skip-cycles', type=int, default=0,
                    help='Skip first N cycles for KE_initial (defaults to cycle 0).')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import parse_conserved_quantities

    cq = parse_conserved_quantities(args.output_dir)
    if not cq or len(cq['cycle']) < 8:
        print(f"FAIL: ConservedQuantities.txt missing or too short in {args.output_dir}")
        sys.exit(1)

    skip = args.skip_cycles
    KE0       = cq['KE'][skip]
    KE_final  = cq['KE'][-1]
    EE_peak   = max(cq['E_energy'])
    BE_peak   = max(cq['B_energy'])
    tot_max   = max(cq['total_energy'])
    tot_min   = min(cq['total_energy'])

    delta_KE_rel  = (KE_final - KE0) / KE0       if KE0 else float('inf')
    EE_rel        = EE_peak / KE0                if KE0 else float('inf')
    BE_rel        = BE_peak / KE0                if KE0 else float('inf')
    delta_tot_rel = (tot_max - tot_min) / KE0    if KE0 else float('inf')

    rel = EE_rel
    ok  = rel < args.tol

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    print()
    print(f"Numerical-heating check — {label}")
    print('─' * 56)
    print(f"  Snapshots               : {len(cq['cycle'])}  (cycles {cq['cycle'][0]}..{cq['cycle'][-1]})")
    print(f"  KE_initial              : {KE0:.4e}")
    print(f"  ΔKE / KE₀               : {delta_KE_rel:+.3e}")
    print(f"  peak(E-field) / KE₀     : {EE_rel:.3e}")
    print(f"  peak(B-field) / KE₀     : {BE_rel:.3e}")
    print(f"  |ΔE_total| / KE₀        : {delta_tot_rel:.3e}    (ECSIM should be ε)")
    print(f"  |EE|/KE₀                : {rel:.3e}    tol = {args.tol:.0e}")
    print('─' * 56)
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
