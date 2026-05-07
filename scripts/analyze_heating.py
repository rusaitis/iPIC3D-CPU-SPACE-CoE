#!/usr/bin/env python3
"""analyze_heating.py — measure numerical heating in a quiet PIC run.

Reads ConservedQuantities.txt from a NumericalHeating-style simulation
(cold uniform Maxwellian, no perturbation, no B0) and reports:

    KE drift          : (⟨KE⟩_final - ⟨KE⟩_initial) / ⟨KE⟩_initial
    saturated E-field : ⟨E-field-energy⟩_saturated / ⟨KE⟩_initial
    saturated B-field : ⟨B-field-energy⟩_saturated / ⟨KE⟩_initial
    relative ΔE_total : |max(total_energy) - min(total_energy)| / ⟨KE⟩_initial

In a perfectly resolved code total energy is bit-exact; KE and B-field stay
flat; E-field stays at machine zero. In a real PIC code at coarse grid (deep
Debye violation) particle-grid aliasing pumps KE into spurious E-field modes.

⟨·⟩ denotes a time-average to suppress incidental peaks. KE_initial averages
over the first --ke0-window-cycles samples (the system needs a few cycles
to settle from the random Maxwellian draw); E/B saturated values average
over the last --saturated-window-cycles samples (after the rapid initial
thermalization phase, typically the first ~50 cycles).

The CIC-vs-TSC contrast: TSC's quadratic B-spline shape function has a
transfer function with much faster k-roll-off than CIC's piecewise-linear,
so high-k particle features alias less back into the resolved low-k grid.

Pass criterion: ⟨E-field-energy⟩_saturated / ⟨KE⟩_initial < tol.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dir', help='Run output directory (contains ConservedQuantities.txt)')
    ap.add_argument('--inp', default=None, help='(Unused — kept for harness compatibility)')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Pass tolerance: ⟨E-energy⟩_saturated / ⟨KE⟩_initial < tol. Default 0.05.')
    ap.add_argument('--label', default=None, help='Test label for the printout')
    ap.add_argument('--ke0-window-cycles', type=int, default=5, dest='ke0_window',
                    help='KE₀ = mean over first N cycles (default 5).')
    ap.add_argument('--saturated-window-cycles', type=int, default=200, dest='sat_window',
                    help='Saturated metric = mean over last N cycles (default 200).')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_utils import parse_conserved_quantities

    try:
        import numpy as np
    except ImportError:
        print("FAIL: numpy required")
        sys.exit(1)

    cq = parse_conserved_quantities(args.output_dir)
    if not cq or len(cq['cycle']) < args.ke0_window + 8:
        print(f"FAIL: ConservedQuantities.txt missing or too short in {args.output_dir}")
        sys.exit(1)

    KE = np.array(cq['KE'])
    EE = np.array(cq['E_energy'])
    BE = np.array(cq['B_energy'])
    TE = np.array(cq['total_energy'])
    n  = len(KE)

    n_ke0 = min(args.ke0_window, n)
    n_sat = min(args.sat_window, n - n_ke0)

    KE0       = float(np.mean(KE[:n_ke0]))
    KE_final  = float(np.mean(KE[-n_sat:]))
    EE_sat    = float(np.mean(EE[-n_sat:]))
    BE_sat    = float(np.mean(BE[-n_sat:]))

    delta_KE_rel  = (KE_final - KE0) / KE0       if KE0 else float('inf')
    EE_rel        = EE_sat / KE0                 if KE0 else float('inf')
    BE_rel        = BE_sat / KE0                 if KE0 else float('inf')
    delta_tot_rel = (float(TE.max()) - float(TE.min())) / KE0 if KE0 else float('inf')

    rel = EE_rel
    ok  = rel < args.tol

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    print()
    print(f"Numerical-heating check — {label}")
    print('─' * 56)
    print(f"  Snapshots               : {n}  (cycles {cq['cycle'][0]}..{cq['cycle'][-1]})")
    print(f"  ⟨KE⟩₀  (first {n_ke0} cycles) : {KE0:.4e}")
    print(f"  Δ⟨KE⟩ / ⟨KE⟩₀ (last {n_sat}c) : {delta_KE_rel:+.3e}")
    print(f"  ⟨E-field⟩_sat / ⟨KE⟩₀  : {EE_rel:.3e}")
    print(f"  ⟨B-field⟩_sat / ⟨KE⟩₀  : {BE_rel:.3e}")
    print(f"  |ΔE_total| / ⟨KE⟩₀     : {delta_tot_rel:.3e}    (ECSIM should be ε)")
    print(f"  |EE|/KE₀                : {rel:.3e}    tol = {args.tol:.0e}")
    print('─' * 56)
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
