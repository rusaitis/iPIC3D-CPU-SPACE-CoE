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
    ap.add_argument('--save-plot', default=None, dest='save_plot',
                    help='Optional path to save KE/EE/BE diagnostic PNG.')
    ap.add_argument('--compare', default=None,
                    help='Optional second run dir to overlay on the same axes (CIC↔TSC contrast).')
    ap.add_argument('--compare-label', default=None, dest='compare_label',
                    help='Label for --compare overlay (default: basename of compare dir).')
    ap.add_argument('--light', action='store_true',
                    help='Use light theme for the saved plot (default dark).')
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

    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from plot_theme import apply_theme
        except ImportError as e:
            print(f"  WARNING: --save-plot disabled, missing dependency: {e}")
        else:
            theme = apply_theme(args)
            ke_color = theme.text_color
            ee_color = '#f38ba8' if theme.mode == 'dark' else '#EE6677'
            be_color = '#89b4fa' if theme.mode == 'dark' else '#4477AA'
            cmp_color = '#a6e3a1' if theme.mode == 'dark' else '#228833'

            cycles = np.array(cq['cycle'], dtype=float)
            t = cycles * 1.0   # cycle index — dt unknown without .inp; cycle is the natural unit

            # Log-y so KE (~1e-6), EE (~1e-9..1e-8), BE all visible together.
            floor = max(1e-300, 1e-6 * float(np.max(KE)))
            KE_pos = np.maximum(KE, floor)
            EE_pos = np.maximum(EE, floor)
            BE_pos = np.maximum(BE, floor)

            fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.0),
                                    constrained_layout=True)

            ax.plot(t, KE_pos, color=ke_color, lw=1.4, label='KE')
            ax.plot(t, EE_pos, color=ee_color, lw=1.4, label='E-field energy')
            ax.plot(t, BE_pos, color=be_color, lw=1.4, label='B-field energy')
            ax.set_yscale('log')

            # KE0 window (first n_ke0 cycles)
            ax.axvspan(float(t[0]), float(t[n_ke0 - 1]), color=cmp_color, alpha=0.18,
                       label=f'<KE>_0 window ({n_ke0}c)')
            # Saturated window (last n_sat cycles)
            ax.axvspan(float(t[-n_sat]), float(t[-1]), color=ee_color, alpha=0.13,
                       label=f'saturated window ({n_sat}c)')

            ax.axhline(KE0, color=ke_color, lw=0.8, ls=':', alpha=0.6)

            if args.compare:
                cq2 = parse_conserved_quantities(args.compare)
                if cq2 and len(cq2['cycle']) > 0:
                    KE2 = np.maximum(np.array(cq2['KE']), floor)
                    EE2 = np.maximum(np.array(cq2['E_energy']), floor)
                    t2 = np.array(cq2['cycle'], dtype=float)
                    cmp_label = args.compare_label or os.path.basename(args.compare.rstrip('/'))
                    ax.plot(t2, KE2, color=cmp_color, lw=1.2, ls='--',
                            label=f'KE  ({cmp_label})', alpha=0.85)
                    ax.plot(t2, EE2, color=cmp_color, lw=1.0, ls=':',
                            label=f'E-field  ({cmp_label})', alpha=0.85)

            ax.set_xlabel('cycle')
            ax.set_ylabel('energy (code units, log scale)')
            ax.legend(loc='center right', fontsize=9)
            ax.text(0.02, 0.98,
                    f'<E>_sat / <KE>_0  = {EE_rel:.3e}\n'
                    f'<B>_sat / <KE>_0  = {BE_rel:.3e}\n'
                    f'd<KE> / <KE>_0    = {delta_KE_rel:+.3e}\n'
                    f'|dE_total| / <KE>_0 = {delta_tot_rel:.3e}',
                    transform=ax.transAxes, va='top', fontsize=9, family='monospace',
                    bbox=dict(facecolor=theme.box_face,
                              edgecolor=theme.box_edge,
                              alpha=theme.box_alpha))

            verdict = 'PASS' if ok else 'FAIL'
            fig.suptitle(f'{label}  —  <E>_sat / <KE>_0 = {EE_rel:.3e}  [{verdict}]',
                         fontsize=12)
            fig.savefig(args.save_plot, dpi=140)
            plt.close(fig)
            print(f"  Saved diagnostic plot → {args.save_plot}")
            print()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
