#!/usr/bin/env python3
"""analyze_reconnection.py — measure GEM-challenge reconnected flux ψ(t).

Reads B_y snapshots from an iPIC3D Double_Harris run, computes the
reconnected magnetic flux at the lower X-line, and compares the peak
reconnection rate to the GEM-challenge consensus value (Birn et al. 2001).

Definition.
    Along the lower sheet midline y_X = Ly/4 :
        A_z(x, t) = -∫₀^x B_y(x', y_X, t) dx'
        ψ(t)      = max_x A_z(x, t) - min_x A_z(x, t)
    A_z is the in-plane vector-potential component (B_x = ∂A_z/∂y,
    B_y = -∂A_z/∂x). ψ measures the flux between the X- and O-points.

Theory.
    GEM-challenge reference (full PIC, hybrid, Hall-MHD, etc.; Birn 2001
    Fig. 4-6): peak reconnection rate dψ/dt | (B0·v_A)  ≈  0.1 - 0.2.
    Below ~0.05 means slow Sweet-Parker-style reconnection (incorrect
    for collisionless plasmas); above ~0.3 means runaway / unphysical.

Pass criterion.
    peak rate / (B0·v_A) ∈ [tol_lo, tol_hi]  (default 0.05 .. 0.3).

Diagnostic plot (`--save-plot`):
    Top  — ψ(t) with peak rate annotated.
    Bot  — dψ/dt(t) with peak marked and the theory band [0.05, 0.3]·B0·v_A
           shaded for visual reference.
"""

import argparse
import math
import os
import sys


def parse_input_file(path):
    keys = {'dt', 'ncycles', 'c', 'Lx', 'Ly', 'Lz', 'nxc', 'nyc', 'nzc',
            'B0x', 'B0y', 'B0z', 'FieldOutputCycle', 'ns'}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if '=' not in line:
                continue
            lhs, rhs = (s.strip() for s in line.split('=', 1))
            if lhs in keys:
                try:
                    out[lhs] = float(rhs.split()[0])
                except ValueError:
                    out[lhs] = rhs
            elif lhs in ('rhoINIT', 'qom'):
                try:
                    out[lhs] = [float(x) for x in rhs.split()]
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', help='Run output directory (contains Fields_*/).')
    ap.add_argument('--inp', required=True, help='Path to the .inp used for the run.')
    ap.add_argument('--y-frac', type=float, default=0.25, dest='y_frac',
                    help='Sheet-midline y-position as fraction of Ly (default 0.25 → lower sheet).')
    ap.add_argument('--rate-lo', type=float, default=0.05, dest='rate_lo',
                    help='Pass lower bound on peak rate / (B0·v_A) (default 0.05).')
    ap.add_argument('--rate-hi', type=float, default=0.30, dest='rate_hi',
                    help='Pass upper bound on peak rate / (B0·v_A) (default 0.30).')
    ap.add_argument('--dpsi-floor-frac', type=float, default=1.0, dest='dpsi_floor_frac',
                    help='Pass floor: Δψ / ψ(0) > frac. Default 1.0 (reconnected '
                         'flux must at least double the perturbation seed).')
    ap.add_argument('--energy-cons-tol', type=float, default=1e-9, dest='energy_cons_tol',
                    help='ECSIM total-energy drift tolerance |ΔE_total|/E_initial. Default 1e-9.')
    ap.add_argument('--ub-decrease-frac', type=float, default=0.005, dest='ub_decrease_frac',
                    help='Pass floor: magnetic-energy decrease must exceed this fraction of U_B(0). '
                         'Default 0.005 (0.5%%) — reconnection releases magnetic energy.')
    ap.add_argument('--smooth-cycles', type=int, default=3, dest='smooth_cycles',
                    help='Boxcar window (in output snapshots) for dψ/dt smoothing.')
    ap.add_argument('--label', default=None, help='Test label for the printout.')
    ap.add_argument('--save-plot', default=None, dest='save_plot',
                    help='Optional path to save the ψ(t) + dψ/dt diagnostic PNG.')
    ap.add_argument('--light', action='store_true',
                    help='Use light theme for the saved plot.')
    args = ap.parse_args()

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_utils import discover_cycles, load_field, parse_conserved_quantities
        import numpy as np
    except ImportError as e:
        print(f"FAIL: missing dependency: {e}")
        sys.exit(2)

    sim = parse_input_file(args.inp)
    cycles = discover_cycles(args.output_dir)
    if len(cycles) < 8:
        print(f"FAIL: only {len(cycles)} field snapshots in {args.output_dir}")
        sys.exit(1)

    Lx = sim['Lx']
    Ly = sim['Ly']
    nxc = int(sim['nxc'])
    nyc = int(sim['nyc'])
    dx = Lx / nxc
    dt = sim['dt']
    sample_dt = dt * (cycles[1] - cycles[0])

    # Sheet-midline node row index. iPIC3D field arrays are node-centred with
    # nxn=nxc+1; the row at y = y_frac·Ly maps to j = round(y_frac · nyc).
    j_X = int(round(args.y_frac * nyc))

    # Reconnected flux ψ(t) = max_x A_z(x, y_X, t) - min_x A_z(x, y_X, t)
    psi = np.empty(len(cycles))
    for n, cyc in enumerate(cycles):
        By = load_field(args.output_dir, cyc, 'B', 'y')   # shape (nxn, nyn, nzn)
        # Lower sheet, mid-z. By along x at j_X gives -∂A_z/∂x.
        line_By = By[:nxc, j_X, 0]   # length nxc, exclude duplicate periodic node
        Az = -np.cumsum(line_By) * dx
        psi[n] = float(Az.max() - Az.min())

    t = np.array(cycles, dtype=float) * dt

    # dψ/dt with a small boxcar smoothing to suppress single-cycle noise.
    dpsi_dt = np.gradient(psi, t)
    if args.smooth_cycles > 1:
        w = args.smooth_cycles
        kernel = np.ones(w) / w
        dpsi_dt = np.convolve(dpsi_dt, kernel, mode='same')

    # Skip the first few snapshots — initial perturbation transient gives a
    # spurious large dψ/dt at t=0 unrelated to the reconnection rate.
    t_skip = 5.0  # code time
    i_skip = int(np.searchsorted(t, t_skip))
    if i_skip >= len(t) - 4:
        i_skip = max(2, len(t) // 10)
    peak_idx = int(i_skip + np.argmax(dpsi_dt[i_skip:]))
    peak_rate = float(dpsi_dt[peak_idx])
    peak_t = float(t[peak_idx])

    # v_A from sim params (same convention as analyze_dispersion.py).
    rho = sim.get('rhoINIT', [])
    qom = sim.get('qom', [])
    mass_density_norm = sum(abs(r) / abs(q) for r, q in zip(rho, qom) if q != 0)
    B0_mag = math.sqrt(sim.get('B0x', 0)**2 + sim.get('B0y', 0)**2 + sim.get('B0z', 0)**2)
    v_A = B0_mag / math.sqrt(mass_density_norm) if mass_density_norm > 0 else float('nan')
    omega_ci = abs(qom[1]) * B0_mag / sim.get('c', 1.0) if len(qom) > 1 else float('nan')

    rate_norm_unit = B0_mag * v_A
    rate_norm = peak_rate / rate_norm_unit if rate_norm_unit > 0 else float('nan')

    # Δψ-floor: ensure substantial reconnection occurred. Defends against a
    # noise-only run that happens to have a single dψ/dt fluctuation in band.
    delta_psi = psi[-1] - psi[0]
    psi_growth_ratio = delta_psi / psi[0] if psi[0] > 0 else float('inf')

    # ECSIM total-energy conservation + magnetic-to-kinetic conversion budget.
    # Reconnection releases U_B → ΔKE + Δthermal. Total energy stays at ε.
    cq = parse_conserved_quantities(args.output_dir)
    if cq and cq['cycle']:
        UB = np.array(cq['B_energy'])
        UE = np.array(cq['E_energy'])
        KE = np.array(cq['KE'])
        TE = np.array(cq['total_energy'])
        delta_UB    = float(UB[-1] - UB[0])
        delta_KE    = float(KE[-1] - KE[0])
        delta_UE    = float(UE[-1] - UE[0])
        UB0         = float(UB[0])
        TE0         = float(TE[0])
        energy_drift = float(TE.max() - TE.min()) / max(abs(TE0), 1e-300)
        ub_decrease_frac = -delta_UB / UB0 if UB0 > 0 else 0.0
    else:
        delta_UB = delta_KE = delta_UE = UB0 = TE0 = float('nan')
        energy_drift = float('nan')
        ub_decrease_frac = float('nan')

    rate_ok    = args.rate_lo <= rate_norm <= args.rate_hi
    dpsi_ok    = psi_growth_ratio > args.dpsi_floor_frac
    energy_ok  = energy_drift < args.energy_cons_tol if energy_drift == energy_drift else True
    ub_ok      = ub_decrease_frac > args.ub_decrease_frac if ub_decrease_frac == ub_decrease_frac else True
    ok = rate_ok and dpsi_ok and energy_ok and ub_ok

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    tag = lambda b: '✓' if b else '✗'
    print()
    print(f"Reconnection check — {label}")
    print('─' * 64)
    print(f"  Snapshots              : {len(cycles)}  (Δt_sample = {sample_dt:.4g})")
    print(f"  Sheet midline j_X      : {j_X}  (y = {j_X * Ly / nyc:.4g} = {args.y_frac:.2f}·Ly)")
    print(f"  ψ(0)                  : {psi[0]:.4e}")
    print(f"  ψ(end)                : {psi[-1]:.4e}")
    print(f"  Δψ                    : {delta_psi:.4e}  (reconnected flux)")
    print(f"  Peak dψ/dt            : {peak_rate:.4e}  at t = {peak_t:.3g}  ({peak_t * omega_ci:.2f} Ω_ci⁻¹)")
    print(f"  B0·v_A                : {rate_norm_unit:.4e}  (B0={B0_mag:.4g}, v_A={v_A:.4g})")
    print(f"  ── Pass criteria ─────────────────────────────────────────")
    print(f"  [{tag(rate_ok)}] peak / (B0·v_A)     = {rate_norm:.3f}     band [{args.rate_lo:.2f}, {args.rate_hi:.2f}]")
    print(f"  [{tag(dpsi_ok)}] Δψ / ψ(0)           = {psi_growth_ratio:.2f}      floor > {args.dpsi_floor_frac:.2f}")
    print(f"  [{tag(ub_ok)}] −ΔU_B / U_B(0)      = {ub_decrease_frac:.4f}    floor > {args.ub_decrease_frac:.3f}")
    print(f"  [{tag(energy_ok)}] |ΔE_total|/E₀       = {energy_drift:.2e}  tol < {args.energy_cons_tol:.0e}")
    if delta_UB == delta_UB:                            # not NaN
        print(f"  Energy budget          : ΔU_B = {delta_UB:+.3e}, ΔKE = {delta_KE:+.3e}, ΔU_E = {delta_UE:+.3e}")
    print('─' * 64)
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
            measured_color = '#f38ba8' if theme.mode == 'dark' else '#EE6677'
            theory_color   = '#a6e3a1' if theme.mode == 'dark' else '#228833'

            has_energy = (delta_UB == delta_UB)         # not NaN
            n_axes = 3 if has_energy else 2
            fig, axes = plt.subplots(n_axes, 1,
                                      figsize=(9.0, 8.5 if has_energy else 6.5),
                                      constrained_layout=True, sharex=True)
            ax1, ax2 = axes[0], axes[1]
            ax3 = axes[2] if has_energy else None

            # Top: ψ(t)
            ax1.plot(t, psi, color=theme.text_color, lw=1.5, label='ψ(t)')
            ax1.axvline(peak_t, color=measured_color, lw=1.0, ls=':', alpha=0.7,
                        label=f'peak rate at t = {peak_t:.2g}')
            ax1.set_ylabel('reconnected flux  ψ')
            ax1.legend(loc='lower right', fontsize=9)

            # Bottom: dψ/dt with theory band
            t_full_max = float(t[-1])
            ax2.fill_between([0, t_full_max],
                              args.rate_lo * rate_norm_unit,
                              args.rate_hi * rate_norm_unit,
                              color=theory_color, alpha=0.18,
                              label=f'GEM band  [{args.rate_lo:.2f}, {args.rate_hi:.2f}]·B0·v_A')
            ax2.plot(t, dpsi_dt, color=theme.text_color, lw=1.3, label='dψ/dt')
            ax2.plot(peak_t, peak_rate, 'o', color=measured_color, ms=8,
                     label=f'peak = {rate_norm:.3f}·B0·v_A')
            ax2.axhline(0, color=theme.zero_line_color, lw=0.6, alpha=0.5)
            if not has_energy:
                ax2.set_xlabel('t [code time]')
            ax2.set_ylabel('reconnection rate  dψ/dt')
            ax2.legend(loc='upper right', fontsize=9)

            # Third panel: magnetic-to-kinetic energy budget — the *physics*
            # behind reconnection. Normalise to U_B(0) so the curves overlay.
            if has_energy:
                assert ax3 is not None and cq
                UB = np.array(cq['B_energy'])
                UE = np.array(cq['E_energy'])
                KE = np.array(cq['KE'])
                ub_color = '#89b4fa' if theme.mode == 'dark' else '#4477AA'
                ke_color = '#f9e2af' if theme.mode == 'dark' else '#CCBB44'
                ue_color = measured_color
                t_cq = np.array(cq['cycle'], dtype=float) * dt
                # Plot −ΔU_B alongside ΔKE so both are positive and overlay:
                # the visual gap between them = thermal heating (energy conservation
                # closes the budget exactly to ε, by ECSIM's structural invariant).
                ax3.plot(t_cq, -(UB - UB[0]) / UB[0],
                         color=ub_color, lw=1.5, label='−ΔU_B / U_B(0)  (magnetic released)')
                ax3.plot(t_cq, (KE - KE[0]) / UB[0],
                         color=ke_color, lw=1.5, label=' ΔKE  / U_B(0)  (kinetic gained)')
                ax3.plot(t_cq, (UE - UE[0]) / UB[0],
                         color=ue_color, lw=1.0, ls=':', alpha=0.85,
                         label=' ΔU_E / U_B(0)  (electric, ~0)')
                ax3.axhline(0, color=theme.zero_line_color, lw=0.6, alpha=0.5)
                ax3.set_xlabel('t [code time]')
                ax3.set_ylabel('Δ energy / U_B(0)')
                ax3.legend(loc='upper right', fontsize=9)
                # Show conversion efficiency: ΔKE / -ΔU_B (close to 1 means
                # almost all released magnetic energy went to bulk kinetic flow).
                conversion_eff = (delta_KE / -delta_UB) if delta_UB < 0 else float('nan')
                ax3.text(0.02, 0.04,
                         f'-ΔU_B / U_B(0) = {ub_decrease_frac:.4f}\n'
                         f' ΔKE  / U_B(0) = {delta_KE / UB0:+.4f}\n'
                         f' ΔKE / -ΔU_B   = {conversion_eff:.3f}   (→1 = pure conversion)\n'
                         f'|ΔE_tot|/E₀   = {energy_drift:.2e}',
                         transform=ax3.transAxes, va='bottom', fontsize=9, family='monospace',
                         bbox=dict(facecolor=theme.box_face,
                                   edgecolor=theme.box_edge,
                                   alpha=theme.box_alpha))
            ax2.text(0.02, 0.96,
                     f'peak / (B0·v_A) = {rate_norm:.3f}\n'
                     f'Δψ / ψ(0)       = {psi_growth_ratio:.2f}\n'
                     f'-ΔU_B / U_B(0)   = {ub_decrease_frac:.4f}\n'
                     f'|ΔE_tot|/E₀     = {energy_drift:.2e}\n'
                     f'B0  = {B0_mag:.4g}, v_A = {v_A:.4g}',
                     transform=ax2.transAxes, va='top', fontsize=9, family='monospace',
                     bbox=dict(facecolor=theme.box_face,
                               edgecolor=theme.box_edge,
                               alpha=theme.box_alpha))

            verdict = 'PASS' if ok else 'FAIL'
            fig.suptitle(f'{label}  —  peak rate = {rate_norm:.3f} · B0·v_A  [{verdict}]',
                         fontsize=12)
            fig.savefig(args.save_plot, dpi=140)
            plt.close(fig)
            print(f"  Saved diagnostic plot → {args.save_plot}")
            print()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
