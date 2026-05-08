#!/usr/bin/env python3
"""analyze_landau.py — fit collisionless damping rate γ_L for a Langmuir wave.

Reads spatial Ex(x) snapshots from `Fields_NNNNN/E_NNNNN.h5`, FFTs each
snapshot along x, tracks the seeded mode m_x amplitude vs t, and fits
log|Ex(m, t)| with a straight line in the early-decay window. Compares
to γ_L computed by numerically solving the linear plasma dispersion
relation

    ε(ω, k) = 1 + (1/(k λ_D)²) · (1 + ξ Z(ξ)) = 0,
    ξ = ω / (k v_th √2)        Z(ξ) = i √π · w(ξ)

with `scipy.special.wofz` and `scipy.optimize.fsolve` from a Bohm-Gross
seed. PIC tolerance default 20%.

Used by Test 8 (Landau damping) of the physics validation suite.

Exit 0 on PASS (|γ_meas - γ_L| / |γ_L| < tol), 1 on FAIL.
"""

import argparse
import math
import os
import sys


def parse_input_file(path):
    """Pull simulation parameters from an iPIC3D .inp file.
    Re-implementation of analyze_growth_rate.parse_input_file (minor superset)."""
    keys = {'dt', 'ncycles', 'c', 'Lx', 'Ly', 'Lz',
            'B0x', 'B0y', 'B0z', 'FieldOutputCycle'}
    arr_keys = {'rhoINIT', 'qom', 'u0', 'uth', 'vth', 'wth', 'custom_parameters'}
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
            elif lhs in arr_keys:
                try:
                    out[lhs] = [float(x) for x in rhs.split()]
                except ValueError:
                    pass
    return out


def langmuir_root(k, lambda_D, v_th, omega_pe):
    """Solve plasma dispersion relation ε(ω, k) = 0 for the lightly-damped
    Langmuir branch. Returns (ω_r, γ_L)."""
    from scipy.special import wofz  # type: ignore[import-not-found]
    from scipy.optimize import fsolve  # type: ignore[import-not-found]

    kD = k * lambda_D
    bg_omega_r = omega_pe * math.sqrt(1.0 + 3.0 * kD * kD)
    # Textbook approximation γ_L ≈ -√(π/8)·(ω_pe/(kλ_D)³)·exp(-1/(2kλ_D²) - 3/2)
    bg_gamma = -math.sqrt(math.pi / 8.0) * (omega_pe / kD**3) \
               * math.exp(-1.0 / (2.0 * kD * kD) - 1.5)

    def Z(xi):
        return 1j * math.sqrt(math.pi) * wofz(xi)

    def epsilon(om):
        xi = om / (k * v_th * math.sqrt(2.0))
        return 1.0 + (1.0 / (kD * kD)) * (1.0 + xi * Z(xi))

    def residual(x):
        om = x[0] + 1j * x[1]
        eps = epsilon(om)
        return [eps.real, eps.imag]

    sol = fsolve(residual, [bg_omega_r, bg_gamma], full_output=True)
    x_root, _, ier, _ = sol
    if ier != 1:
        # Fallback — Bohm-Gross + textbook γ_L. Tolerable accuracy at kλ_D ≤ 0.5.
        return bg_omega_r, bg_gamma
    return float(x_root[0]), float(x_root[1])


def find_decay_window(log_amp, t):
    """Locate the early-decay segment: largest-NEGATIVE-slope window in the
    early half of the time series (the LINEAR regime — late times either hit
    shot-noise recurrence or particle-trapping equilibrium with γ_eff ≠ γ_L).
    Skips first 5% (initial transient)."""
    import numpy as np
    n = len(log_amp)
    n_skip = max(5, n // 20)
    window = max(20, n // 6)
    # Bound search to first 60% of run — Landau decay should saturate by then.
    n_search_end = max(n_skip + window + 1, int(0.6 * n))
    best_slope = float('inf')
    best_start = n_skip
    for i in range(n_skip, n_search_end - window):
        slope = np.polyfit(t[i:i+window], log_amp[i:i+window], 1)[0]
        if slope < best_slope:
            best_slope = slope
            best_start = i
    return best_start, best_start + window


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dir', help='Run output dir (Fields_NNNNN/E_NNNNN.h5)')
    ap.add_argument('--inp', required=True, help='Path to the .inp file used for the run')
    ap.add_argument('--mode', type=int, default=None,
                    help='Mode m_x to track (default: from custom_parameters[1])')
    ap.add_argument('--electron-species', type=int, default=0,
                    help='Index of electron species in .inp (for vth lookup). Default 0.')
    ap.add_argument('--tol', type=float, default=0.20,
                    help='Pass tolerance |Δγ/γ|. Default 0.20 (PIC-typical for Landau).')
    ap.add_argument('--omega-tol', type=float, default=0.05, dest='omega_tol',
                    help='Pass tolerance |Δω_r/ω_r|. Default 0.05 (Bohm-Gross is sharp).')
    ap.add_argument('--label', default=None, help='Test label for the printout')
    ap.add_argument('--save-plot', default=None, dest='save_plot',
                    help='Optional path to save Landau-damping diagnostic PNG.')
    ap.add_argument('--light', action='store_true',
                    help='Use light theme for the saved plot (default dark).')
    args = ap.parse_args()

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_utils import discover_cycles, load_field
        import numpy as np
    except ImportError as e:
        print(f"FAIL: missing dependency: {e}")
        sys.exit(2)

    sim = parse_input_file(args.inp)
    Lx = sim['Lx']
    dt = sim['dt']
    foc = sim.get('FieldOutputCycle', 1)

    rho = sim.get('rhoINIT', [])
    qom = sim.get('qom', [])
    vth_arr = sim.get('uth', [])
    if not (rho and qom and vth_arr):
        print("FAIL: missing rhoINIT/qom/uth in input file")
        sys.exit(1)

    omega_pe2 = sum(abs(r) * abs(q) for r, q in zip(rho, qom) if q < 0)
    omega_pe = math.sqrt(omega_pe2) if omega_pe2 > 0 else 0.0
    if omega_pe <= 0:
        print("FAIL: ω_pe = 0 — no electron species detected")
        sys.exit(1)
    v_th = vth_arr[args.electron_species]
    lambda_D = v_th / omega_pe

    custom = sim.get('custom_parameters', [])
    if args.mode is not None:
        mode = args.mode
    elif len(custom) >= 2:
        mode = int(custom[1])
    else:
        mode = 1
    k = 2.0 * math.pi * mode / Lx
    kD = k * lambda_D

    cycles = discover_cycles(args.output_dir)
    if len(cycles) < 30:
        print(f"FAIL: only {len(cycles)} field snapshots — need ≥ 30 to fit decay")
        sys.exit(1)

    # FFT each snapshot along x at the central (y, z) plane.
    # Keep the COMPLEX coefficient c_m(t) so we can extract ω_r from the time-FFT
    # (forward-branch eigenmode rotates as exp(-iω_r·t) in the complex plane).
    amp = np.zeros(len(cycles))
    t = np.zeros(len(cycles))
    cm_complex = np.zeros(len(cycles), dtype=complex)
    for i, c in enumerate(cycles):
        ex = load_field(args.output_dir, c, 'E', 'x')
        ny = ex.shape[1]
        nz = ex.shape[2]
        line = ex[:, ny // 2, nz // 2]
        coefs = np.fft.rfft(line)
        if mode >= len(coefs):
            print(f"FAIL: mode m={mode} out of range (len(rfft)={len(coefs)})")
            sys.exit(1)
        cm_complex[i] = coefs[mode] / len(line)
        amp[i] = 2.0 * abs(cm_complex[i])
        t[i] = c * dt

    # |Êx(m, t)| oscillates at |cos(ω_r·t)| if both forward+backward branches are
    # excited (period π/ω_r). Extract the slowly-varying envelope via rolling max
    # over one oscillation half-period — clean for log-linear γ_L fit.
    omega_pe_seed = math.sqrt(omega_pe2) if omega_pe2 > 0 else 1.0
    omega_r_bg = omega_pe_seed * math.sqrt(1.0 + 3.0 * (k * (v_th / omega_pe_seed))**2)
    period_t = 2.0 * math.pi / omega_r_bg
    sample_dt = float(t[1] - t[0]) if len(t) > 1 else dt
    half_period_snaps = max(3, int(round(0.5 * period_t / sample_dt)))
    envelope = np.array([amp[max(0, i - half_period_snaps + 1): i + 1].max()
                         for i in range(len(amp))])

    log_amp = np.log(np.clip(envelope, 1e-300, None))
    i_start, i_end = find_decay_window(log_amp, t)
    if i_end - i_start < 10:
        print(f"FAIL: decay window [{i_start}, {i_end}] too narrow")
        sys.exit(1)

    slope, intercept = np.polyfit(t[i_start:i_end], log_amp[i_start:i_end], 1)
    gamma_measured = float(slope)  # |Êx(m, t)| ∝ exp(γ·t), so slope = γ directly

    omega_r_th, gamma_th = langmuir_root(k, lambda_D, v_th, omega_pe)

    # ω_r extraction: time-FFT of the complex spatial-FFT coefficient. The forward
    # Langmuir eigenmode rotates as exp(-iω_r·t) in the complex plane, so the
    # negative-frequency peak of FFT[c_m(t)] sits at ω_r. Use the early-time
    # window (before saturation/recurrence) so the spectrum stays clean.
    n_for_omega = max(50, min(len(cycles), int(round(2 * period_t / sample_dt) * 6)))
    n_for_omega = min(n_for_omega, len(cycles))
    cm_window = cm_complex[:n_for_omega]
    spec = np.fft.fftshift(np.fft.fft(cm_window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_for_omega, d=sample_dt)) * 2.0 * math.pi
    peak_idx = int(np.argmax(np.abs(spec)))
    # Parabolic interpolation around the peak for sub-bin accuracy.
    if 0 < peak_idx < len(spec) - 1:
        y_lo = abs(spec[peak_idx - 1])
        y_hi = abs(spec[peak_idx + 1])
        y_pk = abs(spec[peak_idx])
        denom = y_lo - 2.0 * y_pk + y_hi
        delta = 0.5 * (y_lo - y_hi) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    omega_meas_signed = float(freqs[peak_idx] + delta * df)
    omega_r_meas = abs(omega_meas_signed)

    rel_gamma = abs(gamma_measured - gamma_th) / abs(gamma_th) if gamma_th else float('inf')
    rel_omega = abs(omega_r_meas - omega_r_th) / abs(omega_r_th) if omega_r_th else float('inf')
    ok_gamma = rel_gamma < args.tol
    ok_omega = rel_omega < args.omega_tol
    ok = ok_gamma and ok_omega

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    print()
    print(f"Landau damping check — {label}")
    print('─' * 56)
    print(f"  Field snapshots        : {len(cycles)}  (FieldOutputCycle = {int(foc)})")
    print(f"  Mode m                 : {mode}  →  k = 2π m/Lx = {k:.4g}")
    print(f"  ω_pe                   : {omega_pe:.4g}")
    print(f"  v_th                   : {v_th:.4g}   →   λ_D = {lambda_D:.4g}")
    print(f"  k λ_D                  : {kD:.4g}")
    print(f"  Theory ω_r (Z-fn)      : {omega_r_th:.4g}   ({omega_r_th/omega_pe:.4g}·ω_pe)")
    print(f"  Theory γ_L (Z-fn)      : {gamma_th:.4g}    ({gamma_th/omega_pe:.4g}·ω_pe)")
    print(f"  Decay-fit window       : t [{t[i_start]:.2f}, {t[i_end-1]:.2f}]"
          f"   (snapshots {i_start}..{i_end-1})")
    print(f"  γ_measured (slope)     : {gamma_measured:.4g}")
    print(f"  ω_r_measured (FFT peak): {omega_r_meas:.4g}   ({omega_r_meas/omega_pe:.4g}·ω_pe)")
    print(f"  |Δγ/γ|                 : {rel_gamma:.3e}    tol = {args.tol:.0e}    [{'PASS' if ok_gamma else 'FAIL'}]")
    print(f"  |Δω/ω|                 : {rel_omega:.3e}    tol = {args.omega_tol:.0e}    [{'PASS' if ok_omega else 'FAIL'}]")
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
            measured_color = '#f38ba8' if theme.mode == 'dark' else '#EE6677'
            theory_color   = '#a6e3a1' if theme.mode == 'dark' else '#228833'

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                            constrained_layout=True)

            # Panel 1: log|Êx(m, t)| with fit + theory slope
            ax1.plot(t, log_amp, color=theme.text_color, lw=1.1, alpha=0.85,
                     label=f'log |Ex(m={mode}, t)|')
            ax1.axvspan(t[i_start], t[i_end - 1], color=theory_color, alpha=0.18,
                        label=f'fit window  (t {t[i_start]:.1f}..{t[i_end-1]:.1f})')
            t_w = t[i_start:i_end]
            ax1.plot(t_w, slope * t_w + intercept, color=measured_color, lw=2.0,
                     label=f'fit  g_meas = {gamma_measured:.4g}')
            ax1.plot(t_w, gamma_th * t_w + intercept, color=theory_color, lw=1.8, ls='--',
                     label=f'theory  g_L = {gamma_th:.4g}  (Z-fn)')

            ymin = float(log_amp.min())
            ymax = float(log_amp.max())
            pad = 0.05 * max(1.0, ymax - ymin)
            ax1.set_ylim(ymin - pad, ymax + pad)
            ax1.set_xlabel('t [code time]')
            ax1.set_ylabel(f'log |E_x(m={mode}, t)|')
            ax1.legend(loc='lower left', fontsize=9)
            ax1.text(0.02, 0.98,
                     f'|Δγ/γ| = {rel_gamma:.2e}\n|Δω/ω| = {rel_omega:.2e}\nk·λ_D = {kD:.3f}',
                     transform=ax1.transAxes, va='top', fontsize=9,
                     bbox=dict(facecolor=theme.box_face,
                               edgecolor=theme.box_edge,
                               alpha=theme.box_alpha))

            # Panel 2: time-FFT spectrum |FFT[c_m(t)]| vs ω, with measured peak
            # + theory ω_r marked. Forward-branch eigenmode shows up at one sign.
            spec_amp = np.abs(spec)
            ax2.plot(freqs, spec_amp, color=theme.text_color, lw=1.0,
                     label='|FFT[c_m(t)]|', alpha=0.85)
            ax2.axvline(omega_meas_signed, color=measured_color, lw=1.6, ls='-',
                        label=f'measured  |ω_r| = {omega_r_meas:.4g}')
            # Theory: forward branch sits at +ω_r OR -ω_r depending on FFT sign convention;
            # show both as dashed lines.
            ax2.axvline( omega_r_th, color=theory_color, lw=1.4, ls='--', alpha=0.7,
                        label=f'theory  ±ω_r = ±{omega_r_th:.4g}  (Z-fn)')
            ax2.axvline(-omega_r_th, color=theory_color, lw=1.4, ls='--', alpha=0.7)
            x_pad = max(0.5, 1.5 * omega_r_th)
            ax2.set_xlim(-x_pad, x_pad)
            ax2.set_xlabel('ω')
            ax2.set_ylabel(f'|FFT[c_m(t)]|   (m = {mode}, n = {len(cm_window)} snaps)')
            ax2.legend(loc='upper right', fontsize=9)

            verdict = 'PASS' if ok else 'FAIL'
            fig.suptitle(f'{label}  —  Landau damping  —  |Δγ/γ|={rel_gamma:.2e}  '
                         f'|Δω/ω|={rel_omega:.2e}  [{verdict}]', fontsize=12)
            fig.savefig(args.save_plot, dpi=140)
            plt.close(fig)
            print(f"  Saved diagnostic plot → {args.save_plot}")
            print()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
