#!/usr/bin/env python3
"""analyze_dispersion.py — extract ω from a probed iPIC3D field time series.

Walks the HDF5 field snapshots from an iPIC3D run, samples one chosen
field component at a fixed grid cell across all output cycles, FFTs the
time series, locates the dominant peak ω_measured, and compares it to a
theoretical ω_expected supplied by the caller (`--expected`).

Used by the wave-propagation regression tests:
    Test 1 — Plane EM wave in vacuum:    ω = c·k
    Test 2 — Shear Alfvén wave:          ω = k·v_A
    Test 4 — Whistler R-mode packet:     ω = k²c²·ω_ce / (ω_pe² + k²c²)
    Test 5 — Oblique shear Alfvén wave:  ω = k·v_A·cos_kB

For oblique tests, pass `--mode-y` (or rely on input_param[2]). The eval ctx
exposes kx, ky, k (= |k|), and cos_kB = (k·B0)/(|k|·|B0|).

Two extraction methods:
  --method fft (default): probe one cell, time-FFT, take peak. Fast and
    works for clean 1D waves but is dominated by broadband noise once the
    wave damps below the noise floor.
  --method phase: spatial 2D-FFT each snapshot at the seeded mode (m_x, m_y),
    unwrap the complex phase, and linear-fit the slope on the early
    high-amplitude samples. Robust against the strong PIC damping that
    plagues oblique waves — the wave starts at the correct ω; the fit
    captures it before the damping curves accumulate.

Exit 0 on PASS (|Δω/ω| < tol), 1 on FAIL.
"""

import argparse
import math
import os
import sys


def parse_input_file(path):
    """Pull simulation parameters from an iPIC3D .inp file."""
    keys = {'dt', 'ncycles', 'c', 'Lx', 'Ly', 'Lz', 'nxc', 'nyc', 'nzc',
            'B0x', 'B0y', 'B0z', 'FieldOutputCycle', 'ns', 'nparam'}
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
            elif lhs in ('rhoINIT', 'qom', 'custom_parameters'):
                try:
                    out[lhs] = [float(x) for x in rhs.split()]
                except ValueError:
                    pass
    return out


def evaluate_expected(formula, ctx):
    """Evaluate `formula` against simulation params in ctx.

    Allowed names: pi, c, k, B0, n0, omega_pe, omega_ce, sqrt, log, exp.
    `formula` examples: 'c*k', 'k*v_A', 'k**2*c**2*omega_ce/(omega_pe**2+k**2*c**2)'.
    """
    safe = {
        'pi': math.pi, 'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
        '__builtins__': {},
    }
    safe.update(ctx)
    return eval(formula, safe)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dir', help='Run output directory (contains Fields_*/)')
    ap.add_argument('--inp', required=True, help='Path to the .inp file used for the run')
    ap.add_argument('--probe', default='auto',
                    help='Probe cell as "i,j,k" or "auto" (defaults to nx/4, ny/2, nz/2)')
    ap.add_argument('--field', default='B', choices=['B', 'E'])
    ap.add_argument('--component', default='z', choices=['x', 'y', 'z'])
    ap.add_argument('--mode', type=int, default=None,
                    help='Wave mode m_x for k_x=2π·m_x/Lx. Default: input_param[1] from .inp')
    ap.add_argument('--mode-y', type=int, default=None, dest='mode_y',
                    help='Wave mode m_y for k_y=2π·m_y/Ly. Default: input_param[2] from .inp if present, else 0')
    ap.add_argument('--method', default='fft', choices=['fft', 'phase'],
                    help='ω extraction: "fft" = single-cell time-FFT peak; "phase" = '
                         'phase-slope fit on spatial-FFT mode (use for oblique / heavily-damped waves)')
    ap.add_argument('--phase-fit-points', type=int, default=10, dest='phase_fit_points',
                    help='Number of early samples to use for --method=phase linear fit (default 10)')
    ap.add_argument('--expected', required=True,
                    help='Formula for ω_expected, e.g. "c*k" or "k*v_A"')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Pass tolerance |Δω/ω|. Default 0.05.')
    ap.add_argument('--label', default=None, help='Test label for the printout')
    ap.add_argument('--save-plot', default=None, dest='save_plot',
                    help='Optional path to save a 2-panel theory-agreement diagnostic PNG.')
    ap.add_argument('--light', action='store_true',
                    help='Use light theme for the saved plot (default dark).')
    args = ap.parse_args()

    try:
        import numpy as np
        from plot_utils import discover_cycles, load_field
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from plot_utils import discover_cycles, load_field
        import numpy as np

    sim = parse_input_file(args.inp)
    cycles = discover_cycles(args.output_dir)
    if len(cycles) < 8:
        print(f"FAIL: only {len(cycles)} field snapshots — need ≥ 8 for FFT")
        sys.exit(1)

    if args.probe == 'auto':
        i_probe = int(sim['nxc']) // 4
        j_probe = int(sim['nyc']) // 2
        k_probe = int(sim['nzc']) // 2
    else:
        i_probe, j_probe, k_probe = (int(x) for x in args.probe.split(','))

    sample_dt = sim['dt'] * (cycles[1] - cycles[0])

    # Branch-local outputs — declared up front so the post-extraction plot
    # block can reference them without Pyright "possibly unbound" warnings.
    series = spectrum = freqs = None
    mode_series = phase = t_fit = None
    peak = 0
    N = 0

    if args.method == 'fft':
        series = np.empty(len(cycles))
        for n, cyc in enumerate(cycles):
            data = load_field(args.output_dir, cyc, args.field, args.component)
            series[n] = data[i_probe, j_probe, k_probe]
        series -= series.mean()                            # drop DC
        spectrum = np.abs(np.fft.rfft(series))
        freqs = 2.0 * np.pi * np.fft.rfftfreq(len(series), d=sample_dt)
        spectrum[0] = 0.0                                  # belt-and-braces
        peak = int(np.argmax(spectrum))
        omega_measured = freqs[peak]
        method_note = f"FFT bin peak at {peak}/{len(spectrum) - 1}"
    else:                                                  # method == 'phase'
        # Defer mode_x/mode_y resolution until we've parsed args.mode/args.mode_y.
        cp = sim.get('custom_parameters', [0, 1])
        _mx = args.mode if args.mode is not None else int(cp[1] if len(cp) > 1 else 1)
        _my = args.mode_y if args.mode_y is not None else (int(cp[2]) if len(cp) > 2 else 0)
        mode_series = np.empty(len(cycles), dtype=complex)
        for n, cyc in enumerate(cycles):
            data = load_field(args.output_dir, cyc, args.field, args.component)
            nx_int, ny_int = data.shape[0] - 1, data.shape[1] - 1
            slab = data[:nx_int, :ny_int, k_probe]
            Fk = np.fft.fft2(slab) / (nx_int * ny_int)
            mode_series[n] = Fk[_mx, _my]
        phase = np.unwrap(np.angle(mode_series))
        N = min(args.phase_fit_points, len(cycles) - 1)
        if N < 3:
            print(f"FAIL: only {len(cycles)} snapshots — need ≥ 3 for phase fit")
            sys.exit(1)
        t_fit = np.arange(N) * sample_dt
        slope = np.polyfit(t_fit, phase[:N], 1)[0]
        omega_measured = abs(slope)
        method_note = f"phase-slope fit on first {N} samples (t=0..{t_fit[-1]:.2g}); |Fk| = {abs(mode_series[0]):.3e} → {abs(mode_series[N-1]):.3e}"

    cp = sim.get('custom_parameters', [0, 1])
    mode_x = args.mode if args.mode is not None else int(cp[1] if len(cp) > 1 else 1)
    mode_y = args.mode_y if args.mode_y is not None else (int(cp[2]) if len(cp) > 2 else 0)
    k_x = 2.0 * math.pi * mode_x / sim['Lx']
    k_y = 2.0 * math.pi * mode_y / sim['Ly']
    k_mag = math.sqrt(k_x*k_x + k_y*k_y)

    B0_mag = math.sqrt(sim.get('B0x', 0)**2 + sim.get('B0y', 0)**2 + sim.get('B0z', 0)**2)
    if k_mag > 0 and B0_mag > 0:
        cos_kB = (k_x * sim.get('B0x', 0.0) + k_y * sim.get('B0y', 0.0)) / (k_mag * B0_mag)
    else:
        cos_kB = 1.0
    ctx = {
        'c': sim['c'], 'k': k_mag, 'kx': k_x, 'ky': k_y, 'cos_kB': cos_kB,
        'B0': B0_mag,
        'B0x': sim.get('B0x', 0.0), 'B0y': sim.get('B0y', 0.0), 'B0z': sim.get('B0z', 0.0),
        'Lx': sim['Lx'], 'Ly': sim['Ly'], 'Lz': sim['Lz'],
    }
    rho = sim.get('rhoINIT', [])
    qom = sim.get('qom', [])
    if rho and qom and len(rho) == len(qom):
        # iPIC3D init() sets rhons[s] = rhoINIT[s] / 4π → ρ_s = rhoINIT[s] / (4π · |qom_s|)
        # Total mass density Σ ρ_s = (1/4π) · Σ |rhoINIT_s| / |qom_s|
        # v_A = B / sqrt(4π · ρ_total) = B / sqrt(Σ |rhoINIT_s| / |qom_s|)
        mass_density_norm = sum(abs(r) / abs(q) for r, q in zip(rho, qom) if q != 0)
        if mass_density_norm > 0 and ctx['B0'] > 0:
            ctx['v_A'] = ctx['B0'] / math.sqrt(mass_density_norm)
        ctx['n0'] = max(abs(r) for r in rho)
        # iPIC3D convention: rhoINIT/4π is charge density, n = (rhoINIT/4π)/|q|.
        # Then ω_p² = 4π·n·q²/m = rhoINIT·|qom| (q has unit magnitude in code units).
        # Species 0 is conventionally electrons.
        ctx['omega_pe'] = math.sqrt(abs(rho[0]) * abs(qom[0]))
        if len(qom) > 1 and ctx['B0'] > 0:
            ctx['omega_ce'] = abs(qom[0]) * ctx['B0'] / sim['c']     # |qom_e|·B/c

    omega_expected = evaluate_expected(args.expected, ctx)
    rel = abs(omega_measured - omega_expected) / abs(omega_expected) if omega_expected else float('inf')
    ok = rel < args.tol

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    print()
    print(f"Dispersion check — {label}")
    print('─' * 56)
    print(f"  Probe cell           : ({i_probe}, {j_probe}, {k_probe})")
    print(f"  Field probed         : {args.field}{args.component}")
    print(f"  Snapshots            : {len(cycles)} (Δt_sample = {sample_dt:.4g})")
    if mode_y:
        print(f"  Modes (m_x, m_y)     : ({mode_x}, {mode_y})  →  (k_x, k_y) = ({k_x:.4g}, {k_y:.4g})  |k| = {k_mag:.4g}")
        print(f"  cos(θ_kB)            : {cos_kB:.4g}  (θ_kB = {math.degrees(math.acos(max(-1.0, min(1.0, cos_kB)))):.2f}°)")
    else:
        print(f"  Mode m               : {mode_x}  →  k = {k_mag:.4g}")
    print(f"  ω_expected ({args.expected}) = {omega_expected:.4g}")
    print(f"  ω_measured ({args.method}) = {omega_measured:.4g}    [{method_note}]")
    print(f"  |Δω/ω|               : {rel:.3e}    tol = {args.tol:.0e}")
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                            constrained_layout=True)

            if args.method == 'fft':
                assert series is not None and spectrum is not None and freqs is not None
                t_full = np.arange(len(series)) * sample_dt
                # LSQ fit amplitude+phase against theory frequency: shows visual ω agreement.
                cos_b = np.cos(omega_expected * t_full)
                sin_b = np.sin(omega_expected * t_full)
                M = np.column_stack([cos_b, sin_b])
                coef, *_ = np.linalg.lstsq(M, series, rcond=None)
                theory_curve = coef[0] * cos_b + coef[1] * sin_b

                ax1.plot(t_full, series, color=theme.text_color, lw=1.1,
                         label='measured probe', alpha=0.9)
                ax1.plot(t_full, theory_curve, color=theory_color, lw=1.6, ls='--',
                         label=f'theory ω={omega_expected:.4g}')
                ax1.set_xlabel('t [code time]')
                ax1.set_ylabel(f'δ{args.field}{args.component} (probe)')
                ax1.set_title('Probe time series')
                ax1.legend(loc='upper right', fontsize=9)

                norm = freqs / omega_expected if omega_expected else freqs
                ax2.plot(norm, spectrum, color=theme.text_color, lw=1.2)
                ax2.axvline(1.0, color=theory_color, lw=1.4, ls='--',
                            label=f'ω_theory = {omega_expected:.4g}')
                ax2.plot(omega_measured / omega_expected if omega_expected else 0,
                         spectrum[peak], 'o', color=measured_color, ms=8,
                         label=f'ω_measured = {omega_measured:.4g}')
                ax2.set_xlabel('ω / ω_theory')
                ax2.set_ylabel('|FFT[probe]|')
                ax2.set_title('Frequency spectrum')
                xmax = max(2.5, 1.6 * (omega_measured / omega_expected if omega_expected else 1.0))
                ax2.set_xlim(0, xmax)
                ax2.legend(loc='upper right', fontsize=9)
                bin_w = freqs[1] if len(freqs) > 1 else float('nan')
                ax2.text(0.02, 0.96,
                         f'FFT bin Δω = {bin_w:.4g}\n|Δω/ω| = {rel:.2e}',
                         transform=ax2.transAxes, va='top', fontsize=9,
                         bbox=dict(facecolor=theme.box_face,
                                   edgecolor=theme.box_edge,
                                   alpha=theme.box_alpha))
            else:  # phase method
                assert mode_series is not None and phase is not None and t_fit is not None
                t_full = np.arange(len(cycles)) * sample_dt
                ax1.plot(t_full, np.abs(mode_series),
                         color=theme.text_color, lw=1.2)
                ax1.axvspan(0, t_fit[-1], alpha=0.18, color=theory_color,
                            label=f'fit window (N={N})')
                ax1.set_xlabel('t [code time]')
                ax1.set_ylabel(f'|F_k(δ{args.field}{args.component})|')
                ax1.set_title(f'Spatial-FFT mode amplitude (m_x={mode_x}, m_y={mode_y})')
                if (np.abs(mode_series) > 0).all():
                    ax1.set_yscale('log')
                ax1.legend(loc='upper right', fontsize=9)

                _, intercept = np.polyfit(t_fit, phase[:N], 1)
                ax2.plot(t_full, phase, color=theme.text_color, lw=1.1,
                         label='unwrapped phase', alpha=0.85)
                fit_slope = np.polyfit(t_fit, phase[:N], 1)[0]
                theory_slope = -omega_expected if fit_slope < 0 else omega_expected
                ax2.plot(t_fit, fit_slope * t_fit + intercept, color=measured_color,
                         lw=2.0, label=f'fit slope = {fit_slope:.4g}')
                ax2.plot(t_fit, theory_slope * t_fit + intercept, color=theory_color,
                         lw=1.8, ls='--', label=f'theory slope = {theory_slope:.4g}')
                ax2.set_xlabel('t [code time]')
                ax2.set_ylabel('unwrapped phase [rad]')
                ax2.set_title('Phase-slope fit')
                ax2.legend(loc='best', fontsize=9)
                # Zoom x to a few fit-windows so the slope match is visible —
                # phase past the damping point is noise, not wave physics.
                t_zoom = max(5 * t_fit[-1], 4 * sample_dt * 5)
                xmax_zoom = min(float(t_full[-1]), t_zoom)
                ax2.set_xlim(0, xmax_zoom)
                # Match y to the visible x window so the fit slope isn't squashed.
                visible = phase[t_full <= xmax_zoom]
                if len(visible) > 1:
                    pad = 0.10 * max(1.0, abs(visible.max() - visible.min()))
                    ax2.set_ylim(float(visible.min()) - pad,
                                 float(visible.max()) + pad)
                ax2.text(0.02, 0.96, f'|Δω/ω| = {rel:.2e}',
                         transform=ax2.transAxes, va='top', fontsize=9,
                         bbox=dict(facecolor=theme.box_face,
                                   edgecolor=theme.box_edge,
                                   alpha=theme.box_alpha))

            verdict = 'PASS' if ok else 'FAIL'
            fig.suptitle(f'{label}  ({args.method})  —  |Δω/ω| = {rel:.2e}  [{verdict}]',
                         fontsize=12)
            fig.savefig(args.save_plot, dpi=140)
            plt.close(fig)
            print(f"  Saved diagnostic plot → {args.save_plot}")
            print()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
