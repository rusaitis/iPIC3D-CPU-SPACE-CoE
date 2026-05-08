#!/usr/bin/env python3
"""analyze_growth_rate.py — fit linear growth rate γ from an iPIC3D run.

Reads ConservedQuantities.txt, identifies the exponential-growth phase of
the E-field energy, fits log(E_energy) vs t with a straight line, and
returns 2γ (since E_energy ∝ |E|² ∝ exp(2γt)).

Compares to a theoretical γ_expected supplied by the caller (`--expected`).

Used by Test 3 (two-stream / Weibel instability):
    γ_max = ω_pe / (2√2)   for cold counter-streaming electron beams.

Exit 0 on PASS (|Δγ/γ| < tol), 1 on FAIL.
"""

import argparse
import math
import os
import sys


def parse_input_file(path):
    """Pull simulation parameters from an iPIC3D .inp file."""
    keys = {'dt', 'ncycles', 'c', 'Lx', 'Ly', 'Lz', 'B0x', 'B0y', 'B0z',
            'DiagnosticsOutputCycle'}
    arr_keys = ('rhoINIT', 'qom', 'u0', 'v0', 'w0', 'custom_parameters')
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


def find_linear_regime(log_energy, t):
    """Locate the exponential-growth window — the segment with maximum slope.

    Slides a fixed-size window over the curve (skipping the first 5% as
    initial transient and the last 5% as saturation tail) and returns the
    window that produces the steepest log(E_energy) slope. Robust against
    multi-mode interference, slow ramp-up, and saturation curvature.
    """
    import numpy as np
    n = len(log_energy)
    n_skip = max(5, n // 20)
    window = max(20, n // 6)
    best_slope = -float('inf')
    best_start = n_skip
    for i in range(n_skip, n - window - n_skip):
        slope = np.polyfit(t[i:i+window], log_energy[i:i+window], 1)[0]
        if slope > best_slope:
            best_slope = slope
            best_start = i
    return best_start, best_start + window


def evaluate_expected(formula, ctx):
    """Evaluate a theoretical γ formula against simulation parameters."""
    safe = {
        'pi': math.pi, 'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
        '__builtins__': {},
    }
    safe.update(ctx)
    return eval(formula, safe)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dir', help='Run output directory (contains ConservedQuantities.txt)')
    ap.add_argument('--inp', required=True, help='Path to the .inp file used for the run')
    ap.add_argument('--expected', required=True,
                    help='Formula for γ_expected, e.g. "wpe/(2*sqrt(2))"')
    ap.add_argument('--tol', type=float, default=0.20,
                    help='Pass tolerance |Δγ/γ|. Default 0.20.')
    ap.add_argument('--window', default=None,
                    help='Fit window as "i_start,i_end" diagnostic-row indices (default: auto)')
    ap.add_argument('--label', default=None, help='Test label for the printout')
    ap.add_argument('--field', choices=('electric', 'magnetic'), default='electric',
                    help='Energy series to fit: electric (default, two-stream/Langmuir) or magnetic (Weibel).')
    ap.add_argument('--save-plot', default=None, dest='save_plot',
                    help='Optional path to save log(E)-vs-t diagnostic PNG.')
    ap.add_argument('--light', action='store_true',
                    help='Use light theme for the saved plot (default dark).')
    args = ap.parse_args()

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_utils import parse_conserved_quantities
        import numpy as np
    except ImportError as e:
        print(f"FAIL: missing dependency: {e}")
        sys.exit(2)

    sim = parse_input_file(args.inp)
    cq = parse_conserved_quantities(args.output_dir)
    if not cq or len(cq['cycle']) < 50:
        print(f"FAIL: only {len(cq.get('cycle', []))} ConservedQuantities rows — need ≥ 50 for fit")
        sys.exit(1)

    cycles = np.array(cq['cycle'], dtype=float)
    series_key = 'E_energy' if args.field == 'electric' else 'B_energy'
    if series_key not in cq:
        print(f"FAIL: ConservedQuantities missing column '{series_key}'")
        sys.exit(1)
    energy = np.array(cq[series_key], dtype=float)
    energy = np.clip(energy, 1e-300, None)                  # log domain
    log_energy = np.log(energy)
    t = cycles * sim['dt']

    if args.window is not None:
        i_start, i_end = (int(x) for x in args.window.split(','))
    else:
        i_start, i_end = find_linear_regime(log_energy, t)
    if i_end - i_start < 10:
        print(f"FAIL: linear regime window [{i_start}, {i_end}] too narrow")
        sys.exit(1)

    # Linear fit: log(E_energy) = (2γ)·t + const
    slope, _ = np.polyfit(t[i_start:i_end], log_energy[i_start:i_end], 1)
    gamma_measured = slope / 2.0

    # Theory inputs
    rho = sim.get('rhoINIT', [])
    qom = sim.get('qom', [])
    u0 = sim.get('u0', [])
    v0 = sim.get('v0', [])
    w0 = sim.get('w0', [])
    if not (rho and qom):
        print("FAIL: missing rhoINIT/qom in input file")
        sys.exit(1)

    # Total electron plasma frequency: ω_pe² = Σ_{electron species} rhoINIT_s · |qom_s|.
    # Heuristic: electrons are species with qom < 0.
    omega_pe2 = sum(abs(r) * abs(q) for r, q in zip(rho, qom) if q < 0)
    omega_pe = math.sqrt(omega_pe2) if omega_pe2 > 0 else 0.0
    # Drift velocity: max |drift| over all axes (Weibel beams may use w0 instead of u0).
    drifts = list(u0) + list(v0) + list(w0)
    v_drift = max((abs(d) for d in drifts), default=0.0)
    Ly = sim.get('Ly', 1.0)
    custom = sim.get('custom_parameters', [])
    m_y = float(custom[0]) if custom else 1.0

    ctx = {
        'wpe': omega_pe, 'omega_pe': omega_pe,
        'v_d': v_drift, 'vd': v_drift, 'u_b': v_drift, 'u_d': v_drift,
        'c': sim.get('c', 1.0),
        'Ly': Ly, 'm_y': m_y,
        'k_y': 2.0 * math.pi * m_y / Ly if Ly > 0 else 0.0,
    }
    gamma_expected = evaluate_expected(args.expected, ctx)
    rel = abs(gamma_measured - gamma_expected) / abs(gamma_expected) if gamma_expected else float('inf')
    ok = rel < args.tol

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    field_label = 'E-field' if args.field == 'electric' else 'B-field'
    print()
    print(f"Growth-rate check — {label}  ({field_label})")
    print('─' * 56)
    print(f"  Conserved-quantity rows: {len(cycles)}")
    print(f"  Linear-fit window      : cycles {int(cycles[i_start])}..{int(cycles[i_end-1])}  (rows {i_start}..{i_end-1})")
    print(f"  ω_pe (from species qom<0): {omega_pe:.4g}")
    print(f"  v_drift                : {v_drift:.4g}")
    print(f"  γ_expected ({args.expected}) = {gamma_expected:.4g}")
    print(f"  γ_measured (slope/2)            = {gamma_measured:.4g}")
    print(f"  |Δγ/γ|                 : {rel:.3e}    tol = {args.tol:.0e}")
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

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0),
                                    constrained_layout=True)

            # Skip cycle 0 (often clipped to log = -690 from initial near-zero E
            # field); set a sensible y-window around the fit segment + saturation.
            i_plot0 = max(1, i_start // 4)
            ax.plot(t[i_plot0:], log_energy[i_plot0:],
                    color=theme.text_color, lw=1.1,
                    label=f'log {field_label} energy', alpha=0.85)
            ax.axvspan(t[i_start], t[i_end - 1], color=theory_color, alpha=0.18,
                       label=f'fit window  (cyc {int(cycles[i_start])}..{int(cycles[i_end-1])})')

            t_window = t[i_start:i_end]
            fit_intercept = np.polyfit(t_window, log_energy[i_start:i_end], 1)[1]
            ax.plot(t_window, slope * t_window + fit_intercept, color=measured_color, lw=2.0,
                    label=f'fit  2g_meas = {slope:.4g}  (g_meas = {gamma_measured:.4g})')

            theory_slope = 2.0 * gamma_expected
            ax.plot(t_window, theory_slope * t_window + fit_intercept, color=theory_color, lw=1.8, ls='--',
                    label=f'theory  2g_th = {theory_slope:.4g}  (g_th = {gamma_expected:.4g})')

            # Zoom y to the visible curve only.
            ymin = float(log_energy[i_plot0:].min())
            ymax = float(log_energy[i_plot0:].max())
            pad = 0.05 * max(1.0, ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_xlabel('t [code time]')
            ax.set_ylabel(f'log {field_label} energy')
            ax.legend(loc='lower right', fontsize=9)
            ax.text(0.02, 0.98,
                    f'|Δγ/γ| = {rel:.2e}\nω_pe = {omega_pe:.4g}\nv_drift = {v_drift:.4g}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor=theme.box_face,
                              edgecolor=theme.box_edge,
                              alpha=theme.box_alpha))

            verdict = 'PASS' if ok else 'FAIL'
            fig.suptitle(f'{label}  —  |Δγ/γ| = {rel:.2e}  [{verdict}]', fontsize=12)
            fig.savefig(args.save_plot, dpi=140)
            plt.close(fig)
            print(f"  Saved diagnostic plot → {args.save_plot}")
            print()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
