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
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
