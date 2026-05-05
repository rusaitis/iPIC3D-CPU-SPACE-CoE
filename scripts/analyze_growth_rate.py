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
            elif lhs in ('rhoINIT', 'qom', 'u0'):
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
    e_energy = np.array(cq['E_energy'], dtype=float)
    e_energy = np.clip(e_energy, 1e-300, None)              # log domain
    log_energy = np.log(e_energy)
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
    if not (rho and qom):
        print("FAIL: missing rhoINIT/qom in input file")
        sys.exit(1)

    # Total electron plasma frequency: ω_pe² = Σ_{electron species} rhoINIT_s · |qom_s|.
    # Heuristic: electrons are species with qom < 0.
    omega_pe2 = sum(abs(r) * abs(q) for r, q in zip(rho, qom) if q < 0)
    omega_pe = math.sqrt(omega_pe2) if omega_pe2 > 0 else 0.0
    v_drift = max((abs(d) for d in u0), default=0.0)

    ctx = {
        'wpe': omega_pe, 'omega_pe': omega_pe,
        'v_d': v_drift, 'vd': v_drift,
        'c': sim.get('c', 1.0),
    }
    gamma_expected = evaluate_expected(args.expected, ctx)
    rel = abs(gamma_measured - gamma_expected) / abs(gamma_expected) if gamma_expected else float('inf')
    ok = rel < args.tol

    label = args.label or os.path.basename(args.output_dir.rstrip('/'))
    print()
    print(f"Growth-rate check — {label}")
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
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
