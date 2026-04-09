#!/usr/bin/env python3
"""
plot_preconditioner_study.py -- Generate figures for the v2 preconditioner study.

Produces 3 PNGs:
  1. presentation-iter-vs-dt.png   -- PCNONE iteration count vs dt (stiffness context)
  2. presentation-pc-shootout.png  -- wall-clock speedup for all PCs vs PCNONE
  3. presentation-restart-sweep.png -- field_s vs GMRES restart parameter

Usage:
    pixi run python scripts/plot_preconditioner_study.py [--light] [-o DIR]
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(2)

sys.path.insert(0, SCRIPT_DIR)
from plot_theme import add_theme_arg, apply_theme


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── Figure 1: Iteration count vs dt ─────────────────────────────────────────

def plot_iter_vs_dt(theme, outdir):
    rows = read_csv(os.path.join(PROJECT_DIR, "output/pc_study/phase1_results.csv"))

    dts, iters_none, iters_jac, iters_gamg = [], [], [], []
    kappas = {}  # dt -> kappa (from v1 data)
    kappa_map = {0.125: 2, 0.25: 4, 0.5: 17, 0.625: 25, 1.0: 63}

    for r in rows:
        dt = float(r['dt'])
        pc = r['PC']
        it = float(r['avg_iters'])
        if pc == 'none':
            dts.append(dt)
            iters_none.append(it)
            if dt in kappa_map:
                kappas[dt] = kappa_map[dt]
        elif pc == 'jacobi':
            iters_jac.append(it)
        elif pc == 'gamg':
            iters_gamg.append(it)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(dts, iters_none, 'o-', color='#89b4fa', lw=2, ms=7, label='PCNONE', zorder=3)

    # Jacobi
    dts_jac = [float(r['dt']) for r in rows if r['PC'] == 'jacobi']
    ax.plot(dts_jac, iters_jac, 's--', color='#fab387', lw=1.5, ms=5, label='Jacobi', alpha=0.8)

    # GAMG (only 2 points)
    dts_gamg = [float(r['dt']) for r in rows if r['PC'] == 'gamg']
    ax.plot(dts_gamg, iters_gamg, '^:', color='#f38ba8', lw=1.5, ms=5, label='GAMG', alpha=0.8)

    # Annotate kappa
    for dt, kappa in kappas.items():
        idx = dts.index(dt)
        it = iters_none[idx]
        ax.annotate(f'$\\kappa \\approx {kappa}$',
                    xy=(dt, it), xytext=(10, 12), textcoords='offset points',
                    fontsize=8, color=theme.text_secondary,
                    arrowprops=dict(arrowstyle='->', color=theme.text_tertiary, lw=0.8))

    ax.set_xlabel('$\\Delta t$', fontsize=12)
    ax.set_ylabel('Avg. GMRES iterations / cycle', fontsize=12)
    ax.set_title('Iteration count grows with $\\Delta t$  (100$\\times$100 grid)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0.05, 1.1)
    ax.set_ylim(0, 1000)

    out = os.path.join(outdir, 'presentation-iter-vs-dt.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: PC shootout ────────────────────────────────────────────────────

def plot_pc_shootout(theme, outdir):
    rows = read_csv(os.path.join(PROJECT_DIR, "output/pc_study/v2_phase8_summary.csv"))

    inputs = ['DH_dt0.5', 'DH_dt0.75', 'DH_dt1.0', 'DH_g200_contrast_dt025', 'DH_3D_g32_dt05']
    input_labels = ['dt=0.5\n100$^2$', 'dt=0.75\n100$^2$', 'dt=1.0\n100$^2$',
                    'dt=0.25\n200$^2$ 5:1$\\rho$', 'dt=0.5\n32$^3$']
    pcs = ['Jacobi', 'BJACOBI_ILU', 'ASM_ILU', 'GAMG', 'GAMG_tuned']
    pc_labels = ['Jacobi', 'BJac+ILU', 'ASM+ILU', 'GAMG', 'GAMG\ntuned']
    pc_colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#f38ba8', '#cba6f7']

    # Build speedup matrix
    speedup = {}
    converged = {}
    for r in rows:
        key = (r['input'], r['pc'])
        try:
            speedup[key] = float(r['speedup_vs_pcnone'])
        except (ValueError, KeyError):
            speedup[key] = 0
        converged[key] = r.get('converged', 'yes') == 'yes'

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(inputs))
    width = 0.15
    offsets = np.arange(len(pcs)) - (len(pcs) - 1) / 2

    for j, pc in enumerate(pcs):
        vals = []
        hatches = []
        for inp in inputs:
            sp = speedup.get((inp, pc), 0)
            vals.append(sp)
            hatches.append(not converged.get((inp, pc), True))

        bars = ax.bar(x + offsets[j] * width, vals, width * 0.9,
                      label=pc_labels[j], color=pc_colors[j], alpha=0.85,
                      edgecolor=theme.bar_edge, linewidth=0.5)

        # Mark failed bars
        for i, (bar, failed) in enumerate(zip(bars, hatches)):
            if failed:
                bar.set_hatch('///')
                bar.set_alpha(0.4)

    ax.axhline(1.0, color=theme.refline_color, ls='--', lw=1.2, zorder=1, label='PCNONE baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(input_labels, fontsize=9)
    ax.set_ylabel('Wall-clock speedup vs PCNONE', fontsize=11)
    ax.set_title('No preconditioner beats unpreconditioned GMRES', fontsize=13)
    ax.legend(fontsize=8.5, ncol=3, loc='upper right')
    ax.set_ylim(0, 1.4)

    # Add "FASTER" / "SLOWER" annotations
    ax.text(-0.6, 1.15, 'FASTER', fontsize=8, color='#a6e3a1', fontstyle='italic')
    ax.text(-0.6, 0.35, 'SLOWER', fontsize=8, color='#f38ba8', fontstyle='italic')

    out = os.path.join(outdir, 'presentation-pc-shootout.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Restart sweep ──────────────────────────────────────────────────

def plot_restart_sweep(theme, outdir):
    test_dir = os.path.join(PROJECT_DIR, "tests")
    ids = ['DH_dt0.5', 'DH_dt0.75', 'DH_dt1.0', 'DH_g200_contrast_dt025', 'DH_3D_g32_dt05']
    labels = ['dt=0.5 (142 it)', 'dt=0.75 (248 it)', 'dt=1.0 (369 it)',
              '200$^2$ contrast (79 it)', '32$^3$ 3D (125 it)']
    colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#cba6f7', '#f38ba8']
    restarts = [20, 30, 40, 50]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (ID, label) in enumerate(zip(ids, labels)):
        csv_path = os.path.join(test_dir, f"test_output_v2_pc9b_{ID}", "results.csv")
        if not os.path.exists(csv_path):
            continue
        rows = read_csv(csv_path)
        r = rows[0]

        ref = float(r['PCNONE_r20'])
        vals, stds = [], []
        for restart in restarts:
            key = f'PCNONE_r{restart}'
            t = float(r[key])
            s = float(r.get(f'{key}_std', 0) or 0)
            vals.append(ref / t)  # speedup
            stds.append(s / ref)  # relative std

        ax.errorbar(restarts, vals, yerr=stds, marker='o', ms=6, lw=1.8,
                    color=colors[i], label=label, capsize=3, alpha=0.85)

    ax.axhline(1.0, color=theme.refline_color, ls='--', lw=1, alpha=0.5)
    ax.axvline(40, color=theme.text_tertiary, ls=':', lw=1, alpha=0.5)
    ax.text(41, 0.88, 'r=40\n(new default)', fontsize=8, color=theme.text_secondary)

    ax.set_xlabel('GMRES restart parameter', fontsize=11)
    ax.set_ylabel('Speedup vs restart=20', fontsize=11)
    ax.set_title('Larger restart helps on stiff problems (+10-20%)', fontsize=13)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_xticks(restarts)
    ax.set_ylim(0.82, 1.25)

    out = os.path.join(outdir, 'presentation-restart-sweep.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', default=os.path.join(PROJECT_DIR, 'output/pc_study'),
                        help='Output directory for PNGs')
    add_theme_arg(parser)
    args = parser.parse_args()
    theme = apply_theme(args)

    outdir = args.output
    os.makedirs(outdir, exist_ok=True)

    print("Generating preconditioner study figures...")
    plot_iter_vs_dt(theme, outdir)
    plot_pc_shootout(theme, outdir)
    plot_restart_sweep(theme, outdir)
    print("Done.")


if __name__ == '__main__':
    main()
