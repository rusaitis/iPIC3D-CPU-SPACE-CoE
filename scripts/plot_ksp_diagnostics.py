#!/usr/bin/env python3
"""Parse and plot PETSc KSP runtime diagnostics from iPIC3D output.

Parses the output produced when running iPIC3D with PETSc diagnostic flags:
    -ksp_monitor              (per-iteration residual norms)
    -ksp_view_eigenvalues     (Ritz eigenvalues after each solve)
    -ksp_view_singularvalues  (extreme singular values / condition number)

Usage:
    # Generate the log:
    mpirun -np 8 build/iPIC3D input.inp -solver PETSc \\
        -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor \\
        2>&1 | tee run.log

    # Plot:
    pixi run python scripts/plot_ksp_diagnostics.py run.log
    pixi run python scripts/plot_ksp_diagnostics.py run.log --light -o diagnostics.png
    pixi run python scripts/plot_ksp_diagnostics.py run.log --prefix DH_dt5x
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from plot_theme import add_theme_arg, apply_theme

# Theme-aware accent colors
_ACCENT = {
    "dark":  {"green": "#a6e3a1", "red": "#f38ba8", "blue": "#89b4fa", "yellow": "#f9e2af"},
    "light": {"green": "#228833", "red": "#EE6677", "blue": "#4477AA", "yellow": "#999933"},
}


def parse_log(filename):
    """Parse PETSc diagnostic output into per-cycle data structures.

    Returns:
        cycles: list of cycle numbers
        residuals: dict {cycle: [r0, r1, ...]} per-iteration residual norms
        eigenvalues: dict {cycle: [(re, im), ...]} Ritz eigenvalues
        singular: dict {cycle: (sigma_max, sigma_min, cond)} extreme singular values
        iterations: dict {cycle: n_iter} iteration count
    """
    cycles = []
    residuals = {}
    eigenvalues = {}
    singular = {}
    iterations = {}

    current_cycle = None
    parsing_eigenvalues = False

    with open(filename) as f:
        for line in f:
            line = line.strip()

            m = re.match(r"=+ Cycle (\d+) =+", line)
            if m:
                current_cycle = int(m.group(1))
                cycles.append(current_cycle)
                residuals[current_cycle] = []
                eigenvalues[current_cycle] = []
                parsing_eigenvalues = False
                continue

            if current_cycle is None:
                continue

            m = re.match(r"\s*(\d+)\s+KSP Residual norm\s+(\S+)", line)
            if m:
                residuals[current_cycle].append(float(m.group(2)))
                parsing_eigenvalues = False
                continue

            if line == "Iteratively computed eigenvalues":
                parsing_eigenvalues = True
                continue

            if parsing_eigenvalues:
                m = re.match(r"([\d.e+\-]+)\s+([+\-])\s+([\d.e+\-]+)i", line)
                if m:
                    real = float(m.group(1))
                    sign = 1.0 if m.group(2) == "+" else -1.0
                    imag = sign * float(m.group(3))
                    eigenvalues[current_cycle].append((real, imag))
                    continue
                else:
                    parsing_eigenvalues = False

            m = re.match(
                r"Iteratively computed extreme singular values:\s+max\s+(\S+)\s+min\s+(\S+)\s+max/min\s+(\S+)",
                line,
            )
            if m:
                singular[current_cycle] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue

            m = re.match(r"PETSc KSP converged: cycle=(\d+) iterations=(\d+) residual=(\S+)", line)
            if m:
                iterations[int(m.group(1))] = int(m.group(2))
                continue

            m = re.match(
                r"WARNING: PETSc KSP did NOT converge: cycle=(\d+) reason=\S+ iterations=(\d+)",
                line,
            )
            if m:
                iterations[int(m.group(1))] = int(m.group(2))

    return cycles, residuals, eigenvalues, singular, iterations


def print_summary(cycles, residuals, eigenvalues, singular, iterations):
    """Print a text summary of parsed data."""
    print(f"\nParsed {len(cycles)} cycles")
    n_with_res = sum(1 for c in cycles if residuals.get(c))
    n_with_eig = sum(1 for c in cycles if eigenvalues.get(c))
    n_with_sv = sum(1 for c in cycles if c in singular)
    print(f"  Residual histories: {n_with_res} cycles")
    print(f"  Eigenvalues:        {n_with_eig} cycles")
    print(f"  Singular values:    {n_with_sv} cycles")

    if iterations:
        iters = [iterations[c] for c in cycles if c in iterations]
        print(f"  Iterations:         min={min(iters)}, max={max(iters)}, mean={np.mean(iters):.1f}")

    if singular:
        conds = [singular[c][2] for c in cycles if c in singular]
        print(f"  Condition number:   min={min(conds):.3f}, max={max(conds):.3f}")
    print()


def plot_diagnostics(cycles, residuals, eigenvalues, singular, iterations, theme, args):
    """Create 4-panel diagnostic plot."""
    mode = theme.mode
    accent = _ACCENT[mode]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cycle_arr = np.array([c for c in cycles if c in iterations])

    # --- Panel 1: Iteration count vs cycle ---
    ax = axes[0, 0]
    iter_arr = np.array([iterations[c] for c in cycle_arr])
    ax.plot(cycle_arr, iter_arr, "o-", markersize=4, color=accent["green"])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("KSP iterations")
    ax.set_title("Iterations per cycle")

    # --- Panel 2: Residual convergence curves ---
    ax = axes[0, 1]
    cycles_with_res = [c for c in cycles if residuals.get(c)]
    if cycles_with_res:
        n_show = min(args.max_curves, len(cycles_with_res))
        indices = np.linspace(0, len(cycles_with_res) - 1, n_show, dtype=int)
        cmap = plt.colormaps["viridis"]
        norm = Normalize(vmin=min(cycles_with_res), vmax=max(cycles_with_res))

        for idx in indices:
            c = cycles_with_res[idx]
            res = residuals[c]
            ax.semilogy(range(len(res)), res, "-", color=cmap(norm(c)),
                        alpha=0.8, linewidth=1.5, label=f"cycle {c}")

        ax.set_xlabel("KSP iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title("Convergence curves (selected cycles)")
        ax.legend(fontsize=7, ncol=2)

    # --- Panel 3: Ritz eigenvalues in complex plane ---
    ax = axes[1, 0]
    cycles_with_eig = [c for c in cycles if eigenvalues.get(c)]
    if cycles_with_eig:
        cmap = plt.colormaps["viridis"]
        norm = Normalize(vmin=min(cycles_with_eig), vmax=max(cycles_with_eig))

        for c in cycles_with_eig:
            eigs = eigenvalues[c]
            if not eigs:
                continue
            re_vals = [e[0] for e in eigs]
            im_vals = [e[1] for e in eigs]
            ax.scatter(re_vals, im_vals, s=20, color=cmap(norm(c)),
                       alpha=0.6, edgecolors="none")

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Cycle")
        ax.axhline(0, color=theme.refline_color, linewidth=0.5, alpha=0.5)
        ax.axvline(1, color=theme.refline_color, linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Ritz eigenvalues (complex plane)")
    else:
        ax.text(0.5, 0.5, "No eigenvalue data\n(use -ksp_view_eigenvalues)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Ritz eigenvalues")

    # --- Panel 4: Condition number vs cycle ---
    ax = axes[1, 1]
    cycles_with_sv = [c for c in cycles if c in singular]
    if cycles_with_sv:
        sv_cycles = np.array(cycles_with_sv)
        cond_arr = np.array([singular[c][2] for c in cycles_with_sv])
        smax_arr = np.array([singular[c][0] for c in cycles_with_sv])
        smin_arr = np.array([singular[c][1] for c in cycles_with_sv])

        ax.plot(sv_cycles, cond_arr, "o-", markersize=4, color=accent["red"],
                label="cond (σ_max/σ_min)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Condition number")
        ax.set_title("Condition number vs cycle")
        ax.legend()

        ax2 = ax.twinx()
        ax2.plot(sv_cycles, smax_arr, "--", color=accent["blue"], alpha=0.5,
                 linewidth=1, label="σ_max")
        ax2.plot(sv_cycles, smin_arr, "--", color=accent["yellow"], alpha=0.5,
                 linewidth=1, label="σ_min")
        ax2.set_ylabel("Singular values", alpha=0.6)
        ax2.legend(loc="center right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No singular value data\n(use -ksp_view_singularvalues)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Condition number")

    fig.suptitle("PETSc KSP Diagnostics", fontsize=14)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot PETSc KSP diagnostics from iPIC3D output log"
    )
    parser.add_argument("logfile", help="Log file from iPIC3D run with PETSc diagnostic flags")
    parser.add_argument("-o", "--output", default=None,
                        help="Save figure to FILE")
    parser.add_argument("--prefix", metavar="NAME",
                        help="Run name prefix — auto-saves next to input file as {PREFIX}_ksp_diagnostics.png")
    parser.add_argument("--max-curves", type=int, default=8,
                        help="Max convergence curves to show (default: 8)")
    add_theme_arg(parser)
    args = parser.parse_args()

    # Default output: same directory as input file
    if args.output is None and args.prefix:
        input_dir = os.path.dirname(args.logfile) or "."
        args.output = os.path.join(input_dir, f"{args.prefix}_ksp_diagnostics.png")

    theme = apply_theme(args)

    cycles, residuals, eigenvalues, singular, iterations = parse_log(args.logfile)
    print_summary(cycles, residuals, eigenvalues, singular, iterations)
    plot_diagnostics(cycles, residuals, eigenvalues, singular, iterations, theme, args)


if __name__ == "__main__":
    main()
