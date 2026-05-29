#!/usr/bin/env python3
"""
plot_krylov_bench.py — Krylov-backend comparison (NOT preconditioner).

Compares three matrix-free, unpreconditioned solvers on field-solve wall-clock:
  native GMRES (built-in)  vs  PETSc GMRES  vs  PETSc FGMRES.

Reads the wide results.csv that tests/test.sh emits (one row per grid_size,
five columns per solver: <L>=cumulative "Field solver" seconds, <L>_std,
<L>_iters, <L>_converged, <L>_residual, plus a `cycles` column). Field time
per cycle = <L> / cycles * 1000 [ms].

Three panels:
  1. Double_Harris: field ms/cycle vs grid N (the historical baseline).
  2. Double_Harris: % faster/slower than native GMRES vs grid N.
  3. Single-grid cases (--case "Label=results.csv"): field time per backend
     normalized to native GMRES (=1.0), one group per case, iters annotated.

Usage:
  python scripts/plot_krylov_bench.py [--dh-csv F] \
      [--case "WhistlerPacket=path/results.csv"] [--case "dense DH=...csv"] \
      [--light] [-o out.png]
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_theme import add_theme_arg, apply_theme

# Solver order, display names, and mode-aware colors. We don't use
# plot_theme.get_solver_style here: its built-in prefix match maps both
# "PETScGMRES" and "PETScFGMRES" to the same "PETSc" colour. Pick directly.
SOLVERS = ["GMRES", "PETScGMRES", "PETScFGMRES"]
NICE = {"GMRES": "native GMRES", "PETScGMRES": "PETSc GMRES", "PETScFGMRES": "PETSc FGMRES"}
_PALETTE = {
    "dark":  {"GMRES": "#a6e3a1", "PETScGMRES": "#89b4fa", "PETScFGMRES": "#f9e2af"},
    "light": {"GMRES": "#228833", "PETScGMRES": "#4477AA", "PETScFGMRES": "#CCBB44"},
}
_MARK = {"GMRES": "o", "PETScGMRES": "s", "PETScFGMRES": "D"}
_LS   = {"GMRES": "-", "PETScGMRES": "--", "PETScFGMRES": "-."}


def sty(s, theme):
    """color, marker, linestyle for a solver under the active theme."""
    return _PALETTE[theme.mode][s], _MARK[s], _LS[s]


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def load(path):
    """Return (rows, present_solvers). Each row: dict with derived ms/cycle per solver."""
    if not path or not os.path.isfile(path):
        return [], []
    with open(path) as f:
        raw = list(csv.DictReader(f))
    if not raw:
        return [], []
    present = [s for s in SOLVERS if s in raw[0]]
    rows = []
    for r in raw:
        cyc = _f(r, "cycles")
        rec = {"grid_size": r["grid_size"]}
        for s in present:
            field_s = _f(r, s)
            std_s = _f(r, f"{s}_std")
            rec[s] = {
                "ms": field_s / cyc * 1000.0 if cyc else float("nan"),
                "ms_std": std_s / cyc * 1000.0 if cyc else float("nan"),
                "iters": _f(r, f"{s}_iters"),
                "converged": (r.get(f"{s}_converged", "") == "yes"),
                "residual": _f(r, f"{s}_residual"),
            }
        rows.append(rec)
    return rows, present


def main():
    ap = argparse.ArgumentParser(description="Krylov-backend benchmark comparison plot")
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    ap.add_argument("--dh-csv", default=os.path.join(proj, "tests/test_output_krylov_dh/results.csv"))
    ap.add_argument("--case", action="append", default=[],
                    metavar="LABEL=CSV", help="single-grid case (repeatable)")
    ap.add_argument("-o", "--out", default=os.path.join(here, "output", "krylov_bench.png"))
    add_theme_arg(ap)
    args = ap.parse_args()
    theme = apply_theme(args)

    dh, dh_solvers = load(args.dh_csv)
    # Parse single-grid cases: "Label=path". rpartition so a label containing
    # '=' (e.g. "np=1", "ρ=100") keeps the path intact (split on the last '=').
    cases = []
    for spec in args.case:
        label, _, path = spec.rpartition("=")
        rows, present = load(path)
        if rows:
            cases.append((label or os.path.basename(os.path.dirname(path)), rows[0], present))
    if not dh and not cases:
        sys.exit("No data found (check --dh-csv / --case paths)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axA, axB, axC = axes

    # ── Panel A: Double_Harris field ms/cycle vs grid ──────────────
    if dh:
        N = [int(float(r["grid_size"])) for r in dh]
        for s in dh_solvers:
            c, mk, ls = sty(s, theme)
            y = [r[s]["ms"] for r in dh]
            e = [r[s]["ms_std"] for r in dh]
            axA.errorbar(N, y, yerr=e, label=NICE[s], color=c,
                         marker=mk, ls=ls, lw=2, ms=7, capsize=3)
        axA.set_xlabel("grid size N (N×N)")
        axA.set_ylabel("field-solve time [ms / cycle]")
        axA.set_title("Double_Harris — field-solve wall time")
        axA.set_xticks(N)
        axA.legend(frameon=False, fontsize=9)

        # ── Panel B: % vs native GMRES ─────────────────────────────
        if "GMRES" in dh_solvers:
            for s in dh_solvers:
                if s == "GMRES":
                    continue
                c, mk, ls = sty(s, theme)
                pct = [(r["GMRES"]["ms"] - r[s]["ms"]) / r["GMRES"]["ms"] * 100.0
                       if r["GMRES"]["ms"] else float("nan") for r in dh]
                axB.plot(N, pct, label=NICE[s], color=c,
                         marker=mk, ls=ls, lw=2, ms=7)
            axB.axhline(0, color=theme.zero_line_color, lw=1, alpha=0.6)
            axB.set_xlabel("grid size N (N×N)")
            axB.set_ylabel("% faster than native GMRES")
            axB.set_title("Double_Harris — PETSc speedup vs native")
            axB.set_xticks(N)
            axB.legend(frameon=False, fontsize=9)

    # ── Panel C: single-grid cases, field time normalized to native ────
    if cases:
        ncase = len(cases)
        bw = 0.8 / len(SOLVERS)
        for j, s in enumerate(SOLVERS):
            c = sty(s, theme)[0]
            xs, ys, es, lbl = [], [], [], NICE[s]
            for ci, (_, r, present) in enumerate(cases):
                if s not in present or "GMRES" not in present:
                    continue
                base = r["GMRES"]["ms"]
                rel = r[s]["ms"] / base if base else float("nan")
                rel_e = r[s]["ms_std"] / base if base else float("nan")
                xc = ci + (j - (len(SOLVERS) - 1) / 2.0) * bw
                xs.append(xc); ys.append(rel); es.append(rel_e)
                axC.text(xc, rel + rel_e + 0.03, f"{r[s]['iters']:.0f}\nit", ha="center",
                         va="bottom", fontsize=7.5, color=theme.text_secondary)
                if s != "GMRES":
                    pct = (1.0 - rel) * 100.0
                    axC.text(xc, min(rel, 1.0) / 2.0, f"{pct:+.0f}%", ha="center", va="center",
                             fontsize=8, color=theme.text_color, rotation=90)
            if xs:
                axC.bar(xs, ys, width=bw, yerr=es, color=c, label=lbl, capsize=3,
                        edgecolor=theme.bar_edge, linewidth=0.5)
        axC.axhline(1.0, color=theme.zero_line_color, lw=1, ls="--", alpha=0.7)
        axC.set_xticks(range(ncase))
        axC.set_xticklabels([f"{lab}\n({r['grid_size']})" for lab, r, _ in cases], fontsize=8.5)
        axC.set_ylabel("field-solve time ÷ native GMRES")
        axC.set_title("Other inputs — relative cost (lower = faster)")
        tops = [r[s]["ms"] / r["GMRES"]["ms"] + r[s]["ms_std"] / r["GMRES"]["ms"]
                for _, r, present in cases for s in present if "GMRES" in present]
        axC.set_ylim(0, max(1.35, (max(tops) if tops else 1.0) * 1.18))
        axC.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)

    fig.suptitle("Krylov backend benchmark — matrix-free, unpreconditioned (PCNONE), restart 40",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
