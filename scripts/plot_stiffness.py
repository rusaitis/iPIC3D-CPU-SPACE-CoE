#!/usr/bin/env python3
"""
plot_stiffness.py — Visualize the (dt, N) stiffness frontier for the implicit
Maxwell solve and the preconditioner cost/quality trade-off.

Reads the tidy CSV produced by tests/bench_stiffness.sh:
  study,dt,nxc,nyc,nzc,np,topo,cycles,solver,mean_iters,max_iters,nonconv,
  field_total_s,field_per_cycle_ms,wall_s

Two stacked panels (KSP iterations, field-solver ms/cycle) versus the swept
variable (dt by default, or grid N via --x nxc), one line per solver, one column
per facet value of the other variable.  Non-converged points are ringed.

Usage:
  python scripts/plot_stiffness.py CSV [-o out.png] [--x dt|nxc] [--light]
"""
import argparse
import csv
import os
import sys


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("dt", "field_per_cycle_ms", "field_total_s", "wall_s",
                      "mean_iters", "max_iters"):
                try:
                    r[k] = float(r[k])
                except (ValueError, KeyError):
                    r[k] = float("nan")
            for k in ("nxc", "nyc", "nzc", "np", "cycles", "nonconv"):
                try:
                    r[k] = int(float(r[k]))
                except (ValueError, KeyError):
                    r[k] = 0
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--x", choices=["dt", "nxc"], default="dt",
                    help="swept variable on the x-axis (default dt)")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not args.light:
        plt.style.use("dark_background")

    rows = load(args.csv)
    if not rows:
        print("ERROR: no rows", file=sys.stderr)
        return 1

    xkey = args.x
    facetkey = "nxc" if xkey == "dt" else "dt"
    facets = sorted({r[facetkey] for r in rows})
    solvers = sorted({r["solver"] for r in rows})

    # Stable color per solver; PCNONE always the neutral reference
    palette = {
        "PCNONE": "#888888", "GAMG": "#4c9be8", "GAMG_reuse": "#2ecc71",
        "Helmholtz": "#e67e22", "HYPRE": "#9b59b6",
    }
    def color(s): return palette.get(s, None)

    nfac = len(facets)
    fig, axes = plt.subplots(2, nfac, figsize=(4.6 * nfac, 7.2), squeeze=False,
                             sharex=True)

    for ci, fv in enumerate(facets):
        ax_it, ax_t = axes[0][ci], axes[1][ci]
        for s in solvers:
            pts = sorted((r for r in rows if r["solver"] == s and r[facetkey] == fv),
                         key=lambda r: r[xkey])
            if not pts:
                continue
            xs = [p[xkey] for p in pts]
            its = [p["mean_iters"] for p in pts]
            tms = [p["field_per_cycle_ms"] for p in pts]
            c = color(s)
            ax_it.plot(xs, its, "o-", color=c, label=s, lw=1.8, ms=5)
            ax_t.plot(xs, tms, "o-", color=c, label=s, lw=1.8, ms=5)
            # Ring non-converged points
            for p, x, y in zip(pts, xs, its):
                if p["nonconv"] > 0:
                    ax_it.plot([x], [y], "o", mfc="none", mec="red", ms=12, mew=2)

        flabel = f"{int(fv)}²" if facetkey == "nxc" else f"dt={fv:g}"
        ax_it.set_title(flabel)
        ax_it.set_ylabel("mean KSP iters")
        ax_t.set_ylabel("field solve  (ms/cycle)")
        ax_t.set_yscale("log")
        ax_t.set_xlabel("dt" if xkey == "dt" else "grid N")
        for ax in (ax_it, ax_t):
            ax.grid(True, alpha=0.25)
            if xkey == "dt":
                ax.set_xscale("log")
        ax_it.legend(fontsize=8, loc="best")

    title = args.title or f"Stiffness frontier — {os.path.basename(args.csv)}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
