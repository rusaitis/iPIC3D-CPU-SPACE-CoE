#!/usr/bin/env python3
"""
plot_stiffness_summary.py — Polished summary figures for the preconditioner /
stiffness investigation (see slides/preconditioner_study.md).

Generates five themed figures from the tests/test_output_stiffness/*.csv data:
  fig1_scaling.png   iterations vs field-solve time as grid N grows (dt=0.125)
  fig2_frontier.png  the (dt) stiffness frontier — PCNONE balloons; AMG diverges
  fig3_3d.png        2D vs 3D per-iteration cost — AMG blows up, Helmholtz < GAMG
  fig4_drivers.png   what drives stiffness (magnetization vs density) + stiff-end PCs
  fig5_stress.png    the crossover hunt — GAMG's speedup never clears break-even

Uses the shared Catppuccin theme (scripts/plot_theme.py).

Usage:
  python scripts/plot_stiffness_summary.py [--light] [--csvdir DIR] [--outdir DIR]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_theme import add_theme_arg, apply_theme, register_solver_style, get_solver_style

# Distinct, mode-aware colors for the four solvers in this study.
_PALETTE = {
    "dark":  {"PCNONE": "#a6e3a1", "GAMG": "#f38ba8", "GAMG_reuse": "#cba6f7", "Helmholtz": "#89b4fa"},
    "light": {"PCNONE": "#228833", "GAMG": "#EE6677", "GAMG_reuse": "#AA3377", "Helmholtz": "#4477AA"},
}
_MARK = {"PCNONE": "o", "GAMG": "s", "GAMG_reuse": "D", "Helmholtz": "^"}
_LS   = {"PCNONE": "-", "GAMG": "-", "GAMG_reuse": "--", "Helmholtz": "-."}
_NICE = {"PCNONE": "PCNONE (no precond.)", "GAMG": "GAMG on P",
         "GAMG_reuse": "GAMG + reuse", "Helmholtz": "Helmholtz"}


def register(theme):
    pal = _PALETTE[theme.mode]
    for k, c in pal.items():
        register_solver_style(k, c, _MARK[k], _LS[k])


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("dt", "field_per_cycle_ms", "mean_iters"):
                try: r[k] = float(r[k])
                except (ValueError, KeyError): r[k] = float("nan")
            for k in ("nxc", "nonconv"):
                try: r[k] = int(float(r[k]))
                except (ValueError, KeyError): r[k] = 0
            rows.append(r)
    return rows


def style(s):
    st = get_solver_style(s)
    return st["color"], st["marker"], st["ls"]


def fig_scaling(csvdir, outdir, theme):
    import matplotlib.pyplot as plt
    rows = [r for r in load(f"{csvdir}/scaling125.csv")]
    solvers = ["PCNONE", "GAMG", "GAMG_reuse"]
    fig, (a_it, a_t) = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in solvers:
        pts = sorted((r for r in rows if r["solver"] == s), key=lambda r: r["nxc"])
        N = [p["nxc"] for p in pts]
        c, m, ls = style(s)
        a_it.plot(N, [p["mean_iters"] for p in pts], marker=m, ls=ls, color=c, label=_NICE[s], ms=6)
        a_t.plot(N, [p["field_per_cycle_ms"] for p in pts], marker=m, ls=ls, color=c, label=_NICE[s], ms=6)
    a_it.set(xlabel="grid N (N×N)", ylabel="mean KSP iterations",
             title="Iterations: GAMG halves them at large N")
    a_it.annotate("crossover ≈ N=175", xy=(175, 24.2), xytext=(150, 18),
                  fontsize=8.5, color=theme.text_secondary,
                  arrowprops=dict(arrowstyle="->", color=theme.text_secondary, lw=1))
    a_t.set_yscale("log")
    a_t.set(xlabel="grid N (N×N)", ylabel="field-solve time (ms / cycle)",
            title="Wall-time: GAMG loses 4–5× anyway")
    a_t.annotate("per-iteration cost 5–7× PCNONE\n(AMG V-cycle vs matrix-free matvec)",
                 xy=(200, 857), xytext=(150, 165), fontsize=8.5,
                 color=theme.text_secondary)
    a_it.legend(loc="lower left")
    a_t.legend(loc="upper left")
    fig.suptitle("2D scaling at dt=0.125 — the iterations-vs-wall-time split",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{outdir}/fig1_scaling.png"; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


def fig_frontier(csvdir, outdir, theme):
    import matplotlib.pyplot as plt
    rows = load(f"{csvdir}/dt_frontier.csv")
    fig, (a_it, a_t) = plt.subplots(1, 2, figsize=(11, 4.4))
    c, _, _ = style("PCNONE")
    for N, mk, alpha in ((100, "o", 1.0), (200, "s", 0.75)):
        pts = sorted((r for r in rows if r["nxc"] == N), key=lambda r: r["dt"])
        dt = [p["dt"] for p in pts]
        a_it.plot(dt, [p["mean_iters"] for p in pts], marker=mk, color=c, ms=6,
                  alpha=alpha, label=f"PCNONE {N}²")
        a_t.plot(dt, [p["field_per_cycle_ms"] for p in pts], marker=mk, color=c, ms=6,
                 alpha=alpha, label=f"PCNONE {N}²")
    # AMG-divergence band (dt >= 0.25)
    for ax in (a_it, a_t):
        ax.axvspan(0.18, ax.get_xlim()[1] if False else 2.2, alpha=0.10,
                   color=_PALETTE[theme.mode]["GAMG"])
        ax.set_xscale("log"); ax.set_xlim(0.10, 1.25); ax.legend(loc="upper left")
    a_it.set(xlabel="dt", ylabel="mean KSP iterations",
             title="PCNONE iterations balloon with dt")
    a_t.set_yscale("log")
    a_t.set(xlabel="dt", ylabel="field-solve time (ms / cycle)",
            title="…and so does its cost")
    a_it.annotate("AMG (GAMG / HYPRE / FGMRES)\nDIVERGES for dt ≥ 0.25\n(Helmholtz lags)",
                  xy=(0.5, 137), xytext=(0.135, 220), fontsize=8.5,
                  color=_PALETTE[theme.mode]["GAMG"])
    fig.suptitle("Stiffness frontier — conditioning is set by dt, not grid",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{outdir}/fig2_frontier.png"; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


def fig_3d(csvdir, outdir, theme):
    import matplotlib.pyplot as plt
    import numpy as np
    r2 = {r["solver"]: r for r in load(f"{csvdir}/scaling125.csv") if r["nxc"] == 200}
    r3 = {r["solver"]: r for r in load(f"{csvdir}/threeD_48.csv")}
    solvers = ["PCNONE", "GAMG_reuse", "GAMG", "Helmholtz"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(solvers)); w = 0.38
    t2 = [r2.get(s, {}).get("field_per_cycle_ms", float("nan")) for s in solvers]
    t3 = [r3.get(s, {}).get("field_per_cycle_ms", float("nan")) for s in solvers]
    ax.bar(x - w/2, t2, w, label="2D  (200²)", color=theme.text_tertiary, edgecolor=theme.bar_edge)
    cols = [_PALETTE[theme.mode][s] for s in solvers]
    ax.bar(x + w/2, t3, w, label="3D  (48³)", color=cols, edgecolor=theme.bar_edge)
    # iteration annotations on the 3D bars
    for xi, s in zip(x, solvers):
        it = r3.get(s, {}).get("mean_iters", float("nan"))
        v = r3.get(s, {}).get("field_per_cycle_ms", float("nan"))
        if v == v:
            ax.text(xi + w/2, v, f"{it:.0f} it" if it == it else "div",
                    ha="center", va="bottom", fontsize=8, color=theme.text_color)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([_NICE[s] for s in solvers], fontsize=9)
    ax.set_ylabel("field-solve time (ms / cycle, log)")
    ax.set_title("3D blows up AMG — PCNONE wins, Helmholtz beats GAMG", fontsize=12)
    ax.set_ylim(top=t3[2] * 2.4)  # headroom above the tallest (GAMG) bar
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    out = f"{outdir}/fig3_3d.png"; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


def fig_drivers(csvdir, outdir, theme):
    """fig4: what drives stiffness (B vs density) + preconditioners at the stiff end."""
    import matplotlib.pyplot as plt
    need = [f"{csvdir}/bsweep_uniform.csv", f"{csvdir}/bsweep_whistler.csv",
            f"{csvdir}/stiff_pc.csv"] + [f"{csvdir}/dens_{r}.csv" for r in (1, 10, 100)]
    if not all(os.path.exists(p) for p in need):
        print("fig4: missing B-sweep/density/stiff CSVs — skipping")
        return
    pcn = _PALETTE[theme.mode]["PCNONE"]
    fig, (axB, axR, axP) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel 1 — iterations vs magnetization B0x (flat → B irrelevant)
    def bx(v):
        try: return float(v)
        except ValueError: return 0.0
    for tag, lab, col in (("bsweep_uniform", "uniform  ρ=1", pcn),
                          ("bsweep_whistler", "whistler  ρ=100", _PALETTE[theme.mode]["GAMG"])):
        rows = sorted(load(f"{csvdir}/{tag}.csv"), key=lambda r: bx(r["b0x"]))
        axB.plot([bx(r["b0x"]) for r in rows], [r["mean_iters"] for r in rows],
                 "o-", color=col, ms=6, label=lab)
    axB.set(xlabel="magnetization  B0x", ylabel="mean KSP iterations",
            title="Magnetization: no effect")
    axB.set_ylim(bottom=0); axB.legend(loc="center right")

    # Panel 2 — iterations vs density (rising → the real driver)
    rho = [1, 10, 100]
    its = [load(f"{csvdir}/dens_{r}.csv")[0]["mean_iters"] for r in rho]
    axR.plot(rho, its, "o-", color=pcn, ms=7)
    axR.set_xscale("log")
    axR.set(xlabel="plasma density  ρ (rhoINIT)", ylabel="mean KSP iterations",
            title="Density: the real driver")
    for x, y in zip(rho, its):
        axR.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=9, color=theme.text_secondary)

    # Panel 3 — preconditioners at the stiff end (ρ=100, PCNONE≈218 it)
    rows = {r["solver"]: r for r in load(f"{csvdir}/stiff_pc.csv")}
    order = ["PCNONE", "GAMG_reuse", "GAMG", "Helmholtz"]
    ms = [rows[s]["field_per_cycle_ms"] for s in order]
    cols = [_PALETTE[theme.mode][s] for s in order]
    xs = range(len(order))
    axP.bar(xs, ms, color=cols, edgecolor=theme.bar_edge)
    for x, s in zip(xs, order):
        axP.annotate(f"{rows[s]['mean_iters']:.0f} it", (x, rows[s]["field_per_cycle_ms"]),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8.5, color=theme.text_color)
    axP.axhline(rows["PCNONE"]["field_per_cycle_ms"], ls="--", lw=1, color=theme.refline_color)
    axP.set_xticks(list(xs)); axP.set_xticklabels([_NICE[s] for s in order], fontsize=8, rotation=12)
    axP.set(ylabel="field-solve time (ms / cycle)",
            title="Stiff end: GAMG cuts iters 3.5×, still 1.4× slower")
    axP.set_ylim(top=max(ms) * 1.18)

    for ax in (axB, axR, axP):
        ax.grid(True, alpha=0.25)
    fig.suptitle("Magnetization vs density — and preconditioners where it's finally stiff (ρ=100)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{outdir}/fig4_drivers.png"; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


def fig_stress(csvdir, outdir, theme):
    """fig5: stress test — GAMG's iteration speedup never clears the wall-time break-even."""
    import matplotlib.pyplot as plt
    import numpy as np
    need = [f"{csvdir}/dens_stress.csv", f"{csvdir}/grid_stress.csv"]
    if not all(os.path.exists(p) for p in need):
        print("fig5: missing stress CSVs — skipping")
        return

    def grab(rows, **kw):
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in kw.items()):
                return r
        return None

    dens, grid = load(need[0]), load(need[1])
    raw = [("ρ=100\n100²",   grab(dens, solver="PCNONE", rho="100"),  grab(dens, solver="GAMG", rho="100")),
           ("ρ=1000\n100²",  grab(dens, solver="PCNONE", rho="1000"), grab(dens, solver="GAMG", rho="1000")),
           ("grid 200²\nρ=100", grab(grid, solver="PCNONE", nxc=200), grab(grid, solver="GAMG", nxc=200))]
    labels, ratio, barrier = [], [], []
    for lab, pcn, g in raw:
        if not pcn or not g or g["mean_iters"] != g["mean_iters"]:
            continue
        labels.append(lab)
        ratio.append(pcn["mean_iters"] / g["mean_iters"])                      # iteration speedup achieved
        barrier.append((g["field_per_cycle_ms"] / g["mean_iters"]) /           # per-iter cost ratio = break-even
                       (pcn["field_per_cycle_ms"] / pcn["mean_iters"]))

    lav = _PALETTE[theme.mode]["GAMG_reuse"]
    red = _PALETTE[theme.mode]["GAMG"]
    grn = _PALETTE[theme.mode]["PCNONE"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(labels)); w = 0.52
    ymax = max(barrier) * 1.22

    # the preconditioner-wins zone (above the break-even bar) is never reached
    ax.axhspan(min(barrier), ymax, color=grn, alpha=0.09, zorder=0)
    ax.text((len(labels) - 1) / 2, ymax * 0.95, "preconditioner-wins zone — never reached",
            ha="center", va="top", fontsize=9.5, style="italic", color=theme.text_secondary)

    ax.bar(x, ratio, w, color=lav, edgecolor=theme.bar_edge, zorder=3,
           label="GAMG iteration speedup (achieved)")
    for xi, r in zip(x, ratio):
        ax.text(float(xi), r + 0.05, f"{r:.1f}×", ha="center", va="bottom", fontsize=10, color=theme.text_color)
    for xi, b in zip(x, barrier):
        ax.hlines(b, xi - w / 2 - 0.05, xi + w / 2 + 0.05, color=red, lw=2.6, zorder=5)
        ax.text(float(xi), b + 0.05, f"{b:.1f}×", ha="center", va="bottom", fontsize=9, color=red)
    ax.plot([], [], color=red, lw=2.6, label="wall-time break-even (must clear to win)")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("iterations(PCNONE) / iterations(GAMG)")
    ax.set_ylim(0, ymax)
    ax.set_title("GAMG's iteration speedup never clears the break-even bar",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    out = f"{outdir}/fig5_stress.png"; fig.savefig(out, bbox_inches="tight"); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_theme_arg(ap)
    ap.add_argument("--csvdir", default="tests/test_output_stiffness")
    ap.add_argument("--outdir", default="slides/figs")
    args = ap.parse_args()
    theme = apply_theme(args)
    register(theme)
    os.makedirs(args.outdir, exist_ok=True)
    fig_scaling(args.csvdir, args.outdir, theme)
    fig_frontier(args.csvdir, args.outdir, theme)
    fig_3d(args.csvdir, args.outdir, theme)
    fig_drivers(args.csvdir, args.outdir, theme)
    fig_stress(args.csvdir, args.outdir, theme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
