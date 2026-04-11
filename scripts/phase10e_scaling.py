"""
Phase 10e: grid-size growth-rate scan for the TSC sm=4 instability.

Loads 20x20x4, 40x40x4, 80x80x4 runs of 64 cycles, fits an exponential
growth rate to log|dE| over the growth phase, and plots
  (a) |dE| vs cycle (all three, semilog)
  (b) growth rate vs 1/Δx (log-log)
to distinguish
  rate ∝ 1/Δx       -> dispersion/aliasing
  rate ∝ 1/Δx²      -> diffusion-like
  rate independent  -> particle-noise amplifier
  rate drops w/ 1/Δx -> 20x20 is below TSC resolution threshold.

Usage: pixi run python scripts/phase10e_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase10c_spectral import load_drift, fit_exp_growth


# (label, run_dir, nxc, Lx, color, marker, fit_cyc_start, fit_cyc_end)
# Fit windows hand-picked to cover the clean exponential phase for each run
# (after the pre-instability transient, before nonlinear saturation). Chosen
# from inspection of per-cycle dE trajectories.
RUNS = [
    ("20x20x4 TSC sm=4", Path("/tmp/ipic3d_phase10c_tsc_spectral"), 20, 30.0, "C0", "o", 14, 22),
    ("40x40x4 TSC sm=4", Path("/tmp/ipic3d_phase10e_tsc_40"),       40, 30.0, "C3", "s", 13, 23),
    ("80x80x4 TSC sm=4", Path("/tmp/ipic3d_phase10e_tsc_80"),       80, 30.0, "C2", "^", 20, 32),
]

def find_exp_window(cycles: np.ndarray, dE: np.ndarray,
                    win_len: int = 10,
                    amp_lo: float = 1e-4,
                    amp_hi: float = 5e-1) -> tuple[int, int]:
    """Walk a sliding window of length `win_len` across the log|dE| trajectory,
    pick the window where

      - |dE| is bracketed by [amp_lo, amp_hi] (exponential phase: above the
        pre-instability transient, below nonlinear saturation),
      - log|dE| is most linear (highest R² in a linear fit).

    Returns (start_cycle, end_cycle), both inclusive.
    """
    y_log = np.log(np.abs(dE) + 1e-300)
    best = None  # (r2, start_idx, end_idx)
    n = cycles.size
    for i in range(n - win_len + 1):
        j = i + win_len - 1
        amps = np.abs(dE[i:j + 1])
        if amps.min() < amp_lo or amps.max() > amp_hi:
            continue
        x = cycles[i:j + 1].astype(float)
        y = y_log[i:j + 1]
        p = np.polyfit(x, y, 1)
        y_pred = np.polyval(p, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if best is None or r2 > best[0]:
            best = (float(r2), i, j)
    if best is None:
        # No window qualifies — fall back to the full growing tail.
        growth = np.where(np.abs(dE) > amp_lo)[0]
        if growth.size >= 4:
            return int(cycles[growth[0]]), int(cycles[growth[-1]])
        return int(cycles[0]), int(cycles[-1])
    return int(cycles[best[1]]), int(cycles[best[2]])


def main() -> int:
    print(f"{'label':<22} {'nxc':>4} {'Δx':>8} {'dE(1)':>12} {'dE(13)':>12} "
          f"{'dE(64)':>12}  {'fit window':>14}  {'rate (×/cyc)':>14}")
    print("-" * 105)

    results = []
    for label, run_dir, nxc, Lx, color, marker, s, e in RUNS:
        cyc, dE = load_drift(run_dir)
        if cyc[0] == 0:
            cyc, dE = cyc[1:], dE[1:]

        dx = Lx / nxc
        mask = (cyc >= s) & (cyc <= e)
        slope, r2 = fit_exp_growth(cyc[mask], dE[mask])
        rate = np.exp(slope)

        def pick(target):
            idx = np.where(cyc == target)[0]
            return dE[idx[0]] if idx.size else np.nan

        dE1  = pick(1)
        dE13 = pick(13)
        dE64 = pick(64)

        print(f"{label:<22} {nxc:>4} {dx:>8.3f} "
              f"{dE1:>12.3e} {dE13:>12.3e} {dE64:>12.3e}  "
              f"cyc {s:2d}-{e:2d}  {rate:>8.3f}× (R²={r2:.2f})")

        results.append(dict(
            label=label, nxc=nxc, dx=dx, cyc=cyc, dE=dE,
            slope=slope, rate=rate, r2=r2,
            fit_start=s, fit_end=e,
            dE1=dE1, dE13=dE13, dE64=dE64,
            color=color, marker=marker,
        ))

    # Scaling fit: log(slope) vs log(1/dx) to get the power-law exponent
    dxs    = np.array([r["dx"]    for r in results])
    slopes = np.array([r["slope"] for r in results])
    print()
    print(f"1/Δx values:  {1/dxs}")
    print(f"Growth slopes (per cycle, base-e): {slopes}")
    print(f"Rates (× per cyc): {np.exp(slopes)}")

    # Fit log(slope) = a + b*log(1/dx), i.e. slope ∝ (1/dx)^b
    # Works only if all slopes are positive.
    if np.all(slopes > 0):
        lx = np.log(1.0 / dxs)
        ly = np.log(slopes)
        p = np.polyfit(lx, ly, 1)
        power_exponent = float(p[0])
        print(f"\nPower-law fit  slope ∝ (1/Δx)^{power_exponent:+.3f}")
        if power_exponent > +0.5:
            classification = "DISPERSION (faster at finer Δx)"
        elif power_exponent < -0.5:
            classification = "RESOLUTION-THRESHOLD (slower at finer Δx)"
        else:
            classification = "NEAR-INDEPENDENT (particle-noise or saturated)"
        print(f"Classification: {classification}")
    else:
        power_exponent = np.nan
        classification = "CANNOT FIT (negative slope in some run)"
        print(classification)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for r in results:
        ax.semilogy(r["cyc"], np.abs(r["dE"]), marker=r["marker"], ms=3,
                    color=r["color"],
                    label=f"{r['label']} ({r['rate']:.3f}×/cyc)")
        # Shade each per-run fit window in its own color
        ax.axvspan(r["fit_start"] - 0.4, r["fit_end"] + 0.4,
                   alpha=0.08, color=r["color"])
    ax.set_xlabel("cycle")
    ax.set_ylabel("|dE|")
    ax.set_title("Phase 10e — TSC sm=4 instability at 3 resolutions (Lx fixed)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.loglog(1.0 / dxs, slopes, "o-", ms=7, color="C0", label="TSC sm=4")
    # Add reference slopes for visual comparison
    x_ref = np.array([min(1/dxs), max(1/dxs)])
    if not np.isnan(power_exponent):
        # normalise to pass through the middle point
        mid_i = len(dxs) // 2
        y_at_mid = slopes[mid_i]
        x_at_mid = 1.0 / dxs[mid_i]
        for p_ref, style, label in [(+1.0, ":", "∝ 1/Δx"),
                                    (+2.0, "--", "∝ 1/Δx²"),
                                    ( 0.0, "-.", "const")]:
            y_ref = y_at_mid * (x_ref / x_at_mid) ** p_ref
            ax.loglog(x_ref, y_ref, style, color="gray", alpha=0.6, label=label)
    ax.set_xlabel("1/Δx (cell density)")
    ax.set_ylabel("exp-growth slope / cycle")
    title = f"Growth rate vs resolution — fitted exponent {power_exponent:+.2f}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    out = Path("phase10e_scaling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
