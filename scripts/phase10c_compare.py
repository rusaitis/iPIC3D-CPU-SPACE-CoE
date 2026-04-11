"""
Phase 10c comparison plot: TSC sm=4 vs CIC sm=4 over 64 cycles on 20x20x4.

Overlays cumulative |dE| and per-cycle ΔdE for the two shape functions so the
TSC-specific instability onset is visible at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase10c_spectral import load_drift, fit_exp_growth


def main() -> int:
    tsc_dir = Path("/tmp/ipic3d_phase10c_tsc_spectral")
    cic_dir = Path("/tmp/ipic3d_phase10c_cic_spectral")

    cyc_t, dE_t = load_drift(tsc_dir)
    cyc_c, dE_c = load_drift(cic_dir)
    # drop cycle 0
    cyc_t, dE_t = cyc_t[1:], dE_t[1:]
    cyc_c, dE_c = cyc_c[1:], dE_c[1:]

    mask_t_g = cyc_t >= 13
    mask_c_g = cyc_c >= 13
    slope_t_g, _ = fit_exp_growth(cyc_t[mask_t_g], dE_t[mask_t_g])
    slope_c_g, _ = fit_exp_growth(cyc_c[mask_c_g], dE_c[mask_c_g])

    ratio = np.abs(dE_t[-1]) / np.abs(dE_c[-1])
    print(f"TSC dE(64) = {dE_t[-1]:.3e}")
    print(f"CIC dE(64) = {dE_c[-1]:.3e}")
    print(f"Ratio TSC/CIC at cyc 64 = {ratio:.1f}×")
    print(f"Growth rates (cycles 13-64): TSC exp({slope_t_g:.3f})={np.exp(slope_t_g):.3f}x/cyc,"
          f"  CIC exp({slope_c_g:.3f})={np.exp(slope_c_g):.3f}x/cyc")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.semilogy(cyc_t, np.abs(dE_t), "o-", ms=3, label=f"TSC sm=4  ({np.exp(slope_t_g):.2f}×/cyc)")
    ax.semilogy(cyc_c, np.abs(dE_c), "s-", ms=3, label=f"CIC sm=4  ({np.exp(slope_c_g):.2f}×/cyc)", color="C2")
    ax.axvline(12.5, color="k", ls="--", lw=0.6)
    ax.set_xlabel("cycle")
    ax.set_ylabel("|dE|")
    ax.set_title("Phase 10c — TSC-specific exponential growth")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(cyc_t[1:], np.diff(dE_t), "o-", ms=3, label="TSC")
    ax.plot(cyc_c[1:], np.diff(dE_c), "s-", ms=3, label="CIC", color="C2")
    ax.axvline(12.5, color="k", ls="--", lw=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("Δ(dE)/cyc")
    ax.set_title(f"Per-cycle increment  (ratio at cyc 64: {ratio:.0f}×)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = Path("phase10c_compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
