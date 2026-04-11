"""
Phase 10c spectral analysis of TSC per-cycle energy drift.

Reads ConservedQuantities.txt, extracts the cumulative energy drift (column VI
= Energy(cycle) - Energy(initial)) as a time series, then FFTs both the raw
drift and its first difference (per-cycle increment) to look for a dominant
oscillation frequency. The aim is to distinguish:

  (a) narrow-band near Nyquist -> the 27-point binomial smoother is under-
      damping a high-k mode -> stabilisation design issue.
  (b) broadband drift          -> algebraic leak, no dominant eigenmode
      -> proceed to a GMRES-iteration bisection to separate residual from
      leak at cycle 1.

Usage:  pixi run python scripts/phase10c_spectral.py [run_dir] [-o out.png]
Defaults to /tmp/ipic3d_phase10c_tsc_spectral .
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CYCLE_COL = 0   # column I
DE_COL    = 11  # column VI (Energy - Energy_init)


def load_drift(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "ConservedQuantities.txt"
    cycles: list[int] = []
    dE: list[float] = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                c = int(parts[0])
            except ValueError:
                continue
            if len(parts) <= DE_COL:
                continue
            try:
                v = float(parts[DE_COL])
            except ValueError:
                continue
            cycles.append(c)
            dE.append(v)
    return np.asarray(cycles), np.asarray(dE)


def spectral_summary(signal: np.ndarray, label: str) -> dict:
    """FFT a real signal and report dominant-frequency diagnostics.

    Frequencies are in units of "per cycle". Grid/particle Nyquist (= 0.5)
    is the smoother target; a peak there means the smoother is failing.
    """
    n = signal.size
    x = signal - signal.mean()  # kill DC
    window = np.hanning(n)      # reduce leakage from finite record
    spec = np.fft.rfft(x * window)
    freq = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(spec) ** 2

    # Skip DC bin for peak-finding
    if power.size > 1:
        peak_idx = int(np.argmax(power[1:]) + 1)
    else:
        peak_idx = 0
    peak_freq = float(freq[peak_idx])
    peak_power = float(power[peak_idx])

    total = float(power[1:].sum()) if power.size > 1 else 0.0

    # Concentration metric: what fraction of total variance lives in top-3 bins?
    top3 = np.argsort(power[1:])[-3:] + 1
    top3_power = float(power[top3].sum())
    concentration = top3_power / total if total > 0 else 0.0

    # High-k share: power in the top quarter of the spectrum [0.375, 0.5]
    hi_mask = freq >= 0.375
    hi_power = float(power[hi_mask].sum())
    hi_share = hi_power / total if total > 0 else 0.0

    nyquist_ratio = peak_freq / 0.5  # 1.0 means peak at Nyquist
    classification = (
        "NEAR-NYQUIST"   if nyquist_ratio > 0.75 and concentration > 0.5 else
        "HIGH-K"         if hi_share > 0.5 else
        "BROADBAND"      if concentration < 0.4 else
        "MID-BAND"
    )

    return {
        "label": label,
        "n": n,
        "freq": freq,
        "power": power,
        "peak_freq": peak_freq,
        "peak_cycles_per_period": (1.0 / peak_freq) if peak_freq > 0 else np.inf,
        "peak_power": peak_power,
        "concentration_top3": concentration,
        "high_k_share": hi_share,
        "nyquist_ratio": nyquist_ratio,
        "classification": classification,
    }


def print_summary(s: dict) -> None:
    print(f"\n=== {s['label']} (N={s['n']}) ===")
    print(f"  peak freq         : {s['peak_freq']:.4f} cycles^-1"
          f"  ({s['peak_cycles_per_period']:.2f} cycles/period)")
    print(f"  peak / Nyquist    : {s['nyquist_ratio']:.3f}"
          f"  (1.0 = grid Nyquist, 0.5 cycles^-1)")
    print(f"  top-3 concentration: {s['concentration_top3']:.3f}"
          f"  (>0.5 => narrow-band)")
    print(f"  high-k (f>=0.375) share: {s['high_k_share']:.3f}"
          f"  (>0.5 => smoother under-damping)")
    print(f"  classification    : {s['classification']}")


def fit_exp_growth(cycles: np.ndarray, dE: np.ndarray) -> tuple[float, float]:
    """Linear fit to log|dE| vs cycle. Returns (growth_rate_per_cycle, R^2)."""
    mask = np.abs(dE) > 0
    if mask.sum() < 3:
        return 0.0, 0.0
    y = np.log(np.abs(dE[mask]))
    x = cycles[mask].astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default="/tmp/ipic3d_phase10c_tsc_spectral")
    ap.add_argument("-o", "--output", default="phase10c_spectral.png")
    ap.add_argument("--stable-cycles", type=int, default=12,
                    help="last cycle treated as pre-instability (default 12)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cycles, dE = load_drift(run_dir)
    if cycles.size < 16:
        print(f"ERROR: only {cycles.size} cycles available, need 16+",
              file=sys.stderr)
        return 2

    # Drop cycle 0 (dE == 0 by definition, adds only a DC offset)
    if cycles[0] == 0:
        cycles, dE = cycles[1:], dE[1:]

    dDe = np.diff(dE)

    print(f"Loaded {cycles.size} cycles of dE from {run_dir}")
    print(f"  dE[0]  = {dE[0]:.3e}")
    print(f"  dE[-1] = {dE[-1]:.3e}")
    print(f"  max |dE|       = {np.abs(dE).max():.3e}")
    print(f"  max |ΔdE/cyc| = {np.abs(dDe).max():.3e}")
    print(f"  mean ΔdE/cyc  = {dDe.mean():+.3e}  (secular slope)")
    print(f"  std  ΔdE/cyc  = {dDe.std():.3e}")

    # Exponential growth fit over the full record and over each window.
    slope_all, r2_all = fit_exp_growth(cycles, dE)
    print(f"\n  log|dE| vs cycle (full record): slope = {slope_all:+.3f} / cyc"
          f"  R^2 = {r2_all:.3f}")
    print(f"    -> amplification per cycle = {np.exp(slope_all):.3f}x"
          f"  (>1 = unstable)")

    # Split windows: pre-instability vs growth phase
    stable_cut = args.stable_cycles
    mask_stable = cycles <= stable_cut
    mask_growth = cycles >  stable_cut
    if mask_stable.sum() >= 4:
        s_s, r2_s = fit_exp_growth(cycles[mask_stable], dE[mask_stable])
        print(f"  pre-instability  (cycles 1..{stable_cut}): "
              f"slope = {s_s:+.3f}/cyc  R^2 = {r2_s:.3f}  "
              f"-> {np.exp(s_s):.3f}x/cyc")
    if mask_growth.sum() >= 4:
        s_g, r2_g = fit_exp_growth(cycles[mask_growth], dE[mask_growth])
        print(f"  growth phase     (cycles {stable_cut+1}..{cycles.max()}): "
              f"slope = {s_g:+.3f}/cyc  R^2 = {r2_g:.3f}  "
              f"-> {np.exp(s_g):.3f}x/cyc")

    # Full-record FFT — dominated by growth envelope, but still worth showing
    spec_de_full  = spectral_summary(dE,  "cumulative dE (full)")
    spec_dde_full = spectral_summary(dDe, "per-cycle ΔdE (full)")
    print_summary(spec_de_full)
    print_summary(spec_dde_full)

    # Pre-instability only — this is the window where the "leak" is bounded
    # and any oscillatory component shows through cleanly.
    if mask_stable.sum() >= 8:
        pre_dE     = dE[mask_stable]
        pre_dDe    = np.diff(pre_dE)
        spec_pre   = spectral_summary(pre_dDe, "per-cycle ΔdE (pre-instability)")
        print_summary(spec_pre)
    else:
        spec_pre = None

    plot_spectra_extended(cycles, dE, dDe, spec_de_full, spec_dde_full,
                          spec_pre, stable_cut, slope_all, Path(args.output))

    print("\n--- Interpretation key ---")
    print("  NEAR-NYQUIST / HIGH-K -> smoother under-damps a high-k mode.")
    print("    Fix: stronger/wider filter, different kernel, or new stabiliser.")
    print("  BROADBAND             -> leak is algebraic, no dominant eigenmode.")
    print("    Next step: GMRES NiterGMRES bisection {1,2,5,10,30} at cycle 1")
    print("    to separate solver residual from algebraic leak.")
    print("  EXPONENTIAL GROWTH    -> not a bounded leak; numerical instability.")
    print("    The 10-cycle Phase 9 diagnostics captured a pre-instability")
    print("    transient. Growth rate and onset time are the new diagnostics.")
    return 0


def plot_spectra_extended(cycles, dE, dDe, spec_de, spec_dde, spec_pre,
                          stable_cut, slope_all, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = 3 if spec_pre is not None else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.5 * nrows))

    # Row 0 left: raw dE vs cycle, log y
    ax = axes[0, 0]
    ax.semilogy(cycles, np.abs(dE), "o-", ms=3)
    ax.axvline(stable_cut + 0.5, color="k", ls="--", lw=0.8,
               label=f"instability onset ~cyc {stable_cut+1}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("|dE|")
    ax.set_title(f"|dE| vs cycle — exp rate {np.exp(slope_all):.2f}/cyc")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Row 0 right: per-cycle delta, linear
    ax = axes[0, 1]
    ax.plot(cycles[1:], dDe, "o-", ms=3, color="C1")
    ax.axvline(stable_cut + 0.5, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("Δ(dE)/cyc")
    ax.set_title("Per-cycle energy increment")
    ax.grid(alpha=0.3)

    def _plot_spec(ax, spec, title):
        ax.semilogy(spec["freq"][1:], spec["power"][1:] + 1e-60, "o-", ms=3)
        ax.axvline(0.5, color="r", lw=0.5, ls="--", label="Nyquist")
        ax.axvline(spec["peak_freq"], color="g", lw=0.8, ls=":",
                   label=f"peak {spec['peak_freq']:.3f}")
        ax.set_xlabel("frequency (cycles^-1)")
        ax.set_ylabel("power")
        ax.set_title(f"{title} — {spec['classification']}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    _plot_spec(axes[1, 0], spec_de,  "FFT |cumulative dE|^2 (full)")
    _plot_spec(axes[1, 1], spec_dde, "FFT |ΔdE|^2 (full)")

    if spec_pre is not None:
        _plot_spec(axes[2, 0], spec_pre,
                   f"FFT |ΔdE|^2 (cycles 1..{stable_cut}, pre-instability)")
        axes[2, 1].axis("off")
        axes[2, 1].text(0.02, 0.95,
                        "Pre-instability FFT:\n"
                        "  short record, coarse resolution\n"
                        f"  peak: {spec_pre['peak_freq']:.3f}/cyc\n"
                        f"  cyc/period: {spec_pre['peak_cycles_per_period']:.2f}\n"
                        f"  top-3 conc: {spec_pre['concentration_top3']:.2f}\n"
                        f"  high-k:    {spec_pre['high_k_share']:.2f}\n"
                        f"  verdict:   {spec_pre['classification']}",
                        va="top", family="monospace", fontsize=10,
                        transform=axes[2, 1].transAxes)

    fig.suptitle(f"Phase 10c spectral analysis — {out_path.stem}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
