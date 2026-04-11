"""
Phase 10i: alternative smoother kernel sweep.

Goal: Phase 10h showed stronger smoothing suppresses the TSC instability but
that the optimal num_smoothings shifts with resolution, because the current
(1,2,1)/4 kernel has a fixed 1-cell-per-pass footprint. Phase 10i asks whether
a wider kernel -- the 5-point (1,4,6,4,1)/16 binomial (half-width = 2 cells)
exposed as SmoothKernel=binomial5 -- removes the grid dependence.

Decision gate (from plan-tsc.md §10i):
  * binomial5 sm=4 matches binomial sm=8 at every grid  → success, adopt.
  * binomial5 has its own sweet spot that still shifts  → partial win, try wider.
  * binomial5 is no better than binomial                → kernel width isn't
                                                          the limiting factor,
                                                          escalate to Phase 10j.

Protocol (mirrors Phase 10h):
  * Base inputs are the same `phase10c_tsc_spectral.inp` (20x20), `phase10e_tsc_40.inp`,
    `phase10e_tsc_80.inp` (64 cycles each, TSC, Δx ∈ {1.5, 0.75, 0.375}).
  * Each run also overrides SmoothKernel + num_smoothings.
  * Sweep: SmoothKernel ∈ {binomial, binomial5} × sm ∈ {4, 8} × grid ∈ {20, 40, 80}.
  * For binomial sm ∈ {4, 8}, reuse Phase 10h's existing results if the
    save dirs are still around (see REUSE_BIN below) — only binomial5 runs are new.
  * Fit log|dE| vs cycle over [13, 40] (same window as Phase 10e/10h).
  * Report dE(1), dE(10), dE(32), dE(64), rate = exp(slope), R².

Usage:
  pixi run python scripts/phase10i_kernel.py                   # default full sweep
  pixi run python scripts/phase10i_kernel.py --grid 40 --kernels binomial5 --sm 4,8
  pixi run python scripts/phase10i_kernel.py --reuse-binomial  # skip re-running binomial

Outputs:
  phase10i_results.txt      — table
  phase10i_results.json     — machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO / "build" / "iPIC3D"
WORK = Path("/tmp/ipic3d_phase10i")

# Base inputs match the Phase 10h tree so we can reuse runs.
BASE_INP = {
    20: REPO / "inputfiles" / "phase10c_tsc_spectral.inp",
    40: REPO / "inputfiles" / "phase10e_tsc_40.inp",
    80: REPO / "inputfiles" / "phase10e_tsc_80.inp",
}

# Phase 10h binomial save directories, keyed by (grid, sm).
# If any of these exists we can skip re-running binomial.
REUSE_BIN = {
    (20, 4):  Path("/tmp/ipic3d_phase10c_tsc_spectral"),  # Phase 10c
    (20, 8):  Path("/tmp/ipic3d_phase10h/g20_sm8"),
    (20, 16): Path("/tmp/ipic3d_phase10h/g20_sm16"),
    (40, 4):  Path("/tmp/ipic3d_phase10e_tsc_40"),        # Phase 10e
    (40, 8):  Path("/tmp/ipic3d_phase10h/g40_sm8"),
    (40, 16): Path("/tmp/ipic3d_phase10h/g40_sm16"),
    (80, 4):  Path("/tmp/ipic3d_phase10e_tsc_80"),        # Phase 10e
    (80, 8):  Path("/tmp/ipic3d_phase10h/g80_sm8"),
    (80, 16): Path("/tmp/ipic3d_phase10h/g80_sm16"),
}

CYCLE_COL = 0
DE_COL = 11   # column VI = Energy(cycle) - Energy(initial)

MPI_RANKS = 4
DEFAULT_KERNELS = ["binomial", "binomial5"]
DEFAULT_SM = [4, 8]
DEFAULT_GRIDS = [20, 40, 80]
FIT_WINDOW_DEFAULT = (13, 40)


def load_drift(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "ConservedQuantities.txt"
    cycles: list[int] = []
    dE: list[float] = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < DE_COL + 1:
                continue
            try:
                c = int(parts[CYCLE_COL])
            except ValueError:
                continue
            try:
                v = float(parts[DE_COL])
            except ValueError:
                continue
            cycles.append(c)
            dE.append(v)
    return np.asarray(cycles), np.asarray(dE)


def fit_exp_growth(cycles: np.ndarray, dE: np.ndarray,
                   window: tuple[int, int]) -> dict:
    lo, hi = window
    mask = (cycles >= lo) & (cycles <= hi) & (np.abs(dE) > 0)
    if mask.sum() < 3:
        return {"slope": 0.0, "rate": 1.0, "r2": 0.0, "n": int(mask.sum()),
                "window": list(window)}
    y = np.log(np.abs(dE[mask]))
    x = cycles[mask].astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "rate": float(np.exp(slope)), "r2": r2,
            "n": int(mask.sum()), "window": list(window)}


def pick_value_at(cycles: np.ndarray, dE: np.ndarray, target: int) -> float | None:
    idx = np.where(cycles == target)[0]
    if idx.size == 0:
        return None
    return float(dE[idx[0]])


def make_input(base_inp: Path, grid: int, kernel: str, sm: int, work: Path) -> tuple[Path, Path]:
    text = base_inp.read_text()
    tag = f"g{grid}_{kernel}_sm{sm}"
    save_dir = work / tag
    save_str = str(save_dir)

    text = re.sub(r"^SaveDirName\s*=.*$",
                  f"SaveDirName                    = {save_str}", text, flags=re.M)
    text = re.sub(r"^RestartDirName\s*=.*$",
                  f"RestartDirName                 = {save_str}", text, flags=re.M)
    text = re.sub(r"^SimulationName\s*=.*$",
                  f"SimulationName                 = Phase10i_{tag}", text, flags=re.M)
    text = re.sub(r"^num_smoothings\s*=.*$",
                  f"num_smoothings                 = {sm}", text, flags=re.M)
    if re.search(r"^SmoothKernel\s*=", text, flags=re.M):
        text = re.sub(r"^SmoothKernel\s*=.*$",
                      f"SmoothKernel                   = {kernel}", text, flags=re.M)
    else:
        text += f"\nSmoothKernel                   = {kernel}\n"

    inp_path = work / f"{tag}.inp"
    inp_path.write_text(text)
    return inp_path, save_dir


def run_one(inp_path: Path, log_path: Path) -> int:
    cmd = ["mpirun", "-np", str(MPI_RANKS), str(BUILD_BIN), str(inp_path)]
    with log_path.open("w") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))


def process(save_dir: Path, grid: int, kernel: str, sm: int,
            window: tuple[int, int]) -> dict:
    cycles, dE = load_drift(save_dir)
    if cycles.size == 0:
        return {"grid": grid, "kernel": kernel, "sm": sm, "error": "no cycles"}
    if cycles[0] == 0:
        cycles_nozero = cycles[1:]
        dE_nozero = dE[1:]
    else:
        cycles_nozero = cycles
        dE_nozero = dE
    fit = fit_exp_growth(cycles_nozero, dE_nozero, window)
    return {
        "grid": grid,
        "kernel": kernel,
        "sm": sm,
        "n_cycles": int(cycles.size),
        "dE_1": pick_value_at(cycles, dE, 1),
        "dE_10": pick_value_at(cycles, dE, 10),
        "dE_32": pick_value_at(cycles, dE, 32),
        "dE_40": pick_value_at(cycles, dE, 40),
        "dE_64": pick_value_at(cycles, dE, 64),
        "fit_slope": fit["slope"],
        "fit_rate": fit["rate"],
        "fit_r2": fit["r2"],
        "fit_window": fit["window"],
        "fit_n": fit["n"],
    }


def print_table(results: list[dict], out=sys.stdout) -> None:
    print("", file=out)
    print("Phase 10i alternative-smoother-kernel sweep", file=out)
    print("=" * 124, file=out)
    header = (f"{'grid':<8}{'kernel':<12}{'sm':<4}"
              f"{'dE(1)':<14}{'dE(10)':<14}{'dE(32)':<14}{'dE(64)':<14}"
              f"{'fit window':<14}{'rate (x/cyc)':<14}{'R^2':<8}")
    print(header, file=out)
    print("-" * 124, file=out)
    prev_grid = None
    for rec in results:
        if prev_grid is not None and rec["grid"] != prev_grid:
            print("-" * 124, file=out)
        prev_grid = rec["grid"]

        def fmt(v):
            return f"{v:<14.3e}" if isinstance(v, (int, float)) and v is not None else f"{'—':<14}"

        window_str = f"{rec['fit_window'][0]}..{rec['fit_window'][1]}"
        rate = rec.get("fit_rate", 1.0)
        r2 = rec.get("fit_r2", 0.0)
        print(f"{rec['grid']}x{rec['grid']:<3} {rec['kernel']:<12}{rec['sm']:<4}"
              f"{fmt(rec.get('dE_1'))}{fmt(rec.get('dE_10'))}"
              f"{fmt(rec.get('dE_32'))}{fmt(rec.get('dE_64'))}"
              f"{window_str:<14}{rate:<14.4f}{r2:<8.3f}", file=out)
    print("=" * 124, file=out)


def parse_csv_list_int(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def parse_csv_list_str(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=parse_csv_list_int, default=DEFAULT_GRIDS,
                    help="grid sizes (nxc=nyc). Default 20,40,80.")
    ap.add_argument("--kernels", type=parse_csv_list_str, default=DEFAULT_KERNELS,
                    help="comma-separated SmoothKernel values. Default binomial,binomial5.")
    ap.add_argument("--sm", type=parse_csv_list_int, default=DEFAULT_SM,
                    help="comma-separated num_smoothings values. Default 4,8.")
    ap.add_argument("--window", type=parse_csv_list_int, default=list(FIT_WINDOW_DEFAULT),
                    help="fit window cycle range [lo,hi]. Default 13,40.")
    ap.add_argument("--reuse-binomial", action="store_true",
                    help="reuse Phase 10c/10e/10h save dirs for SmoothKernel=binomial runs")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for phase10i_results{,suffix}.{txt,json}")
    args = ap.parse_args()

    window = (int(args.window[0]), int(args.window[1]))

    if not BUILD_BIN.exists():
        print(f"missing binary: {BUILD_BIN}", file=sys.stderr)
        return 1

    WORK.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for grid in args.grids:
        base_inp = BASE_INP.get(grid)
        if base_inp is None or not base_inp.exists():
            print(f"skipping grid {grid}: missing base input {base_inp}", file=sys.stderr)
            continue
        for kernel in args.kernels:
            for sm in args.sm:
                reuse_dir = None
                if args.reuse_binomial and kernel == "binomial":
                    candidate = REUSE_BIN.get((grid, sm))
                    if candidate is not None and (candidate / "ConservedQuantities.txt").exists():
                        reuse_dir = candidate
                        print(f"=== grid {grid}, kernel={kernel}, sm={sm}: reusing {candidate}")

                if reuse_dir is None:
                    inp_path, save_dir = make_input(base_inp, grid, kernel, sm, WORK)
                    log_path = WORK / f"g{grid}_{kernel}_sm{sm}.log"
                    print(f"=== grid {grid}, kernel={kernel}, sm={sm}: running ({inp_path.name}) ===",
                          flush=True)
                    t0 = time.time()
                    rc = run_one(inp_path, log_path)
                    wall = time.time() - t0
                    print(f"    wall={wall:.1f}s  rc={rc}")
                    if rc != 0:
                        print(f"    FAILED (see {log_path})", file=sys.stderr)
                        continue
                else:
                    save_dir = reuse_dir

                rec = process(save_dir, grid, kernel, sm, window)
                results.append(rec)
                print(f"    dE(1)={rec.get('dE_1')!s:<20}  dE(32)={rec.get('dE_32')!s:<20}  "
                      f"dE(64)={rec.get('dE_64')!s:<20}  rate={rec.get('fit_rate')!s:<10} "
                      f"R²={rec.get('fit_r2')!s:<8}", flush=True)

    suffix = args.out_suffix
    out_txt = REPO / f"phase10i_results{suffix}.txt"
    out_json = REPO / f"phase10i_results{suffix}.json"
    with out_txt.open("w") as fh:
        print_table(results, out=fh)
        print("\nper-run detail:", file=fh)
        for rec in results:
            print(f"  {rec}", file=fh)
    out_json.write_text(json.dumps(results, indent=2, default=float))

    print_table(results)
    print(f"\nWrote {out_txt} and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
