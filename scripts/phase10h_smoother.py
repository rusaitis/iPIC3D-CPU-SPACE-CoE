"""
Phase 10h: smoother-strength sweep at fixed high resolution.

Goal: Phase 10e showed the TSC+sm=4 instability growth rate decays with
Δx^0.703 at fixed num_smoothings=4. Phase 10f confirmed the cycle-1 seed is
algebraic, not solver residual. This sweep asks whether increasing
num_smoothings at fixed Δx has the same effect as refining — does "stronger
smoothing" kill the growth rate without needing finer grids?

Protocol:
  * Base input: inputfiles/phase10e_tsc_40.inp (40x40x4, TSC, 64 cycles,
    Δx=0.75, same physics as Phase 10e/10f).
  * num_smoothings ∈ {4, 8, 16, 32} — four runs at 40x40.
  * For each run, load dE(cycle) from ConservedQuantities.txt col VI and
    fit log|dE| vs cycle over a fixed window [13, 40] (Phase 10e's 40x40
    growth phase started around cycle 13).
  * Report growth rate (exp(slope)) and dE at select cycles.
  * Re-use existing /tmp/ipic3d_phase10e_tsc_40 for sm=4 if present.

After 40x40 finishes, optionally re-run 1–2 interesting sm values at 80x80x4
(~11 min each) to check the combined (Δx, sm) trend.

Usage:
  pixi run python scripts/phase10h_smoother.py                 # 40x40 sweep
  pixi run python scripts/phase10h_smoother.py --grid 80 --sm 4,16   # follow-ups

Outputs:
  phase10h_results.txt  — table of rates
  phase10h_results.json — machine-readable
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
INP_40 = REPO / "inputfiles" / "phase10e_tsc_40.inp"
INP_80 = REPO / "inputfiles" / "phase10e_tsc_80.inp"
INP_20 = REPO / "inputfiles" / "phase10c_tsc_spectral.inp"
WORK = Path("/tmp/ipic3d_phase10h")

# For the --reuse-sm4 fallback: which existing dir holds a 64-cycle TSC sm=4 run.
REUSE_DIRS = {
    20: Path("/tmp/ipic3d_phase10c_tsc_spectral"),
    40: Path("/tmp/ipic3d_phase10e_tsc_40"),
    80: Path("/tmp/ipic3d_phase10e_tsc_80"),
}

CYCLE_COL = 0
DE_COL = 11  # col VI = Energy(cycle) - Energy(initial)

DEFAULT_SM_LIST = [4, 8, 16, 32]
FIT_WINDOW_DEFAULT = (13, 40)  # cycles [c_lo, c_hi] inclusive
MPI_RANKS = 4


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
    """Linear fit to log|dE| vs cycle inside [window[0], window[1]] inclusive."""
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


def make_input(base_inp: Path, grid: int, sm: int, work: Path) -> tuple[Path, Path]:
    """Return (inp_path, save_dir) for a (grid, sm) run."""
    text = base_inp.read_text()
    save_dir = work / f"g{grid}_sm{sm}"
    save_str = str(save_dir)

    text = re.sub(r"^SaveDirName\s*=.*$",
                  f"SaveDirName                    = {save_str}", text, flags=re.M)
    text = re.sub(r"^RestartDirName\s*=.*$",
                  f"RestartDirName                 = {save_str}", text, flags=re.M)
    text = re.sub(r"^SimulationName\s*=.*$",
                  f"SimulationName                 = Phase10h_g{grid}_sm{sm}", text, flags=re.M)
    text = re.sub(r"^num_smoothings\s*=.*$",
                  f"num_smoothings                 = {sm}", text, flags=re.M)

    inp_path = work / f"g{grid}_sm{sm}.inp"
    inp_path.write_text(text)
    return inp_path, save_dir


def run_one(inp_path: Path, log_path: Path) -> int:
    cmd = ["mpirun", "-np", str(MPI_RANKS), str(BUILD_BIN), str(inp_path)]
    with log_path.open("w") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))


def process(save_dir: Path, sm: int, grid: int, window: tuple[int, int]) -> dict:
    cycles, dE = load_drift(save_dir)
    if cycles.size == 0:
        return {"grid": grid, "sm": sm, "error": "no cycles"}
    if cycles[0] == 0:
        cycles_nozero = cycles[1:]
        dE_nozero = dE[1:]
    else:
        cycles_nozero = cycles
        dE_nozero = dE
    fit = fit_exp_growth(cycles_nozero, dE_nozero, window)
    return {
        "grid": grid,
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
    print("Phase 10h smoother-strength sweep — growth rate vs num_smoothings", file=out)
    print("=" * 112, file=out)
    header = (f"{'grid':<8}{'sm':<6}{'dE(1)':<14}{'dE(10)':<14}{'dE(32)':<14}"
              f"{'dE(64)':<14}{'fit window':<14}{'rate (x/cyc)':<14}{'R^2':<8}")
    print(header, file=out)
    print("-" * 112, file=out)
    prev_grid = None
    for rec in results:
        if prev_grid is not None and rec["grid"] != prev_grid:
            print("-" * 112, file=out)
        prev_grid = rec["grid"]

        def fmt(v):
            return f"{v:<14.3e}" if isinstance(v, (int, float)) and v is not None else f"{'—':<14}"

        window_str = f"{rec['fit_window'][0]}..{rec['fit_window'][1]}"
        rate = rec.get("fit_rate", 1.0)
        r2 = rec.get("fit_r2", 0.0)
        print(f"{rec['grid']}x{rec['grid']:<3} {rec['sm']:<6}"
              f"{fmt(rec.get('dE_1'))}{fmt(rec.get('dE_10'))}"
              f"{fmt(rec.get('dE_32'))}{fmt(rec.get('dE_64'))}"
              f"{window_str:<14}{rate:<14.4f}{r2:<8.3f}", file=out)
    print("=" * 112, file=out)


def parse_sm_list(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=40,
                    help="grid size (nxc=nyc, 40 or 80). Default 40.")
    ap.add_argument("--sm", type=parse_sm_list, default=DEFAULT_SM_LIST,
                    help="comma-separated num_smoothings values. Default 4,8,16,32.")
    ap.add_argument("--window", type=parse_sm_list, default=list(FIT_WINDOW_DEFAULT),
                    help="fit window cycle range [lo,hi]. Default 13,40.")
    ap.add_argument("--reuse-sm4", action="store_true",
                    help="reuse /tmp/ipic3d_phase10e_tsc_40 (or _80) for sm=4 if present")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for phase10h_results{,suffix}.{txt,json}")
    args = ap.parse_args()

    grid = args.grid
    sm_list = args.sm
    window = (int(args.window[0]), int(args.window[1]))
    base_inp = {20: INP_20, 40: INP_40, 80: INP_80}.get(grid)
    if base_inp is None:
        print(f"unsupported grid: {grid}", file=sys.stderr)
        return 1
    if not base_inp.exists():
        print(f"missing base input: {base_inp}", file=sys.stderr)
        return 1
    if not BUILD_BIN.exists():
        print(f"missing binary: {BUILD_BIN}", file=sys.stderr)
        return 1

    WORK.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for sm in sm_list:
        reuse_dir = None
        if args.reuse_sm4 and sm == 4:
            candidate = REUSE_DIRS.get(grid)
            if candidate is not None and (candidate / "ConservedQuantities.txt").exists():
                reuse_dir = candidate
                print(f"=== grid {grid}, sm=4: reusing {candidate} ===")

        if reuse_dir is None:
            inp_path, save_dir = make_input(base_inp, grid, sm, WORK)
            log_path = WORK / f"g{grid}_sm{sm}.log"
            print(f"=== grid {grid}, sm={sm}: running ({inp_path.name}) ===", flush=True)
            t0 = time.time()
            rc = run_one(inp_path, log_path)
            wall = time.time() - t0
            print(f"    wall={wall:.1f}s  rc={rc}")
            if rc != 0:
                print(f"    FAILED (see {log_path})", file=sys.stderr)
                continue
        else:
            save_dir = reuse_dir

        rec = process(save_dir, sm, grid, window)
        results.append(rec)
        print(f"    dE(1)={rec['dE_1']:.3e}  dE(32)={rec['dE_32']:.3e}  "
              f"dE(64)={rec['dE_64']:.3e}  rate={rec['fit_rate']:.4f}/cyc "
              f"R²={rec['fit_r2']:.3f} (window {rec['fit_window'][0]}..{rec['fit_window'][1]})",
              flush=True)

    suffix = args.out_suffix
    out_txt = REPO / f"phase10h_results{suffix}.txt"
    out_json = REPO / f"phase10h_results{suffix}.json"
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
