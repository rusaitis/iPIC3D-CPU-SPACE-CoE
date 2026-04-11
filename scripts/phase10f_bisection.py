"""
Phase 10f: GMRES-iteration bisection at cycle 1.

Goal: distinguish whether the TSC sm=4 cycle-1 energy leak is dominated by
solver residual or by an algebraic (scheme-level) floor.

Protocol:
  * Base input: phase8c_tsc_kahan.inp (20x20x4, TSC sm=4, Quadratic, GMREStol=1e-15)
  * ncycles overridden to 1 (we only care about dE(1))
  * NiterGMRES in {5, 10, 20, 40, 80} — forces m=N, max_iter=1 in the built-in GMRES
    so each run stops after exactly N Krylov iterations with no restart.
  * 5 runs per NiterGMRES value; report median and full range of dE(1).
  * Optional: same sweep at 40x40x4 and 80x80x4 (Finding 23 cycle-1 seed is
    Δx^-1.4, so the signal should hold across resolutions).

dE(1) here = column VI (Energy(cycle) - Energy(initial)) at cycle 1, matching
the convention used in Phases 8/9/10 of plan-tsc.md.

GMRES residual reduction at the final iteration is also scraped from stdout
("GMRES converged at restart ... with error: X" or "GMRES not converged !! ...
Final error: X") so we can report how far the solver got for each Niter.

Usage:
  pixi run python scripts/phase10f_bisection.py

Outputs:
  phase10f_results.txt  — plain-text table of medians
  phase10f_results.json — machine-readable (grid, niter, run_idx, dE1, rel_res)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO / "build" / "iPIC3D"
INP_BASE = REPO / "inputfiles" / "phase8c_tsc_kahan.inp"
WORK = Path("/tmp/ipic3d_phase10f")

CYCLE_COL = 0
DE_COL = 11  # column VI = Energy(cycle) - Energy(initial)

NITER_LIST = [5, 10, 20, 40, 80]
N_RUNS_PER = 5
MPI_RANKS = 4


def base_inp_text() -> str:
    return INP_BASE.read_text()


def make_input(niter: int, run_idx: int, grid: int, save_root: Path) -> Path:
    """Produce a per-config input file:
       - ncycles=1
       - unique SaveDirName per run (MPI determinism caveats in Phase 10b.0)
       - NiterGMRES = niter (or omitted if niter < 0)
       - optionally override nxc/nyc for the grid sweep
    """
    text = base_inp_text()
    save_dir = save_root / f"g{grid}_n{niter}_r{run_idx}"
    save_dir_str = str(save_dir)

    # Rewrite paths
    text = re.sub(r"^SaveDirName\s*=.*$",    f"SaveDirName                    = {save_dir_str}", text, flags=re.M)
    text = re.sub(r"^RestartDirName\s*=.*$", f"RestartDirName                 = {save_dir_str}", text, flags=re.M)
    text = re.sub(r"^ncycles\s*=.*$",        "ncycles                        = 1", text, flags=re.M)
    text = re.sub(r"^SimulationName\s*=.*$", f"SimulationName                 = Phase10f_g{grid}_n{niter}_r{run_idx}", text, flags=re.M)

    if grid != 20:
        # Scale nxc, nyc while keeping Lx, Ly fixed -> Δx halves with 2x grid
        text = re.sub(r"^nxc\s*=.*$", f"nxc                            = {grid}", text, flags=re.M)
        text = re.sub(r"^nyc\s*=.*$", f"nyc                            = {grid}", text, flags=re.M)

    # Append NiterGMRES override (base inp has no such line)
    text += f"\nNiterGMRES                     = {niter}\n"

    inp_path = save_root / f"g{grid}_n{niter}_r{run_idx}.inp"
    inp_path.write_text(text)
    return inp_path


def parse_de1(save_dir: Path) -> float | None:
    cq = save_dir / "ConservedQuantities.txt"
    if not cq.exists():
        return None
    with cq.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < DE_COL + 1:
                continue
            try:
                cyc = int(parts[CYCLE_COL])
            except ValueError:
                continue
            if cyc == 1:
                return float(parts[DE_COL])
    return None


_RE_CONV = re.compile(r"GMRES converged at restart (\d+); iteration (\d+) with error:\s*([0-9eE+.\-]+)")
_RE_UNCONV = re.compile(r"GMRES not converged !!.*?Final error:\s*([0-9eE+.\-]+)")


def parse_solver_info(log_path: Path) -> dict:
    """Extract the FIRST GMRES message (corresponds to the one cycle-1 field solve).

    Returns dict with:
      converged: bool
      rel_residual: float (= final residual / initial residual)
      iterations: int if converged else None
    """
    text = log_path.read_text()
    m = _RE_CONV.search(text)
    if m:
        return {
            "converged": True,
            "rel_residual": float(m.group(3)),
            "iterations": int(m.group(2)) + 1,  # iteration index is 0-based inside GMRES.cpp
        }
    m = _RE_UNCONV.search(text)
    if m:
        return {
            "converged": False,
            "rel_residual": float(m.group(1)),
            "iterations": None,
        }
    return {"converged": None, "rel_residual": None, "iterations": None}


def run_one(inp_path: Path, log_path: Path) -> int:
    cmd = ["mpirun", "-np", str(MPI_RANKS), str(BUILD_BIN), str(inp_path)]
    with log_path.open("w") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))


def sweep(grid: int, niter_list: list[int], n_runs: int, work: Path) -> list[dict]:
    results: list[dict] = []
    for niter in niter_list:
        for r in range(n_runs):
            inp_path = make_input(niter, r, grid, work)
            log_path = work / f"g{grid}_n{niter}_r{r}.log"
            t0 = time.time()
            rc = run_one(inp_path, log_path)
            wall = time.time() - t0
            save_dir = work / f"g{grid}_n{niter}_r{r}"
            dE1 = parse_de1(save_dir) if rc == 0 else None
            solver_info = parse_solver_info(log_path)
            rec = {
                "grid": grid,
                "niter": niter,
                "run": r,
                "rc": rc,
                "dE1": dE1,
                "converged": solver_info["converged"],
                "rel_residual": solver_info["rel_residual"],
                "iterations": solver_info["iterations"],
                "wall_s": round(wall, 2),
            }
            results.append(rec)
            conv_tag = "c" if solver_info["converged"] else ("x" if solver_info["converged"] is False else "?")
            print(f"  [{grid:>3}x{grid:>3}] Niter={niter:<3} run={r}  dE1={dE1!s:<25} "
                  f"res={solver_info['rel_residual']!s:<12} iters={solver_info['iterations']!s:<5} "
                  f"{conv_tag} wall={wall:.1f}s", flush=True)
    return results


def summarise(results: list[dict]) -> dict:
    """Group by (grid, niter), compute median / min / max / range / std of dE1."""
    grouped: dict[tuple, list[float]] = {}
    for rec in results:
        if rec["dE1"] is None:
            continue
        key = (rec["grid"], rec["niter"])
        grouped.setdefault(key, []).append(rec["dE1"])
    summary = {}
    for (grid, niter), vals in sorted(grouped.items()):
        vals_sorted = sorted(vals)
        summary[f"{grid}_{niter}"] = {
            "grid": grid,
            "niter": niter,
            "n": len(vals),
            "median": median(vals),
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "spread": vals_sorted[-1] / vals_sorted[0] if vals_sorted[0] > 0 else float("inf"),
            "values": vals_sorted,
        }
    return summary


def print_table(summary: dict, out=sys.stdout) -> None:
    print("", file=out)
    print("Phase 10f GMRES-iter bisection — dE(1) vs NiterGMRES", file=out)
    print("=" * 86, file=out)
    header = f"{'grid':<8}{'Niter':<8}{'n':<4}{'median dE(1)':<18}{'min':<14}{'max':<14}{'spread':<10}"
    print(header, file=out)
    print("-" * 86, file=out)
    grids_seen = set()
    for key in sorted(summary.keys(), key=lambda k: (summary[k]["grid"], summary[k]["niter"])):
        s = summary[key]
        g = s["grid"]
        if g not in grids_seen and grids_seen:
            print("-" * 86, file=out)
        grids_seen.add(g)
        print(f"{g}x{g:<6}{s['niter']:<8}{s['n']:<4}"
              f"{s['median']:<18.4e}{s['min']:<14.4e}{s['max']:<14.4e}"
              f"{s['spread']:<10.2f}", file=out)
    print("=" * 86, file=out)


def main() -> int:
    grids = [20, 40, 80] if "--grids-all" in sys.argv else [20]
    if "--grid-40" in sys.argv:
        grids = [40]
    if "--grid-80" in sys.argv:
        grids = [80]
    if "--all" in sys.argv:
        grids = [20, 40, 80]

    WORK.mkdir(parents=True, exist_ok=True)
    if not BUILD_BIN.exists():
        print(f"missing binary: {BUILD_BIN}", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for grid in grids:
        n_runs = N_RUNS_PER
        if grid == 80:
            n_runs = 3  # cycle-1 variance at 80x80 still cheap but keep it sane
        print(f"\n=== Grid {grid}x{grid}x4, NiterGMRES sweep ({n_runs} runs each) ===")
        results = sweep(grid, NITER_LIST, n_runs, WORK)
        all_results.extend(results)

    summary = summarise(all_results)

    out_txt = REPO / "phase10f_results.txt"
    out_json = REPO / "phase10f_results.json"
    with out_txt.open("w") as fh:
        print_table(summary, out=fh)
        print("\nper-run detail:", file=fh)
        for rec in all_results:
            print(f"  {rec}", file=fh)
    out_json.write_text(json.dumps({"runs": all_results, "summary": summary}, indent=2, default=float))

    print_table(summary)
    print(f"\nWrote {out_txt} and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
