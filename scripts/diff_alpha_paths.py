#!/usr/bin/env python3
"""Diff mover vs gather per-particle α tensors (Step 64).

Reads `alpha_{gather|mover}_cyc{N}_s{s}_r{r}.bin` binary dumps produced by
`Particles3D::computeMoments` and `Particles3D::ECSIM_velocity` when the
`DumpAlphaBothPaths` input-file flag is on.

Layout per particle (16 doubles):
  [pidx, x, y, z, Bx, By, Bz, α00, α01, α02, α10, α11, α12, α20, α21, α22]

Expected outcome of the audit (ECSIM condition #4):
  Within a single cycle, the gather samples fieldForPcls AFTER the last
  `set_fieldForPcls` call in `CalculateMoments` (top of cycle; buffer holds
  B^n), and the mover samples it AFTER `set_fieldForPcls` in `ParticlesMover`
  (mid-cycle; buffer holds B^{n+1}). So α_gather(N) and α_mover(N) share
  particle positions x_n but use *different* B fields by one solve step —
  they should NOT be byte-identical.

  The interesting cross-cycle comparison is α_mover(N-1) vs α_gather(N):
  same B buffer (fieldForPcls is untouched between mover exit and the next
  cycle's `set_fieldForPcls`, which re-runs but re-reads the same field
  state), but particle positions shifted by dt·v_avg via ECSIM_position.

  If condition #4 holds, the ONLY differences between α_gather and α_mover
  at matched inputs should be ε_mach-level FP noise from evaluation order.
  This script reports max/mean/median |Δ| for Om and each α component so
  you can spot any systematic non-FP divergence.

Usage:
  diff_alpha_paths.py OUTPUT_DIR [--cycle N] [--species s] [--rank r]
                                 [--cross-cycle]
"""
import argparse
import sys
from pathlib import Path
import numpy as np

N_DOUBLES = 16
COLS = [
    "pidx", "x", "y", "z",
    "Bx", "By", "Bz",
    "a00", "a01", "a02",
    "a10", "a11", "a12",
    "a20", "a21", "a22",
]


def load(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size % N_DOUBLES:
        raise SystemExit(f"{path}: {raw.size} doubles not a multiple of {N_DOUBLES}")
    return raw.reshape(-1, N_DOUBLES)


def diff_table(a: np.ndarray, b: np.ndarray, title: str) -> None:
    print(f"\n=== {title} ===")
    if a.shape != b.shape:
        print(f"  shape mismatch: {a.shape} vs {b.shape}")
        return
    n = a.shape[0]
    print(f"  particles: {n}")
    print(f"  {'col':>6s} | {'max_abs':>12s} | {'mean_abs':>12s} | {'median_abs':>12s} | {'max_rel':>12s}")
    for i, name in enumerate(COLS):
        d = a[:, i] - b[:, i]
        ad = np.abs(d)
        denom = np.maximum(np.abs(a[:, i]), np.abs(b[:, i]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, ad / denom, 0.0)
        print(f"  {name:>6s} | {ad.max():12.4e} | {ad.mean():12.4e} | {np.median(ad):12.4e} | {rel.max():12.4e}")


def cycle_files(outdir: Path, which: str, cycle: int, species: int, rank: int) -> Path:
    return outdir / f"alpha_{which}_cyc{cycle}_s{species}_r{rank}.bin"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path, help="Simulation output directory containing the α dumps")
    ap.add_argument("--cycle", type=int, default=1, help="Cycle to diff (default 1)")
    ap.add_argument("--species", type=int, default=0, help="Species index (default 0)")
    ap.add_argument("--rank", type=int, default=0, help="MPI rank (default 0)")
    ap.add_argument("--cross-cycle", action="store_true",
                    help="Also diff α_mover(cycle-1) vs α_gather(cycle): same B buffer, shifted x.")
    args = ap.parse_args()

    outdir = args.outdir
    if not outdir.is_dir():
        raise SystemExit(f"not a directory: {outdir}")

    gather_path = cycle_files(outdir, "gather", args.cycle, args.species, args.rank)
    mover_path  = cycle_files(outdir, "mover",  args.cycle, args.species, args.rank)

    if not gather_path.is_file():
        raise SystemExit(f"missing: {gather_path}")
    if not mover_path.is_file():
        raise SystemExit(f"missing: {mover_path}")

    g = load(gather_path)
    m = load(mover_path)

    diff_table(g, m, f"Within-cycle: gather (cyc {args.cycle}) vs mover (cyc {args.cycle}) — same x, B buffer differs by one solve")

    if args.cross_cycle:
        prev = args.cycle - 1
        if prev < 0:
            raise SystemExit("--cross-cycle requires --cycle >= 1")
        prev_mover = cycle_files(outdir, "mover", prev, args.species, args.rank)
        if not prev_mover.is_file():
            raise SystemExit(f"missing: {prev_mover}")
        pm = load(prev_mover)
        diff_table(pm, g, f"Cross-cycle: mover (cyc {prev}) vs gather (cyc {args.cycle}) — same B buffer, x shifted by ECSIM_position")

    return 0


if __name__ == "__main__":
    sys.exit(main())
