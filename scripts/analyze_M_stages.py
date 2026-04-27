#!/usr/bin/env python3
"""Analyze the 4-stage M dump from communicateGhostP2G_mass_matrix.

Reads `maxwell_stage_c1_m0_Mstage{0..3}_*_x_r0.bin` produced when
`DumpMassMatrixStages = 1`, prints i-line cross sections across stages, and
the boundary-band magnitude relative to interior.

Run: pixi run python scripts/analyze_M_stages.py output/phaseE_staged_dump
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


STAGES = [
    ("stage0", "Mstage0_pre_halo"),
    ("stage1", "Mstage1_post_addFace"),
    ("stage2", "Mstage2_post_selfcopy"),
    ("stage3", "Mstage3_post_unify"),
]


def load(path: str, nxn: int, nyn: int, nzn: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float64)
    if a.size != nxn * nyn * nzn:
        raise SystemExit(f"size mismatch {path}: {a.size} vs {nxn*nyn*nzn}")
    return a.reshape(nxn, nyn, nzn)


def deduce_shape(path: str) -> tuple[int, int, int]:
    n = os.path.getsize(path) // 8
    for nxn in range(20, 80):
        for nyn in range(20, 80):
            for nzn in range(1, 10):
                if nxn * nyn * nzn == n and nxn == nyn:
                    return nxn, nyn, nzn
    raise SystemExit(f"cannot deduce shape for {n} doubles")


def stage_summary(arr: np.ndarray, label: str) -> None:
    nxn, nyn, nzn = arr.shape
    j_mid = nyn // 2
    k_mid = nzn // 2
    line = arr[:, j_mid, k_mid]
    interior_mid = nxn // 2
    interior_val = line[interior_mid]
    print(f"\n=== {label} (j_mid={j_mid}, k_mid={k_mid}) ===")
    print(f"  i:    " + " ".join(f"{i:8d}" for i in range(nxn)))
    print(f"  M:    " + " ".join(f"{v:8.3e}" for v in line))
    print(f"  M/Mc: " + " ".join(f"{(v/interior_val if interior_val else 0):8.3f}"
                                  for v in line))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="dump directory (SaveDirName)")
    ap.add_argument("--comp", default="x", choices=["x", "y", "z"],
                    help="which Mxx/Myy/Mzz component to inspect")
    args = ap.parse_args()

    base = os.path.abspath(args.dir)
    sample = os.path.join(base,
        f"maxwell_stage_c1_m0_{STAGES[0][1]}_{args.comp}_r0.bin")
    nxn, nyn, nzn = deduce_shape(sample)
    print(f"Detected shape: nxn={nxn} nyn={nyn} nzn={nzn}")

    arrs = {}
    for tag, name in STAGES:
        p = os.path.join(base, f"maxwell_stage_c1_m0_{name}_{args.comp}_r0.bin")
        arrs[tag] = load(p, nxn, nyn, nzn)

    for tag, name in STAGES:
        stage_summary(arrs[tag], f"{tag}: {name}")

    # Δ between adjacent stages: max abs diff and where
    print("\n=== stage-to-stage delta (max |Δ| over full slab) ===")
    prev_tag = None
    prev_arr = None
    for tag, _ in STAGES:
        a = arrs[tag]
        if prev_arr is not None:
            d = a - prev_arr
            idx = np.unravel_index(np.argmax(np.abs(d)), d.shape)
            print(f"  {prev_tag} -> {tag}: max|Δ|={np.max(np.abs(d)):.3e} "
                  f"at i={idx[0]} j={idx[1]} k={idx[2]} "
                  f"(prev={prev_arr[idx]:.3e}, curr={a[idx]:.3e})")
        prev_tag = tag
        prev_arr = a

    # Interior median per stage (strict interior, away from any boundary band).
    # Also report ratios stage_n / stage_0 to expose any global multiplicative shifts.
    print("\n=== strict-interior median (boundary band excluded) ===")
    ng_guess = 2 if (nxn - 32) >= 5 else 1   # quick heuristic for nxc=32 layouts
    pad = ng_guess + 1                        # exclude both ghost and 1-cell boundary band
    if nxn > 2 * pad and nyn > 2 * pad:
        z0, z1 = (1, nzn - 1) if nzn > 2 else (0, nzn)
        s0_med = float(np.median(arrs["stage0"][pad:nxn - pad, pad:nyn - pad, z0:z1]))
    else:
        s0_med = float(np.median(arrs["stage0"]))
    for tag, _ in STAGES:
        if nxn > 2 * pad and nyn > 2 * pad:
            z0, z1 = (1, nzn - 1) if nzn > 2 else (0, nzn)
            med = float(np.median(arrs[tag][pad:nxn - pad, pad:nyn - pad, z0:z1]))
        else:
            med = float(np.median(arrs[tag]))
        rel = med / s0_med if s0_med else float("nan")
        print(f"  {tag}: median={med:.3e}  (×{rel:.3f} vs stage0)")


if __name__ == "__main__":
    main()
