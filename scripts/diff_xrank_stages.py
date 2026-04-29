#!/usr/bin/env python3
"""Phase E.13/E.14 cross-rank stage diff.

For each MaxwellImage stage dumped at np=4 X+Y=2×2, compare rank 0's HI X dup
(i=nx-ng-1) against rank 1's LO X dup (i=ng). They share the same physical
cell across the X-face boundary, so a non-zero diff at any stage localises
where cross-rank ULP divergence first appears in the matvec composition.
"""

import os
import sys
import numpy as np

DIR = "output/probeC_dump_stages"
NX, NY, NZ = 13, 13, 13         # nxn=nyn=nzn at this config
NG = 2

STAGES = [
    "A_post_halo_in",
    "B_post_curl2",
    "B2_pre_ME_temp",
    "C_raw_ME",
    "D_SMS_invVOL",
    "E_full_Ae",
]
COMPS = ["x", "y", "z"]


def load(stage, comp, rank):
    p = f"{DIR}/maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin"
    a = np.fromfile(p, dtype=np.float64)
    if a.size != NX * NY * NZ:
        sys.exit(f"size mismatch {p}: {a.size} vs {NX*NY*NZ}")
    return a.reshape((NX, NY, NZ))


def stage_diffs(stage):
    rows = []
    for comp in COMPS:
        r0 = load(stage, comp, 0)   # rank 0 (cx=0, cy=0)
        r1 = load(stage, comp, 1)   # rank 1 (cx=1, cy=0) — X-HI neighbour of r0
        # r0's HI X dup at i = NX-NG-1; r1's LO X dup at i = NG.
        # Both are the same physical cell at every (j, k).
        a = r0[NX - NG - 1, :, :]
        b = r1[NG, :, :]
        diff = a - b
        amax = np.max(np.abs(diff))
        l2_rel = np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-300)
        rows.append((comp, amax, l2_rel))
    return rows


def main():
    print(f"{'stage':<22} {'comp':<5} {'max|Δ|':>14} {'l2_rel':>14}")
    print("-" * 60)
    for st in STAGES:
        for comp, amax, l2 in stage_diffs(st):
            mark = "  ULP" if amax < 1e-12 else "  *"
            print(f"{st:<22} {comp:<5} {amax:14.3e} {l2:14.3e}{mark}")


if __name__ == "__main__":
    main()
