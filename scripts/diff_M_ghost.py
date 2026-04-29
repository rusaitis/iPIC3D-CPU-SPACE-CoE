#!/usr/bin/env python3
"""Check whether M ghost cells are paired-rank consistent.

After communicateNode_P (Mstage2_post_selfcopy), r0's HI X ghost (i=11, 12)
should equal r2's LO X strict (i=3, 4) — same physical X via FACE PHASE
copy. r0's HI X strict (i=8, 9) should equal r2's LO X ghost (i=0, 1)
similarly. Same for Y axis with r1.

If these aren't paired, the M·E kernel reads inconsistent M values when
the stencil reaches into ghost (e.g., at output face dup pair r0[10,j,k]
BWD reads M[g][9,...] strict vs r2's BWD reads M[g][1,...] ghost).
"""

import os
import sys
import numpy as np

NX, NY, NZ = 13, 13, 13
NG = 2


def load(out_dir, stage, comp, rank):
    p = os.path.join(out_dir,
                     f"maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin")
    a = np.fromfile(p, dtype=np.float64)
    return a.reshape((NX, NY, NZ))


def diff_section(label, a, b):
    diff = a - b
    amax = float(np.max(np.abs(diff)))
    mark = "ε" if amax < 1e-12 else "✗"
    print(f"  {label:<60} {amax:>10.2e}{mark}")


def report_x_pairing(stage, out_dir):
    print(f"\n=== {stage} : X-axis ghost ↔ strict pairing (r0 vs r2) ===")
    r0 = load(out_dir, stage, "x", 0)
    r2 = load(out_dir, stage, "x", 2)
    sl_y = slice(NG + 1, NY - 1 - NG)
    sl_z = slice(NG + 1, NZ - 1 - NG)
    # r0's HI X ghost outer (i=12) should = r2's LO X strict near-LO 2nd (i=4)
    diff_section("r0[i=12, j∈strict, k∈strict] vs r2[i=4]",
                 r0[12, sl_y, sl_z], r2[4, sl_y, sl_z])
    # r0's HI X ghost inner (i=11) should = r2's LO X strict near-LO 1st (i=3)
    diff_section("r0[i=11, ...] vs r2[i=3]",
                 r0[11, sl_y, sl_z], r2[3, sl_y, sl_z])
    # r0's HI X dup (i=10) should = r2's LO X dup (i=2) -- post-SOR
    diff_section("r0[i=10 dup] vs r2[i=2 dup]",
                 r0[10, sl_y, sl_z], r2[2, sl_y, sl_z])
    # r0's HI X strict near-HI (i=9) should = r2's LO X ghost inner (i=1)
    diff_section("r0[i=9 strict] vs r2[i=1 ghost]",
                 r0[9, sl_y, sl_z], r2[1, sl_y, sl_z])
    # r0's HI X strict (i=8) should = r2's LO X ghost outer (i=0)
    diff_section("r0[i=8 strict] vs r2[i=0 ghost]",
                 r0[8, sl_y, sl_z], r2[0, sl_y, sl_z])


def report_y_pairing(stage, out_dir):
    print(f"\n=== {stage} : Y-axis ghost ↔ strict pairing (r0 vs r1) ===")
    r0 = load(out_dir, stage, "x", 0)
    r1 = load(out_dir, stage, "x", 1)
    sl_x = slice(NG + 1, NX - 1 - NG)
    sl_z = slice(NG + 1, NZ - 1 - NG)
    diff_section("r0[*, j=12 ghost] vs r1[*, j=4 strict]",
                 r0[sl_x, 12, sl_z], r1[sl_x, 4, sl_z])
    diff_section("r0[*, j=11] vs r1[*, j=3]",
                 r0[sl_x, 11, sl_z], r1[sl_x, 3, sl_z])
    diff_section("r0[*, j=10 dup] vs r1[*, j=2 dup]",
                 r0[sl_x, 10, sl_z], r1[sl_x, 2, sl_z])
    diff_section("r0[*, j=9 strict] vs r1[*, j=1 ghost]",
                 r0[sl_x, 9, sl_z], r1[sl_x, 1, sl_z])
    diff_section("r0[*, j=8] vs r1[*, j=0]",
                 r0[sl_x, 8, sl_z], r1[sl_x, 0, sl_z])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/leo/projects/ipic3d/output/probe1_dump"
    stages = sys.argv[2:] if len(sys.argv) > 2 else \
        ["A_post_halo_in", "B2_pre_ME_temp"]
    for stage in stages:
        report_x_pairing(stage, out)
        report_y_pairing(stage, out)


if __name__ == "__main__":
    main()
