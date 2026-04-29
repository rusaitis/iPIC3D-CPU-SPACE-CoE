#!/usr/bin/env python3
"""Phase E.17 — paired-rank dup-cell diff inside MaxwellImage.

For each MaxwellImage stage at np=4 X+Y=2x2, compare paired-rank duplicate
cells where r0 and r2 share the X-axis dup column (HI X dup of r0 ↔
LO X dup of r2), and r0 and r1 share the Y-axis dup row, etc. The
duplicate cells should hold identical physical values; FP-ULP divergence
between them tells us where the operator-side leak erupts.

Stages (in matvec order):
  A_post_halo_in   — input E after halo (pre-curl²)
  B_post_curl2     — after curl² composition
  B2_pre_ME_temp   — temp before mass_matrix call
  C_raw_ME         — raw α·M·α matvec output
  D_SMS_invVOL     — after invVOL/smoothing
  E_full_Ae        — final A·E
"""

import os
import sys

import numpy as np

NX, NY, NZ = 13, 13, 13
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


def load(out_dir, stage, comp, rank):
    p = os.path.join(out_dir,
                     f"maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin")
    a = np.fromfile(p, dtype=np.float64)
    if a.size != NX * NY * NZ:
        sys.exit(f"size mismatch {p}: {a.size} vs {NX*NY*NZ}")
    return a.reshape((NX, NY, NZ))


def diff_x_dup(d_r0, d_r2):
    """r0[HI X dup] vs r2[LO X dup] over j∈[ng+1..ny-2-ng], k∈[ng+1..nz-2-ng]."""
    a = d_r0[NX - NG - 1, NG + 1:NY - 1 - NG, NG + 1:NZ - 1 - NG]
    b = d_r2[NG,          NG + 1:NY - 1 - NG, NG + 1:NZ - 1 - NG]
    return float(np.max(np.abs(a - b)))


def diff_y_dup(d_r0, d_r1):
    """r0[HI Y dup] vs r1[LO Y dup] over i∈[ng+1..nx-2-ng], k∈[ng+1..nz-2-ng]."""
    a = d_r0[NG + 1:NX - 1 - NG, NY - NG - 1, NG + 1:NZ - 1 - NG]
    b = d_r1[NG + 1:NX - 1 - NG, NG,          NG + 1:NZ - 1 - NG]
    return float(np.max(np.abs(a - b)))


def diff_xy_corner(d_r0, d_r3):
    """r0[HI X, HI Y] vs r3[LO X, LO Y] over k∈[ng+1..nz-2-ng]."""
    a = d_r0[NX - NG - 1, NY - NG - 1, NG + 1:NZ - 1 - NG]
    b = d_r3[NG,          NG,          NG + 1:NZ - 1 - NG]
    return float(np.max(np.abs(a - b)))


def report(label, out_dir):
    print(f"\n=== {label} : {out_dir} ===")
    print(f"{'stage':<22} {'comp':<5} {'X-dup':>12} {'Y-dup':>12} {'XY-corner':>12}")
    print("-" * 70)
    for stage in STAGES:
        for comp in COMPS:
            try:
                r0 = load(out_dir, stage, comp, 0)
                r1 = load(out_dir, stage, comp, 1)
                r2 = load(out_dir, stage, comp, 2)
                r3 = load(out_dir, stage, comp, 3)
            except FileNotFoundError as e:
                print(f"{stage:<22} {comp:<5}  missing: {e}")
                continue
            dx = diff_x_dup(r0, r2)
            dy = diff_y_dup(r0, r1)
            dc = diff_xy_corner(r0, r3)
            mark_x = "ε" if dx < 1e-12 else "✗"
            mark_y = "ε" if dy < 1e-12 else "✗"
            mark_c = "ε" if dc < 1e-12 else "✗"
            print(f"{stage:<22} {comp:<5} {dx:>10.2e}{mark_x} "
                  f"{dy:>10.2e}{mark_y} {dc:>10.2e}{mark_c}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/leo/projects/ipic3d/output/probe1_dump"
    report("Probe 1 — paired-rank dup diff", out)


if __name__ == "__main__":
    main()
