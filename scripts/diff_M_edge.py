#!/usr/bin/env python3
"""Check M[g] cross-rank consistency at EDGE cells (X-ghost ∩ Y-ghost).

For pair (np=4 X+Y=2x2): r2[i=0, j=1, k=5] and r0[i=8, j=1, k=5] should
both equal the physical value at (X=6dx, Y=15dy, k=3dx) which lives in r1's
strict at r1[i=8, j=9, k=5]. Same with r3 → r1.

If r2[i=0, j=1] doesn't match r0[i=8, j=1], the EDGE PHASE diagonal-corner
ghost routing is broken at multi-axis cross-rank.
"""

import os
import sys
import numpy as np

NX, NY, NZ = 13, 13, 13


def load(out_dir, stage, comp, rank):
    p = os.path.join(out_dir,
                     f"maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin")
    return np.fromfile(p, dtype=np.float64).reshape((NX, NY, NZ))


def report_edge(stage, out_dir, comp="x"):
    print(f"\n=== {stage} {comp} : EDGE-cell pair consistency ===")
    r0 = load(out_dir, stage, comp, 0)
    r1 = load(out_dir, stage, comp, 1)
    r2 = load(out_dir, stage, comp, 2)
    r3 = load(out_dir, stage, comp, 3)

    # All 4 ranks should hold same physical value at LO X ∩ LO Y corner of r2,
    # which is (X=6dx, Y=15dy, k strict) — physically in r1's strict at (i=8, j=9, k).
    # On each rank, that cell maps to:
    #   r0[i=8, j=1, k=*]   (HI X strict ∩ LO Y ghost via face Y)
    #   r1[i=8, j=9, k=*]   (interior strict)
    #   r2[i=0, j=1, k=*]   (LO X ghost ∩ LO Y ghost — edge corner)
    #   r3[i=4, j=9, k=*]   (LO X near-LO ∩ HI Y near-HI strict — interior)
    # Wait, only r1[8, 9, k] is the actual interior with that physical X,Y.
    # r3 doesn't cover (X=6dx, Y=15dy) at all. Let me check what r3[4, 9, k]
    # is physically: X = (4-2)*dx + 8*dx = 10*dx, Y = (9-2)*dy + 8*dy = 15*dy.
    # So r3[4, 9, k] is at X=10dx, NOT 6dx.
    # That means r0 and r2's edge ghost should hold r1's value (NOT r3's).

    sl_z = slice(3, 10)
    print("Reference: r1[i=8, j=9, k strict] (interior strict at X=6dx, Y=15dy)")
    ref = r1[8, 9, sl_z]
    print(f"  r1 sample at k=5: {ref[2]:.6e}")

    print(f"\nr0[i=8, j=1, k strict] (HI X strict ∩ LO Y ghost):")
    val = r0[8, 1, sl_z]
    print(f"  sample at k=5: {val[2]:.6e}")
    diff = np.max(np.abs(val - ref))
    print(f"  max|Δ| vs r1[8,9,k]: {diff:.2e}")

    print(f"\nr2[i=0, j=1, k strict] (LO X ghost ∩ LO Y ghost — EDGE):")
    val = r2[0, 1, sl_z]
    print(f"  sample at k=5: {val[2]:.6e}")
    diff = np.max(np.abs(val - ref))
    print(f"  max|Δ| vs r1[8,9,k]: {diff:.2e}")

    print(f"\nr3[i=4, j=9, k strict] (different physical position, NOT same as ref):")
    val = r3[4, 9, sl_z]
    print(f"  sample at k=5: {val[2]:.6e}")
    diff = np.max(np.abs(val - ref))
    print(f"  max|Δ| vs r1[8,9,k]: {diff:.2e}")

    # Direct pair: r0[8, 1, k] should equal r2[0, 1, k]
    print(f"\nDirect pair: r0[8, 1, k] vs r2[0, 1, k] (both should hold same physical value):")
    diff = np.max(np.abs(r0[8, 1, sl_z] - r2[0, 1, sl_z]))
    print(f"  max|Δ|: {diff:.2e}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/leo/projects/ipic3d/output/probe1_M"
    for stage in ["Mstage2_post_selfcopy", "Mstage2_g50"]:
        report_edge(stage, out)


if __name__ == "__main__":
    main()
