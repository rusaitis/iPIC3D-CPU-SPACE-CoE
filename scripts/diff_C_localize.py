#!/usr/bin/env python3
"""Localize the C_raw_ME divergence at face dup pair r0[i=10] vs r2[i=2].

Print per-(j, k) divergence for all j ∈ [3..9], k ∈ [3..9] to see if max is
at a specific position (e.g., j=3 or j=9 boundary, vs j=5 mid).
"""

import os
import sys
import numpy as np

NX, NY, NZ = 13, 13, 13


def load(out_dir, stage, comp, rank):
    p = os.path.join(out_dir,
                     f"maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin")
    return np.fromfile(p, dtype=np.float64).reshape((NX, NY, NZ))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/leo/projects/ipic3d/output/probe1_dump"
    stage = "C_raw_ME"
    comp = "x"
    r0 = load(out, stage, comp, 0)
    r2 = load(out, stage, comp, 2)
    print(f"=== {stage} {comp} : r0[i=10, j, k] vs r2[i=2, j, k] ===")
    print(f"{'j\\k':<4} ", end="")
    for k in range(3, 10):
        print(f"{k:>10d}", end="")
    print()
    for j in range(3, 10):
        print(f"j={j:<2d} ", end="")
        for k in range(3, 10):
            d = abs(r0[10, j, k] - r2[2, j, k])
            print(f"{d:>10.2e}", end="")
        print()


if __name__ == "__main__":
    main()
