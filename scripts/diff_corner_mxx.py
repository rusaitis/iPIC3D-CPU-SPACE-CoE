#!/usr/bin/env python3
"""Phase E.14 corner-completion verification.

Compare Mxx Stage 1 (post addFace+SOR) X-face dup pairing at np=4 X+Y=2x2:
  rank 0's HI X dup (i=10) vs rank 2's LO X dup (i=2).
At j strict (3..9) the two should match at ULP either way (face-only path).
At j ∈ {2, 10} (Y dup rows) the two should match only with MultiAxisCornerSOR=1.
"""

import os
import sys
import numpy as np

NX, NY, NZ = 13, 13, 13
NG = 2

ROWS = [
    ("LO Y dup",  2),
    ("strict y3", 3),
    ("strict y5", 5),
    ("strict y9", 9),
    ("HI Y dup",  10),
]


def load(out_dir, stage, comp, rank):
    p = os.path.join(out_dir,
                     f"maxwell_stage_c1_m0_{stage}_{comp}_r{rank}.bin")
    a = np.fromfile(p, dtype=np.float64)
    if a.size != NX * NY * NZ:
        sys.exit(f"size mismatch {p}: {a.size} vs {NX*NY*NZ}")
    return a.reshape((NX, NY, NZ))


def report(label, out_dir):
    print(f"\n=== {label} : {out_dir} ===")
    r0 = load(out_dir, "Mstage1_post_addFace", "x", 0)
    r2 = load(out_dir, "Mstage1_post_addFace", "x", 2)
    a = r0[NX - NG - 1, :, :]   # i=10 on r0, j range, k strict mask later
    b = r2[NG, :, :]            # i=2 on r2

    print(f"{'j':<10} {'max|Δ|':>14}")
    print("-" * 30)
    for label_j, j in ROWS:
        diff = a[j, NG + 1:NZ - 1 - NG] - b[j, NG + 1:NZ - 1 - NG]
        amax = float(np.max(np.abs(diff))) if diff.size else 0.0
        mark = "  ε" if amax < 1e-12 else "  ✗"
        print(f"j={j:<3d} {label_j:<8} {amax:14.3e}{mark}")


def main():
    base = "/Users/leo/projects/ipic3d/output"
    report("BASELINE (no flag)", os.path.join(base, "probeM_dump"))
    report("CORNER ON",          os.path.join(base, "probeM_corner_on"))
    face_dir = os.path.join(base, "probeM_face_on")
    if os.path.isdir(face_dir):
        report("FACE ON (corner+face)", face_dir)


if __name__ == "__main__":
    main()
