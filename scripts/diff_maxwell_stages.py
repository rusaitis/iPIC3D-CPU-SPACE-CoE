#!/usr/bin/env python3
"""Cross-diff per-stage MaxwellImage dumps (Step 38).

Reads `maxwell_stage_c{cycle}_m{matvec}_{stage}_{x|y|z}.bin` binary dumps from
two directories and prints per-stage l2/max divergence.

iPIC3D instrumentation is in `fields/EMfields3D.cpp::MaxwellImage`, gated by
the `DumpMaxwellImageStages` input-file flag.

Stages:
  A_post_halo_in   tempX,Y,Z right after input halo refresh
  B_post_curl2     imageX,Y,Z after (I + (cθΔt)²·curl²)·E assembly
  C_raw_ME         temp2X,Y,Z raw M·E × (dt·θ·4π), pre outer halo/smooth
  D_SMS_invVOL     temp2X,Y,Z post outer halo + outer smooth + invVOL
  E_full_Ae        imageX,Y,Z final A·E (pre optional Symm-out)

The script reads full nxn×nyn×nzn arrays and diffs only the interior
[1:-1, 1:-1, 1:-1] to ignore ghost-cell differences.

Usage:
  diff_maxwell_stages.py DIR1 DIR2 [--shape NXN NYN NZN] [--cycle 1] [--matvec 0]
"""
import argparse
import sys
from pathlib import Path
import numpy as np


STAGES = [
    "A_post_halo_in",
    "B_post_curl2",
    "B2_pre_ME_temp",
    "C_raw_ME",
    "D_SMS_invVOL",
    "E_full_Ae",
]


def load(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float64)
    want = shape[0] * shape[1] * shape[2]
    if a.size != want:
        raise SystemExit(
            f"{path}: got {a.size} doubles, expected {want} for shape {shape}"
        )
    return a.reshape(*shape)


def diff_pair(a: np.ndarray, b: np.ndarray) -> dict:
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    ai, bi = a[interior], b[interior]
    d = ai - bi
    denom_max = max(float(np.abs(ai).max()), 1e-300)
    denom_l2 = max(float(np.linalg.norm(ai)), 1e-300)
    return {
        "max_abs": float(np.abs(d).max()),
        "max_rel": float(np.abs(d).max()) / denom_max,
        "l2_abs": float(np.linalg.norm(d)),
        "l2_rel": float(np.linalg.norm(d)) / denom_l2,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir1", type=Path)
    p.add_argument("dir2", type=Path)
    p.add_argument("--shape", nargs=3, type=int, metavar=("NXN", "NYN", "NZN"),
                   default=[259, 131, 4],
                   help="node-grid shape; default matches DoubleGEM 256×128 CIC")
    p.add_argument("--cycle", type=int, default=1)
    p.add_argument("--matvec", type=int, default=0)
    args = p.parse_args(argv)

    shape = tuple(args.shape)
    label1 = args.dir1.name or str(args.dir1)
    label2 = args.dir2.name or str(args.dir2)

    print(f"Diffing {label1} vs {label2}  (interior of {shape})\n")
    print(f"{'Stage':<20}  {'comp':<4}  "
          f"{'max_abs':>12}  {'max_rel':>12}  "
          f"{'l2_abs':>12}  {'l2_rel':>12}")
    print("-" * 90)

    for stage in STAGES:
        for comp in "xyz":
            name = f"maxwell_stage_c{args.cycle}_m{args.matvec}_{stage}_{comp}.bin"
            p1 = args.dir1 / name
            p2 = args.dir2 / name
            if not p1.exists() or not p2.exists():
                missing = [str(p) for p in (p1, p2) if not p.exists()]
                print(f"{stage:<20}  {comp:<4}  MISSING: {', '.join(missing)}")
                continue
            a = load(p1, shape)
            b = load(p2, shape)
            d = diff_pair(a, b)
            print(f"{stage:<20}  {comp:<4}  "
                  f"{d['max_abs']:12.3e}  {d['max_rel']:12.3e}  "
                  f"{d['l2_abs']:12.3e}  {d['l2_rel']:12.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
