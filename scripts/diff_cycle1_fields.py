#!/usr/bin/env python3
"""Step 32 — cross-code byte diff of iPIC3D vs ECSIM field dumps at cycle N.

Consumes raw IEEE-754 double binaries produced by
`EMfields3D::dump_cycle_fields()` (iPIC3D) and the matching ECSIM hook.

Layout assumption: row-major C order, k (z) fastest. Each .bin is one array.
Companion `fields_cycle{N}.meta.txt` encodes shape.

Usage:
    pixi run python scripts/diff_cycle1_fields.py A_dir B_dir [--cycle N] [--top N]

Cycle 0 is the post-init (pre-solve) snapshot; cycle 1 is after the first
field solve + gather. Default is cycle 1.

Output: per-field max-abs, max-rel, L2 differences over the unique interior
[n_ghost, n-n_ghost-1). A single "verdict" tag per field:
    - MATCH     : max_rel < 1e-14 (bit-identical after periodic aliasing)
    - CLOSE     : max_rel < 1e-10 (O(truncation))
    - DIVERGE   : max_rel >= 1e-10 — localises the bug.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def parse_meta(meta_path: pathlib.Path) -> dict:
    """Parse `fields_cycle{N}.meta.txt` — node + mass shape, field names."""
    meta: dict = {"arr3": [], "arr4": [], "n_ghost": 1}
    for line in meta_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("# grid_node_shape"):
            meta["node_shape"] = tuple(int(x) for x in s.split()[2:5])
        elif s.startswith("# grid_mass_shape"):
            meta["mass_shape"] = tuple(int(x) for x in s.split()[2:6])
        elif s.startswith("# n_ghost"):
            meta["n_ghost"] = int(s.split()[2])
        elif s.startswith("# arr3:"):
            meta["arr3"] = s.split(":", 1)[1].split()
        elif s.startswith("# arr4:"):
            meta["arr4"] = s.split(":", 1)[1].split()
    return meta


def load_arr(dir_path: pathlib.Path, cycle: int, name: str, shape: tuple[int, ...]) -> np.ndarray:
    path = dir_path / f"fields_cycle{cycle}_{name}.bin"
    a = np.fromfile(path, dtype="<f8")
    expected = int(np.prod(shape))
    if a.size != expected:
        raise ValueError(f"{path.name}: size {a.size} != expected {expected} for shape {shape}")
    return a.reshape(shape)


def diff(a: np.ndarray, b: np.ndarray, n_ghost: int) -> tuple[float, float, float]:
    """Return (max_abs, max_rel, l2_rel) over the unique interior.

    Unique interior: exclude ghost nodes AND the high-side periodic duplicate
    at index n-n_ghost-1 (matches the convention in `get_E_field_energy`).
    """
    if a.ndim == 3:
        s = (slice(n_ghost, -n_ghost - 1),) * 3
    elif a.ndim == 4:
        s = (slice(None),) + (slice(n_ghost, -n_ghost - 1),) * 3
    else:
        raise ValueError(f"unexpected ndim {a.ndim}")

    ai, bi = a[s], b[s]
    d = ai - bi
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    scale = np.maximum(np.abs(ai), np.abs(bi))
    nz = scale > 0
    max_rel = float((np.abs(d[nz]) / scale[nz]).max()) if nz.any() else 0.0
    l2 = float(np.sqrt((d * d).sum()) / np.sqrt((ai * ai).sum() + 1e-300))
    return max_abs, max_rel, l2


def verdict(max_rel: float) -> str:
    if max_rel < 1e-14:
        return "MATCH"
    if max_rel < 1e-10:
        return "CLOSE"
    return "DIVERGE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a_dir", type=pathlib.Path, help="first dump directory (e.g. iPIC3D)")
    ap.add_argument("b_dir", type=pathlib.Path, help="second dump directory (e.g. ECSIM)")
    ap.add_argument("--cycle", type=int, default=1, help="cycle number to diff (default 1)")
    ap.add_argument(
        "--top", type=int, default=5,
        help="print N largest-divergence (i,j,k) points per field (default 5)"
    )
    args = ap.parse_args()

    cyc = args.cycle
    meta_a = parse_meta(args.a_dir / f"fields_cycle{cyc}.meta.txt")
    meta_b = parse_meta(args.b_dir / f"fields_cycle{cyc}.meta.txt")
    if meta_a["node_shape"] != meta_b["node_shape"]:
        print(
            f"ERROR: node shape mismatch {meta_a['node_shape']} vs {meta_b['node_shape']}",
            file=sys.stderr,
        )
        return 2
    if meta_a["mass_shape"] != meta_b["mass_shape"]:
        print(
            f"ERROR: mass shape mismatch {meta_a['mass_shape']} vs {meta_b['mass_shape']}",
            file=sys.stderr,
        )
        return 2

    ng = meta_a["n_ghost"]
    node = meta_a["node_shape"]
    mass = meta_a["mass_shape"]
    arr3 = sorted(set(meta_a["arr3"]) & set(meta_b["arr3"]))
    arr4 = sorted(set(meta_a["arr4"]) & set(meta_b["arr4"]))
    print(f"A = {args.a_dir}")
    print(f"B = {args.b_dir}")
    print(f"Grid: node {node}, mass {mass}, n_ghost={ng}")
    print()
    print(f"{'field':<8} {'shape':<16} {'max_abs':>11} {'max_rel':>11} {'l2_rel':>11}  verdict")
    print("-" * 74)
    for name in arr3:
        try:
            A = load_arr(args.a_dir, cyc, name, node)
            B = load_arr(args.b_dir, cyc, name, node)
        except (FileNotFoundError, ValueError) as e:
            print(f"{name:<8}  SKIP ({e})")
            continue
        ma, mr, l2 = diff(A, B, ng)
        print(f"{name:<8} {str(A.shape):<16} {ma:>11.3e} {mr:>11.3e} {l2:>11.3e}  {verdict(mr)}")
    print()
    for name in arr4:
        try:
            A = load_arr(args.a_dir, cyc, name, mass)
            B = load_arr(args.b_dir, cyc, name, mass)
        except (FileNotFoundError, ValueError) as e:
            print(f"{name:<8}  SKIP ({e})")
            continue
        ma, mr, l2 = diff(A, B, ng)
        print(f"{name:<8} {str(A.shape):<16} {ma:>11.3e} {mr:>11.3e} {l2:>11.3e}  {verdict(mr)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
