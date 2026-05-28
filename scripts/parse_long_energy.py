#!/usr/bin/env python3
"""Summarize ECSIM energy conservation for the long-run sweep.

For each case, read ConservedQuantities.txt and report, over all cycles:
  final |dE/E0|, max |dE/E0|, and whether the drift is secular (monotonic ramp)
  vs bounded.  Columns (0-indexed): 10 = total_energy, 11 = delta_energy.
"""
import os

CASES = [
    ("uniform", "np=1"), ("uniform_np4", "np=4"),
    ("harris",  "np=1"), ("harris_np4",  "np=4"),
]
ROOT = os.path.join(os.path.dirname(__file__), "..", "output", "test_energy_long")

def load(case):
    path = os.path.join(ROOT, case, "ConservedQuantities.txt")
    rows = [l.split() for l in open(path) if l.strip() and l.strip()[0].isdigit()]
    cyc   = [int(r[0])     for r in rows]
    ratio = [abs(float(r[11])) / abs(float(r[10])) for r in rows]
    return cyc, ratio

def secular(ratio):
    # crude: compare mean of last fifth vs first fifth; secular if grows >10x
    n = len(ratio); q = max(1, n // 5)
    head = sum(ratio[:q]) / q
    tail = sum(ratio[-q:]) / q
    return tail > 10 * head and tail > 1e-12

print(f"{'case':<20}{'np':<6}{'cycles':<8}{'final|dE/E0|':<16}{'max|dE/E0|':<16}{'trend':<10}")
print("-" * 76)
for case, np_ in CASES:
    try:
        cyc, ratio = load(case)
    except FileNotFoundError:
        print(f"{case:<20}{np_:<6}{'-- no output (crash?) --'}")
        continue
    rmax = max(ratio); rfin = ratio[-1]
    trend = "SECULAR" if secular(ratio) else "bounded"
    print(f"{case:<20}{np_:<6}{cyc[-1]:<8}{rfin:<16.3e}{rmax:<16.3e}{trend:<10}")

# coarse trajectory (every ~100 cycles) for the periodic machine-precision cases
print("\ntrajectory  |dE/E0|  (every ~100 cycles)")
for case, np_ in CASES:
    try:
        cyc, ratio = load(case)
    except FileNotFoundError:
        continue
    pts = [f"c{cyc[i]}={ratio[i]:.1e}" for i in range(0, len(cyc), max(1, len(cyc)//5))]
    print(f"  {case:<18} " + "  ".join(pts))
