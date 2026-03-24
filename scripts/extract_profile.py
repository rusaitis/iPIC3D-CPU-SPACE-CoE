#!/usr/bin/env python3
"""
extract_profile.py — Extract per-cycle timing profile from an iPIC3D log file.

Parses cumulative timing counters (Field solver, Particle mover, Moment gatherer,
Write data) and solver iteration counts from the simulation log, producing a CSV
with per-cycle deltas.

Usage:
    python scripts/extract_profile.py <logfile> <output.csv>
"""

import csv
import sys

from log_patterns import ITER_PATTERNS, TIMING_PATTERNS


def extract_profile(logfile, outcsv):
    """Parse logfile for timing data and write per-cycle profile CSV."""
    fields = []       # list of (field, mover, moment, write) cumulative tuples
    iterations = []   # per-solve iteration counts

    with open(logfile) as f:
        cur = {}
        for line in f:
            # Check convergence patterns for iteration counts
            matched = False
            for pat, extract in ITER_PATTERNS:
                m = pat.search(line)
                if m:
                    iterations.append(extract(m))
                    matched = True
                    break

            if not matched:
                # Check timing patterns
                for key, pat in TIMING_PATTERNS:
                    m = pat.search(line)
                    if m:
                        cur[key] = float(m.group(1))
                if len(cur) == 4:
                    fields.append((cur['field'], cur['mover'],
                                   cur['moment'], cur['write']))
                    cur = {}

    if not fields:
        return 0

    with open(outcsv, 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(['cycle', 'field_solver_s', 'particle_mover_s',
                     'moment_gatherer_s', 'write_data_s', 'iterations'])
        prev = (0, 0, 0, 0)
        cycle_num = 0
        for i, vals in enumerate(fields):
            delta = tuple(v - p for v, p in zip(vals, prev))
            # Detect counter reset (e.g., from restart): negative deltas
            if any(d < -1e-9 for d in delta):
                prev = (0, 0, 0, 0)
                delta = vals
            cycle_num += 1
            iters = iterations[i] if i < len(iterations) else ''
            w.writerow([cycle_num] + [f'{d:.6f}' for d in delta] + [iters])
            prev = vals

    return len(fields)


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <logfile> <output.csv>', file=sys.stderr)
        sys.exit(2)

    logfile, outcsv = sys.argv[1], sys.argv[2]
    n = extract_profile(logfile, outcsv)
    if n > 0:
        print(f'Extracted {n} cycles → {outcsv}')
    else:
        print(f'No timing data found in {logfile}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
