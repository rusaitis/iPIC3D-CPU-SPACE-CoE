#!/usr/bin/env python3
"""
check_prec_iters.py — Compare per-cycle KSP iteration counts across solver logs.

The "did the preconditioner actually help?" gate.  Parses one or more run logs,
reports mean/max iterations per cycle, and (optionally) fails if a solver did not
converge or exceeded an iteration cap.

Note: a preconditioner only conditions the Krylov solve — it must NOT change the
converged field (that is validate.py's job).  The value of a preconditioner is a
LOWER and (under mesh refinement) BOUNDED iteration count.  At a well-conditioned
regime (small dt, coarse 2D grid) no preconditioner beats PCNONE; the scalar-
Helmholtz win shows up under refinement / in 3D — see tests/bench_preconditioners.sh.

Usage:
  python scripts/check_prec_iters.py LOG [LOG ...]            # report table
  python scripts/check_prec_iters.py run_none.log run_helm.log
  python scripts/check_prec_iters.py *.log --assert-converged --max-iters 200
  python scripts/check_prec_iters.py *.log --baseline PCNONE --max-ratio 1.5

Exit 0 = all checks passed (or report-only), 1 = a check failed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_patterns import ITER_PATTERNS, CONV_FAIL


def parse_iters(log_path):
    """Return the list of per-cycle KSP iteration counts found in a log."""
    iters = []
    n_fail = 0
    with open(log_path) as f:
        for line in f:
            for pat in CONV_FAIL:
                if pat.search(line):
                    n_fail += 1
                    break
            for regex, extract in ITER_PATTERNS:
                m = regex.search(line)
                if m:
                    iters.append(extract(m))
                    break
    return iters, n_fail


def label_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='+', help='run log file(s) to compare')
    ap.add_argument('--assert-converged', action='store_true',
                    help='fail if any solver reports a non-converged cycle')
    ap.add_argument('--max-iters', type=float, default=None,
                    help='fail if any solver mean iterations exceeds this cap')
    ap.add_argument('--baseline', default=None,
                    help='label (filename stem) of the baseline solver for ratio check')
    ap.add_argument('--max-ratio', type=float, default=None,
                    help='fail if mean(solver)/mean(baseline) exceeds this for any solver')
    args = ap.parse_args()

    stats = {}  # label -> (mean, max, count, n_fail)
    for log in args.logs:
        if not os.path.isfile(log):
            print(f"  SKIP  {log}: not found")
            continue
        iters, n_fail = parse_iters(log)
        lbl = label_of(log)
        if not iters:
            print(f"  SKIP  {lbl}: no KSP iteration lines")
            continue
        mean = sum(iters) / len(iters)
        stats[lbl] = (mean, max(iters), len(iters), n_fail)

    if not stats:
        print("ERROR: no parseable iteration data in any log", file=sys.stderr)
        return 1

    width = max(len(l) for l in stats)
    print(f"\n  {'solver'.ljust(width)}   cycles   mean_iters   max_iters   nonconv")
    print(f"  {'-' * width}   ------   ----------   ---------   -------")
    for lbl in sorted(stats):
        mean, mx, n, nf = stats[lbl]
        print(f"  {lbl.ljust(width)}   {n:6d}   {mean:10.1f}   {mx:9d}   {nf:7d}")
    print()

    failed = []
    if args.assert_converged:
        for lbl, (_, _, _, nf) in stats.items():
            if nf > 0:
                failed.append(f"{lbl}: {nf} non-converged cycle(s)")
    if args.max_iters is not None:
        for lbl, (mean, _, _, _) in stats.items():
            if mean > args.max_iters:
                failed.append(f"{lbl}: mean iters {mean:.1f} > cap {args.max_iters}")
    if args.baseline is not None and args.max_ratio is not None:
        if args.baseline not in stats:
            failed.append(f"baseline '{args.baseline}' not among logs")
        else:
            base = stats[args.baseline][0]
            for lbl, (mean, _, _, _) in stats.items():
                if lbl == args.baseline:
                    continue
                ratio = mean / base if base else float('inf')
                tag = 'ok' if ratio <= args.max_ratio else 'FAIL'
                print(f"  ratio {lbl}/{args.baseline} = {ratio:.2f}  [{tag}]")
                if ratio > args.max_ratio:
                    failed.append(f"{lbl}: iters/baseline {ratio:.2f} > {args.max_ratio}")

    if failed:
        print("\nFAIL:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
