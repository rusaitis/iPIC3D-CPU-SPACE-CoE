#!/usr/bin/env bash
set -euo pipefail

# test_restart.sh — Verify that restart produces identical results.
#
# Runs the same simulation two ways:
#   A) 20 cycles uninterrupted
#   B) 10 cycles, checkpoint, restart for 10 more
# Then compares ConservedQuantities.txt — values should match within round-off.
#
# Usage:
#   bash tests/test_restart.sh [--exe PATH] [--cycles N] [--np N]
#
# END_HELP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIM_SH="$PROJECT_DIR/scripts/sim.sh"
PYTHON="${PYTHON:-python3}"

TOTAL_CYCLES=20
HALF_CYCLES=$((TOTAL_CYCLES / 2))
NP=""
EXE="${IPIC3D_BUILD_DIR:-build}/iPIC3D"
INPUT="inputfiles/Double_Harris.inp"
TOL=1e-12

# ── Parse args ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exe)    EXE="$2"; shift 2 ;;
        --cycles) TOTAL_CYCLES="$2"; HALF_CYCLES=$((TOTAL_CYCLES / 2)); shift 2 ;;
        --np)     NP="$2"; shift 2 ;;
        --input)  INPUT="$2"; shift 2 ;;
        -h|--help) sed -n '/^# test_restart/,/^# END_HELP/{ s/^# \{0,1\}//; p; }' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Setup ────────────────────────────────────────────────────────────────

TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_restart_test.XXXXXX")
trap 'rm -rf "$TMPDIR_BASE"' EXIT

DIR_A="$TMPDIR_BASE/straight"
DIR_B="$TMPDIR_BASE/restarted"

NP_FLAG=()
[[ -n "$NP" ]] && NP_FLAG=(-np "$NP")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Restart continuity test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total cycles: $TOTAL_CYCLES (${HALF_CYCLES} + ${HALF_CYCLES})"
echo "  Tolerance:    $TOL"
echo ""

# ── Run A: straight through ──────────────────────────────────────────────

echo "  [1/3] Running $TOTAL_CYCLES cycles uninterrupted..."
bash "$SIM_SH" "$INPUT" --no-build "${NP_FLAG[@]}" \
    --cycles "$TOTAL_CYCLES" -o "$DIR_A" \
    --restart-cycle 0 > "$TMPDIR_BASE/run_a.log" 2>&1 \
    || { echo "FAILED (run A). Log:"; tail -20 "$TMPDIR_BASE/run_a.log"; exit 1; }

# ── Run B: first half ────────────────────────────────────────────────────

echo "  [2/3] Running $HALF_CYCLES cycles (first half)..."
bash "$SIM_SH" "$INPUT" --no-build "${NP_FLAG[@]}" \
    --cycles "$HALF_CYCLES" -o "$DIR_B" \
    --restart-cycle "$HALF_CYCLES" > "$TMPDIR_BASE/run_b1.log" 2>&1 \
    || { echo "FAILED (run B1). Log:"; tail -20 "$TMPDIR_BASE/run_b1.log"; exit 1; }

# ── Run B: restart second half ───────────────────────────────────────────

echo "  [3/3] Restarting for $HALF_CYCLES more cycles..."
bash "$SIM_SH" "$DIR_B" --no-build "${NP_FLAG[@]}" \
    --cycles "$TOTAL_CYCLES" > "$TMPDIR_BASE/run_b2.log" 2>&1 \
    || { echo "FAILED (run B2). Log:"; tail -20 "$TMPDIR_BASE/run_b2.log"; exit 1; }

# ── Compare ──────────────────────────────────────────────────────────────

echo ""
echo "  Comparing ConservedQuantities.txt..."

RESULT=$("$PYTHON" - "$DIR_A" "$DIR_B" "$TOL" <<PYCOMPARE
import sys, os
sys.path.insert(0, os.path.join('${PROJECT_DIR}', 'scripts'))
from plot_utils import parse_conserved_quantities

dir_a, dir_b, tol_str = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(tol_str)

cq_a = parse_conserved_quantities(dir_a)
cq_b = parse_conserved_quantities(dir_b)

if not cq_a or not cq_b:
    print("FAIL: Could not read ConservedQuantities.txt")
    sys.exit(1)

# Compare energy at each cycle that exists in both
cycles_a = {c: i for i, c in enumerate(cq_a['cycle'])}
cycles_b = {c: i for i, c in enumerate(cq_b['cycle'])}
common = sorted(set(cycles_a) & set(cycles_b))

if not common:
    print("FAIL: No common cycles between runs")
    sys.exit(1)

max_diff = 0.0
worst_cycle = 0
for c in common:
    ea = cq_a['total_energy'][cycles_a[c]]
    eb = cq_b['total_energy'][cycles_b[c]]
    diff = abs(ea - eb)
    if diff > max_diff:
        max_diff = diff
        worst_cycle = c

passed = max_diff < tol
status = "PASS" if passed else "FAIL"
print(f"{status}: max |E_A - E_B| = {max_diff:.2e} at cycle {worst_cycle} (tol={tol})")
print(f"  Compared {len(common)} common cycles ({common[0]}..{common[-1]})")
sys.exit(0 if passed else 1)
PYCOMPARE
) || { echo "  $RESULT"; exit 1; }

echo "  $RESULT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Restart test PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
