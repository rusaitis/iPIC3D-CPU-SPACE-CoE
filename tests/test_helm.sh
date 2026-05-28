#!/usr/bin/env bash
#
# test_helm.sh — Correctness gate for the scalar-Helmholtz preconditioner.
#
# A preconditioner must NOT change the converged field — only the iteration count.
# This runs PETSc with PrecType=None (baseline) and PrecType=Helmholtz (both inner
# engines, AMG and Chebyshev) on the same case, at np=1 and np=4, and lets test.sh's
# validate.py (run with --strict via --field-output) enforce:
#   - energy conservation     (|ΔE/E₀| < 1e-6)
#   - all cycles converged
#   - cross-solver L2 vs ref   (< 1e-10)  ← the decisive "PC didn't bias the solve" check
# Finally it reports per-cycle KSP iterations per solver (informational; the iteration
# WIN is a mesh-scaling property — see tests/bench_preconditioners.sh).
#
# Usage: bash tests/test_helm.sh [--exe PATH] [--grid N] [--cycles N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXE="${EXE:-$PROJECT_DIR/build/iPIC3D}"
GRID=32
CYCLES=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exe)    EXE="$2";    shift 2 ;;
        --grid)   GRID="$2";   shift 2 ;;
        --cycles) CYCLES="$2"; shift 2 ;;
        --help|-h) sed -n '2,/^$/{ s/^# *//; p }' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

TEST_SH="$SCRIPT_DIR/test.sh"
PYTHON="${PYTHON:-python}"

SOLVER_ARGS=(
    --add-solver "PETSc_none:PETSc:-prec None"
    --add-solver "Helmholtz_amg:PETSc:-prec Helmholtz -helm-inner AMG"
    --add-solver "Helmholtz_cheb:PETSc:-prec Helmholtz -helm-inner Chebyshev"
)

for NP in 1 4; do
    echo ""
    echo "================================================================"
    echo "  Helmholtz correctness gate: ${GRID}x${GRID}, $CYCLES cycles, np=$NP"
    echo "================================================================"
    if [[ "$NP" -eq 1 ]]; then XL=1; YL=1; else XL=2; YL=2; fi
    bash "$TEST_SH" --exe "$EXE" \
        --grid "$GRID" "$GRID" --cycles "$CYCLES" --np "$NP" \
        --field-output "$CYCLES" \
        --name "helm_np${NP}" --clean \
        "${SOLVER_ARGS[@]}"

    # Iteration report (None baseline vs both Helmholtz engines)
    OUT="$SCRIPT_DIR/test_output_helm_np${NP}"
    LOGS=()
    for lbl in PETSc_none Helmholtz_amg Helmholtz_cheb; do
        for f in "$OUT"/${lbl}*.log; do [[ -f "$f" ]] && LOGS+=("$f"); done
    done
    if [[ ${#LOGS[@]} -gt 0 ]]; then
        "$PYTHON" "$PROJECT_DIR/scripts/check_prec_iters.py" "${LOGS[@]}" --assert-converged || true
    fi
done

echo ""
echo "Helmholtz correctness gate complete (energy + convergence + L2 enforced by validate.py)."
