#!/usr/bin/env bash
#
# bench_krylov.sh — Benchmark the Krylov backend, NOT the preconditioner.
#
# Compares three matrix-free solvers on field-solve wall-clock (ms/cycle),
# all unpreconditioned (PCNONE), same restart (40) and tolerance (input GMREStol):
#   - GMRES        iPIC3D's built-in modified-Gram-Schmidt GMRES
#   - PETScGMRES   PETSc KSPGMRES  (-pc_type none)
#   - PETScFGMRES  PETSc KSPFGMRES (-pc_type none)
#
# Three runs:
#   1. Double_Harris grid sweep 100->200 (np=8) — low iters (~26), the
#      historical "PETSc is 5-15% faster" baseline; shows the gap vs field size.
#   2. WhistlerPacket at its native 256x4x4 grid (np=1) — a different input with
#      no MPI reductions, so the orthogonalization cost shows in serial too.
#   3. dense Double_Harris (rhoINIT=100, generated on the fly) at 100x100 np=8 —
#      the high-iteration regime (~600 iters, ~15 GMRES(40) restarts).
#
# The "Field solver : <s>" total comes from the same unified per-cycle timer
# (iPIC3D.cpp) for every backend, so field_s/cycle is apples-to-apples.
#
# Usage:
#   bash tests/bench_krylov.sh [OPTIONS]
#
# Options:
#   --exe PATH     iPIC3D executable (default: build/iPIC3D)
#   --cycles N     Cycles per run for runs 1-2 (default: 20)
#   --dense-cycles N  Cycles for the high-iteration dense run (default: 8)
#   --avg N        Timed repetitions per solver, averaged (default: 3)
#   --warmup N     Warm-up cycles before timing (default: 2)
#   --dh-only      Run only the Double_Harris grid sweep
#   --whistler-only Run only the WhistlerPacket run
#   --no-dense     Skip the (slow) high-iteration dense run
#   --no-plot      Skip the comparison figure
#   --dry-run      Print commands without running
#   (extra args are passed through to test.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SH="$SCRIPT_DIR/test.sh"
PLOT_PY="$PROJECT_DIR/scripts/plot_krylov_bench.py"
PYTHON="${PYTHON:-python3}"

# ── Defaults ──────────────────────────────────────────────────────
EXE="$PROJECT_DIR/build/iPIC3D"
CYCLES=20
DENSE_CYCLES=8
DENSE_RHO=100
AVG=3
WARMUP=2
RUN_DH=true
RUN_WH=true
RUN_DENSE=true
NO_PLOT=""
DRY_RUN=""
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exe)           EXE="$2"; shift 2 ;;
        --cycles)        CYCLES="$2"; shift 2 ;;
        --dense-cycles)  DENSE_CYCLES="$2"; shift 2 ;;
        --avg)           AVG="$2"; shift 2 ;;
        --warmup)        WARMUP="$2"; shift 2 ;;
        --dh-only)       RUN_WH=false; RUN_DENSE=false; shift ;;
        --whistler-only) RUN_DH=false; RUN_DENSE=false; shift ;;
        --no-dense)      RUN_DENSE=false; shift ;;
        --no-plot)       NO_PLOT="--no-plot"; shift ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# *//; s/^#//; p }' "$0"
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# The two PETSc backends: unpreconditioned, restart matched to the native GMRES(40).
ADD_PETSC=(
    --add-solver "PETScGMRES:PETSc:-ksp_type gmres -pc_type none -ksp_gmres_restart 40 -ksp_max_it 1000"
    --add-solver "PETScFGMRES:PETSc:-ksp_type fgmres -pc_type none -ksp_gmres_restart 40 -ksp_max_it 1000"
)
SOLVER_FILTER="GMRES,PETScGMRES,PETScFGMRES"

DH_CSV="$SCRIPT_DIR/test_output_krylov_dh/results.csv"
WH_CSV="$SCRIPT_DIR/test_output_krylov_whistler/results.csv"
DENSE_CSV="$SCRIPT_DIR/test_output_krylov_dense/results.csv"

# ── Run 1: Double_Harris grid sweep (np=8, low iters) ─────────────
run_dh() {
    echo ""
    echo "================================================================"
    echo "  Run 1: Double_Harris grid sweep 100->200 (np=8, $CYCLES cycles)"
    echo "================================================================"
    bash "$TEST_SH" --exe "$EXE" \
        --input "$PROJECT_DIR/inputfiles/Double_Harris.inp" --np 8 \
        --grid-min 100 --grid-max 200 --grid-step 50 \
        --cycles "$CYCLES" --avg "$AVG" --warmup "$WARMUP" \
        --clean --no-plot --name krylov_dh \
        "${ADD_PETSC[@]}" --solvers "$SOLVER_FILTER" \
        $DRY_RUN ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

# ── Run 2: WhistlerPacket at native grid (np=1, high iters) ───────
run_whistler() {
    echo ""
    echo "================================================================"
    echo "  Run 2: WhistlerPacket native 256x4x4 (np=1, $CYCLES cycles)"
    echo "================================================================"
    # Pin the native 256x4x4 grid explicitly: test.sh's native-grid auto-read
    # strips non-digits from the `nxc = 256  # k·dx ≈ 0.39` line and would read 256039.
    bash "$TEST_SH" --exe "$EXE" \
        --input "$PROJECT_DIR/inputfiles/WhistlerPacket.inp" --np 1 \
        --grid 256 4 4 \
        --cycles "$CYCLES" --avg "$AVG" --warmup "$WARMUP" \
        --clean --no-plot --name krylov_whistler \
        "${ADD_PETSC[@]}" --solvers "$SOLVER_FILTER" \
        $DRY_RUN ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

# ── Run 3: dense Double_Harris (np=8, high iters ~600) ────────────
# Generated on the fly from Double_Harris.inp by setting rhoINIT=100 across all
# species — the mass term dominates the operator, ballooning the iteration count.
run_dense() {
    echo ""
    echo "================================================================"
    echo "  Run 3: dense Double_Harris rhoINIT=$DENSE_RHO, 100x100 (np=8, $DENSE_CYCLES cycles)"
    echo "================================================================"
    local base="$PROJECT_DIR/inputfiles/Double_Harris.inp"
    local nsp rhostr tmp s
    nsp=$(grep -E '^ns[[:space:]]*=' "$base" | head -1 | sed -E 's/.*=[[:space:]]*([0-9]+).*/\1/')
    rhostr=""
    for ((s=0; s<nsp; s++)); do rhostr="$rhostr ${DENSE_RHO}.0"; done
    tmp="$(mktemp -t dh_dense.XXXXXX)"
    sed -E "s|^rhoINIT[[:space:]]*=.*|rhoINIT =$rhostr|" "$base" > "$tmp"
    echo "  (dense input: rhoINIT =$rhostr)"
    bash "$TEST_SH" --exe "$EXE" \
        --input "$tmp" --np 8 --grid 100 100 \
        --cycles "$DENSE_CYCLES" --avg "$AVG" --warmup "$WARMUP" \
        --clean --no-plot --name krylov_dense \
        "${ADD_PETSC[@]}" --solvers "$SOLVER_FILTER" \
        $DRY_RUN ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    rm -f "$tmp"
}

[[ "$RUN_DH" == true ]] && run_dh
[[ "$RUN_WH" == true ]] && run_whistler
[[ "$RUN_DENSE" == true ]] && run_dense

# ── Summary + plot ────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results"
echo "================================================================"
for csv in "$DH_CSV" "$WH_CSV" "$DENSE_CSV"; do
    [[ -f "$csv" ]] || continue
    echo ""
    echo "$csv"
    head -1 "$csv"
    tail -n +2 "$csv"
done

if [[ -z "$NO_PLOT" && -z "$DRY_RUN" && -f "$PLOT_PY" ]]; then
    echo ""
    echo "Plotting comparison figure..."
    plot_args=(--dh-csv "$DH_CSV")
    [[ -f "$WH_CSV" ]]    && plot_args+=(--case "WhistlerPacket (np=1)=$WH_CSV")
    [[ -f "$DENSE_CSV" ]] && plot_args+=(--case "dense DH ρ=100 (np=8)=$DENSE_CSV")
    "$PYTHON" "$PLOT_PY" "${plot_args[@]}" \
        || echo "WARNING: plotting failed (non-fatal)." >&2
fi

echo ""
echo "Done."
