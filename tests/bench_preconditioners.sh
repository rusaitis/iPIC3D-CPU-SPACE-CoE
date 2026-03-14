#!/usr/bin/env bash
#
# bench_preconditioners.sh — Benchmark AMG preconditioner configurations
#
# Two studies:
#   1. Preconditioner sweep at 100x100 (GAMG, HYPRE, FieldSplit, etc.)
#   2. Mesh scaling 50->300 (GAMG vs unpreconditioned — AMG should stay bounded)
#
# Usage:
#   bash tests/bench_preconditioners.sh [OPTIONS]
#
# Options:
#   --exe PATH     Path to iPIC3D executable (default: build/iPIC3D)
#   --sweep        Run only the preconditioner sweep study
#   --scaling      Run only the mesh scaling study
#   --np N         MPI processes (default: 8)
#   --cycles N     Cycles per run (default: 20)
#   --no-plot      Skip plotting
#   --dry-run      Print commands without running
#   (no study flag = run both)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────
EXE="$PROJECT_DIR/build/iPIC3D"
INPUT="$PROJECT_DIR/inputfiles/Double_Harris_PETSc_prec.inp"
NP=8
CYCLES=20
RUN_SWEEP=false
RUN_SCALING=false
NO_PLOT=""
DRY_RUN=""
EXTRA_ARGS=()

# ── Argument parsing ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exe)       EXE="$2";    shift 2 ;;
        --sweep)     RUN_SWEEP=true;  shift ;;
        --scaling)   RUN_SCALING=true; shift ;;
        --np)        NP="$2";     shift 2 ;;
        --cycles)    CYCLES="$2"; shift 2 ;;
        --no-plot)   NO_PLOT="--no-plot"; shift ;;
        --dry-run)   DRY_RUN="--dry-run"; shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# *//; p }' "$0"
            exit 0 ;;
        *)  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Default: run both studies
if [[ "$RUN_SWEEP" == false && "$RUN_SCALING" == false ]]; then
    RUN_SWEEP=true
    RUN_SCALING=true
fi

TEST_SH="$SCRIPT_DIR/test.sh"

# ── Study 1: Preconditioner sweep ─────────────────────────────────
# Fixed 100x100 grid, 20 cycles, compare all PC types.
# GMRES and PETSc_gmres (default ILU) are auto-added by test.sh.
run_sweep() {
    echo ""
    echo "================================================================"
    echo "  Study 1: Preconditioner sweep (100x100, $CYCLES cycles, $NP MPI)"
    echo "================================================================"
    echo ""

    bash "$TEST_SH" --exe "$EXE" \
        --input "$INPUT" \
        --grid 100 100 --cycles "$CYCLES" --np "$NP" \
        --name pc_sweep --clean \
        --add-solver "PCNONE:PETSc:-pc_type none" \
        --add-solver "Jacobi:PETSc:-pc_type jacobi" \
        --add-solver "GAMG:PETSc:-pc_type gamg" \
        --add-solver "GAMG_tuned:PETSc:-pc_type gamg -pc_gamg_threshold -1.0 -pc_gamg_agg_nsmooths 3 -mg_levels_ksp_max_it 2" \
        --add-solver "HYPRE:PETSc:-pc_type hypre -pc_hypre_type boomeramg" \
        --add-solver "HYPRE_nodal:PETSc:-pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 2" \
        --add-solver "FieldSplit:PETSc:-pc_type fieldsplit -pc_fieldsplit_block_size 3 -pc_fieldsplit_type symmetric_multiplicative -fieldsplit_pc_type hypre" \
        $NO_PLOT $DRY_RUN "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ── Study 2: Mesh scaling ─────────────────────────────────────────
# Sweep grid sizes 50->300 to show AMG iteration counts stay bounded
# while unpreconditioned GMRES grows with mesh refinement.
run_scaling() {
    echo ""
    echo "================================================================"
    echo "  Study 2: Mesh scaling (50->300, $CYCLES cycles, $NP MPI)"
    echo "================================================================"
    echo ""

    bash "$TEST_SH" --exe "$EXE" \
        --input "$INPUT" \
        --grid-min 50 --grid-max 300 --grid-step 50 \
        --cycles "$CYCLES" --np "$NP" \
        --name pc_scaling --clean \
        --solvers GMRES,PETSc_gmres,PCNONE,GAMG \
        --add-solver "PCNONE:PETSc:-pc_type none" \
        --add-solver "GAMG:PETSc:-pc_type gamg" \
        $NO_PLOT $DRY_RUN "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ── Run selected studies ──────────────────────────────────────────
if [[ "$RUN_SWEEP" == true ]]; then
    run_sweep
fi

if [[ "$RUN_SCALING" == true ]]; then
    run_scaling
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Results"
echo "================================================================"

SWEEP_CSV="$SCRIPT_DIR/test_output_pc_sweep/results.csv"
SCALING_CSV="$SCRIPT_DIR/test_output_pc_scaling/results.csv"

if [[ -f "$SWEEP_CSV" ]]; then
    echo ""
    echo "Preconditioner sweep: $SWEEP_CSV"
    # Print header + data, extracting solver label and avg iterations
    head -1 "$SWEEP_CSV"
    tail -n +2 "$SWEEP_CSV" | sort -t, -k1,1
fi

if [[ -f "$SCALING_CSV" ]]; then
    echo ""
    echo "Mesh scaling: $SCALING_CSV"
    head -1 "$SCALING_CSV"
    tail -n +2 "$SCALING_CSV" | sort -t, -k1,1
fi

echo ""
echo "Done."
