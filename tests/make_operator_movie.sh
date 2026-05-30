#!/usr/bin/env bash
#
# make_operator_movie.sh — render an "operator movie" (fields + per-node P
# stiffness + Ritz eigenvalues) from a STANDARD input file, with the movie
# overrides applied on the fly. No committed "_movie.inp" duplicates.
#
# A temp .inp is derived from the base input via make_np_variant (tests/common.sh):
# forced to np=1 (so the P block index maps directly to the grid), PETSc solver
# with an explicit-matrix dump every PrecDumpCycle cycles, and phdf5 output. The
# sim runs with `-pc_type none` so the Krylov solve stays matrix-free on the bare
# operator A (production regime) — the Ritz eigenvalues are then A's. The log is
# written OUTSIDE SaveDirName because iPIC3D wipes that directory at startup.
#
# Usage:
#   bash tests/make_operator_movie.sh --inp inputfiles/Double_Harris.inp \
#        --save-dir output/data_reconnection_movie --mode reconnection \
#        [--cycles 500] [--field-output 5] [--out movie.mp4] [--log run.log] \
#        [--skip-sim] [--fps N] [--stride N] [--matrix-view stiffmap|blocknorm] [--light]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"   # make_np_variant

PYTHON="${PYTHON:-python3}"
EXE="${EXE:-${IPIC3D_BUILD_DIR:-build}/iPIC3D}"

INP=""
MODE="auto"
SAVE_DIR=""
CYCLES=500
FIELD_OUT=5
OUT=""
LOG=""
GEN_INP=""
SKIP_SIM=0
EXTRA=()   # forwarded to movie_operator.py

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inp)          INP="$2";        shift 2 ;;
        --mode)         MODE="$2";       shift 2 ;;
        --save-dir)     SAVE_DIR="$2";   shift 2 ;;
        --cycles)       CYCLES="$2";     shift 2 ;;
        --field-output) FIELD_OUT="$2";  shift 2 ;;
        --out)          OUT="$2";        shift 2 ;;
        --log)          LOG="$2";        shift 2 ;;
        --exe)          EXE="$2";        shift 2 ;;
        --skip-sim)     SKIP_SIM=1;      shift ;;
        --fps)          EXTRA+=(--fps "$2");          shift 2 ;;
        --stride)       EXTRA+=(--stride "$2");       shift 2 ;;
        --matrix-view)  EXTRA+=(--matrix-view "$2");  shift 2 ;;
        --light)        EXTRA+=(--light);             shift ;;
        --help|-h)
            sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$INP" ]]      || { echo "ERROR: --inp is required (a standard input file)" >&2; exit 1; }
[[ -f "$INP" ]]      || { echo "ERROR: input file not found: $INP" >&2; exit 1; }
[[ -n "$SAVE_DIR" ]] || { echo "ERROR: --save-dir is required" >&2; exit 1; }
[[ -n "$OUT" ]]      || OUT="output/operator_movie.mp4"
[[ -n "$LOG" ]]      || LOG="output/$(basename "${OUT%.*}").log"
GEN_INP="output/$(basename "${OUT%.*}").inp"   # derived input (kept for the render + inspection)

command -v ffmpeg &>/dev/null || {
    echo "ERROR: ffmpeg is required but not found. Install with: brew install ffmpeg" >&2
    exit 1
}

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")" "$(dirname "$GEN_INP")"

# Derive the movie input from the standard file (np=1 + PETSc matrix dump + phdf5).
# make_np_variant sets XLEN/YLEN/ZLEN, SaveDirName/RestartDirName, SimulationName
# (= basename of SAVE_DIR), then appends these last-wins KEY=VAL overrides.
echo "Deriving $GEN_INP from $INP (np=1, PETSc, PrecDumpCycle=$FIELD_OUT)"
make_np_variant "$INP" "$GEN_INP" 1 1 1 "$SAVE_DIR" \
    "ncycles=$CYCLES" \
    "FieldOutputCycle=$FIELD_OUT" \
    "PrecDumpCycle=$FIELD_OUT" \
    "ParticlesOutputCycle=0" \
    "WriteMethod=phdf5" \
    "FieldOutputTag=B + E" \
    "SolverType=PETSc" \
    "PrecMatrix=true" \
    "PrecDiagnostics=true"

if [[ "$SKIP_SIM" -eq 0 ]]; then
    echo "Running $GEN_INP (np=1) → $SAVE_DIR   [log: $LOG]"
    mpirun -np 1 "$EXE" "$GEN_INP" \
        -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor \
        > "$LOG" 2>&1
    echo "Simulation finished."
else
    echo "Skipping simulation (--skip-sim); using existing $SAVE_DIR + $LOG"
fi

echo "Rendering operator movie → $OUT"
"$PYTHON" "$ROOT_DIR/scripts/movie_operator.py" "$SAVE_DIR" \
    --inp "$GEN_INP" --log "$LOG" --mode "$MODE" -o "$OUT" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done: $OUT"
echo "(P dumps in $SAVE_DIR can be removed once happy: rm $SAVE_DIR/*_P_cycle*.bin)"
