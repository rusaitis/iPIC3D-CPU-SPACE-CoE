#!/usr/bin/env bash
set -euo pipefail

# sim.sh — Build and run an iPIC3D simulation
#
# Usage:
#   pixi run sim -- <input.inp> [options]     Fresh run (builds first)
#   pixi run sim -- <output_dir> [options]    Restart from checkpoint
#   pixi run sim                              Defaults to Double_Harris.inp
#
# The first positional argument determines the mode:
#   - A file (.inp)  → fresh run: build, run, archive .inp to output dir
#   - A directory     → restart: skip build, resume from checkpoint files
#   - (none)          → fresh run with inputfiles/Double_Harris.inp
#
# Options:
#   -np N            Override MPI process count (default: auto from XLEN*YLEN*ZLEN)
#   -o DIR           Override output directory (SaveDirName)
#   --solver TYPE    Override SolverType (GMRES / PETSc)
#   --cycles N       Override ncycles
#   --restart-cycle N Override RestartOutputCycle (0 = final only)
#   --input FILE     Use specific .inp instead of archived copy (restart mode)
#   --log FILE       Override log file path (default: <output_dir>/run.log)
#   --no-log         Disable log capture (uses exec mpirun for clean signals)
#   --diag           Enable PETSc diagnostics: PrecMatrix, KSP monitor, eigenvalues
#   --no-build       Skip build step on fresh run
#   --dry-run        Print the mpirun command without executing
#   -h, --help       Show this help
#
# PETSc flags (-ksp_*, -pc_*, etc.) pass through to iPIC3D:
#   pixi run sim -- input.inp --solver PETSc -ksp_monitor -ksp_type bcgs
#
# Examples:
#   pixi run sim -- inputfiles/Double_Harris.inp
#   pixi run sim -- inputfiles/Double_Harris.inp -o output/my_run
#   pixi run sim -- output/data_reconnection
#   pixi run sim -- output/data_reconnection --cycles 500 --solver PETSc
#   pixi run sim -- output/data_reconnection -o output/petsc_run --solver PETSc
#   pixi run sim -- inputfiles/Double_Harris.inp --solver PETSc --diag
#
# END_HELP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared helpers (compute_topology, etc.)
source "${PROJECT_DIR}/tests/common.sh" || die "Failed to source tests/common.sh"

DEFAULT_INPUT="inputfiles/Double_Harris.inp"

# ─── Helpers ──────────────────────────────────────────────────────────

show_help() {
    sed -n '/^# sim.sh/,/^# END_HELP/{ s/^# \{0,1\}//; p; }' "${BASH_SOURCE[0]}"
}

# Extract a value from an .inp file: parse_inp KEY file
# Returns the first whitespace-delimited token after "KEY = ..." (empty if not found)
parse_inp() {
    local key="$1" file="$2"
    { grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" || true; } \
        | head -1 \
        | sed 's/.*=[[:space:]]*//' \
        | awk '{print $1}'
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ─── Defaults ─────────────────────────────────────────────────────────

BUILD_DIR="${IPIC3D_BUILD_DIR:-build}"
EXECUTABLE="${PROJECT_DIR}/${BUILD_DIR}/iPIC3D"

NP_OVERRIDE=""
OUTPUT_DIR=""
SOLVER_OVERRIDE=""
CYCLES_OVERRIDE=""
RESTART_CYCLE_OVERRIDE=""
INPUT_OVERRIDE=""
LOG_FILE=""
LOG_EXPLICIT=false
NO_LOG=false
DIAG_MODE=false
DO_BUILD=true
DRY_RUN=false

MODE=""           # "fresh" or "restart"
TARGET=""         # the file or dir argument
RESTART_DIR=""    # set in restart mode
INPUT_FILE=""     # resolved .inp path
PASSTHROUGH_ARGS=()

# ─── Parse arguments ──────────────────────────────────────────────────

# Consume leading -- from pixi (pixi run sim -- args)
[[ "${1:-}" == "--" ]] && shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -np)         NP_OVERRIDE="${2:?'-np requires a value'}"; shift 2 ;;
        -o)          OUTPUT_DIR="${2:?'-o requires a value'}"; shift 2 ;;
        --solver)    SOLVER_OVERRIDE="${2:?'--solver requires a value'}"; shift 2 ;;
        --cycles)    CYCLES_OVERRIDE="${2:?'--cycles requires a value'}"; shift 2 ;;
        --restart-cycle) RESTART_CYCLE_OVERRIDE="${2:?'--restart-cycle requires a value'}"; shift 2 ;;
        --input)     INPUT_OVERRIDE="${2:?'--input requires a value'}"; shift 2 ;;
        --log)       LOG_FILE="${2:?'--log requires a filename'}"; LOG_EXPLICIT=true; shift 2 ;;
        --no-log)    NO_LOG=true; shift ;;
        --diag)      DIAG_MODE=true; shift ;;
        --no-build)  DO_BUILD=false; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   show_help; exit 0 ;;
        --)          shift; break ;;     # everything after -- goes to passthrough
        -*)          PASSTHROUGH_ARGS+=("$1"); shift ;;
        *)
            if [[ -z "$TARGET" ]]; then
                TARGET="$1"
            else
                PASSTHROUGH_ARGS+=("$1")
            fi
            shift ;;
    esac
done
# Remaining args after -- go to passthrough
PASSTHROUGH_ARGS+=("$@")

# ─── Mode detection ──────────────────────────────────────────────────

if [[ -z "$TARGET" ]]; then
    # No argument → default fresh run
    MODE="fresh"
    INPUT_FILE="${PROJECT_DIR}/${DEFAULT_INPUT}"
    echo "No input specified — defaulting to ${DEFAULT_INPUT}"
    [[ -f "$INPUT_FILE" ]] || die "Default input file not found: $INPUT_FILE"

elif [[ -f "$TARGET" ]]; then
    # File → fresh run
    MODE="fresh"
    INPUT_FILE="$TARGET"

elif [[ -d "$TARGET" ]]; then
    # Directory → restart
    MODE="restart"
    RESTART_DIR="$TARGET"
    DO_BUILD=false   # skip build on restart

    # Verify restart files exist
    [[ -f "${RESTART_DIR}/restart0.hdf" ]] \
        || die "No restart files found in ${RESTART_DIR}/ (expected restart0.hdf)"

    # Find the input file: --input override > archived .inp > default
    if [[ -n "$INPUT_OVERRIDE" ]]; then
        INPUT_FILE="$INPUT_OVERRIDE"
        [[ -f "$INPUT_FILE" ]] || die "Input file not found: $INPUT_FILE"
    else
        # Look for archived .inp in the restart dir
        archived_inp=""
        for f in "${RESTART_DIR}"/*.inp; do
            [[ -f "$f" ]] || continue
            archived_inp="$f"
            break
        done
        if [[ -n "$archived_inp" ]]; then
            INPUT_FILE="$archived_inp"
            echo "Using archived input: ${INPUT_FILE}"
        else
            INPUT_FILE="${PROJECT_DIR}/${DEFAULT_INPUT}"
            echo "No archived .inp found in ${RESTART_DIR}/ — defaulting to ${DEFAULT_INPUT}"
            [[ -f "$INPUT_FILE" ]] || die "Default input file not found: $INPUT_FILE"
        fi
    fi
else
    die "'${TARGET}' is neither a file nor a directory"
fi

# ─── Parse simulation config from .inp ────────────────────────────────

# Auto-detect np from XLEN * YLEN * ZLEN
XLEN=$(parse_inp "XLEN" "$INPUT_FILE")
YLEN=$(parse_inp "YLEN" "$INPUT_FILE")
ZLEN=$(parse_inp "ZLEN" "$INPUT_FILE")
XLEN="${XLEN:-1}"; YLEN="${YLEN:-1}"; ZLEN="${ZLEN:-1}"
NP=$(( XLEN * YLEN * ZLEN ))

# If -np overrides, auto-adjust MPI topology to match
if [[ -n "$NP_OVERRIDE" && "$NP_OVERRIDE" -ne "$NP" ]]; then
    NP="$NP_OVERRIDE"
    NP_XY=$(( NP / ZLEN ))
    if [[ $((NP_XY * ZLEN)) -ne $NP ]]; then
        die "np=$NP is not divisible by ZLEN=$ZLEN from input file"
    fi
    read -r XLEN YLEN <<< "$(compute_topology "$NP_XY")"
    echo "Adjusting topology: XLEN=${XLEN} YLEN=${YLEN} ZLEN=${ZLEN} for np=${NP}"
elif [[ -n "$NP_OVERRIDE" ]]; then
    NP="$NP_OVERRIDE"
fi

# Resolve output directory
if [[ -n "$OUTPUT_DIR" ]]; then
    SAVE_DIR="$OUTPUT_DIR"
elif [[ "$MODE" == "restart" && -z "$OUTPUT_DIR" ]]; then
    # Restart with no -o: write back to the same dir
    SAVE_DIR="$RESTART_DIR"
else
    SAVE_DIR=$(parse_inp "SaveDirName" "$INPUT_FILE")
    SAVE_DIR="${SAVE_DIR:-data}"
fi

# Resolve solver
SOLVER=$(parse_inp "SolverType" "$INPUT_FILE")
SOLVER="${SOLVER:-GMRES}"
[[ -n "$SOLVER_OVERRIDE" ]] && SOLVER="$SOLVER_OVERRIDE"

# Resolve cycles (for display only; override applied via sed)
NCYCLES=$(parse_inp "ncycles" "$INPUT_FILE")
NCYCLES="${NCYCLES:-100}"
[[ -n "$CYCLES_OVERRIDE" ]] && NCYCLES="$CYCLES_OVERRIDE"

# ─── Default logging ─────────────────────────────────────────────────

if [[ "$NO_LOG" == true ]]; then
    LOG_FILE=""
elif [[ "$LOG_EXPLICIT" == false ]]; then
    LOG_FILE="${SAVE_DIR}/run.log"
fi

# ─── Build ────────────────────────────────────────────────────────────

if [[ "$DO_BUILD" == true ]]; then
    if [[ ! -f "${PROJECT_DIR}/${BUILD_DIR}/CMakeCache.txt" ]]; then
        die "Build not configured (no ${BUILD_DIR}/CMakeCache.txt).
    Run 'pixi run build' first (or './build.sh')."
    fi
    echo "Building iPIC3D..."
    cmake --build "${PROJECT_DIR}/${BUILD_DIR}" --parallel
    echo ""
fi

# Verify executable
if [[ ! -x "$EXECUTABLE" ]]; then
    die "Executable not found: ${EXECUTABLE}
    Build first with: pixi run build"
fi

# ─── Prepare working .inp ─────────────────────────────────────────────

NEEDS_SED=false
SED_ARGS=()

# MPI topology override (when -np changed the factorization)
if [[ -n "$NP_OVERRIDE" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*XLEN.*=.*|XLEN                           = ${XLEN}|")
    SED_ARGS+=(-e "s|^[[:space:]]*YLEN.*=.*|YLEN                           = ${YLEN}|")
    SED_ARGS+=(-e "s|^[[:space:]]*ZLEN.*=.*|ZLEN                           = ${ZLEN}|")
fi

# Output dir override
if [[ -n "$OUTPUT_DIR" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*SaveDirName.*=.*|SaveDirName                    = ${SAVE_DIR}|")
    if [[ "$MODE" == "restart" ]]; then
        SED_ARGS+=(-e "s|^[[:space:]]*RestartDirName.*=.*|RestartDirName                 = ${RESTART_DIR}|")
    else
        SED_ARGS+=(-e "s|^[[:space:]]*RestartDirName.*=.*|RestartDirName                 = ${SAVE_DIR}|")
    fi
fi

# Restart to same dir: ensure SaveDirName/RestartDirName match the dir
if [[ "$MODE" == "restart" && -z "$OUTPUT_DIR" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*SaveDirName.*=.*|SaveDirName                    = ${RESTART_DIR}|")
    SED_ARGS+=(-e "s|^[[:space:]]*RestartDirName.*=.*|RestartDirName                 = ${RESTART_DIR}|")
fi

# Solver override
if [[ -n "$SOLVER_OVERRIDE" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*SolverType.*=.*|SolverType                     = ${SOLVER_OVERRIDE}|")
fi

# Cycles override
if [[ -n "$CYCLES_OVERRIDE" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*ncycles.*=.*|ncycles                        = ${CYCLES_OVERRIDE}|")
fi

# RestartOutputCycle override
if [[ -n "$RESTART_CYCLE_OVERRIDE" ]]; then
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*RestartOutputCycle.*=.*|RestartOutputCycle             = ${RESTART_CYCLE_OVERRIDE}|")
fi

# Diagnostics mode
if [[ "$DIAG_MODE" == true ]]; then
    if [[ "$SOLVER" != "PETSc" ]]; then
        echo "WARNING: --diag enables PETSc diagnostics but solver is ${SOLVER}."
        echo "         Use --solver PETSc to enable PETSc."
    fi
    NEEDS_SED=true
    SED_ARGS+=(-e "s|^[[:space:]]*PrecMatrix.*=.*|PrecMatrix                     = true|")
    SED_ARGS+=(-e "s|^[[:space:]]*PrecDiagnostics.*=.*|PrecDiagnostics                = true|")
    PASSTHROUGH_ARGS+=(-ksp_monitor -ksp_view_eigenvalues -ksp_view_singularvalues)
fi

# Original filename for archiving
INP_BASENAME="$(basename "$INPUT_FILE")"

CLEANUP_DIR=""
RUN_INPUT="$INPUT_FILE"

if [[ "$NEEDS_SED" == true ]]; then
    CLEANUP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_sim.XXXXXX")
    trap 'rm -rf "$CLEANUP_DIR"' EXIT

    RUN_INPUT="${CLEANUP_DIR}/${INP_BASENAME}"
    sed "${SED_ARGS[@]}" "$INPUT_FILE" > "$RUN_INPUT"

    # --diag: append PrecMatrix/PrecDiagnostics if not already in the .inp
    if [[ "$DIAG_MODE" == true ]]; then
        grep -q '^[[:space:]]*PrecMatrix' "$RUN_INPUT" \
            || echo "PrecMatrix                     = true" >> "$RUN_INPUT"
        grep -q '^[[:space:]]*PrecDiagnostics' "$RUN_INPUT" \
            || echo "PrecDiagnostics                = true" >> "$RUN_INPUT"
    fi
fi

# ─── OpenMPI detection ────────────────────────────────────────────────

command -v mpirun >/dev/null 2>&1 || die "mpirun not found in PATH"

MPIRUN_FLAGS=()
if mpirun --version 2>&1 | grep -qi "open.mpi"; then
    MPIRUN_FLAGS=(--oversubscribe)
fi

# Guard against NCYCLES=0 (would cause division by zero in progress bar)
[[ "$NCYCLES" -gt 0 ]] 2>/dev/null || NCYCLES=1

# ─── Assemble command ─────────────────────────────────────────────────

IPIC_ARGS=("$RUN_INPUT")
[[ "$MODE" == "restart" ]] && IPIC_ARGS+=("restart")
[[ -n "$SOLVER_OVERRIDE" ]] && IPIC_ARGS+=("-solver" "$SOLVER_OVERRIDE")
[[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]] && IPIC_ARGS+=("${PASSTHROUGH_ARGS[@]}")

# ─── Summary ──────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ "$MODE" == "restart" ]]; then
    echo "  iPIC3D — RESTART"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  From:     ${RESTART_DIR}"
    [[ "$SAVE_DIR" != "$RESTART_DIR" ]] && echo "  To:       ${SAVE_DIR}"
else
    echo "  iPIC3D — Fresh run"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
echo "  Input:    ${INP_BASENAME}"
echo "  Output:   ${SAVE_DIR}"
echo "  Solver:   ${SOLVER}"
echo "  Cycles:   ${NCYCLES}"
echo "  np:       ${NP} (XLEN=${XLEN} YLEN=${YLEN} ZLEN=${ZLEN})"
[[ -n "$LOG_FILE" ]] && echo "  Log:      ${LOG_FILE}"
[[ "$DIAG_MODE" == true ]] && echo "  Diag:     PrecMatrix + KSP monitor + eigenvalues"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  mpirun ${MPIRUN_FLAGS[*]:-} -np ${NP} ${EXECUTABLE} ${IPIC_ARGS[*]}"
echo ""

# ─── Run ──────────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == true ]]; then
    echo "(dry run — not executing)"
    exit 0
fi

# ─── Create output dir & archive .inp ─────────────────────────────────

mkdir -p "$SAVE_DIR"
cp "$RUN_INPUT" "${SAVE_DIR}/${INP_BASENAME}"

PYTHON="${PYTHON:-python3}"
EXTRACT_PROFILE="${SCRIPT_DIR}/extract_profile.py"

if [[ -n "$LOG_FILE" ]]; then
    # Write to a temp log during the run (iPIC3D's checkOutputFolder clears
    # SaveDirName on fresh runs, which would delete a log placed there early).
    TMP_LOG=$(mktemp "${TMPDIR:-/tmp}/ipic3d_run.XXXXXX")

    mpirun "${MPIRUN_FLAGS[@]}" -np "$NP" "$EXECUTABLE" "${IPIC_ARGS[@]}" \
        > "$TMP_LOG" 2>&1 &
    MPIRUN_PID=$!
    trap 'kill $MPIRUN_PID 2>/dev/null; wait $MPIRUN_PID 2>/dev/null; rm -f "$TMP_LOG"; exit 130' INT TERM

    # Progress bar helper: prints ████░░░░ N/M | Xs/cycle | ETA
    BAR_WIDTH=20
    show_progress() {
        local cycle="$1" total="$2" elapsed="$3"
        local filled=$(( cycle * BAR_WIDTH / total ))
        local empty=$(( BAR_WIDTH - filled ))
        local bar=""
        for ((b=0; b<filled; b++)); do bar+="█"; done
        for ((b=0; b<empty;  b++)); do bar+="░"; done
        local per_cycle eta_s eta
        per_cycle=$(awk "BEGIN{printf \"%.1f\", $elapsed/$cycle}")
        eta_s=$(awk "BEGIN{printf \"%d\", ($total-$cycle)*$elapsed/$cycle}")
        if [[ "$eta_s" -ge 60 ]]; then
            eta="$(( eta_s / 60 ))m $(( eta_s % 60 ))s"
        else
            eta="${eta_s}s"
        fi
        printf "\r  %s %d/%d | %ss/cycle | ETA %s   " "$bar" "$cycle" "$total" "$per_cycle" "$eta"
    }

    START_TIME=$(date +%s)
    prev_cycle=0
    while kill -0 "$MPIRUN_PID" 2>/dev/null; do
        cycle=$(grep -c '=================== Cycle' "$TMP_LOG" 2>/dev/null) || cycle=0
        if [[ "$cycle" -gt 0 && "$cycle" -ne "$prev_cycle" ]]; then
            now=$(date +%s)
            elapsed=$((now - START_TIME))
            show_progress "$cycle" "$NCYCLES" "$elapsed"
            prev_cycle=$cycle
        fi
        sleep 2
    done
    wait "$MPIRUN_PID"
    RUN_EXIT=$?

    # Final status line
    now=$(date +%s); elapsed=$((now - START_TIME))
    cycle=$(grep -c '=================== Cycle' "$TMP_LOG" 2>/dev/null) || cycle=0
    if [[ $RUN_EXIT -eq 0 ]]; then
        printf "\r  "
        for ((b=0; b<BAR_WIDTH; b++)); do printf "█"; done
        printf " %d/%d — done in %dm %ds                \n" "$cycle" "$NCYCLES" $((elapsed/60)) $((elapsed%60))
    else
        printf "\r  FAILED at cycle ~%d — see %s                \n" "$cycle" "$LOG_FILE"
        tail -5 "$TMP_LOG"
    fi

    # Move log to final location and extract profile
    mkdir -p "$(dirname "$LOG_FILE")"
    cat "$TMP_LOG" >> "$LOG_FILE"
    rm -f "$TMP_LOG"

    if [[ -f "$LOG_FILE" && -f "$EXTRACT_PROFILE" ]]; then
        PROFILE_CSV="${SAVE_DIR}/profile_run.csv"
        "$PYTHON" "$EXTRACT_PROFILE" "$LOG_FILE" "$PROFILE_CSV" 2>/dev/null || true
    fi

    exit "$RUN_EXIT"
else
    exec mpirun "${MPIRUN_FLAGS[@]}" -np "$NP" "$EXECUTABLE" "${IPIC_ARGS[@]}"
fi
