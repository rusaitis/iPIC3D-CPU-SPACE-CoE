#!/usr/bin/env bash
#
# test.sh — Solver regression test: compare field solver configurations
#
# Auto-detects PETSc availability from the build. When PETSc is present,
# runs GMRES vs PETSc comparison; otherwise runs GMRES only.
#
# Two modes:
#   Single-grid (default): Runs all solvers on one grid config.
#     Outputs: convergence info, timing breakdown, comparison table, CSV, plot.
#   Scaling (--grid-min/--grid-max): Sweeps grid sizes.
#     Outputs: summary table, CSV, plot.
#
# Usage:
#   # Single-grid mode
#   ./tests/test.sh [--np N] [--cycles N] [--grid NXC NYC [NZC]] [--topo X Y [Z]] ...
#
#   # Scaling mode
#   ./tests/test.sh --grid-min 50 --grid-max 400 [--grid-step N] ...
#
# Requirements:
#   - Build iPIC3D:  pixi run build  (or ./build.sh)
#   - MPI available (mpirun)
#   - python3 (for statistics, plotting, convergence parsing; override with PYTHON env var)
#   - Bash 4.0+ (brew install bash on macOS)

# ---- Shell guard (POSIX-compatible — must run before set -euo pipefail) ----
if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: This script requires bash. Run with: bash $0" >&2
    echo "       (current shell: $(ps -p $$ -o comm= 2>/dev/null || echo unknown))" >&2
    exit 1
fi

set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: Bash 4.0+ required (found ${BASH_VERSION})." >&2
    echo "       macOS: brew install bash" >&2
    echo "       Linux: sudo apt install bash  (or equivalent)" >&2
    exit 1
fi

# ========================= Section 1: Setup ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
source "$SCRIPT_DIR/common.sh"

# ========================= Section 2: Defaults =============================
EXECUTABLE="$PROJECT_DIR/build/iPIC3D"
INPUT_FILE="$PROJECT_DIR/inputfiles/Double_Harris.inp"
NP=8
CYCLES=""
CYCLES_SET_EXPLICITLY=false
AVG=1
COOLDOWN=0
NXC=""
NYC=""
NZC=""
XLEN=""
YLEN=""
ZLEN=""
GRID_MIN=""
GRID_MAX=""
GRID_STEP=50
DRY_RUN=false
RANDOMIZE=false
ALL_SOLVERS=false
NAME_TAG=""
FIELD_OUTPUT=""
PARTICLE_OUTPUT=""
DIAG_OUTPUT=""
OUTPUT_DIR=""
WRITE_METHOD="phdf5"
MAKE_MOVIE=false
CLEAN_OUTPUT=false
TIMEOUT=""
SOLVER_FILTER=""
VERBOSE=true
WARMUP=0
NO_PLOT=false
NO_VALIDATE=false
NO_COLOR_FLAG=false
BASELINE=false
PETSC_OPTIONS=""
TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_test.XXXXXX") \
    || { echo "ERROR: Failed to create temp directory" >&2; exit 1; }
PYTHON="${PYTHON:-python3}"

# MPI flags: --oversubscribe is needed for OpenMPI 5+ when np > physical cores
# Only add the flag for OpenMPI — MPICH/Intel MPI/Cray don't recognize it.
MPIRUN_FLAGS=()
if mpirun --version 2>&1 | grep -qi "open.mpi"; then
    MPIRUN_FLAGS=(--oversubscribe)
fi

# ========================= Solver Configuration ============================
#
# Each solver is defined by three parallel arrays:
#   SOLVER_LABELS  — short unique label used in filenames and tables
#   SOLVER_TYPES   — runtime solver type passed to -solver flag ("GMRES" or "PETSc")
#   SOLVER_EXTRA   — extra CLI args appended after -solver (PETSc KSP options, etc.)
#
# To add a new solver variant:
#   1. Append to all three arrays below (keep them in sync)
#   2. Guard behind ALL_SOLVERS or a new flag if experimental
#
# Default: GMRES only. PETSc solvers added below if available.
SOLVER_LABELS=("GMRES")
SOLVER_TYPES=( "GMRES")
SOLVER_EXTRA=( "")
ADD_SOLVER_SPECS=()
REF_SOLVER_LABEL=""

# ========================= Section 3: Argument parsing =====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)
            [[ $# -ge 2 ]] || { echo "Error: --np requires a positive integer" >&2; exit 2; }
            validate_pos_int "--np" "$2"
            NP="$2"; shift 2 ;;
        --cycles)
            [[ $# -ge 2 ]] || { echo "Error: --cycles requires a positive integer" >&2; exit 2; }
            validate_pos_int "--cycles" "$2"
            CYCLES="$2"; CYCLES_SET_EXPLICITLY=true; shift 2 ;;
        --avg)
            [[ $# -ge 2 ]] || { echo "Error: --avg requires a positive integer" >&2; exit 2; }
            validate_pos_int "--avg" "$2"
            AVG="$2"; shift 2 ;;
        --cooldown)
            [[ $# -ge 2 ]] || { echo "Error: --cooldown requires a non-negative integer" >&2; exit 2; }
            validate_nonneg_int "--cooldown" "$2"
            COOLDOWN="$2"; shift 2 ;;
        --grid)
            [[ $# -ge 3 && "$2" =~ ^[0-9]+$ && "$2" -gt 0 && "$3" =~ ^[0-9]+$ && "$3" -gt 0 ]] \
                || { echo "Error: --grid requires two positive integers: NXC NYC [NZC]" >&2; exit 2; }
            NXC="$2"; NYC="$3"
            if [[ $# -ge 4 && "$4" =~ ^[0-9]+$ && "$4" -gt 0 ]]; then
                NZC="$4"; shift 4
            else
                shift 3
            fi ;;
        --topo)
            [[ $# -ge 3 && "$2" =~ ^[0-9]+$ && "$2" -gt 0 && "$3" =~ ^[0-9]+$ && "$3" -gt 0 ]] \
                || { echo "Error: --topo requires two positive integers: XLEN YLEN [ZLEN]" >&2; exit 2; }
            XLEN="$2"; YLEN="$3"
            if [[ $# -ge 4 && "$4" =~ ^[0-9]+$ && "$4" -gt 0 ]]; then
                ZLEN="$4"; shift 4
            else
                shift 3
            fi ;;
        --grid-min)
            [[ $# -ge 2 ]] || { echo "Error: --grid-min requires a positive integer" >&2; exit 2; }
            validate_pos_int "--grid-min" "$2"
            GRID_MIN="$2"; shift 2 ;;
        --grid-max)
            [[ $# -ge 2 ]] || { echo "Error: --grid-max requires a positive integer" >&2; exit 2; }
            validate_pos_int "--grid-max" "$2"
            GRID_MAX="$2"; shift 2 ;;
        --grid-step)
            [[ $# -ge 2 ]] || { echo "Error: --grid-step requires a positive integer" >&2; exit 2; }
            validate_pos_int "--grid-step" "$2"
            GRID_STEP="$2"; shift 2 ;;
        --all-solvers)  ALL_SOLVERS=true;   shift ;;
        --solvers)
            [[ $# -ge 2 ]] || { echo "Error: --solvers requires a comma-separated list" >&2; exit 2; }
            SOLVER_FILTER="$2"; shift 2 ;;
        --ref-solver)
            [[ $# -ge 2 ]] || { echo "Error: --ref-solver requires a solver label" >&2; exit 2; }
            REF_SOLVER_LABEL="$2"; shift 2 ;;
        --add-solver)
            [[ $# -ge 2 ]] || { echo "Error: --add-solver requires LABEL:TYPE[:EXTRA]" >&2; exit 2; }
            ADD_SOLVER_SPECS+=("$2"); shift 2 ;;
        --randomize)    RANDOMIZE=true;     shift ;;
        --clean)        CLEAN_OUTPUT=true;  shift ;;
        --movie)        MAKE_MOVIE=true;    shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "Error: --timeout requires a positive integer (seconds)" >&2; exit 2; }
            validate_pos_int "--timeout" "$2"
            TIMEOUT="$2"; shift 2 ;;
        --verbose)      VERBOSE=true;       shift ;;
        --quiet)        VERBOSE=false;      shift ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "Error: --warmup requires a non-negative integer" >&2; exit 2; }
            validate_nonneg_int "--warmup" "$2"
            WARMUP="$2"; shift 2 ;;
        --no-plot)      NO_PLOT=true;       shift ;;
        --no-validate)  NO_VALIDATE=true;   shift ;;
        --no-color)     NO_COLOR_FLAG=true; shift ;;
        --baseline)     BASELINE=true;     shift ;;
        --petsc-options)
            [[ $# -ge 2 ]] || { echo "Error: --petsc-options requires a quoted string" >&2; exit 2; }
            PETSC_OPTIONS="$2"; shift 2 ;;
        --exe)
            [[ $# -ge 2 ]] || { echo "Error: --exe requires a path" >&2; exit 2; }
            EXECUTABLE="$2"; shift 2 ;;
        --input)
            [[ $# -ge 2 ]] || { echo "Error: --input requires a file path" >&2; exit 2; }
            INPUT_FILE="$2"; shift 2 ;;
        --name)
            [[ $# -ge 2 ]] || { echo "Error: --name requires a tag string" >&2; exit 2; }
            NAME_TAG="$2"; shift 2 ;;
        --field-output)
            [[ $# -ge 2 ]] || { echo "Error: --field-output requires a non-negative integer" >&2; exit 2; }
            validate_nonneg_int "--field-output" "$2"
            FIELD_OUTPUT="$2"; shift 2 ;;
        --particle-output)
            [[ $# -ge 2 ]] || { echo "Error: --particle-output requires a non-negative integer" >&2; exit 2; }
            validate_nonneg_int "--particle-output" "$2"
            PARTICLE_OUTPUT="$2"; shift 2 ;;
        --diag-output)
            [[ $# -ge 2 ]] || { echo "Error: --diag-output requires a non-negative integer" >&2; exit 2; }
            validate_nonneg_int "--diag-output" "$2"
            DIAG_OUTPUT="$2"; shift 2 ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Error: --output-dir requires a path" >&2; exit 2; }
            OUTPUT_DIR="$2"; shift 2 ;;
        --write-method)
            [[ $# -ge 2 ]] || { echo "Error: --write-method requires a value (phdf5, shdf5, pvtk)" >&2; exit 2; }
            case "$2" in phdf5|shdf5|pvtk) ;; *) echo "Error: --write-method must be one of: phdf5, shdf5, pvtk (got '$2')" >&2; exit 2 ;; esac
            WRITE_METHOD="$2"; shift 2 ;;
        --help|-h)
            cat <<HELPEOF
Usage: $0 [OPTIONS]

Single-grid mode (default):
  $0 [--np N] [--cycles N] [--grid NXC NYC [NZC]] [--topo X Y [Z]] ...

Scaling mode (sweep grid sizes):
  $0 --grid-min N --grid-max N [--grid-step N] ...

Options:
  --np N               MPI processes (default: 8)
  --cycles N           Simulation cycles (default: 10 single / 20 scaling)
  --avg N              Repetitions per solver; results are averaged (default: 1)
  --cooldown N         Seconds to wait between runs for cooling (default: 0)
  --grid NXC NYC [NZC] Override grid size; NZC defaults to 1 (2D) if omitted
  --topo X Y [Z]       Override MPI decomposition; Z defaults to 1 if omitted
  --grid-min N         Smallest XY grid size (triggers scaling mode)
  --grid-max N         Largest XY grid size (triggers scaling mode)
  --grid-step N        Grid size increment (default: 50)
  --all-solvers        Add PETSc fgmres, bcgs, tfqmr (requires PETSc build)
  --solvers LIST       Comma-separated solver labels to run (e.g. GMRES,PETSc_bcgs)
  --ref-solver LABEL   Reference solver for comparisons (default: first solver)
  --add-solver SPEC    Add custom solver: LABEL:TYPE[:EXTRA] (repeatable)
                       Examples: --add-solver "GMRES_2:GMRES:"
                                 --add-solver "PETSc_cg:PETSc:-ksp_type cg"
  --randomize          Shuffle solver run order
  --timeout N          Kill simulation after N seconds (requires timeout/gtimeout)
  --verbose            Show full output (default)
  --quiet              Suppress per-run progress lines
  --warmup N           Run N warm-up cycles before timed runs (default: 0)
  --no-plot            Skip matplotlib plotting
  --no-validate        Skip automated validation checks
  --no-color           Disable colored output (auto-detected when piped)
  --petsc-options STR  Extra PETSc KSP options for all PETSc runs (quoted string)
  --exe PATH           Path to iPIC3D executable (default: build/iPIC3D)
  --input FILE         Input file path (default: inputfiles/Double_Harris.inp)
  --name TAG           Output dir suffix: test_output_TAG/
  --field-output N     Save field data every N cycles (default: CYCLES in single-grid, off in scaling)
  --particle-output N  Save particle data every N cycles (0 = off)
  --diag-output N      Save energy diagnostics every N cycles (default: 5)
  --movie              Generate mp4 movie of field comparison across cycles (requires ffmpeg)
  --baseline           Add a second GMRES run (baseline for floating-point comparison)
  --output-dir DIR     Override output directory (default: tests/test_output/)
  --write-method M     HDF5 write method: phdf5, shdf5, pvtk (default: phdf5)
  --clean              Remove output directory before running (start fresh)
  --dry-run            Print commands without running them

Output:
  CSV and plots are saved to tests/test_output/ (or test_output_TAG/).
  CSV file: results.csv   Plot: results.png

Examples:
  # quick test with defaults (8 procs, 10 cycles, 2D)
  $0

  # explicit 2D grid and MPI topology
  $0 --np 4 --grid 200 200 --topo 2 2

  # 3D run: 8 procs in a 2x2x2 topology
  $0 --np 8 --grid 100 100 100 --topo 2 2 2

  # scaling sweep
  $0 --grid-min 50 --grid-max 400 --all-solvers

  # named run (non-overwriting)
  $0 --grid-min 100 --grid-max 400 --name overnight
HELPEOF
            exit 0 ;;
        --)  shift ;;  # consume -- (pixi inserts it before extra args)
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ========================= Color Setup ======================================
# Auto-detect: disable color when piped, when --no-color given, or when
# the NO_COLOR env var is set (https://no-color.org/).
if [[ "$NO_COLOR_FLAG" == true ]] || [[ "${NO_COLOR+set}" == "set" ]] || ! [[ -t 1 ]]; then
    _color=false
else
    _color=true
fi

if [[ "$_color" == true ]]; then
    RED=$'\033[31m'  GREEN=$'\033[32m'  YELLOW=$'\033[33m'  BLUE=$'\033[34m'
    BOLD=$'\033[1m'  DIM=$'\033[2m'     RESET=$'\033[0m'
    ARROW='▸'  CHECK='✓'  CROSS='✗'  DOT='·'  MDASH='─'  HDASH='━'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' DIM='' RESET=''
    ARROW='>'  CHECK='[OK]'  CROSS='[FAIL]'  DOT='-'  MDASH='-'  HDASH='='
fi

# Single-line section header: ── TITLE ──────────────────────────
section() {
    local title="$1" width="${2:-60}"
    local prefix="${MDASH}${MDASH} ${BOLD}${title}${RESET} "
    local prefix_plain="${MDASH}${MDASH} ${title} "
    local pad=$(( width - ${#prefix_plain} ))
    (( pad < 2 )) && pad=2
    printf -v trail '%*s' "$pad" ''; trail=${trail// /$MDASH}
    echo "${DIM}${prefix}${trail}${RESET}"
}

# ========================= Section 4: Mode detection =======================
SCALING_MODE=false
if [[ -n "$GRID_MIN" || -n "$GRID_MAX" ]]; then
    SCALING_MODE=true
    GRID_MIN="${GRID_MIN:-50}"
    GRID_MAX="${GRID_MAX:-400}"
fi

# Default cycles: 10 for single-grid, 20 for scaling
if [[ -z "$CYCLES" ]]; then
    if [[ "$SCALING_MODE" == true ]]; then
        CYCLES=20
    else
        CYCLES=10
    fi
fi

# In single-grid mode, default field output to every CYCLES so visual comparison always works
if [[ "$SCALING_MODE" != true && -z "$FIELD_OUTPUT" ]]; then
    FIELD_OUTPUT="$CYCLES"
fi

DIAG_OUTPUT="${DIAG_OUTPUT:-5}"

# ZLEN defaults to 1 (2D)
ZLEN="${ZLEN:-1}"

# If --topo was not given, auto-compute most square-like XLEN * YLEN factorization
if [[ -z "$XLEN" || -z "$YLEN" ]]; then
    NP_XY=$(( NP / ZLEN ))
    read -r XLEN YLEN <<< "$(compute_topology "$NP_XY")"
fi

# Validate that topology matches --np
expected_np=$((XLEN * YLEN * ZLEN))
if [[ $expected_np -ne $NP ]]; then
    echo "ERROR: Topology ${XLEN}x${YLEN}x${ZLEN} requires $expected_np processes, but --np is $NP" >&2
    exit 2
fi

# Parse grid values from input file once (single awk pass)
read -r file_nxc file_nyc file_nzc <<< "$(awk -F= '
    /^[[:space:]]*#/{next}
    /^nxc/ && !nxc {gsub(/[^0-9]/,"",$2); nxc=$2}
    /^nyc/ && !nyc {gsub(/[^0-9]/,"",$2); nyc=$2}
    /^nzc/ && !nzc {gsub(/[^0-9]/,"",$2); nzc=$2}
    END {print (nxc?nxc:100), (nyc?nyc:100), (nzc?nzc:1)}
' "$INPUT_FILE")"

# Ensure the grid is divisible by the MPI topology in each dimension.
# Round up to the nearest multiple if needed.
if [[ -z "$NXC" ]]; then
    rem=$((file_nxc % XLEN))
    [[ $rem -ne 0 ]] && NXC=$(( file_nxc + XLEN - rem ))
fi
if [[ -z "$NYC" ]]; then
    rem=$((file_nyc % YLEN))
    [[ $rem -ne 0 ]] && NYC=$(( file_nyc + YLEN - rem ))
fi
if [[ -z "$NZC" && "$ZLEN" -gt 1 ]]; then
    rem=$((file_nzc % ZLEN))
    [[ $rem -ne 0 ]] && NZC=$(( file_nzc + ZLEN - rem ))
fi

# Effective grid values for config generation
eff_nxc="${NXC:-$file_nxc}"
eff_nyc="${NYC:-$file_nyc}"
eff_nzc="${NZC:-$file_nzc}"

# Validate scaling range
if [[ "$SCALING_MODE" == true ]]; then
    [[ $GRID_MIN -lt $GRID_MAX ]] \
        || { echo "Error: --grid-min ($GRID_MIN) must be less than --grid-max ($GRID_MAX)" >&2; exit 2; }
fi

# Output directory and file paths
if [[ -n "$OUTPUT_DIR" ]]; then
    PERSISTENT_OUTPUT_DIR="$OUTPUT_DIR"
elif [[ -n "$NAME_TAG" ]]; then
    PERSISTENT_OUTPUT_DIR="$SCRIPT_DIR/test_output_${NAME_TAG}"
else
    PERSISTENT_OUTPUT_DIR="$SCRIPT_DIR/test_output"
fi
if [[ "$CLEAN_OUTPUT" == true && "$DRY_RUN" != true && -d "$PERSISTENT_OUTPUT_DIR" ]]; then
    # Safety: reject system directories
    _clean_dir_real=$(cd "$PERSISTENT_OUTPUT_DIR" && pwd -P) || { echo "ERROR: Cannot access $PERSISTENT_OUTPUT_DIR" >&2; exit 1; }
    case "$_clean_dir_real" in
        /|/usr|/usr/*|/var|/var/*|/etc|/etc/*|/bin|/bin/*|/sbin|/sbin/*|"$HOME")
            echo "ERROR: --clean refuses to delete system directory: $PERSISTENT_OUTPUT_DIR" >&2
            exit 1 ;;
    esac
    # Safety: verify directory contains only expected output files
    _bad_files=$(find "$PERSISTENT_OUTPUT_DIR" -maxdepth 2 -type f \
        ! -name '*.h5' ! -name '*.hdf' ! -name '*.hdf5' ! -name '*.csv' ! -name '*.png' \
        ! -name '*.jpg' ! -name '*.mp4' ! -name '*.txt' ! -name '*.md' \
        ! -name '*.log' ! -name '*.inp' ! -name '.DS_Store' ! -name 'Thumbs.db' \
        2>/dev/null | head -5)
    if [[ -n "$_bad_files" ]]; then
        echo "ERROR: --clean found unexpected files in $PERSISTENT_OUTPUT_DIR:" >&2
        echo "$_bad_files" >&2
        echo "       Only .h5/.hdf/.hdf5/.csv/.png/.jpg/.mp4/.txt/.md/.log/.inp files (and OS metadata) are expected." >&2
        echo "       Remove manually if intended." >&2
        exit 1
    fi
    rm -rf "$PERSISTENT_OUTPUT_DIR"
fi
mkdir -p "$PERSISTENT_OUTPUT_DIR"

CSV_FILE="$PERSISTENT_OUTPUT_DIR/results.csv"
PLOT_FILE="$PERSISTENT_OUTPUT_DIR/results.png"

# ========================= Section 5: PETSc detection & solver list =========
# Auto-detect PETSc support from the build
BUILD_DIR_PATH=$(dirname "$EXECUTABLE")
HAS_PETSC=false
if grep -q 'USE_PETSC:BOOL=ON' "$BUILD_DIR_PATH/CMakeCache.txt" 2>/dev/null; then
    HAS_PETSC=true
fi

# Add default PETSc solver if available
if [[ "$HAS_PETSC" == true ]]; then
    SOLVER_LABELS+=("PETSc_gmres")
    SOLVER_TYPES+=( "PETSc")
    SOLVER_EXTRA+=( "")
fi

# Extra solvers added with --all-solvers
if [[ "$ALL_SOLVERS" == true ]]; then
    if [[ "$HAS_PETSC" == true ]]; then
        SOLVER_LABELS+=("PETSc_fgmres"  "PETSc_bcgs"  "PETSc_tfqmr")
        SOLVER_TYPES+=( "PETSc"         "PETSc"       "PETSc")
        SOLVER_EXTRA+=( "-ksp_type fgmres -ksp_gmres_restart 50 -ksp_max_it 1000" \
                        "-ksp_type bcgs -ksp_max_it 1000" \
                        "-ksp_type tfqmr -ksp_max_it 1000")
    else
        echo "WARNING: --all-solvers ignored — build does not include PETSc" >&2
    fi
fi

# Baseline GMRES run for floating-point comparison
if [[ "$BASELINE" == true ]]; then
    SOLVER_LABELS+=("GMRES_2")
    SOLVER_TYPES+=( "GMRES")
    SOLVER_EXTRA+=( "")
fi

# Process --add-solver specs (LABEL:TYPE[:EXTRA])
for spec in ${ADD_SOLVER_SPECS[@]+"${ADD_SOLVER_SPECS[@]}"}; do
    IFS=: read -r _as_label _as_type _as_extra <<< "$spec"
    if [[ -z "$_as_label" || -z "$_as_type" ]]; then
        echo "ERROR: --add-solver requires LABEL:TYPE[:EXTRA], got: '$spec'" >&2
        exit 2
    fi
    SOLVER_LABELS+=("$_as_label")
    SOLVER_TYPES+=("$_as_type")
    SOLVER_EXTRA+=("${_as_extra:-}")
done

# Append --petsc-options to all PETSc solver extra args
if [[ -n "$PETSC_OPTIONS" ]]; then
    for ((si=0; si<${#SOLVER_LABELS[@]}; si++)); do
        if [[ "${SOLVER_TYPES[$si]}" == "PETSc" ]]; then
            SOLVER_EXTRA[$si]="${SOLVER_EXTRA[$si]:+${SOLVER_EXTRA[$si]} }${PETSC_OPTIONS}"
        fi
    done
fi

# Filter solvers if --solvers was given
if [[ -n "$SOLVER_FILTER" ]]; then
    IFS=',' read -ra _wanted <<< "$SOLVER_FILTER"
    # Validate: error if user explicitly requests PETSc solver but build lacks it
    if [[ "$HAS_PETSC" == false ]]; then
        for _w in "${_wanted[@]}"; do
            if [[ "$_w" == PETSc* ]]; then
                echo "ERROR: Solver '$_w' requires a PETSc build, but USE_PETSC is OFF." >&2
                echo "       Build with: pixi run build  (or ./build.sh --petsc)" >&2
                exit 2
            fi
        done
    fi
    _new_labels=() _new_types=() _new_extra=()
    for ((si=0; si<${#SOLVER_LABELS[@]}; si++)); do
        for _w in "${_wanted[@]}"; do
            if [[ "${SOLVER_LABELS[$si]}" == "$_w" ]]; then
                _new_labels+=("${SOLVER_LABELS[$si]}")
                _new_types+=("${SOLVER_TYPES[$si]}")
                _new_extra+=("${SOLVER_EXTRA[$si]}")
                break
            fi
        done
    done
    if [[ ${#_new_labels[@]} -eq 0 ]]; then
        echo "ERROR: --solvers matched no known solver labels." >&2
        echo "       Available: ${SOLVER_LABELS[*]}" >&2
        exit 2
    fi
    SOLVER_LABELS=("${_new_labels[@]}")
    SOLVER_TYPES=("${_new_types[@]}")
    SOLVER_EXTRA=("${_new_extra[@]}")
fi

[[ ${#SOLVER_LABELS[@]} -eq ${#SOLVER_TYPES[@]} && ${#SOLVER_TYPES[@]} -eq ${#SOLVER_EXTRA[@]} ]] \
    || { echo "ERROR: Solver arrays out of sync (BUG)" >&2; exit 1; }
NUM_SOLVERS=${#SOLVER_LABELS[@]}

# Determine reference solver: default to first solver in the list
if [[ -z "$REF_SOLVER_LABEL" ]]; then
    REF_SOLVER_LABEL="${SOLVER_LABELS[0]}"
else
    # Validate that the specified ref solver exists
    _ref_found=false
    for _sl in "${SOLVER_LABELS[@]}"; do
        [[ "$_sl" == "$REF_SOLVER_LABEL" ]] && _ref_found=true && break
    done
    if [[ "$_ref_found" != true ]]; then
        echo "ERROR: --ref-solver '$REF_SOLVER_LABEL' not found in solver list: ${SOLVER_LABELS[*]}" >&2
        exit 2
    fi
fi
# Find the index of the reference solver
REF_SOLVER_IDX=0
for ((si=0; si<NUM_SOLVERS; si++)); do
    [[ "${SOLVER_LABELS[$si]}" == "$REF_SOLVER_LABEL" ]] && REF_SOLVER_IDX=$si && break
done

# ========================= Section 6: Prerequisite checks ==================
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. It is required for parsing, plotting, and statistics." >&2
    exit 1
fi
if [[ "$DRY_RUN" != true ]]; then
    if [[ ! -x "$EXECUTABLE" ]]; then
        echo "ERROR: Executable not found at $EXECUTABLE"
        echo "       Build first with:  pixi run build  (or ./build.sh)"
        exit 1
    fi
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "ERROR: Input file not found: $INPUT_FILE"
        exit 1
    fi
fi

warn_oversubscription "$NP"

# ========================= Section 7: Helper functions =====================
cleanup() { [[ -n "${TMPDIR_BASE:-}" ]] && rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT INT TERM HUP

# High-resolution timestamp. Uses $EPOCHREALTIME (Bash 5.0+, zero overhead)
# when available, falls back to python3.
if [[ -n "${EPOCHREALTIME:-}" ]]; then
    timestamp() { printf '%s' "$EPOCHREALTIME"; }
else
    timestamp() { "$PYTHON" -c 'import time; print(f"{time.time():.6f}")'; }
fi

# Elapsed time: elapsed <start> <end>  →  stdout "X.XXX"
elapsed() { awk -v s="$1" -v e="$2" 'BEGIN{printf "%.2f", e - s}'; }

# Compute mean and stdev for six space-separated value strings.
# Args: field_str wall_str moments_str mover_str iters_str resid_str
# Stdout: "avg_field std_field avg_moments avg_mover avg_iters avg_resid avg_wall"
compute_stats() {
    "$PYTHON" -c "
import sys, statistics
def mean_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.mean(vals):.2f}' if vals else 'NA'
def std_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.stdev(vals):.2f}' if len(vals) > 1 else 'NA'

field   = sys.argv[1].split()
wall    = sys.argv[2].split()
moments = sys.argv[3].split()
mover   = sys.argv[4].split()
iters   = sys.argv[5].split()
resid   = sys.argv[6].split()

print(f'{mean_str(field)} {std_str(field)} {mean_str(moments)} {mean_str(mover)} {mean_str(iters)} {mean_str(resid)} {mean_str(wall)}')
" "$1" "$2" "$3" "$4" "$5" "$6"
}

# Create a test input file with reduced cycles and controlled output.
# Args: outdir nxc nyc nzc [solver_label]
make_input() {
    local outdir="$1" nxc="$2" nyc="$3" nzc="$4"
    local solver_label="${5:-}"
    local inp="$outdir/test_input.inp"

    # Determine save directory: persistent if field/particle output requested
    local save_dir="$outdir/output"
    if [[ -n "$FIELD_OUTPUT" || -n "$PARTICLE_OUTPUT" ]]; then
        local config_tag="${nxc}x${nyc}x${nzc}"
        if [[ -n "$solver_label" ]]; then
            save_dir="$PERSISTENT_OUTPUT_DIR/${solver_label}_${config_tag}"
        else
            save_dir="$PERSISTENT_OUTPUT_DIR/${config_tag}"
        fi
    fi

    local field_out="${FIELD_OUTPUT:-0}"
    local particle_out="${PARTICLE_OUTPUT:-0}"

    mkdir -p "$save_dir"
    local sed_args=(
        -e "s|^ncycles.*=.*|ncycles                        = $CYCLES|"
        -e "s|^nxc.*=.*|nxc                            = $nxc|"
        -e "s|^nyc.*=.*|nyc                            = $nyc|"
        -e "s|^nzc.*=.*|nzc                            = $nzc|"
        -e "s|^XLEN.*=.*|XLEN                           = $XLEN|"
        -e "s|^YLEN.*=.*|YLEN                           = $YLEN|"
        -e "s|^ZLEN.*=.*|ZLEN                           = $ZLEN|"
        -e "s|^SaveDirName.*=.*|SaveDirName                    = $save_dir|"
        -e "s|^RestartDirName.*=.*|RestartDirName                 = $save_dir|"
        -e "s|^FieldOutputCycle.*=.*|FieldOutputCycle               = $field_out|"
        -e "s|^ParticlesOutputCycle.*=.*|ParticlesOutputCycle           = $particle_out|"
        -e "s|^RestartOutputCycle.*=.*|RestartOutputCycle             = 0|"
        -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle         = $DIAG_OUTPUT|"
        -e "s|^WriteMethod.*=.*|WriteMethod                    = $WRITE_METHOD|"
    )
    sed "${sed_args[@]}" "$INPUT_FILE" > "$inp" || { echo "ERROR: Failed to create test input from $INPUT_FILE" >&2; return 1; }
    echo "$inp"
}

# Extract per-cycle timing profile from a log file.
# Args: logfile output_csv
extract_profile() {
    local logfile="$1" outcsv="$2"
    "$PYTHON" "$SCRIPTS_DIR/extract_profile.py" "$logfile" "$outcsv" 2>/dev/null || true
}

# Parse convergence info from a log file.
# Stdout: "avg_iters converged avg_residual"
#   avg_iters: average iterations per solve call
#   converged: "yes" if all calls converged, "no" if any failed
#   avg_residual: average final residual
parse_convergence() {
    local logfile="$1"
    PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}" "$PYTHON" -c "
from log_patterns import CONVERGENCE_PATTERNS, GMRES_ZERO_ITER

logfile = sys.argv[1]
iters_list, residuals, any_failed = [], [], False

with open(logfile) as f:
    for line in f:
        if GMRES_ZERO_ITER.search(line):
            iters_list.append(0); residuals.append(0.0); continue
        for name, success_re, fail_re, iter_fn, res_grp, fail_iter_fn, fail_res_grp in CONVERGENCE_PATTERNS:
            m = success_re.search(line)
            if m:
                iters_list.append(iter_fn(m)); residuals.append(float(m.group(res_grp))); break
            m = fail_re.search(line)
            if m:
                any_failed = True
                if fail_iter_fn is not None: iters_list.append(fail_iter_fn(m))
                if fail_res_grp is not None: residuals.append(float(m.group(fail_res_grp)))
                break

if not iters_list and not residuals:
    print('NA NA NA')
else:
    avg_iters = sum(iters_list) / len(iters_list) if iters_list else 0
    converged = 'no' if any_failed else 'yes'
    avg_residual = sum(residuals) / len(residuals) if residuals else 0
    print(f'{avg_iters:.1f} {converged} {avg_residual:.2e}')
" "$logfile"
}

# Run a short warm-up simulation (WARMUP cycles) to prime caches/JIT.
# Args: nxc nyc nzc label solver_type [extra_args...]

run_warmup() {
    local nxc="$1" nyc="$2" nzc="$3" label="$4" solver_type="$5"
    shift 5
    local extra_args=("$@")

    if [[ "$WARMUP" -le 0 || "$DRY_RUN" == true ]]; then return 0; fi

    local warmup_dir="$TMPDIR_BASE/warmup_${label}_${nxc}x${nyc}x${nzc}"
    mkdir -p "$warmup_dir"

    # Create a short input file with WARMUP cycles and no output
    local save_cycles="$CYCLES" save_field="$FIELD_OUTPUT" save_particle="$PARTICLE_OUTPUT"
    CYCLES="$WARMUP"; FIELD_OUTPUT=0; PARTICLE_OUTPUT=0
    local inp
    inp=$(make_input "$warmup_dir" "$nxc" "$nyc" "$nzc" "$label")
    CYCLES="$save_cycles"; FIELD_OUTPUT="$save_field"; PARTICLE_OUTPUT="$save_particle"

    local cmd=(mpirun "${MPIRUN_FLAGS[@]}" -np "$NP" "$EXECUTABLE" "$inp" -solver "$solver_type")
    [[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")

    [[ "$VERBOSE" == true ]] && echo "    ${DIM}warm-up (${WARMUP} cycles)...${RESET}" >&2
    "${cmd[@]}" > "$warmup_dir/warmup.log" 2>&1 || true
}

# Run a single simulation attempt.
# Stdout: "field_t moments_t mover_t iters converged residual"
# Args: nxc nyc nzc label solver_type run_id [extra_args...]
run_solver() {
    local nxc="$1" nyc="$2" nzc="$3" label="$4" solver_type="$5" run_id="$6"
    shift 6
    local extra_args=("$@")

    local rundir="$TMPDIR_BASE/${label}_${nxc}x${nyc}x${nzc}_r${run_id}"
    mkdir -p "$rundir"
    local logfile="$rundir/output.log"

    local cmd=(mpirun "${MPIRUN_FLAGS[@]}" -np "$NP" "$EXECUTABLE")
    if [[ "$DRY_RUN" != true ]]; then
        local inp
        inp=$(make_input "$rundir" "$nxc" "$nyc" "$nzc" "$label")
        cmd+=("$inp" -solver "$solver_type")
    else
        cmd+=("test_input.inp" -solver "$solver_type")
    fi
    [[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")

    if [[ "$DRY_RUN" == true ]]; then
        echo "0.000 NA NA NA yes NA"
        return 0
    fi

    # Optionally wrap with timeout
    local run_cmd=("${cmd[@]}")
    if [[ -n "$TIMEOUT" ]]; then
        local timeout_bin=""
        if command -v timeout &>/dev/null; then
            timeout_bin="timeout"
        elif command -v gtimeout &>/dev/null; then
            timeout_bin="gtimeout"
        fi
        if [[ -n "$timeout_bin" ]]; then
            run_cmd=("$timeout_bin" "$TIMEOUT" "${cmd[@]}")
        fi
    fi

    if ! "${run_cmd[@]}" > "$logfile" 2>&1; then
        echo "FAILED"
        # Show last 20 lines for diagnostics
        if [[ -f "$logfile" ]]; then
            echo "--- Last 20 lines of $logfile ---" >&2
            tail -20 "$logfile" >&2
            echo "--- End of log snippet ---" >&2
        fi
        # Hint about conda/mamba MPI conflicts
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            echo "${YELLOW}WARNING:${RESET} conda/mamba environment '${CONDA_DEFAULT_ENV}' is active." >&2
            echo "         Its MPI may conflict with the MPI used to build iPIC3D." >&2
            echo "         Try: conda deactivate && rerun this script." >&2
        fi
        return 1
    fi

    # Extract timings
    local field_t moments_t mover_t
    field_t=$(awk '/Field solver/{v=$(NF-1)} END{print v}' "$logfile") || field_t="NA"
    moments_t=$(awk '/Moment gatherer/{v=$(NF-1)} END{print v}' "$logfile") || moments_t="NA"
    mover_t=$(awk '/Particle mover/{v=$(NF-1)} END{print v}' "$logfile") || mover_t="NA"
    [[ -z "$field_t" ]] && field_t="NA"
    [[ -z "$moments_t" ]] && moments_t="NA"
    [[ -z "$mover_t" ]] && mover_t="NA"

    # Parse convergence
    local conv_result iters converged residual
    conv_result=$(parse_convergence "$logfile")
    read -r iters converged residual <<< "$conv_result"

    # Save per-cycle profile
    local config_tag="${nxc}x${nyc}x${nzc}"
    local profile_csv="$PERSISTENT_OUTPUT_DIR/profile_${label}_${config_tag}.csv"
    extract_profile "$logfile" "$profile_csv"

    # Persist log to output dir for post-hoc analysis (KSP diagnostics, etc.)
    cp "$logfile" "$PERSISTENT_OUTPUT_DIR/${label}_${config_tag}.log"

    echo "${field_t} ${moments_t} ${mover_t} ${iters} ${converged} ${residual}"
}

# Run AVG repetitions, averaging results.
# Stdout: "avg_field std_field avg_moments avg_mover avg_iters converged avg_residual avg_wall logfile"
# Args: nxc nyc nzc label solver_type [extra_args...]
run_averaged() {
    local nxc="$1" nyc="$2" nzc="$3" label="$4" solver_type="$5"
    shift 5
    local extra_args=("$@")

    # Indent prefix: deeper when inside a grid loop (scaling mode)
    local I="  "
    [[ "$SCALING_MODE" == true ]] && I="    "

    local runs_info=""
    [[ $AVG -gt 1 ]] && runs_info=" ${DOT} Runs: $AVG"
    [[ "$VERBOSE" == true ]] && echo "${I}${ARROW} ${BOLD}$label${RESET} ($solver_type) ${DOT} ${nxc}x${nyc}x${nzc} ${DOT} ${NP} procs ${DOT} ${CYCLES} cycles${runs_info}" >&2

    local field_arr=() wall_arr=() moments_arr=() mover_arr=()
    local iters_arr=() residual_arr=()
    local all_converged=true
    local last_logfile=""
    local failed=0

    # Optional warm-up run
    run_warmup "$nxc" "$nyc" "$nzc" "$label" "$solver_type" "${extra_args[@]+"${extra_args[@]}"}"

    for ((r=1; r<=AVG; r++)); do
        if [[ $r -gt 1 && $COOLDOWN -gt 0 ]]; then
            echo "${DIM}${I}  cooldown ${COOLDOWN}s...${RESET}" >&2
            sleep "$COOLDOWN"
        fi

        local start_time
        start_time=$(timestamp)

        local result
        result=$(run_solver "$nxc" "$nyc" "$nzc" "$label" "$solver_type" "$r" "${extra_args[@]+"${extra_args[@]}"}") || true

        local end_time
        end_time=$(timestamp)

        if [[ "$result" == "FAILED" || -z "$result" ]]; then
            failed=1; break
        fi

        local ft mt pt it cv rs
        read -r ft mt pt it cv rs <<< "$result"

        local wall_t
        wall_t=$(elapsed "$start_time" "$end_time")
        wall_arr+=("$wall_t")

        [[ "$ft" != "NA" && -n "$ft" ]] && field_arr+=("$ft")
        [[ "$mt" != "NA" && -n "$mt" ]] && moments_arr+=("$mt")
        [[ "$pt" != "NA" && -n "$pt" ]] && mover_arr+=("$pt")
        [[ "$it" != "NA" && -n "$it" ]] && iters_arr+=("$it")
        [[ "$rs" != "NA" && -n "$rs" ]] && residual_arr+=("$rs")
        [[ "$cv" == "no" ]] && all_converged=false

        last_logfile="$TMPDIR_BASE/${label}_${nxc}x${nyc}x${nzc}_r${r}/output.log"

        if [[ "$VERBOSE" == true ]]; then
            if [[ $AVG -gt 1 ]]; then
                printf "${I}  Run %d/%d ${DOT} ${ft}s field ${DOT} ${wall_t}s wall\n" "$r" "$AVG" >&2
            else
                echo "${I}  ${ft}s field ${DOT} ${wall_t}s wall" >&2
            fi
        fi
    done

    if [[ $failed -eq 1 || ${#field_arr[@]} -eq 0 ]]; then
        echo "${I}  ${RED}FAILED${RESET}" >&2
        echo "NA NA NA NA NA no NA NA none"
        return 1
    fi

    # Compute means — build safe strings from arrays (empty-safe)
    local field_str wall_str moments_str mover_str iters_str resid_str
    field_str="${field_arr[*]+"${field_arr[*]}"}"
    wall_str="${wall_arr[*]+"${wall_arr[*]}"}"
    moments_str="${moments_arr[*]+"${moments_arr[*]}"}"
    mover_str="${mover_arr[*]+"${mover_arr[*]}"}"
    iters_str="${iters_arr[*]+"${iters_arr[*]}"}"
    resid_str="${residual_arr[*]+"${residual_arr[*]}"}"

    local stats
    stats=$(compute_stats "$field_str" "$wall_str" "$moments_str" "$mover_str" "$iters_str" "$resid_str")

    local avg_field std_field avg_moments avg_mover avg_iters avg_residual avg_wall
    read -r avg_field std_field avg_moments avg_mover avg_iters avg_residual avg_wall <<< "$stats"

    local converged_str="yes"
    [[ "$all_converged" == false ]] && converged_str="no"

    if [[ $AVG -gt 1 ]]; then
        echo "${I}  ${BOLD}mean${RESET}: ${avg_field} ${DOT} ${std_field} s ${DOT} wall ${avg_wall} s" >&2
    fi

    echo "$avg_field $std_field $avg_moments $avg_mover $avg_iters $converged_str $avg_residual $avg_wall $last_logfile"
}

# Extract timing from the log (last occurrence = cumulative over all cycles).
# Args: logfile label field_t
extract_timing() {
    local logfile="$1" label="$2" field_t="$3"

    local suffix=""
    [[ $AVG -gt 1 ]] && suffix=" (mean over $AVG runs)"
    echo "  [$label] Module timings (cumulative over $CYCLES cycles)${suffix}:"

    local init_t moment_t mover_t write_t
    init_t=$(awk '/Time needed for initialisation:/{v=$(NF-1)} END{print v}' "$logfile") || init_t="N/A"
    mover_t=$(awk '/Particle mover/{v=$(NF-1)} END{print v}' "$logfile") || mover_t="N/A"
    moment_t=$(awk '/Moment gatherer/{v=$(NF-1)} END{print v}' "$logfile") || moment_t="N/A"
    write_t=$(awk '/Write data/{v=$(NF-1)} END{print v}' "$logfile") || write_t="N/A"

    printf "    %-22s %10s s\n" "Initialisation:" "$init_t"
    printf "    %-22s %10s s\n" "Field solver:" "$field_t"
    printf "    %-22s %10s s\n" "Particle mover:" "$mover_t"
    printf "    %-22s %10s s\n" "Moment gatherer:" "$moment_t"
    printf "    %-22s %10s s\n" "Write data:" "$write_t"
}

# Extract convergence info from the log.
# Args: logfile label
extract_convergence() {
    local logfile="$1" label="$2"

    echo "  ${BOLD}[$label]${RESET} Convergence:"

    # Solver convergence pattern registry: name, converged_pattern, failed_pattern
    local -a _cv_names=("GMRES" "PETSc")
    local -a _cv_conv=("GMRES converged" "PETSc KSP converged")
    local -a _cv_fail=("GMRES not converged" "PETSc KSP did NOT converge")
    # Add new solvers here:
    # _cv_names+=("CG"); _cv_conv+=("CG converged"); _cv_fail+=("CG did NOT converge")

    local _any_found=false
    for ((_ci=0; _ci<${#_cv_names[@]}; _ci++)); do
        local _name="${_cv_names[$_ci]}"
        local _conv_pat="${_cv_conv[$_ci]}"
        local _fail_pat="${_cv_fail[$_ci]}"
        local _conv_count _fail_count
        _conv_count=$(grep -c "$_conv_pat" "$logfile" 2>/dev/null) || _conv_count=0
        _fail_count=$(grep -c "$_fail_pat" "$logfile" 2>/dev/null) || _fail_count=0

        if [[ $_conv_count -gt 0 || $_fail_count -gt 0 ]]; then
            _any_found=true
            local _cv_mark="${GREEN}${CHECK}${RESET}"
            [[ $_fail_count -gt 0 ]] && _cv_mark="${RED}${CROSS}${RESET}"
            echo "    ${_cv_mark} ${_name}: $_conv_count converged, $_fail_count failed"
            grep "$_conv_pat" "$logfile" | head -1 | sed 's/^/    First: /'
            grep "$_conv_pat" "$logfile" | tail -1 | sed 's/^/    Last:  /'
        fi
    done

    if [[ "$_any_found" != true ]]; then
        echo "    (no convergence messages found)"
    fi
}

# Print ETA estimate.
# Args: t1 (seconds for one run_averaged call), g0 (grid size of that call)
# Uses globals: CONFIGS, NUM_SOLVERS, t_sweep_start, SCALING_MODE
show_eta() {
    local t1="$1" g0="$2"
    export _C_BOLD="$BOLD" _C_RESET="$RESET" _C_YELLOW="$YELLOW"
    "$PYTHON" - "$t1" "$g0" "$NUM_SOLVERS" "$t_sweep_start" "$SCALING_MODE" "${CONFIGS[*]}" <<'PYETA' || true
import os, sys, time
from datetime import datetime, timedelta

_bold   = os.environ.get('_C_BOLD', '')
_reset  = os.environ.get('_C_RESET', '')
_yellow = os.environ.get('_C_YELLOW', '')

t1        = float(sys.argv[1])
g0        = float(sys.argv[2])
n_solvers = int(sys.argv[3])
elapsed   = time.time() - float(sys.argv[4])
scaling   = sys.argv[5] == "true"

if scaling:
    # N^2-scaled ETA
    configs = sys.argv[6].split()
    # Parse first field (nxc) from each config triplet
    grids = []
    for i in range(0, len(configs), 3):
        grids.append(int(configs[i]))
    total_est = t1 * n_solvers * sum((g / g0) ** 2 for g in grids)
else:
    # Linear scaling: all solvers on same grid
    total_est = t1 * n_solvers

remaining_est = max(0.0, total_est - elapsed)

def fmt(s):
    s = int(s)
    if s < 60:   return f'{s}s'
    if s < 3600: return f'{s//60}m{s%60:02d}s'
    return f'{s//3600}h{(s%3600)//60:02d}m'

eta = datetime.now() + timedelta(seconds=remaining_est)
print(f'  {_yellow}{_bold}ETA{_reset}  {_yellow}~{fmt(total_est)} total  |  ~{fmt(remaining_est)} remaining  |  done by {eta:%H:%M}{_reset}')
PYETA
}

# ========================= Section 8: Configuration generator ==============
CONFIGS=()
if [[ "$SCALING_MODE" == true ]]; then
    # Compute LCM of XLEN and YLEN for grid divisibility
    lcm=$("$PYTHON" -c "import sys; from math import gcd; a,b=int(sys.argv[1]),int(sys.argv[2]); print(a*b//gcd(a,b))" "$XLEN" "$YLEN")

    prev_grids=()
    for (( g=GRID_MIN; g<=GRID_MAX; g+=GRID_STEP )); do
        # Round up to LCM for divisibility
        rem=$((g % lcm))
        if [[ $rem -ne 0 ]]; then
            rounded=$(( g + lcm - rem ))
        else
            rounded=$g
        fi
        # Deduplicate
        is_dup=false
        for prev in ${prev_grids[@]+"${prev_grids[@]}"}; do
            [[ "$prev" == "$rounded" ]] && is_dup=true && break
        done
        if [[ "$is_dup" == false ]]; then
            prev_grids+=("$rounded")
            CONFIGS+=("$rounded $rounded $eff_nzc")
        fi
    done
else
    CONFIGS+=("$eff_nxc $eff_nyc $eff_nzc")
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: No grid configurations generated." >&2
    exit 1
fi

# ========================= Section 9: Summary box ==========================
echo ""

if [[ "$SCALING_MODE" == true ]]; then
    box_title="iPIC3D Scaling Study: Field Solver Comparison"
else
    if [[ $NUM_SOLVERS -gt 1 ]]; then
        # Build "A vs B, C" label from solver list
        _other_labels=""
        for _sl in "${SOLVER_LABELS[@]}"; do
            [[ "$_sl" != "$REF_SOLVER_LABEL" ]] && _other_labels+="${_other_labels:+, }$_sl"
        done
        box_title="iPIC3D Solver Comparison: ${REF_SOLVER_LABEL} vs ${_other_labels}"
    else
        box_title="iPIC3D Solver Test: ${SOLVER_LABELS[0]}"
    fi
fi

# Collect all content lines, then size the box to fit
box_lines=()
box_lines+=("  Input: $(basename "$INPUT_FILE")")
box_lines+=("  Procs: $NP   Cycles: $CYCLES   Topo: ${XLEN}x${YLEN}x${ZLEN}")

if [[ "$SCALING_MODE" == true ]]; then
    config_grids=""
    for cfg in "${CONFIGS[@]}"; do
        read -r cnxc cnyc cnzc <<< "$cfg"
        config_grids+="${cnxc} "
    done
    box_lines+=("  XY grids: ${config_grids% }")
    box_lines+=("  NZC: $eff_nzc")
else
    read -r show_nxc show_nyc show_nzc <<< "${CONFIGS[0]}"
    box_lines+=("  Grid: ${show_nxc}x${show_nyc}x${show_nzc}")
fi

box_lines+=("  Solvers: ${SOLVER_LABELS[*]}  (ref: ${REF_SOLVER_LABEL})")
for ((si=0; si<NUM_SOLVERS; si++)); do
    if [[ -n "${SOLVER_EXTRA[$si]}" ]]; then
        box_lines+=("    ${SOLVER_LABELS[$si]}: ${SOLVER_EXTRA[$si]}")
    fi
done
[[ $AVG -gt 1 ]] && box_lines+=("  Runs: $AVG   Cooldown: ${COOLDOWN}s between runs")
[[ -n "$NAME_TAG" ]] && box_lines+=("  Name tag: $NAME_TAG")
[[ -n "$FIELD_OUTPUT" || -n "$PARTICLE_OUTPUT" ]] && box_lines+=("  Field output: ${FIELD_OUTPUT:-off}   Particle output: ${PARTICLE_OUTPUT:-off}")
[[ "$MAKE_MOVIE" == true ]] && box_lines+=("  Movie: yes (generate mp4 comparison movie)")
[[ "$BASELINE" == true ]] && box_lines+=("  Baseline: GMRES_2 (same solver, separate run)")
[[ "$RANDOMIZE" == true ]] && box_lines+=("  Randomize: yes (solver order shuffled)")
[[ -n "$TIMEOUT" ]] && box_lines+=("  Timeout: ${TIMEOUT}s per simulation")
[[ "$WARMUP" -gt 0 ]] && box_lines+=("  Warm-up: $WARMUP cycles before timed runs")
[[ -n "$PETSC_OPTIONS" ]] && box_lines+=("  PETSc options: $PETSC_OPTIONS")
[[ "$NO_PLOT" == true ]] && box_lines+=("  Plotting: disabled (--no-plot)")
[[ "$VERBOSE" == false ]] && box_lines+=("  Output: quiet mode")
[[ "$DRY_RUN" == true ]] && box_lines+=("  ${YELLOW}*** DRY-RUN MODE — no simulations will be executed ***${RESET}")

section "$box_title"
for line in "${box_lines[@]}"; do
    echo "   ${line#  }"
done
echo ""

# ========================= Section 10: Main loop ==========================
# Result accumulators (parallel arrays indexed by config × solver)
ALL_LABELS=()
ALL_FIELD=()
ALL_STD=()
ALL_MOMENTS=()
ALL_MOVER=()
ALL_ITERS=()
ALL_CONVERGED=()
ALL_RESIDUAL=()
ALL_WALL=()
ALL_LOGS=()
ALL_CONFIG_TAGS=()    # "nxc nyc nzc" for each result

t_sweep_start=$(timestamp)
first_run_t=""

config_count=0
total_configs=${#CONFIGS[@]}

for cfg in "${CONFIGS[@]}"; do
    read -r cnxc cnyc cnzc <<< "$cfg"
    config_count=$((config_count + 1))
    config_tag="${cnxc}x${cnyc}x${cnzc}"

    if [[ "$SCALING_MODE" == true ]]; then
        section "[$config_count/$total_configs] Grid ${config_tag} ${DOT} ${CYCLES} cycles ${DOT} ${NP} procs"
    fi

    if [[ "$RANDOMIZE" == true ]]; then
        # ── Interleaved mode: AVG rounds, each with all solvers in shuffled order ──
        # This avoids running all reps of one solver consecutively, reducing
        # bias from CPU throttling.

        # Indent prefix for scaling vs single-grid
        RI="  "
        [[ "$SCALING_MODE" == true ]] && RI="    "

        # Per-solver accumulators using temp files (one file per metric per solver)
        declare -A acc_converged acc_lastlog acc_failed
        _acc_dir="$TMPDIR_BASE/_acc_${config_tag}"
        mkdir -p "$_acc_dir"
        for ((si=0; si<NUM_SOLVERS; si++)); do
            : > "$_acc_dir/field_$si"
            : > "$_acc_dir/wall_$si"
            : > "$_acc_dir/moments_$si"
            : > "$_acc_dir/mover_$si"
            : > "$_acc_dir/iters_$si"
            : > "$_acc_dir/resid_$si"
            acc_converged[$si]=true
            acc_lastlog[$si]="none"
            acc_failed[$si]=0
        done

        # Optional warm-up for each solver
        if [[ "$WARMUP" -gt 0 && "$DRY_RUN" != true ]]; then
            for ((si=0; si<NUM_SOLVERS; si++)); do
                _wu_extra=()
                [[ -n "${SOLVER_EXTRA[$si]}" ]] && read -ra _wu_extra <<< "${SOLVER_EXTRA[$si]}"
                run_warmup "$cnxc" "$cnyc" "$cnzc" "${SOLVER_LABELS[$si]}" "${SOLVER_TYPES[$si]}" "${_wu_extra[@]+"${_wu_extra[@]}"}"
            done
        fi

        for ((r=1; r<=AVG; r++)); do
            if [[ $r -gt 1 && $COOLDOWN -gt 0 ]]; then
                echo "${DIM}${RI}cooldown ${COOLDOWN}s...${RESET}"
                sleep "$COOLDOWN"
            fi

            # Shuffle solver indices for this round
            solver_order=($("$PYTHON" -c "import sys,random; x=list(range(int(sys.argv[1]))); random.shuffle(x); print(*x)" "$NUM_SOLVERS"))
            order_str=""
            for si in "${solver_order[@]}"; do order_str+="${SOLVER_LABELS[$si]} "; done
            if [[ $AVG -gt 1 ]]; then
                echo "${RI}Round $r/$AVG: [${order_str% }]"
            else
                echo "${RI}Order: [${order_str% }]"
            fi

            for si in "${solver_order[@]}"; do
                label="${SOLVER_LABELS[$si]}"
                solver_type="${SOLVER_TYPES[$si]}"
                extra_args=()
                if [[ -n "${SOLVER_EXTRA[$si]}" ]]; then
                    read -ra extra_args <<< "${SOLVER_EXTRA[$si]}"
                fi

                printf "${RI}  ${ARROW} ${BOLD}%-18s${RESET}" "$label"

                [[ -z "$first_run_t" ]] && t_r_before=$(timestamp)

                start_time=$(timestamp)
                result=$(run_solver "$cnxc" "$cnyc" "$cnzc" "$label" "$solver_type" "${r}" "${extra_args[@]+"${extra_args[@]}"}") || true
                end_time=$(timestamp)

                if [[ "$result" == "FAILED" || -z "$result" ]]; then
                    echo "  ${RED}${CROSS}${RESET}"
                    acc_failed[$si]=1
                    continue
                fi

                read -r ft mt pt it cv rs <<< "$result"

                wall_t=$(elapsed "$start_time" "$end_time")

                echo "  ${ft}s field ${DOT} ${wall_t}s wall  ${GREEN}${CHECK}${RESET}"

                # ETA after first completed run_solver
                if [[ -z "$first_run_t" && "$ft" != "NA" && "$DRY_RUN" != true ]]; then
                    t_r_after=$(timestamp)
                    # Scale one run_solver to one full config: ×AVG×NUM_SOLVERS
                    first_run_t=$(awk -v ta="$t_r_after" -v tb="$t_r_before" -v avg="$AVG" -v ns="$NUM_SOLVERS" 'BEGIN{printf "%.2f", (ta - tb) * avg * ns}')
                    show_eta "$first_run_t" "$cnxc"
                fi

                echo "$wall_t" >> "$_acc_dir/wall_$si"
                [[ "$ft" != "NA" && -n "$ft" ]] && echo "$ft" >> "$_acc_dir/field_$si"
                [[ "$mt" != "NA" && -n "$mt" ]] && echo "$mt" >> "$_acc_dir/moments_$si"
                [[ "$pt" != "NA" && -n "$pt" ]] && echo "$pt" >> "$_acc_dir/mover_$si"
                [[ "$it" != "NA" && -n "$it" ]] && echo "$it" >> "$_acc_dir/iters_$si"
                [[ "$rs" != "NA" && -n "$rs" ]] && echo "$rs" >> "$_acc_dir/resid_$si"
                [[ "$cv" == "no" ]] && acc_converged[$si]=false
                acc_lastlog[$si]="$TMPDIR_BASE/${label}_${cnxc}x${cnyc}x${cnzc}_r${r}/output.log"
            done
        done

        # Aggregate per-solver results
        if [[ $AVG -gt 1 ]]; then
            echo "${RI}Averages:"
        fi
        for ((si=0; si<NUM_SOLVERS; si++)); do
            label="${SOLVER_LABELS[$si]}"
            local_failed="${acc_failed[$si]}"
            # Read temp files into space-separated strings
            field_str=$(tr '\n' ' ' < "$_acc_dir/field_$si")
            wall_str=$(tr '\n' ' ' < "$_acc_dir/wall_$si")
            moments_str=$(tr '\n' ' ' < "$_acc_dir/moments_$si")
            mover_str=$(tr '\n' ' ' < "$_acc_dir/mover_$si")
            iters_str=$(tr '\n' ' ' < "$_acc_dir/iters_$si")
            resid_str=$(tr '\n' ' ' < "$_acc_dir/resid_$si")
            all_conv="${acc_converged[$si]}"
            last_log="${acc_lastlog[$si]}"

            if [[ "$local_failed" -eq 1 || -z "${field_str// /}" ]]; then
                ALL_LABELS+=("$label")
                ALL_FIELD+=("NA"); ALL_STD+=("NA"); ALL_MOMENTS+=("NA"); ALL_MOVER+=("NA")
                ALL_ITERS+=("NA"); ALL_CONVERGED+=("no"); ALL_RESIDUAL+=("NA")
                ALL_WALL+=("NA"); ALL_LOGS+=("none"); ALL_CONFIG_TAGS+=("$cnxc $cnyc $cnzc")
                continue
            fi

            stats=$(compute_stats "$field_str" "$wall_str" "$moments_str" "$mover_str" "$iters_str" "$resid_str")
            read -r r_field r_std r_moments r_mover r_iters r_residual r_wall <<< "$stats"
            r_converged="yes"
            [[ "$all_conv" == false ]] && r_converged="no"

            if [[ $AVG -gt 1 ]]; then
                printf "${RI}  %-18s avg: %s +/- %s s\n" "$label" "$r_field" "$r_std"
            fi

            ALL_LABELS+=("$label")
            ALL_FIELD+=("$r_field"); ALL_STD+=("$r_std")
            ALL_MOMENTS+=("$r_moments"); ALL_MOVER+=("$r_mover")
            ALL_ITERS+=("$r_iters"); ALL_CONVERGED+=("$r_converged"); ALL_RESIDUAL+=("$r_residual")
            ALL_WALL+=("$r_wall"); ALL_LOGS+=("$last_log")
            ALL_CONFIG_TAGS+=("$cnxc $cnyc $cnzc")
        done

    else
        # ── Sequential mode: run all AVG reps of each solver together ──
        for ((i=0; i<NUM_SOLVERS; i++)); do
            label="${SOLVER_LABELS[$i]}"
            solver_type="${SOLVER_TYPES[$i]}"
            extra_args=()
            if [[ -n "${SOLVER_EXTRA[$i]}" ]]; then
                read -ra extra_args <<< "${SOLVER_EXTRA[$i]}"
            fi

            [[ -z "$first_run_t" ]] && t_r_before=$(timestamp)

            local_result=$(run_averaged "$cnxc" "$cnyc" "$cnzc" "$label" "$solver_type" "${extra_args[@]+"${extra_args[@]}"}")

            if [[ -z "$first_run_t" && "$DRY_RUN" != true ]]; then
                t_r_after=$(timestamp)
                first_run_t=$(elapsed "$t_r_before" "$t_r_after")
                show_eta "$first_run_t" "$cnxc"
            fi

            read -r r_field r_std r_moments r_mover r_iters r_converged r_residual r_wall r_log <<< "$local_result"

            ALL_LABELS+=("$label")
            ALL_FIELD+=("$r_field")
            ALL_STD+=("$r_std")
            ALL_MOMENTS+=("$r_moments")
            ALL_MOVER+=("$r_mover")
            ALL_ITERS+=("$r_iters")
            ALL_CONVERGED+=("$r_converged")
            ALL_RESIDUAL+=("$r_residual")
            ALL_WALL+=("$r_wall")
            ALL_LOGS+=("$r_log")
            ALL_CONFIG_TAGS+=("$cnxc $cnyc $cnzc")
        done
    fi

    # Print per-config speedup in scaling mode
    if [[ "$SCALING_MODE" == true ]]; then
        # Find reference solver field time for this config
        base_idx=$(( (config_count - 1) * NUM_SOLVERS ))
        ref_t="${ALL_FIELD[$((base_idx + REF_SOLVER_IDX))]}"
        if [[ "$ref_t" != "NA" ]]; then
            for ((si=0; si<NUM_SOLVERS; si++)); do
                [[ $si -eq $REF_SOLVER_IDX ]] && continue
                idx=$(( base_idx + si ))
                other_t="${ALL_FIELD[$idx]}"
                if [[ "$other_t" != "NA" ]]; then
                    speedup=$(awk -v p="$other_t" -v g="$ref_t" 'BEGIN{if(p>0) printf "%.2f", g/p; else print "N/A"}')
                    echo "    ${SOLVER_LABELS[$si]} speedup: ${speedup}x"
                fi
            done
        fi
        echo ""
    fi
done

# ========================= Section 11: Results =============================
echo ""
printf -v _results_pad '%*s' 22 ''; _results_pad=${_results_pad// /$HDASH}
echo "${BOLD}${_results_pad} RESULTS ${_results_pad}${RESET}"
echo ""

# Export color codes for embedded Python blocks
export _C_GREEN="$GREEN" _C_RED="$RED" _C_RESET="$RESET" _C_BOLD="$BOLD"

if [[ "$SCALING_MODE" != true ]]; then
    # ---- Single-grid mode: detailed output ----
    # Convergence
    section "CONVERGENCE"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [${ALL_LABELS[$i]}] Convergence: (dry-run, no log)"
        elif [[ "${ALL_LOGS[$i]}" != "none" && -f "${ALL_LOGS[$i]}" ]]; then
            extract_convergence "${ALL_LOGS[$i]}" "${ALL_LABELS[$i]}"
        else
            echo "  [${ALL_LABELS[$i]}] Convergence: (no log available)"
        fi
        echo ""
    done

    # Timing
    section "TIMING"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [${ALL_LABELS[$i]}] Timing: (dry-run, no log)"
        elif [[ "${ALL_LOGS[$i]}" != "none" && -f "${ALL_LOGS[$i]}" ]]; then
            extract_timing "${ALL_LOGS[$i]}" "${ALL_LABELS[$i]}" "${ALL_FIELD[$i]}"
        else
            echo "  [${ALL_LABELS[$i]}] Timing: (no log available)"
        fi
        echo ""
    done

    # Comparison table
    section "COMPARISON SUMMARY" 79

    # Header
    printf "  ${BOLD}%-20s${RESET}" ""
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " ${BOLD}%14s${RESET}" "${ALL_LABELS[$i]}"
    done
    echo ""
    printf "  %-20s" "--------------------"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %14s" "--------------"
    done
    echo ""

    # Wall-clock row
    printf "  %-20s" "Wall-clock total"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %12s s" "${ALL_WALL[$i]}"
    done
    echo ""

    # Field solver row
    printf "  %-20s" "Field solver"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %12s s" "${ALL_FIELD[$i]}"
    done
    echo ""

    # Iterations row
    printf "  %-20s" "Avg iterations"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %14s" "${ALL_ITERS[$i]}"
    done
    echo ""

    # Converged row
    printf "  %-20s" "All converged?"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        val="${ALL_CONVERGED[$i]}"
        case "$val" in
            yes) printf " ${GREEN}%14s${RESET}" "yes" ;;
            no)  printf " ${RED}%14s${RESET}" "no" ;;
            *)   printf " %14s" "$val" ;;
        esac
    done
    echo ""

    # Residual row
    printf "  %-20s" "Avg residual"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %14s" "${ALL_RESIDUAL[$i]}"
    done
    echo ""

    # Init row
    printf "  %-20s" "Initialisation"
    for ((i=0; i<NUM_SOLVERS; i++)); do
        if [[ "$DRY_RUN" == true || "${ALL_LOGS[$i]}" == "none" ]]; then
            printf " %12s s" "N/A"
        elif [[ -f "${ALL_LOGS[$i]}" ]]; then
            init_t=$(awk '/Time needed for initialisation:/{v=$(NF-1)} END{print v}' "${ALL_LOGS[$i]}") || init_t="N/A"
            printf " %12s s" "${init_t:-N/A}"
        else
            printf " %12s s" "N/A"
        fi
    done
    echo ""
    echo ""

    # Speedup vs reference solver
    echo "  Speedup vs ${REF_SOLVER_LABEL} (field solver time):"
    field_data=""
    for ((i=0; i<NUM_SOLVERS; i++)); do
        field_data+="${ALL_LABELS[$i]} ${ALL_FIELD[$i]}"$'\n'
    done

    _REF_LABEL="$REF_SOLVER_LABEL" "$PYTHON" - <<EOF 2>/dev/null || echo "  (Could not compute speedup ratios)"
import os
_green = os.environ.get('_C_GREEN', '')
_red   = os.environ.get('_C_RED', '')
_reset = os.environ.get('_C_RESET', '')
_ref_label = os.environ.get('_REF_LABEL', 'GMRES')

data = {}
for line in """$field_data""".strip().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            data[parts[0]] = float(parts[1])
        except ValueError:
            pass
base = data.get(_ref_label)
if base:
    for label, val in data.items():
        if label != _ref_label and val > 0:
            sp = base / val
            c = _green if sp >= 1.0 else _red
            print(f'    {label:20s}  {c}{sp:.2f}x{_reset}  ({val:.2f}s vs {base:.2f}s)')
EOF
    echo ""

else
    # ---- Scaling mode: summary table ----
    section "SUMMARY"

    # Header
    printf "  ${BOLD}%-14s${RESET}" "Grid"
    for label in "${SOLVER_LABELS[@]}"; do
        printf " ${BOLD}%14s${RESET}" "$label"
    done
    printf " ${BOLD}%10s${RESET}\n" "Best"

    printf "  %-14s" "--------------"
    for label in "${SOLVER_LABELS[@]}"; do
        printf " %14s" "--------------"
    done
    printf " %10s\n" "----------"

    # Data rows
    for ((ci=0; ci<total_configs; ci++)); do
        base=$(( ci * NUM_SOLVERS ))
        read -r cnxc cnyc cnzc <<< "${ALL_CONFIG_TAGS[$base]}"
        printf "  %-14s" "${cnxc}x${cnyc}x${cnzc}"

        best_label=""
        best_time=999999
        for ((si=0; si<NUM_SOLVERS; si++)); do
            idx=$(( base + si ))
            t="${ALL_FIELD[$idx]}"
            printf " %12s s" "$t"
            if [[ "$t" != "NA" ]]; then
                is_better=$(awk -v t="$t" -v bt="$best_time" 'BEGIN{print (t < bt) ? "yes" : "no"}')
                if [[ "$is_better" == "yes" ]]; then
                    best_time="$t"
                    best_label="${ALL_LABELS[$idx]}"
                fi
            fi
        done
        printf " %10s\n" "$best_label"
    done
    echo ""

    # Average speedup vs reference solver across all grid sizes
    echo "  Average speedup vs ${REF_SOLVER_LABEL} (field solver time):"

    # Build per-solver speedup data: label ref_t1 other_t1 ref_t2 other_t2 ...
    speedup_input=""
    for ((si=0; si<NUM_SOLVERS; si++)); do
        [[ $si -eq $REF_SOLVER_IDX ]] && continue
        speedup_input+="${SOLVER_LABELS[$si]}"
        for ((ci=0; ci<total_configs; ci++)); do
            base=$(( ci * NUM_SOLVERS ))
            ref_t="${ALL_FIELD[$((base + REF_SOLVER_IDX))]}"
            other_t="${ALL_FIELD[$((base + si))]}"
            speedup_input+=" $ref_t $other_t"
        done
        speedup_input+=$'\n'
    done

    _SPEEDUP_DATA="$speedup_input" "$PYTHON" - <<'PYSPEEDUP' 2>/dev/null || echo "  (Could not compute average speedup)"
import os

_green = os.environ.get('_C_GREEN', '')
_red   = os.environ.get('_C_RED', '')
_reset = os.environ.get('_C_RESET', '')

results = []
for line in os.environ['_SPEEDUP_DATA'].strip().splitlines():
    parts = line.split()
    if len(parts) < 3:
        continue
    label = parts[0]
    speedups = []
    for i in range(1, len(parts), 2):
        gt, pt = parts[i], parts[i+1]
        try:
            g, p = float(gt), float(pt)
            if g > 0 and p > 0:
                speedups.append(g / p)
        except ValueError:
            pass
    if speedups:
        avg = sum(speedups) / len(speedups)
        results.append((label, avg, speedups))

# Sort by average speedup descending
results.sort(key=lambda x: x[1], reverse=True)

for label, avg, sps in results:
    lo, hi = min(sps), max(sps)
    c = _green if avg >= 1.0 else _red
    if avg >= 1.0:
        print(f'    {label:20s}  {c}{avg:.2f}x{_reset} avg  (range: {lo:.2f}x .. {hi:.2f}x)')
    else:
        pct = (1 - avg) * 100
        print(f'    {label:20s}  {c}{avg:.2f}x{_reset} avg  ({pct:.0f}% slower, range: {lo:.2f}x .. {hi:.2f}x)')
PYSPEEDUP
    echo ""
fi

# ========================= Section 12: CSV + Plot + Visual comparison ======

# ---- Write CSV ----
{
    # Header
    printf "grid_size"
    for label in "${SOLVER_LABELS[@]}"; do
        printf ",%s,%s_std,%s_iters,%s_converged,%s_residual" \
            "$label" "$label" "$label" "$label" "$label"
    done
    printf ",np,cycles,nzc,ref_solver,ref_moments_s,ref_mover_s"
    printf ",ref_field_pct,ref_moments_pct,ref_mover_pct"
    echo ""

    # Data rows
    for ((ci=0; ci<total_configs; ci++)); do
        base=$(( ci * NUM_SOLVERS ))
        read -r cnxc cnyc cnzc <<< "${ALL_CONFIG_TAGS[$base]}"

        # grid_size column: numeric N for scaling, NxNxN string for single-grid
        if [[ "$SCALING_MODE" == true ]]; then
            printf "%s" "$cnxc"
        else
            printf "%s" "${cnxc}x${cnyc}x${cnzc}"
        fi

        local_ref_field="NA"
        local_ref_moments="NA"
        local_ref_mover="NA"

        for ((si=0; si<NUM_SOLVERS; si++)); do
            idx=$(( base + si ))
            printf ",%s,%s,%s,%s,%s" \
                "${ALL_FIELD[$idx]}" "${ALL_STD[$idx]}" \
                "${ALL_ITERS[$idx]}" "${ALL_CONVERGED[$idx]}" "${ALL_RESIDUAL[$idx]}"

            if [[ $si -eq $REF_SOLVER_IDX ]]; then
                local_ref_field="${ALL_FIELD[$idx]}"
                local_ref_moments="${ALL_MOMENTS[$idx]}"
                local_ref_mover="${ALL_MOVER[$idx]}"
            fi
        done

        # Reference solver breakdown percentages
        ref_pcts=$("$PYTHON" -c "
import sys
f_s, m_s, p_s = sys.argv[1], sys.argv[2], sys.argv[3]
f = float(f_s) if f_s != 'NA' else None
m = float(m_s) if m_s != 'NA' else None
p = float(p_s) if p_s != 'NA' else None
vals = [v for v in [f, m, p] if v is not None]
total = sum(vals) if vals else 0
if total > 0 and m is not None and p is not None:
    print(f'{f/total*100:.1f},{m/total*100:.1f},{p/total*100:.1f}')
else:
    print('NA,NA,NA')
" "$local_ref_field" "$local_ref_moments" "$local_ref_mover" 2>/dev/null) || ref_pcts="NA,NA,NA"

        printf ",%s,%s,%s,%s,%s,%s,%s\n" \
            "$NP" "$CYCLES" "$cnzc" "$REF_SOLVER_LABEL" \
            "$local_ref_moments" "$local_ref_mover" "$ref_pcts"
    done
} > "$CSV_FILE"

# Write ref_solver.txt so plot scripts can auto-discover the reference
echo "$REF_SOLVER_LABEL" > "$PERSISTENT_OUTPUT_DIR/ref_solver.txt"

echo "  ${ARROW} CSV saved to: $CSV_FILE"

# ---- Generate plot ----
if [[ "$NO_PLOT" != true ]]; then
    echo "  ${ARROW} Generating plot..."
    export CSV_FILE PLOT_FILE NP CYCLES
    export PROFILE_DIR="$PERSISTENT_OUTPUT_DIR"
    export SOLVER_LABEL_LIST="${SOLVER_LABELS[*]}"
    export REF_SOLVER="$REF_SOLVER_LABEL"

    # Build breakdown data for env export
    BD_GRIDS_arr=() BD_FIELD_arr=() BD_MOMENTS_arr=() BD_MOVER_arr=()
    for ((ci=0; ci<total_configs; ci++)); do
        base=$(( ci * NUM_SOLVERS ))
        read -r cnxc cnyc cnzc <<< "${ALL_CONFIG_TAGS[$base]}"
        BD_GRIDS_arr+=("$cnxc")
        BD_FIELD_arr+=("${ALL_FIELD[$base]}")
        BD_MOMENTS_arr+=("${ALL_MOMENTS[$base]}")
        BD_MOVER_arr+=("${ALL_MOVER[$base]}")
    done
    export NZC="$eff_nzc"
    export BD_GRIDS_LIST="${BD_GRIDS_arr[*]}"
    export BD_FIELD_LIST="${BD_FIELD_arr[*]}"
    export BD_MOMENTS_LIST="${BD_MOMENTS_arr[*]}"
    export BD_MOVER_LIST="${BD_MOVER_arr[*]}"

    "$PYTHON" "$SCRIPTS_DIR/plot_timing.py" "$CSV_FILE" || echo "  ${YELLOW}WARNING:${RESET} Plot generation failed (see above)."
    "$PYTHON" "$SCRIPTS_DIR/plot_energy.py" "$CSV_FILE" || echo "  ${YELLOW}WARNING:${RESET} Energy plot generation failed."
else
    echo "  ${ARROW} Plotting skipped (--no-plot)."
fi

# ---- Visual comparison, movie & L2 (when field output was saved) ----
# Loop over all grid configs so each gets its own comparison plots.
if [[ -n "$FIELD_OUTPUT" && "$FIELD_OUTPUT" != "0" && "$DRY_RUN" != true ]]; then
    for v_cfg in "${CONFIGS[@]}"; do
        read -r v_nxc v_nyc v_nzc <<< "$v_cfg"
        v_config_tag="${v_nxc}x${v_nyc}x${v_nzc}"
        ref_dir="$PERSISTENT_OUTPUT_DIR/${REF_SOLVER_LABEL}_${v_config_tag}"
        test_args=()
        for ((si=0; si<NUM_SOLVERS; si++)); do
            [[ $si -eq $REF_SOLVER_IDX ]] && continue
            test_dir="$PERSISTENT_OUTPUT_DIR/${SOLVER_LABELS[$si]}_${v_config_tag}"
            if [[ -d "$test_dir" ]]; then
                test_args+=(--test "$test_dir")
            fi
        done

        [[ -d "$ref_dir" && ${#test_args[@]} -gt 0 ]] || continue

        if [[ -f "$SCRIPTS_DIR/plot_field_comparison.py" ]]; then
            echo "  ${ARROW} Visual comparison ${DOT} ${v_config_tag}"
            "$PYTHON" "$SCRIPTS_DIR/plot_field_comparison.py" \
                --ref "$ref_dir" "${test_args[@]}" || \
                echo "  ${YELLOW}WARNING:${RESET} Visual comparison failed for ${v_config_tag}."
        fi

        # ---- Movie generation (when --movie flag was passed) ----
        if [[ "$MAKE_MOVIE" == true ]]; then
            if command -v ffmpeg &>/dev/null; then
                echo "  ${ARROW} Generating movie ${DOT} ${v_config_tag}"
                bash "$SCRIPT_DIR/make_movie.sh" \
                    --ref "$ref_dir" "${test_args[@]}" || \
                    echo "  ${YELLOW}WARNING:${RESET} Movie generation failed for ${v_config_tag}."
            else
                echo "  ${YELLOW}WARNING:${RESET} ffmpeg not found, skipping movie generation."
            fi
        fi

        # ---- L2 time-series plot ----
        if [[ -f "$SCRIPTS_DIR/plot_l2_timeseries.py" ]]; then
            l2_test_args=()
            for ((si=0; si<NUM_SOLVERS; si++)); do
                [[ $si -eq $REF_SOLVER_IDX ]] && continue
                test_dir="$PERSISTENT_OUTPUT_DIR/${SOLVER_LABELS[$si]}_${v_config_tag}"
                [[ -d "$test_dir" ]] && l2_test_args+=(--test "$test_dir")
            done
            if [[ ${#l2_test_args[@]} -gt 0 ]]; then
                echo "  ${ARROW} L2 time-series plot ${DOT} ${v_config_tag}"
                "$PYTHON" "$SCRIPTS_DIR/plot_l2_timeseries.py" \
                    --ref "$ref_dir" "${l2_test_args[@]}" || \
                    echo "  ${YELLOW}WARNING:${RESET} L2 time-series plot failed for ${v_config_tag}."
            fi
        fi
    done
fi

# ---- Validation ----
if [[ "$NO_VALIDATE" != true && "$DRY_RUN" != true ]]; then
    echo ""
    echo "  ${ARROW} Running validation checks..."
    VALIDATE_ARGS=()
    if [[ -n "$FIELD_OUTPUT" && "$FIELD_OUTPUT" != "0" ]]; then
        VALIDATE_ARGS+=(--strict)
    fi
    if ! "$PYTHON" "$SCRIPTS_DIR/validate.py" "$PERSISTENT_OUTPUT_DIR" "${VALIDATE_ARGS[@]}"; then
        echo "  ${RED}${CROSS}${RESET} Validation FAILED"
        exit 1
    fi
fi

echo ""
printf -v _trail_sep '%*s' 60 ''; _trail_sep=${_trail_sep// /$MDASH}
echo "${DIM}${_trail_sep}${RESET}"
