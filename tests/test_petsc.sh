#!/usr/bin/env bash
#
# test_petsc.sh — Compare GMRES vs PETSc field solver performance
#
# Two modes:
#   Single-grid (default): Runs all solvers on one grid config.
#     Outputs: convergence info, timing breakdown, comparison table, CSV, plot.
#   Scaling (--grid-min/--grid-max): Sweeps grid sizes.
#     Outputs: summary table, CSV, plot.
#
# Usage:
#   # Single-grid mode
#   ./tests/test_petsc.sh [--np N] [--cycles N] [--grid NXC NYC [NZC]] [--topo X Y [Z]] ...
#
#   # Scaling mode
#   ./tests/test_petsc.sh --grid-min 50 --grid-max 400 [--grid-step N] ...
#
# Requirements:
#   - Build with PETSc:  ./build.sh --petsc
#   - MPI available (mpirun)

set -euo pipefail

# ========================= Section 1: Setup ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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
OUTPUT_DIR=""
WRITE_METHOD="phdf5"
MAKE_MOVIE=false
CLEAN_OUTPUT=false
TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_test_petsc.XXXXXX")

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
        --randomize)    RANDOMIZE=true;     shift ;;
        --clean)        CLEAN_OUTPUT=true;  shift ;;
        --movie)        MAKE_MOVIE=true;    shift ;;
        --dry-run)      DRY_RUN=true;       shift ;;
        --input)        INPUT_FILE="$2";    shift 2 ;;
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
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Error: --output-dir requires a path" >&2; exit 2; }
            OUTPUT_DIR="$2"; shift 2 ;;
        --write-method)
            [[ $# -ge 2 ]] || { echo "Error: --write-method requires a value (phdf5, shdf5, pvtk)" >&2; exit 2; }
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
  --all-solvers        Add PETSc fgmres, bcgs, tfqmr
  --randomize          Shuffle solver run order
  --input FILE         Input file path (default: inputfiles/Double_Harris.inp)
  --name TAG           Output dir suffix: test_petsc_output_TAG/
  --field-output N     Save field data every N cycles (default: CYCLES in single-grid, off in scaling)
  --particle-output N  Save particle data every N cycles (0 = off)
  --movie              Generate mp4 movie of field comparison across cycles (requires ffmpeg)
  --output-dir DIR     Override output directory (default: tests/test_petsc_output/)
  --write-method M     HDF5 write method: phdf5, shdf5, pvtk (default: phdf5)
  --clean              Remove output directory before running (start fresh)
  --dry-run            Print commands without running them

Output:
  CSV and plots are saved to tests/test_petsc_output/ (or test_petsc_output_TAG/).
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
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

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

# ZLEN defaults to 1 (2D)
ZLEN="${ZLEN:-1}"

# If --topo was not given, auto-compute most square-like XLEN * YLEN factorization
if [[ -z "$XLEN" || -z "$YLEN" ]]; then
    NP_XY=$(( NP / ZLEN ))
    read -r XLEN YLEN <<< "$(compute_topology "$NP_XY")"
fi

# Ensure the grid is divisible by the MPI topology in each dimension.
# Read values from the input file and round up to the nearest multiple if needed.
if [[ -z "$NXC" ]]; then
    file_nxc=$(grep -m1 '^nxc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')
    file_nxc=${file_nxc:-100}
    rem=$((file_nxc % XLEN))
    if [[ $rem -ne 0 ]]; then
        NXC=$(( file_nxc + XLEN - rem ))
    fi
fi
if [[ -z "$NYC" ]]; then
    file_nyc=$(grep -m1 '^nyc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')
    file_nyc=${file_nyc:-100}
    rem=$((file_nyc % YLEN))
    if [[ $rem -ne 0 ]]; then
        NYC=$(( file_nyc + YLEN - rem ))
    fi
fi
if [[ -z "$NZC" && "$ZLEN" -gt 1 ]]; then
    file_nzc=$(grep -m1 '^nzc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')
    file_nzc=${file_nzc:-1}
    rem=$((file_nzc % ZLEN))
    if [[ $rem -ne 0 ]]; then
        NZC=$(( file_nzc + ZLEN - rem ))
    fi
fi

# Effective grid values for config generation
eff_nxc="${NXC:-$(grep -m1 '^nxc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')}"
eff_nyc="${NYC:-$(grep -m1 '^nyc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')}"
eff_nzc="${NZC:-$(grep -m1 '^nzc' "$INPUT_FILE" | awk -F= '{print $2}' | awk '{print $1}')}"
eff_nxc="${eff_nxc:-100}"
eff_nyc="${eff_nyc:-100}"
eff_nzc="${eff_nzc:-1}"

# Validate scaling range
if [[ "$SCALING_MODE" == true ]]; then
    [[ $GRID_MIN -lt $GRID_MAX ]] \
        || { echo "Error: --grid-min ($GRID_MIN) must be less than --grid-max ($GRID_MAX)" >&2; exit 2; }
fi

# Output directory and file paths
if [[ -n "$OUTPUT_DIR" ]]; then
    PERSISTENT_OUTPUT_DIR="$OUTPUT_DIR"
elif [[ -n "$NAME_TAG" ]]; then
    PERSISTENT_OUTPUT_DIR="$SCRIPT_DIR/test_petsc_output_${NAME_TAG}"
else
    PERSISTENT_OUTPUT_DIR="$SCRIPT_DIR/test_petsc_output"
fi
if [[ "$CLEAN_OUTPUT" == true && "$DRY_RUN" != true && -d "$PERSISTENT_OUTPUT_DIR" ]]; then
    rm -rf "$PERSISTENT_OUTPUT_DIR"
fi
mkdir -p "$PERSISTENT_OUTPUT_DIR"

CSV_FILE="$PERSISTENT_OUTPUT_DIR/results.csv"
PLOT_FILE="$PERSISTENT_OUTPUT_DIR/results.png"

# ========================= Section 5: Solver configuration =================
SOLVER_LABELS=("GMRES"      "PETSc_gmres")
SOLVER_TYPES=( "GMRES"      "PETSc")
SOLVER_EXTRA=( ""           "")

if [[ "$ALL_SOLVERS" == true ]]; then
    SOLVER_LABELS+=("PETSc_fgmres"                                          "PETSc_bcgs"                      "PETSc_tfqmr")
    SOLVER_TYPES+=( "PETSc"                                                  "PETSc"                           "PETSc")
    SOLVER_EXTRA+=( "-ksp_type fgmres -ksp_gmres_restart 50 -ksp_max_it 1000" "-ksp_type bcgs -ksp_max_it 1000" "-ksp_type tfqmr -ksp_max_it 1000")
fi

NUM_SOLVERS=${#SOLVER_LABELS[@]}

# ========================= Section 6: Prerequisite checks ==================
if [[ "$DRY_RUN" != true ]]; then
    if [[ ! -x "$EXECUTABLE" ]]; then
        echo "ERROR: Executable not found at $EXECUTABLE"
        echo "       Build first with:  ./build.sh --petsc"
        exit 1
    fi
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "ERROR: Input file not found: $INPUT_FILE"
        exit 1
    fi
fi

warn_oversubscription "$NP"

# ========================= Section 7: Helper functions =====================
cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT INT TERM

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
        -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle         = $CYCLES|"
        -e "s|^WriteMethod.*=.*|WriteMethod                    = $WRITE_METHOD|"
    )
    sed "${sed_args[@]}" "$INPUT_FILE" > "$inp"
    echo "$inp"
}

# Extract per-cycle timing profile from a log file.
# Args: logfile output_csv
extract_profile() {
    local logfile="$1" outcsv="$2"
    python3 - "$logfile" "$outcsv" <<'PYPROFILE'
import re, sys, csv

logfile, outcsv = sys.argv[1], sys.argv[2]
fields = []  # list of (field, mover, moment, write) cumulative tuples

with open(logfile) as f:
    cur = {}
    for line in f:
        for key, pat in [('field', r'Field solver\s*:\s*([\d.eE+\-]+)'),
                         ('mover', r'Particle mover\s*:\s*([\d.eE+\-]+)'),
                         ('moment', r'Moment gatherer\s*:\s*([\d.eE+\-]+)'),
                         ('write', r'Write data\s*:\s*([\d.eE+\-]+)')]:
            m = re.search(pat, line)
            if m:
                cur[key] = float(m.group(1))
        if len(cur) == 4:
            fields.append((cur['field'], cur['mover'], cur['moment'], cur['write']))
            cur = {}

with open(outcsv, 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['cycle', 'field_solver_s', 'particle_mover_s', 'moment_gatherer_s', 'write_data_s'])
    prev = (0, 0, 0, 0)
    for i, vals in enumerate(fields):
        delta = tuple(v - p for v, p in zip(vals, prev))
        w.writerow([i + 1] + [f'{d:.6f}' for d in delta])
        prev = vals
PYPROFILE
}

# Parse convergence info from a log file.
# Stdout: "avg_iters converged avg_residual"
#   avg_iters: average iterations per solve call
#   converged: "yes" if all calls converged, "no" if any failed
#   avg_residual: average final residual
parse_convergence() {
    local logfile="$1"
    python3 - "$logfile" <<'PYCONV'
import re, sys

logfile = sys.argv[1]
iters_list = []
residuals = []
any_failed = False

with open(logfile) as f:
    for line in f:
        # GMRES: "GMRES converged at restart R; iteration I with error: E"
        m = re.search(r'GMRES converged at restart (\d+); iteration (\d+) with error:\s*([\d.eE+\-]+)', line)
        if m:
            restart, iteration = int(m.group(1)), int(m.group(2))
            total_iters = restart * 20 + iteration  # m=20 from EMfields3D.cpp
            iters_list.append(total_iters)
            residuals.append(float(m.group(3)))
            continue

        # GMRES: "GMRES converged without iterations"
        if 'GMRES converged without iterations' in line:
            iters_list.append(0)
            residuals.append(0.0)
            continue

        # GMRES: "GMRES not converged !! Final error: E"
        m = re.search(r'GMRES not converged.*Final error:\s*([\d.eE+\-]+)', line)
        if m:
            any_failed = True
            residuals.append(float(m.group(1)))
            continue

        # PETSc: "PETSc KSP converged in I iterations, residual = E"
        m = re.search(r'PETSc KSP converged in (\d+) iterations, residual\s*=\s*([\d.eE+\-]+)', line)
        if m:
            iters_list.append(int(m.group(1)))
            residuals.append(float(m.group(2)))
            continue

        # PETSc: "WARNING: PETSc KSP did NOT converge (reason=R) after I iterations, residual = E"
        m = re.search(r'PETSc KSP did NOT converge.*after (\d+) iterations, residual\s*=\s*([\d.eE+\-]+)', line)
        if m:
            any_failed = True
            iters_list.append(int(m.group(1)))
            residuals.append(float(m.group(2)))
            continue

if not iters_list and not residuals:
    print("NA yes NA")
else:
    avg_iters = sum(iters_list) / len(iters_list) if iters_list else 0
    converged = "no" if any_failed else "yes"
    avg_residual = sum(residuals) / len(residuals) if residuals else 0
    print(f"{avg_iters:.1f} {converged} {avg_residual:.2e}")
PYCONV
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

    local cmd=(mpirun -np "$NP" "$EXECUTABLE")
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

    if ! "${cmd[@]}" > "$logfile" 2>&1; then
        echo "FAILED"
        return 1
    fi

    # Extract timings
    local field_t moments_t mover_t
    field_t=$(grep "Field solver" "$logfile" | tail -1 | awk '{print $(NF-1)}') || field_t="NA"
    moments_t=$(grep "Moment gatherer" "$logfile" | tail -1 | awk '{print $(NF-1)}') || moments_t="NA"
    mover_t=$(grep "Particle mover" "$logfile" | tail -1 | awk '{print $(NF-1)}') || mover_t="NA"
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
    [[ $AVG -gt 1 ]] && runs_info=",  Runs: $AVG"
    echo "${I}--- $label ($solver_type) — ${nxc}x${nyc}x${nzc},  ${NP} procs,  ${CYCLES} cyc${runs_info}" >&2

    local field_arr=() wall_arr=() moments_arr=() mover_arr=()
    local iters_arr=() residual_arr=()
    local all_converged=true
    local last_logfile=""
    local failed=0

    for ((r=1; r<=AVG; r++)); do
        if [[ $r -gt 1 && $COOLDOWN -gt 0 ]]; then
            echo "${I}  Cooling down for ${COOLDOWN}s..." >&2
            sleep "$COOLDOWN"
        fi

        local start_time
        start_time=$(python3 -c 'import time; print(time.time())')

        local result
        result=$(run_solver "$nxc" "$nyc" "$nzc" "$label" "$solver_type" "$r" "${extra_args[@]+"${extra_args[@]}"}")

        local end_time
        end_time=$(python3 -c 'import time; print(time.time())')

        if [[ "$result" == "FAILED" || -z "$result" ]]; then
            failed=1; break
        fi

        local ft mt pt it cv rs
        read -r ft mt pt it cv rs <<< "$result"

        local wall_t
        wall_t=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")
        wall_arr+=("$wall_t")

        [[ "$ft" != "NA" && -n "$ft" ]] && field_arr+=("$ft")
        [[ "$mt" != "NA" && -n "$mt" ]] && moments_arr+=("$mt")
        [[ "$pt" != "NA" && -n "$pt" ]] && mover_arr+=("$pt")
        [[ "$it" != "NA" && -n "$it" ]] && iters_arr+=("$it")
        [[ "$rs" != "NA" && -n "$rs" ]] && residual_arr+=("$rs")
        [[ "$cv" == "no" ]] && all_converged=false

        last_logfile="$TMPDIR_BASE/${label}_${nxc}x${nyc}x${nzc}_r${r}/output.log"

        if [[ $AVG -gt 1 ]]; then
            printf "${I}  Run %d/%d — wall: %s s,  field: %s s\n" "$r" "$AVG" "$wall_t" "$ft" >&2
        else
            echo "${I}  Wall: ${wall_t} s,  Field: ${ft} s" >&2
        fi
    done

    if [[ $failed -eq 1 || ${#field_arr[@]} -eq 0 ]]; then
        echo "${I}  FAILED" >&2
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
    stats=$(python3 -c "
import statistics
def mean_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.mean(vals):.3f}' if vals else 'NA'
def std_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.stdev(vals):.3f}' if len(vals) > 1 else 'NA'

field   = '$field_str'.split()
wall    = '$wall_str'.split()
moments = '$moments_str'.split()
mover   = '$mover_str'.split()
iters   = '$iters_str'.split()
resid   = '$resid_str'.split()

avg_field   = mean_str(field)
std_field   = std_str(field)
avg_moments = mean_str(moments)
avg_mover   = mean_str(mover)
avg_iters   = mean_str(iters)
avg_resid   = mean_str(resid)
avg_wall    = mean_str(wall)

print(f'{avg_field} {std_field} {avg_moments} {avg_mover} {avg_iters} {avg_resid} {avg_wall}')
")

    local avg_field std_field avg_moments avg_mover avg_iters avg_residual avg_wall
    read -r avg_field std_field avg_moments avg_mover avg_iters avg_residual avg_wall <<< "$stats"

    local converged_str="yes"
    [[ "$all_converged" == false ]] && converged_str="no"

    if [[ $AVG -gt 1 ]]; then
        echo "${I}  Mean field: ${avg_field} +/- ${std_field} s,  Mean wall: ${avg_wall} s" >&2
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
    init_t=$(grep "Time needed for initialisation:" "$logfile" | tail -1 | awk '{print $(NF-1)}') || init_t="N/A"
    mover_t=$(grep "Particle mover" "$logfile" | tail -1 | awk '{print $(NF-1)}') || mover_t="N/A"
    moment_t=$(grep "Moment gatherer" "$logfile" | tail -1 | awk '{print $(NF-1)}') || moment_t="N/A"
    write_t=$(grep "Write data" "$logfile" | tail -1 | awk '{print $(NF-1)}') || write_t="N/A"

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

    echo "  [$label] Convergence:"

    local gmres_lines gmres_fail petsc_conv petsc_fail
    gmres_lines=$(grep -c "GMRES converged" "$logfile" 2>/dev/null) || gmres_lines=0
    gmres_fail=$(grep -c "GMRES not converged" "$logfile" 2>/dev/null) || gmres_fail=0
    petsc_conv=$(grep -c "PETSc KSP converged" "$logfile" 2>/dev/null) || petsc_conv=0
    petsc_fail=$(grep -c "PETSc KSP did NOT converge" "$logfile" 2>/dev/null) || petsc_fail=0

    if [[ $gmres_lines -gt 0 || $gmres_fail -gt 0 ]]; then
        echo "    GMRES: $gmres_lines converged, $gmres_fail failed"
        grep "GMRES converged" "$logfile" | head -1 | sed 's/^/    First: /'
        grep "GMRES converged" "$logfile" | tail -1 | sed 's/^/    Last:  /'
    fi

    if [[ $petsc_conv -gt 0 || $petsc_fail -gt 0 ]]; then
        echo "    PETSc: $petsc_conv converged, $petsc_fail failed"
        grep "PETSc KSP converged" "$logfile" | head -1 | sed 's/^/    First: /'
        grep "PETSc KSP converged" "$logfile" | tail -1 | sed 's/^/    Last:  /'
    fi

    if [[ $gmres_lines -eq 0 && $gmres_fail -eq 0 && $petsc_conv -eq 0 && $petsc_fail -eq 0 ]]; then
        echo "    (no convergence messages found)"
    fi
}

# Print ETA estimate.
# Args: t1 (seconds for one run_averaged call), g0 (grid size of that call)
# Uses globals: CONFIGS, NUM_SOLVERS, t_sweep_start, SCALING_MODE
show_eta() {
    local t1="$1" g0="$2"
    python3 - <<PYETA || true
import time
from datetime import datetime, timedelta

t1        = $t1
g0        = $g0
n_solvers = $NUM_SOLVERS
elapsed   = time.time() - $t_sweep_start
scaling   = "$SCALING_MODE" == "true"

if scaling:
    # N^2-scaled ETA
    configs = """${CONFIGS[*]}""".split()
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
title = 'ESTIMATED TOTAL RUN TIME'
msg = f'~{fmt(total_est)} total  |  ~{fmt(remaining_est)} remaining  |  ETA {eta:%H:%M}'
pad = max(len(title), len(msg)) + 4
print(f'  +{"-" * pad}+')
print(f'  |  {title:^{pad-4}}  |')
print(f'  +{"-" * pad}+')
print(f'  |  {msg:^{pad-4}}  |')
print(f'  +{"-" * pad}+')
PYETA
}

# ========================= Section 8: Configuration generator ==============
CONFIGS=()
if [[ "$SCALING_MODE" == true ]]; then
    # Compute LCM of XLEN and YLEN for grid divisibility
    lcm=$(python3 -c "import math; print(math.lcm($XLEN, $YLEN))")

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
    box_title="iPIC3D Solver Comparison: GMRES vs PETSc"
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

box_lines+=("  Solvers: ${SOLVER_LABELS[*]}")
for ((si=0; si<NUM_SOLVERS; si++)); do
    if [[ -n "${SOLVER_EXTRA[$si]}" ]]; then
        box_lines+=("    ${SOLVER_LABELS[$si]}: ${SOLVER_EXTRA[$si]}")
    fi
done
[[ $AVG -gt 1 ]] && box_lines+=("  Runs: $AVG   Cooldown: ${COOLDOWN}s between runs")
[[ -n "$NAME_TAG" ]] && box_lines+=("  Name tag: $NAME_TAG")
[[ -n "$FIELD_OUTPUT" || -n "$PARTICLE_OUTPUT" ]] && box_lines+=("  Field output: ${FIELD_OUTPUT:-off}   Particle output: ${PARTICLE_OUTPUT:-off}")
[[ "$MAKE_MOVIE" == true ]] && box_lines+=("  Movie: yes (generate mp4 comparison movie)")
[[ "$RANDOMIZE" == true ]] && box_lines+=("  Randomize: yes (solver order shuffled)")
[[ "$DRY_RUN" == true ]] && box_lines+=("  *** DRY-RUN MODE — no simulations will be executed ***")

# Compute box width: max of title and all content lines, plus 2 for padding
BOX_W=$(( ${#box_title} + 4 ))  # title + margins
for line in "${box_lines[@]}"; do
    (( ${#line} + 2 > BOX_W )) && BOX_W=$(( ${#line} + 2 ))
done

box_border=$(printf -- '-%.0s' $(seq 1 $BOX_W))
box_pad=$(( (BOX_W - ${#box_title}) / 2 ))
echo "+${box_border}+"
printf "|%*s%-*s|\n" $box_pad "" $((BOX_W - box_pad)) "$box_title"
echo "+${box_border}+"
for line in "${box_lines[@]}"; do
    printf "|%-${BOX_W}s|\n" "$line"
done
echo "+${box_border}+"
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

t_sweep_start=$(python3 -c 'import time; print(time.time())')
first_run_t=""

config_count=0
total_configs=${#CONFIGS[@]}

for cfg in "${CONFIGS[@]}"; do
    read -r cnxc cnyc cnzc <<< "$cfg"
    config_count=$((config_count + 1))
    config_tag="${cnxc}x${cnyc}x${cnzc}"

    if [[ "$SCALING_MODE" == true ]]; then
        echo "--------------------------------------------------------"
        echo "  [$config_count/$total_configs]  Grid: ${config_tag}  (${CYCLES} cycles, ${NP} procs)"
        echo "--------------------------------------------------------"
    fi

    if [[ "$RANDOMIZE" == true ]]; then
        # ── Interleaved mode: AVG rounds, each with all solvers in shuffled order ──
        # This avoids running all reps of one solver consecutively, reducing
        # bias from CPU throttling.

        # Indent prefix for scaling vs single-grid
        RI="  "
        [[ "$SCALING_MODE" == true ]] && RI="    "

        # Per-solver accumulators (indexed by solver position)
        for ((si=0; si<NUM_SOLVERS; si++)); do
            eval "acc_field_$si=()"
            eval "acc_wall_$si=()"
            eval "acc_moments_$si=()"
            eval "acc_mover_$si=()"
            eval "acc_iters_$si=()"
            eval "acc_resid_$si=()"
            eval "acc_converged_$si=true"
            eval "acc_lastlog_$si='none'"
            eval "acc_failed_$si=0"
        done

        for ((r=1; r<=AVG; r++)); do
            if [[ $r -gt 1 && $COOLDOWN -gt 0 ]]; then
                echo "${RI}Cooling down for ${COOLDOWN}s..."
                sleep "$COOLDOWN"
            fi

            # Shuffle solver indices for this round
            solver_order=($(python3 -c "import random; x=list(range($NUM_SOLVERS)); random.shuffle(x); print(*x)"))
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

                printf "${RI}  %-18s ... " "$label"

                [[ -z "$first_run_t" ]] && t_r_before=$(python3 -c 'import time; print(time.time())')

                start_time=$(python3 -c 'import time; print(time.time())')
                result=$(run_solver "$cnxc" "$cnyc" "$cnzc" "$label" "$solver_type" "${r}" "${extra_args[@]+"${extra_args[@]}"}")
                end_time=$(python3 -c 'import time; print(time.time())')

                if [[ "$result" == "FAILED" || -z "$result" ]]; then
                    echo "FAILED"
                    eval "acc_failed_$si=1"
                    continue
                fi

                read -r ft mt pt it cv rs <<< "$result"

                wall_t=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")

                echo "${ft} s  (wall: ${wall_t} s)"

                # ETA after first completed run_solver
                if [[ -z "$first_run_t" && "$ft" != "NA" && "$DRY_RUN" != true ]]; then
                    t_r_after=$(python3 -c 'import time; print(time.time())')
                    # Scale one run_solver to one full config: ×AVG×NUM_SOLVERS
                    first_run_t=$(python3 -c "print(($t_r_after - $t_r_before) * $AVG * $NUM_SOLVERS)")
                    show_eta "$first_run_t" "$cnxc"
                fi

                eval "acc_field_$si+=('$ft')"
                eval "acc_wall_$si+=('$wall_t')"
                [[ "$ft" != "NA" && -n "$ft" ]] || true
                [[ "$mt" != "NA" && -n "$mt" ]] && eval "acc_moments_$si+=('$mt')"
                [[ "$pt" != "NA" && -n "$pt" ]] && eval "acc_mover_$si+=('$pt')"
                [[ "$it" != "NA" && -n "$it" ]] && eval "acc_iters_$si+=('$it')"
                [[ "$rs" != "NA" && -n "$rs" ]] && eval "acc_resid_$si+=('$rs')"
                [[ "$cv" == "no" ]] && eval "acc_converged_$si=false"
                eval "acc_lastlog_$si='$TMPDIR_BASE/${label}_${cnxc}x${cnyc}x${cnzc}_r${r}/output.log'"
            done
        done

        # Aggregate per-solver results
        if [[ $AVG -gt 1 ]]; then
            echo "${RI}Averages:"
        fi
        for ((si=0; si<NUM_SOLVERS; si++)); do
            label="${SOLVER_LABELS[$si]}"
            eval "local_failed=\$acc_failed_$si"
            eval "field_vals=(\"\${acc_field_$si[@]+\"\${acc_field_$si[@]}\"}\")"
            eval "wall_vals=(\"\${acc_wall_$si[@]+\"\${acc_wall_$si[@]}\"}\")"
            eval "moments_vals=(\"\${acc_moments_$si[@]+\"\${acc_moments_$si[@]}\"}\")"
            eval "mover_vals=(\"\${acc_mover_$si[@]+\"\${acc_mover_$si[@]}\"}\")"
            eval "iters_vals=(\"\${acc_iters_$si[@]+\"\${acc_iters_$si[@]}\"}\")"
            eval "resid_vals=(\"\${acc_resid_$si[@]+\"\${acc_resid_$si[@]}\"}\")"
            eval "all_conv=\$acc_converged_$si"
            eval "last_log=\$acc_lastlog_$si"

            if [[ "$local_failed" -eq 1 || ${#field_vals[@]} -eq 0 ]]; then
                ALL_LABELS+=("$label")
                ALL_FIELD+=("NA"); ALL_STD+=("NA"); ALL_MOMENTS+=("NA"); ALL_MOVER+=("NA")
                ALL_ITERS+=("NA"); ALL_CONVERGED+=("no"); ALL_RESIDUAL+=("NA")
                ALL_WALL+=("NA"); ALL_LOGS+=("none"); ALL_CONFIG_TAGS+=("$cnxc $cnyc $cnzc")
                continue
            fi

            # Safe strings for python
            field_str="${field_vals[*]+"${field_vals[*]}"}"
            wall_str="${wall_vals[*]+"${wall_vals[*]}"}"
            moments_str="${moments_vals[*]+"${moments_vals[*]}"}"
            mover_str="${mover_vals[*]+"${mover_vals[*]}"}"
            iters_str="${iters_vals[*]+"${iters_vals[*]}"}"
            resid_str="${resid_vals[*]+"${resid_vals[*]}"}"

            stats=$(python3 -c "
import statistics
def mean_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.mean(vals):.3f}' if vals else 'NA'
def std_str(arr):
    vals = [float(x) for x in arr if x not in ('NA','')]
    return f'{statistics.stdev(vals):.3f}' if len(vals) > 1 else 'NA'

field   = '$field_str'.split()
wall    = '$wall_str'.split()
moments = '$moments_str'.split()
mover   = '$mover_str'.split()
iters   = '$iters_str'.split()
resid   = '$resid_str'.split()

print(f'{mean_str(field)} {std_str(field)} {mean_str(moments)} {mean_str(mover)} {mean_str(iters)} {mean_str(resid)} {mean_str(wall)}')
")
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

            [[ -z "$first_run_t" ]] && t_r_before=$(python3 -c 'import time; print(time.time())')

            local_result=$(run_averaged "$cnxc" "$cnyc" "$cnzc" "$label" "$solver_type" "${extra_args[@]+"${extra_args[@]}"}")

            if [[ -z "$first_run_t" && "$DRY_RUN" != true ]]; then
                t_r_after=$(python3 -c 'import time; print(time.time())')
                first_run_t=$(python3 -c "print($t_r_after - $t_r_before)")
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
        # Find GMRES field time for this config
        base_idx=$(( (config_count - 1) * NUM_SOLVERS ))
        gmres_t="${ALL_FIELD[$base_idx]}"
        if [[ "$gmres_t" != "NA" ]]; then
            for ((si=1; si<NUM_SOLVERS; si++)); do
                idx=$(( base_idx + si ))
                other_t="${ALL_FIELD[$idx]}"
                if [[ "$other_t" != "NA" ]]; then
                    speedup=$(python3 -c "g=float('$gmres_t'); p=float('$other_t'); print(f'{g/p:.2f}' if p > 0 else 'N/A')")
                    echo "    ${SOLVER_LABELS[$si]} speedup: ${speedup}x"
                fi
            done
        fi
        echo ""
    fi
done

# ========================= Section 11: Results =============================
echo ""
echo "========================= RESULTS ========================="
echo ""

if [[ "$SCALING_MODE" != true ]]; then
    # ---- Single-grid mode: detailed output ----
    # Convergence
    echo "--------------------------------------------------------"
    echo "  CONVERGENCE"
    echo "--------------------------------------------------------"
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
    echo "--------------------------------------------------------"
    echo "  TIMING"
    echo "--------------------------------------------------------"
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
    echo "-------------------------------------------------------------------------------"
    echo "  COMPARISON SUMMARY"
    echo "-------------------------------------------------------------------------------"

    # Header
    printf "  %-20s" ""
    for ((i=0; i<NUM_SOLVERS; i++)); do
        printf " %14s" "${ALL_LABELS[$i]}"
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
        printf " %14s" "${ALL_CONVERGED[$i]}"
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
            init_t=$(grep "Time needed for initialisation:" "${ALL_LOGS[$i]}" | tail -1 | awk '{print $(NF-1)}') || init_t="N/A"
            printf " %12s s" "${init_t:-N/A}"
        else
            printf " %12s s" "N/A"
        fi
    done
    echo ""
    echo ""

    # Speedup vs GMRES
    echo "  Speedup vs built-in GMRES (field solver time):"
    field_data=""
    for ((i=0; i<NUM_SOLVERS; i++)); do
        field_data+="${ALL_LABELS[$i]} ${ALL_FIELD[$i]}"$'\n'
    done

    python3 - <<EOF 2>/dev/null || echo "  (Could not compute speedup ratios)"
data = {}
for line in """$field_data""".strip().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            data[parts[0]] = float(parts[1])
        except ValueError:
            pass
base = data.get('GMRES')
if base:
    for label, val in data.items():
        if label != 'GMRES' and val > 0:
            print(f'    {label:20s}  {base/val:.2f}x  ({val:.3f}s vs {base:.3f}s)')
EOF
    echo ""

else
    # ---- Scaling mode: summary table ----
    echo "--------------------------------------------------------"
    echo "  SUMMARY"
    echo "--------------------------------------------------------"

    # Header
    printf "  %-14s" "Grid"
    for label in "${SOLVER_LABELS[@]}"; do
        printf " %14s" "$label"
    done
    printf " %10s\n" "Best"

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
                is_better=$(python3 -c "print('yes' if float('$t') < float('$best_time') else 'no')")
                if [[ "$is_better" == "yes" ]]; then
                    best_time="$t"
                    best_label="${ALL_LABELS[$idx]}"
                fi
            fi
        done
        printf " %10s\n" "$best_label"
    done
    echo ""

    # Average speedup vs GMRES across all grid sizes, sorted by highest speedup
    echo "  Average speedup vs built-in GMRES (field solver time):"

    # Build per-solver speedup data: label gmres_t1 other_t1 gmres_t2 other_t2 ...
    speedup_input=""
    for ((si=1; si<NUM_SOLVERS; si++)); do
        speedup_input+="${SOLVER_LABELS[$si]}"
        for ((ci=0; ci<total_configs; ci++)); do
            base=$(( ci * NUM_SOLVERS ))
            gmres_t="${ALL_FIELD[$base]}"
            other_t="${ALL_FIELD[$((base + si))]}"
            speedup_input+=" $gmres_t $other_t"
        done
        speedup_input+=$'\n'
    done

    python3 - <<PYSPEEDUP 2>/dev/null || echo "  (Could not compute average speedup)"
results = []
for line in """$speedup_input""".strip().splitlines():
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
    if avg >= 1.0:
        print(f'    {label:20s}  {avg:.2f}x avg  (range: {lo:.2f}x .. {hi:.2f}x)')
    else:
        pct = (1 - avg) * 100
        print(f'    {label:20s}  {avg:.2f}x avg  ({pct:.0f}% slower, range: {lo:.2f}x .. {hi:.2f}x)')
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
    printf ",np,cycles,nzc,gmres_moments_s,gmres_mover_s"
    printf ",gmres_field_pct,gmres_moments_pct,gmres_mover_pct"
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

        local_gmres_field="NA"
        local_gmres_moments="NA"
        local_gmres_mover="NA"

        for ((si=0; si<NUM_SOLVERS; si++)); do
            idx=$(( base + si ))
            printf ",%s,%s,%s,%s,%s" \
                "${ALL_FIELD[$idx]}" "${ALL_STD[$idx]}" \
                "${ALL_ITERS[$idx]}" "${ALL_CONVERGED[$idx]}" "${ALL_RESIDUAL[$idx]}"

            if [[ $si -eq 0 ]]; then
                local_gmres_field="${ALL_FIELD[$idx]}"
                local_gmres_moments="${ALL_MOMENTS[$idx]}"
                local_gmres_mover="${ALL_MOVER[$idx]}"
            fi
        done

        # GMRES breakdown percentages
        gmres_pcts=$(python3 -c "
f = float('$local_gmres_field')   if '$local_gmres_field'   != 'NA' else None
m = float('$local_gmres_moments') if '$local_gmres_moments' != 'NA' else None
p = float('$local_gmres_mover')   if '$local_gmres_mover'   != 'NA' else None
vals = [v for v in [f, m, p] if v is not None]
total = sum(vals) if vals else 0
if total > 0 and m is not None and p is not None:
    print(f'{f/total*100:.1f},{m/total*100:.1f},{p/total*100:.1f}')
else:
    print('NA,NA,NA')
" 2>/dev/null) || gmres_pcts="NA,NA,NA"

        printf ",%s,%s,%s,%s,%s,%s\n" \
            "$NP" "$CYCLES" "$cnzc" \
            "$local_gmres_moments" "$local_gmres_mover" "$gmres_pcts"
    done
} > "$CSV_FILE"

echo "  CSV saved to:  $CSV_FILE"

# ---- Generate plot ----
echo "  Generating plot..."
export CSV_FILE PLOT_FILE NP CYCLES
export PROFILE_DIR="$PERSISTENT_OUTPUT_DIR"
export SOLVER_LABEL_LIST="${SOLVER_LABELS[*]}"

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

python3 "$SCRIPT_DIR/plot_petsc_test_results.py" "$CSV_FILE" || echo "  WARNING: Plot generation failed (see above)."

# ---- Visual comparison & movie (when field output was saved) ----
# Build directory paths for GMRES and PETSc runs (used by both comparison and movie)
read -r v_nxc v_nyc v_nzc <<< "${CONFIGS[0]}"
v_config_tag="${v_nxc}x${v_nyc}x${v_nzc}"
gmres_dir="$PERSISTENT_OUTPUT_DIR/GMRES_${v_config_tag}"
petsc_args=()
for ((si=1; si<NUM_SOLVERS; si++)); do
    petsc_dir="$PERSISTENT_OUTPUT_DIR/${SOLVER_LABELS[$si]}_${v_config_tag}"
    if [[ -d "$petsc_dir" ]]; then
        petsc_args+=(--petsc "$petsc_dir")
    fi
done

if [[ -n "$FIELD_OUTPUT" && "$FIELD_OUTPUT" != "0" && "$DRY_RUN" != true ]]; then
    if [[ -f "$SCRIPT_DIR/plot_petsc_visual_comparison.py" ]]; then
        if [[ -d "$gmres_dir" && ${#petsc_args[@]} -gt 0 ]]; then
            echo "  Running visual comparison..."
            python3 "$SCRIPT_DIR/plot_petsc_visual_comparison.py" \
                --gmres "$gmres_dir" "${petsc_args[@]}" || \
                echo "  WARNING: Visual comparison failed."
        fi
    fi

    # ---- Movie generation (when --movie flag was passed) ----
    if [[ "$MAKE_MOVIE" == true ]]; then
        if command -v ffmpeg &>/dev/null; then
            if [[ -d "$gmres_dir" && ${#petsc_args[@]} -gt 0 ]]; then
                echo "  Generating comparison movie..."
                bash "$SCRIPT_DIR/make_petsc_movie.sh" \
                    --gmres "$gmres_dir" "${petsc_args[@]}" || \
                    echo "  WARNING: Movie generation failed."
            fi
        else
            echo "  WARNING: ffmpeg not found, skipping movie generation."
        fi
    fi
fi

echo ""
echo "--------------------------------------------------------"
