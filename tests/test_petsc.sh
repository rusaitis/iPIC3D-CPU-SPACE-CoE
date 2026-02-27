#!/usr/bin/env bash
#
# test_petsc.sh — Compare GMRES vs PETSc field solver performance
#
# Runs Double_Harris.inp under three configurations:
#   1. Built-in GMRES solver (default)
#   2. PETSc KSP solver (default settings: GMRES + no preconditioner)
#   3. PETSc KSP solver with tuned options (flexible GMRES, larger restart)
#
# Outputs: convergence info and per-module timing comparison.
#
# Usage:
#   ./tests/test_petsc.sh [--np N] [--cycles N] [--input FILE]
#
# Requirements:
#   - Build with PETSc:  ./build.sh --petsc
#   - MPI available (mpirun)

set -euo pipefail

# --------------------------- defaults ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXECUTABLE="$PROJECT_DIR/build/iPIC3D"
INPUT_FILE="$PROJECT_DIR/inputfiles/Double_Harris.inp"
NP=8          # MPI processes
CYCLES=10     # override ncycles for faster test
NXC=""        # grid cells X (empty = use input file default)
NYC=""        # grid cells Y (empty = use input file default)
XLEN=""       # MPI decomposition X (empty = use input file default)
YLEN=""       # MPI decomposition Y (empty = use input file default)
TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_test_petsc.XXXXXX")

# --------------------------- parse args --------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)      NP="$2";         shift 2 ;;
        --cycles)  CYCLES="$2";     shift 2 ;;
        --grid)    NXC="$2"; NYC="$3"; shift 3 ;;
        --topo)    XLEN="$2"; YLEN="$3"; shift 3 ;;
        --input)   INPUT_FILE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--np N] [--cycles N] [--grid NXC NYC] [--topo XLEN YLEN] [--input FILE]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If --topo was not given, auto-compute XLEN * YLEN = NP (ZLEN=1).
# Picks the most square-like factorization.
if [[ -z "$XLEN" || -z "$YLEN" ]]; then
    YLEN=$(python3 -c "
import math
n = $NP
best = (1, n)
for i in range(1, int(math.isqrt(n)) + 1):
    if n % i == 0:
        best = (i, n // i)
print(best[0])
")
    XLEN=$(( NP / YLEN ))
fi

# If --grid was not given, ensure the input file grid is divisible by the topology.
# Read nxc/nyc from the input file and round up to the nearest multiple if needed.
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

# --------------------------- checks ------------------------------
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Executable not found at $EXECUTABLE"
    echo "       Build first with:  ./build.sh --petsc"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# --------------------------- helpers -----------------------------
cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT

# Create a test input file with reduced cycles and no file output
make_test_input() {
    local outdir="$1"
    local inp="$outdir/test_input.inp"
    # Copy original and override settings for speed
    local sed_args=(
        -e "s|^ncycles.*=.*|ncycles                        = $CYCLES|"
        -e "s|^SaveDirName.*=.*|SaveDirName                    = $outdir/output|"
        -e "s|^RestartDirName.*=.*|RestartDirName                 = $outdir/output|"
        -e "s|^FieldOutputCycle.*=.*|FieldOutputCycle               = 0|"
        -e "s|^ParticlesOutputCycle.*=.*|ParticlesOutputCycle           = 0|"
        -e "s|^RestartOutputCycle.*=.*|RestartOutputCycle             = 0|"
        -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle         = $CYCLES|"
    )
    if [[ -n "$NXC" ]]; then
        sed_args+=(-e "s|^nxc.*=.*|nxc                            = $NXC|")
    fi
    if [[ -n "$NYC" ]]; then
        sed_args+=(-e "s|^nyc.*=.*|nyc                            = $NYC|")
    fi
    if [[ -n "$XLEN" ]]; then
        sed_args+=(-e "s|^XLEN.*=.*|XLEN                           = $XLEN|")
    fi
    if [[ -n "$YLEN" ]]; then
        sed_args+=(-e "s|^YLEN.*=.*|YLEN                           = $YLEN|")
    fi
    sed "${sed_args[@]}" "$INPUT_FILE" > "$inp"
    mkdir -p "$outdir/output"
    echo "$inp"
}

# Run a simulation and capture output
# Args: label solver_flag [extra_petsc_args...]
run_sim() {
    local label="$1"
    local solver="$2"
    shift 2
    local extra_args=()
    if [[ $# -gt 0 ]]; then
        extra_args=("$@")
    fi

    local rundir="$TMPDIR_BASE/$label"
    mkdir -p "$rundir"
    local inp
    inp=$(make_test_input "$rundir")
    local logfile="$rundir/output.log"

    echo "------------------------------------------------------------"
    echo "  Running: $label"
    echo "  Solver:  $solver"
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        echo "  PETSc flags: ${extra_args[*]}"
    fi
    echo "  Processes: $NP,  Cycles: $CYCLES"
    echo "------------------------------------------------------------"

    local start_time
    start_time=$(date +%s.%N 2>/dev/null || python3 -c 'import time; print(time.time())')

    # Run the simulation
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        mpirun -np "$NP" "$EXECUTABLE" "$inp" -solver "$solver" "${extra_args[@]}" \
            > "$logfile" 2>&1
    else
        mpirun -np "$NP" "$EXECUTABLE" "$inp" -solver "$solver" \
            > "$logfile" 2>&1
    fi

    local run_status=$?
    if [[ $run_status -ne 0 ]]; then
        echo "  FAILED (exit code $run_status) — see $logfile"
        cat "$logfile"
        return 1
    fi
    local end_time
    end_time=$(date +%s.%N 2>/dev/null || python3 -c 'import time; print(time.time())')
    local wall_time
    wall_time=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")

    echo "  Wall-clock time: ${wall_time} s"
    echo ""

    # Store log path and wall time for later parsing
    eval "LOG_${label}=$logfile"
    eval "WALL_${label}=$wall_time"
}

# Extract timing from the log (last occurrence = cumulative over all cycles)
extract_timing() {
    local logfile="$1"
    local label="$2"

    echo "  [$label] Module timings (cumulative over $CYCLES cycles):"

    local field_t moment_t mover_t write_t init_t
    init_t=$(grep "Time needed for initialisation:" "$logfile" | tail -1 | awk '{print $(NF-1)}') || init_t="N/A"
    field_t=$(grep "Field solver" "$logfile" | tail -1 | awk '{print $(NF-1)}') || field_t="N/A"
    mover_t=$(grep "Particle mover" "$logfile" | tail -1 | awk '{print $(NF-1)}') || mover_t="N/A"
    moment_t=$(grep "Moment gatherer" "$logfile" | tail -1 | awk '{print $(NF-1)}') || moment_t="N/A"
    write_t=$(grep "Write data" "$logfile" | tail -1 | awk '{print $(NF-1)}') || write_t="N/A"

    printf "    %-22s %10s s\n" "Initialisation:" "$init_t"
    printf "    %-22s %10s s\n" "Field solver:" "$field_t"
    printf "    %-22s %10s s\n" "Particle mover:" "$mover_t"
    printf "    %-22s %10s s\n" "Moment gatherer:" "$moment_t"
    printf "    %-22s %10s s\n" "Write data:" "$write_t"

    # Store for comparison
    eval "FIELD_${label}=$field_t"
    eval "INIT_${label}=$init_t"
}

# Extract convergence info from the log
extract_convergence() {
    local logfile="$1"
    local label="$2"

    echo "  [$label] Convergence:"

    # GMRES convergence lines
    local gmres_lines
    gmres_lines=$(grep -c "GMRES converged" "$logfile" 2>/dev/null) || gmres_lines=0
    local gmres_fail
    gmres_fail=$(grep -c "GMRES not converged" "$logfile" 2>/dev/null) || gmres_fail=0

    # PETSc convergence lines
    local petsc_conv
    petsc_conv=$(grep -c "PETSc KSP converged" "$logfile" 2>/dev/null) || petsc_conv=0
    local petsc_fail
    petsc_fail=$(grep -c "PETSc KSP did NOT converge" "$logfile" 2>/dev/null) || petsc_fail=0

    if [[ $gmres_lines -gt 0 || $gmres_fail -gt 0 ]]; then
        echo "    GMRES: $gmres_lines converged, $gmres_fail failed"
        # Show first and last convergence line
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

# ===============================================================
#                          MAIN
# ===============================================================
echo ""
GRID_INFO=""
if [[ -n "$NXC" && -n "$NYC" ]]; then
    GRID_INFO="Grid: ${NXC}x${NYC}"
else
    GRID_INFO="Grid: (from input file)"
fi
BOX_W=63
box_border=$(printf -- '-%.0s' $(seq 1 $BOX_W))
box_title="iPIC3D Solver Comparison: GMRES vs PETSc"
box_pad=$(( (BOX_W - ${#box_title}) / 2 ))
echo "+${box_border}+"
printf "|%*s%-*s|\n" $box_pad "" $((BOX_W - box_pad)) "$box_title"
echo "+${box_border}+"
printf "|%-${BOX_W}s|\n" "  Input: $(basename "$INPUT_FILE")"
printf "|%-${BOX_W}s|\n" "  Procs: $NP   Cycles: $CYCLES   $GRID_INFO"
echo "+${box_border}+"
echo ""

# List of all run labels (used for results loop)
ALL_RUNS=()

# --Run 1: Built-in GMRES ──
run_sim "GMRES" "GMRES"
ALL_RUNS+=("GMRES")

# --Run 2: PETSc default (GMRES + no preconditioner) ──
run_sim "PETSc_gmres" "PETSc"
ALL_RUNS+=("PETSc_gmres")

# --Run 3: PETSc FGMRES with larger restart ──
run_sim "PETSc_fgmres" "PETSc" \
    -ksp_type fgmres \
    -ksp_gmres_restart 50 \
    -ksp_max_it 1000

ALL_RUNS+=("PETSc_fgmres")

# --Run 4: PETSc BiCGStab ──
run_sim "PETSc_bcgs" "PETSc" \
    -ksp_type bcgs \
    -ksp_max_it 1000

ALL_RUNS+=("PETSc_bcgs")

# --Run 5: PETSc TFQMR ──
run_sim "PETSc_tfqmr" "PETSc" \
    -ksp_type tfqmr \
    -ksp_max_it 1000

ALL_RUNS+=("PETSc_tfqmr")

# ===============================================================
#                        RESULTS
# ===============================================================
echo ""
echo "========================= RESULTS ========================="
echo ""

# --Convergence ──
echo "--------------------------------------------------------"
echo "  CONVERGENCE"
echo "--------------------------------------------------------"
for label in "${ALL_RUNS[@]}"; do
    local_log="LOG_${label}"
    extract_convergence "${!local_log}" "$label"
    echo ""
done

# --Timing ──
echo "--------------------------------------------------------"
echo "  TIMING"
echo "--------------------------------------------------------"
for label in "${ALL_RUNS[@]}"; do
    local_log="LOG_${label}"
    extract_timing "${!local_log}" "$label"
    echo ""
done

# --Comparison table ──
echo "-------------------------------------------------------------------------------"
echo "  COMPARISON SUMMARY"
echo "-------------------------------------------------------------------------------"

# Header row
printf "  %-20s" ""
for label in "${ALL_RUNS[@]}"; do
    printf " %14s" "$label"
done
echo ""
printf "  %-20s" "--------------------"
for label in "${ALL_RUNS[@]}"; do
    printf " %14s" "--------------"
done
echo ""

# Wall-clock row
printf "  %-20s" "Wall-clock total"
for label in "${ALL_RUNS[@]}"; do
    local_wall="WALL_${label}"
    printf " %12s s" "${!local_wall}"
done
echo ""

# Field solver row
printf "  %-20s" "Field solver"
for label in "${ALL_RUNS[@]}"; do
    local_field="FIELD_${label}"
    printf " %12s s" "${!local_field}"
done
echo ""

# Init row
printf "  %-20s" "Initialisation"
for label in "${ALL_RUNS[@]}"; do
    local_init="INIT_${label}"
    printf " %12s s" "${!local_init}"
done
echo ""
echo ""

# --Speedup vs GMRES ──
echo "  Speedup vs built-in GMRES (field solver time):"
{
    # Write label=field_time pairs to a temp file for Python
    local_datafile="$TMPDIR_BASE/field_times.txt"
    for label in "${ALL_RUNS[@]}"; do
        local_field="FIELD_${label}"
        echo "${label} ${!local_field}" >> "$local_datafile"
    done
    python3 -c "
data = {}
for line in open('$local_datafile'):
    parts = line.strip().split()
    if len(parts) == 2:
        try: data[parts[0]] = float(parts[1])
        except: pass
base = data.get('GMRES')
if base:
    for label, val in data.items():
        if label == 'GMRES': continue
        if val > 0:
            ratio = base / val
            print(f'    {label:20s} {ratio:.2f}x  ({val:.3f}s vs {base:.3f}s)')
" 2>/dev/null || echo "  (Could not compute speedup ratios)"
}
echo ""

echo "--------------------------------------------------------"
echo "  Logs saved in: $TMPDIR_BASE"
echo "  (Will be cleaned up on exit)"
echo "--------------------------------------------------------"
