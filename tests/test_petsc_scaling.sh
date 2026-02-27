#!/usr/bin/env bash
#
# test_petsc_scaling.sh — Field solver time vs grid size for multiple solvers
#
# By default, compares built-in GMRES vs PETSc GMRES.
# Use --all-solvers to also include BiCGStab, FGMRES, and TFQMR.
#
# Usage:
#   ./tests/test_petsc_scaling.sh [--np N] [--cycles N] [--all-solvers]
#
# Generates: tests/scaling_results.csv and tests/scaling_plot.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXECUTABLE="$PROJECT_DIR/build/iPIC3D"
INPUT_FILE="$PROJECT_DIR/inputfiles/Double_Harris.inp"
NP=8
CYCLES=20
GRID_MIN=50
GRID_MAX=400
GRID_STEP=50
CSV_FILE="$SCRIPT_DIR/scaling_results.csv"
PLOT_FILE="$SCRIPT_DIR/scaling_plot.png"
TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_scaling.XXXXXX")
ALL_SOLVERS=false

# --------------------------- solver configs -------------------
# Parallel arrays: SOLVER_LABELS[i] / SOLVER_TYPES[i]
# Extra PETSc flags are handled in run_one() via a case statement.
SOLVER_LABELS=("GMRES" "PETSc")
SOLVER_TYPES=("GMRES" "PETSc")

# --------------------------- parse args -----------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)           NP="$2";       shift 2 ;;
        --cycles)       CYCLES="$2";   shift 2 ;;
        --grid-min)     GRID_MIN="$2"; shift 2 ;;
        --grid-max)     GRID_MAX="$2"; shift 2 ;;
        --grid-step)    GRID_STEP="$2"; shift 2 ;;
        --all-solvers)  ALL_SOLVERS=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--np N] [--cycles N] [--grid-min N] [--grid-max N] [--grid-step N] [--all-solvers]"
            echo ""
            echo "  --grid-min N    Smallest grid size (default: 50)"
            echo "  --grid-max N    Largest grid size (default: 400)"
            echo "  --grid-step N   Grid size increment (default: 50)"
            echo "  --all-solvers   Also test PETSc BiCGStab, FGMRES, and TFQMR"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Generate grid sizes from min/max/step
GRID_SIZES=()
for (( g=GRID_MIN; g<=GRID_MAX; g+=GRID_STEP )); do
    GRID_SIZES+=("$g")
done

# Auto-compute topology: XLEN * YLEN = NP (ZLEN=1), most square-like factorization.
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

# Ensure all grid sizes are divisible by both XLEN and YLEN.
# Round up any that aren't.
VALID_GRIDS=()
for g in "${GRID_SIZES[@]}"; do
    rem_x=$((g % XLEN))
    rem_y=$((g % YLEN))
    adj=$g
    if [[ $rem_x -ne 0 || $rem_y -ne 0 ]]; then
        # Round up to nearest common multiple of XLEN and YLEN
        lcm=$(python3 -c "import math; print(math.lcm($XLEN, $YLEN))")
        rem=$((g % lcm))
        if [[ $rem -ne 0 ]]; then
            adj=$((g + lcm - rem))
        fi
    fi
    VALID_GRIDS+=("$adj")
done
GRID_SIZES=("${VALID_GRIDS[@]}")

if [[ "$ALL_SOLVERS" == true ]]; then
    SOLVER_LABELS+=("PETSc_bcgs" "PETSc_fgmres" "PETSc_tfqmr")
    SOLVER_TYPES+=("PETSc" "PETSc" "PETSc")
fi

NUM_SOLVERS=${#SOLVER_LABELS[@]}

# --------------------------- checks ---------------------------
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Executable not found at $EXECUTABLE"
    echo "       Build first with:  ./build.sh --petsc"
    exit 1
fi

cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT

# --------------------------- helpers --------------------------
make_input() {
    local outdir="$1" nxc="$2" nyc="$3"
    local inp="$outdir/test_input.inp"
    mkdir -p "$outdir/output"
    sed \
        -e "s|^ncycles.*=.*|ncycles                        = $CYCLES|" \
        -e "s|^nxc.*=.*|nxc                            = $nxc|" \
        -e "s|^nyc.*=.*|nyc                            = $nyc|" \
        -e "s|^XLEN.*=.*|XLEN                           = $XLEN|" \
        -e "s|^YLEN.*=.*|YLEN                           = $YLEN|" \
        -e "s|^SaveDirName.*=.*|SaveDirName                    = $outdir/output|" \
        -e "s|^RestartDirName.*=.*|RestartDirName                 = $outdir/output|" \
        -e "s|^FieldOutputCycle.*=.*|FieldOutputCycle               = 0|" \
        -e "s|^ParticlesOutputCycle.*=.*|ParticlesOutputCycle           = 0|" \
        -e "s|^RestartOutputCycle.*=.*|RestartOutputCycle             = 0|" \
        -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle         = $CYCLES|" \
        "$INPUT_FILE" > "$inp"
    echo "$inp"
}

# Run one simulation and print the field solver time (or "FAILED")
run_one() {
    local nxc="$1" label="$2" solver_type="$3"
    local rundir="$TMPDIR_BASE/${label}_${nxc}x${nxc}"
    mkdir -p "$rundir"
    local inp logfile
    inp=$(make_input "$rundir" "$nxc" "$nxc")
    logfile="$rundir/output.log"

    local cmd=(mpirun -np "$NP" "$EXECUTABLE" "$inp" -solver "$solver_type")
    case "$label" in
        PETSc_bcgs)    cmd+=(-ksp_type bcgs -ksp_max_it 1000) ;;
        PETSc_fgmres)  cmd+=(-ksp_type fgmres -ksp_gmres_restart 50 -ksp_max_it 1000) ;;
        PETSc_tfqmr)   cmd+=(-ksp_type tfqmr -ksp_max_it 1000) ;;
    esac

    if ! "${cmd[@]}" > "$logfile" 2>&1; then
        echo "FAILED"
        return 1
    fi

    # Extract field solver time (last line = cumulative)
    grep "Field solver" "$logfile" | tail -1 | awk '{print $(NF-1)}'
}

# =============================================================
#                          MAIN
# =============================================================
echo ""
BOX_W=63
box_border=$(printf -- '-%.0s' $(seq 1 $BOX_W))
box_title="iPIC3D Scaling Study: Field Solver Comparison"
box_pad=$(( (BOX_W - ${#box_title}) / 2 ))
echo "+${box_border}+"
printf "|%*s%-*s|\n" $box_pad "" $((BOX_W - box_pad)) "$box_title"
echo "+${box_border}+"
printf "|%-${BOX_W}s|\n" "  Procs: $NP  Cycles: $CYCLES  Topo: ${XLEN}x${YLEN}x1"
printf "|%-${BOX_W}s|\n" "  Grids: ${GRID_SIZES[*]}"
printf "|%-${BOX_W}s|\n" "  Solvers: ${SOLVER_LABELS[*]}"
echo "+${box_border}+"
echo ""

# CSV header: grid_size,solver1,solver2,...
{
    printf "grid_size"
    for label in "${SOLVER_LABELS[@]}"; do
        printf ",%s" "$label"
    done
    echo ""
} > "$CSV_FILE"

total=${#GRID_SIZES[@]}
count=0

for nxc in "${GRID_SIZES[@]}"; do
    count=$((count + 1))
    echo "--------------------------------------------------------"
    echo "  [$count/$total]  Grid: ${nxc}x${nxc}  (${CYCLES} cycles, ${NP} procs)"
    echo "--------------------------------------------------------"

    csv_line="$nxc"
    times=()

    for i in $(seq 0 $((NUM_SOLVERS - 1))); do
        label="${SOLVER_LABELS[$i]}"
        solver_type="${SOLVER_TYPES[$i]}"
        printf "    %-18s ... " "$label"
        t=$(run_one "$nxc" "$label" "$solver_type") || t="NA"
        echo "${t} s"
        times+=("$t")
        csv_line="${csv_line},${t}"
    done

    # Print speedup vs GMRES (first solver)
    gmres_t="${times[0]}"
    if [[ "$gmres_t" != "NA" ]]; then
        for i in $(seq 1 $((NUM_SOLVERS - 1))); do
            if [[ "${times[$i]}" != "NA" ]]; then
                speedup=$(python3 -c "print(f'{float(\"$gmres_t\")/float(\"${times[$i]}\"):.2f}')")
                echo "    ${SOLVER_LABELS[$i]} speedup: ${speedup}x"
            fi
        done
    fi

    echo "$csv_line" >> "$CSV_FILE"
    echo ""
done

# =============================================================
#                      RESULTS TABLE
# =============================================================
echo "--------------------------------------------------------"
echo "  SUMMARY"
echo "--------------------------------------------------------"

# Header
printf "  %-10s" "Grid"
for label in "${SOLVER_LABELS[@]}"; do
    printf " %14s" "$label"
done
printf " %10s\n" "Best"

printf "  %-10s" "----------"
for label in "${SOLVER_LABELS[@]}"; do
    printf " %14s" "--------------"
done
printf " %10s\n" "----------"

# Data rows (read from CSV)
tail -n +2 "$CSV_FILE" | while IFS=',' read -r line; do
    # Split CSV line into array
    IFS=',' read -ra cols <<< "$line"
    grid="${cols[0]}"
    printf "  %-10s" "${grid}x${grid}"

    best_label=""
    best_time=999999
    for i in $(seq 1 $((NUM_SOLVERS))); do
        t="${cols[$i]}"
        printf " %12s s" "$t"
        if [[ "$t" != "NA" ]]; then
            is_better=$(python3 -c "print('yes' if float('$t') < float('$best_time') else 'no')")
            if [[ "$is_better" == "yes" ]]; then
                best_time="$t"
                best_label="${SOLVER_LABELS[$((i-1))]}"
            fi
        fi
    done
    printf " %10s\n" "$best_label"
done
echo ""

# =============================================================
#                      GENERATE PLOT
# =============================================================
echo "  Generating plot..."

# Pass solver labels to Python via environment
export CSV_FILE PLOT_FILE
export SOLVER_LABEL_LIST="${SOLVER_LABELS[*]}"

python3 << 'PYEOF'
import csv
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("  WARNING: matplotlib not installed, skipping plot generation.")
    print("  Install with: pip3 install matplotlib")
    exit(0)

csv_path = os.environ["CSV_FILE"]
plot_path = os.environ["PLOT_FILE"]
solver_labels = os.environ["SOLVER_LABEL_LIST"].split()

# Read CSV
grids = []
solver_times = {label: [] for label in solver_labels}

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        grids.append(int(row["grid_size"]))
        for label in solver_labels:
            val = row.get(label, "NA")
            solver_times[label].append(float(val) if val != "NA" else None)

# Colors and markers for each solver
styles = {
    "GMRES":         {"color": "#2196F3", "marker": "o", "ls": "-"},
    "PETSc":         {"color": "#FF5722", "marker": "s", "ls": "-"},
    "PETSc_bcgs":    {"color": "#4CAF50", "marker": "^", "ls": "--"},
    "PETSc_fgmres":  {"color": "#9C27B0", "marker": "D", "ls": "--"},
    "PETSc_tfqmr":   {"color": "#FF9800", "marker": "v", "ls": "--"},
}
default_style = {"color": "#607D8B", "marker": "x", "ls": ":"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --Left: Field solver time vs grid size ──
for label in solver_labels:
    times = solver_times[label]
    valid = [(g, t) for g, t in zip(grids, times) if t is not None]
    if not valid:
        continue
    vg, vt = zip(*valid)
    s = styles.get(label, default_style)
    ax1.plot(vg, vt, marker=s["marker"], linestyle=s["ls"], color=s["color"],
             linewidth=2, markersize=7, label=label)

ax1.set_xlabel('Grid size (N x N)', fontsize=12)
ax1.set_ylabel('Field solver time (s)', fontsize=12)
ax1.set_title('Field Solver Time vs Grid Size', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(grids)
ax1.set_xticklabels([str(g) for g in grids], fontsize=9)

# --Right: Speedup vs GMRES ──
gmres_times = solver_times.get("GMRES", [])
petsc_labels = [l for l in solver_labels if l != "GMRES"]

if gmres_times and petsc_labels:
    x_positions = list(range(len(grids)))
    bar_width = 0.8 / len(petsc_labels)

    for j, label in enumerate(petsc_labels):
        times = solver_times[label]
        speedups = []
        for gt, pt in zip(gmres_times, times):
            if gt is not None and pt is not None and pt > 0:
                speedups.append(gt / pt)
            else:
                speedups.append(0)

        s = styles.get(label, default_style)
        offsets = [x + (j - len(petsc_labels)/2 + 0.5) * bar_width for x in x_positions]
        bars = ax2.bar(offsets, speedups, width=bar_width, color=s["color"],
                       alpha=0.8, edgecolor='black', linewidth=0.5, label=label)

        for bar, sp in zip(bars, speedups):
            if sp > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, sp + 0.02,
                         f'{sp:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Grid size (N x N)', fontsize=12)
    ax2.set_ylabel('Speedup vs built-in GMRES', fontsize=12)
    ax2.set_title('PETSc Speedup vs Built-in GMRES', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([str(g) for g in grids], fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Plot saved to: {plot_path}")
PYEOF

echo ""
echo "  CSV saved to:  $CSV_FILE"
echo "--------------------------------------------------------"
