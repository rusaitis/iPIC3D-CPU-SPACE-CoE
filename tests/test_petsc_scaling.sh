#!/usr/bin/env bash
#
# test_petsc_scaling.sh — Field solver time vs grid size: built-in GMRES vs PETSc GMRES
#
# Usage:
#   ./tests/test_petsc_scaling.sh [--np N] [--cycles N]
#
# Generates: tests/scaling_results.csv and tests/scaling_plot.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXECUTABLE="$PROJECT_DIR/build/iPIC3D"
INPUT_FILE="$PROJECT_DIR/inputfiles/Double_Harris.inp"
NP=8
CYCLES=20
XLEN=4
YLEN=2
CSV_FILE="$SCRIPT_DIR/scaling_results.csv"
PLOT_FILE="$SCRIPT_DIR/scaling_plot.png"
TMPDIR_BASE=$(mktemp -d "${TMPDIR:-/tmp}/ipic3d_scaling.XXXXXX")

# Grid sizes to test (must be divisible by XLEN=4 and YLEN=2)
GRID_SIZES=(50 100 150 200 250 300 350 400)

# ─────────────────────────── parse args ─────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)      NP="$2";     shift 2 ;;
        --cycles)  CYCLES="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--np N] [--cycles N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─────────────────────────── checks ─────────────────────────────
if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Executable not found at $EXECUTABLE"
    echo "       Build first with:  ./build.sh --petsc"
    exit 1
fi

cleanup() { rm -rf "$TMPDIR_BASE"; }
trap cleanup EXIT

# ─────────────────────────── helpers ────────────────────────────
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

run_one() {
    local nxc="$1" solver="$2" label="$3"
    local rundir="$TMPDIR_BASE/${label}_${nxc}x${nxc}"
    mkdir -p "$rundir"
    local inp logfile
    inp=$(make_input "$rundir" "$nxc" "$nxc")
    logfile="$rundir/output.log"

    if [[ "$solver" == "GMRES" ]]; then
        mpirun -np "$NP" "$EXECUTABLE" "$inp" -solver GMRES > "$logfile" 2>&1
    else
        mpirun -np "$NP" "$EXECUTABLE" "$inp" -solver PETSc > "$logfile" 2>&1
    fi

    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "FAILED"
        return 1
    fi

    # Extract field solver time (last line = cumulative)
    grep "Field solver" "$logfile" | tail -1 | awk '{print $(NF-1)}'
}

# ═══════════════════════════════════════════════════════════════
#                          MAIN
# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║    iPIC3D Scaling Study: Built-in GMRES vs PETSc GMRES      ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
printf "║  Procs: %-4d  Cycles: %-4d  Topo: %dx%dx1                    ║\n" "$NP" "$CYCLES" "$XLEN" "$YLEN"
printf "║  Grids: %s ║\n" "$(printf '%s ' "${GRID_SIZES[@]}")"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# CSV header
echo "grid_size,gmres_field_time,petsc_field_time" > "$CSV_FILE"

total=${#GRID_SIZES[@]}
count=0

for nxc in "${GRID_SIZES[@]}"; do
    count=$((count + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$count/$total]  Grid: ${nxc}x${nxc}  (${CYCLES} cycles, ${NP} procs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run GMRES
    printf "    Built-in GMRES ... "
    gmres_t=$(run_one "$nxc" "GMRES" "GMRES") || { echo "FAILED"; gmres_t="NA"; }
    echo "${gmres_t} s"

    # Run PETSc
    printf "    PETSc GMRES    ... "
    petsc_t=$(run_one "$nxc" "PETSc" "PETSc") || { echo "FAILED"; petsc_t="NA"; }
    echo "${petsc_t} s"

    # Speedup
    if [[ "$gmres_t" != "NA" && "$petsc_t" != "NA" ]]; then
        speedup=$(python3 -c "print(f'{float(\"$gmres_t\")/float(\"$petsc_t\"):.2f}')")
        echo "    Speedup: ${speedup}x"
    fi

    echo "$nxc,$gmres_t,$petsc_t" >> "$CSV_FILE"
    echo ""
done

# ═══════════════════════════════════════════════════════════════
#                      RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  %-12s %14s %14s %10s\n" "Grid" "GMRES (s)" "PETSc (s)" "Speedup"
printf "  %-12s %14s %14s %10s\n" "────────────" "──────────────" "──────────────" "──────────"

# Read CSV (skip header)
tail -n +2 "$CSV_FILE" | while IFS=',' read -r grid gt pt; do
    if [[ "$gt" != "NA" && "$pt" != "NA" ]]; then
        sp=$(python3 -c "print(f'{float(\"$gt\")/float(\"$pt\"):.2f}')")
    else
        sp="N/A"
    fi
    printf "  %-12s %12s s %12s s %8sx\n" "${grid}x${grid}" "$gt" "$pt" "$sp"
done
echo ""

# ═══════════════════════════════════════════════════════════════
#                      GENERATE PLOT
# ═══════════════════════════════════════════════════════════════
echo "  Generating plot..."

export CSV_FILE PLOT_FILE
python3 << 'PYEOF'
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath("PYEOF_PLACEHOLDER"))
csv_path = os.environ["CSV_FILE"]
plot_path = os.environ["PLOT_FILE"]

grids, gmres_times, petsc_times = [], [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["gmres_field_time"] == "NA" or row["petsc_field_time"] == "NA":
            continue
        grids.append(int(row["grid_size"]))
        gmres_times.append(float(row["gmres_field_time"]))
        petsc_times.append(float(row["petsc_field_time"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: Field solver time vs grid size ──
ax1.plot(grids, gmres_times, 'o-', color='#2196F3', linewidth=2, markersize=7, label='Built-in GMRES')
ax1.plot(grids, petsc_times, 's-', color='#FF5722', linewidth=2, markersize=7, label='PETSc GMRES')
ax1.set_xlabel('Grid size (N × N)', fontsize=12)
ax1.set_ylabel('Field solver time (s)', fontsize=12)
ax1.set_title('Field Solver Time vs Grid Size', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(grids)
ax1.set_xticklabels([f'{g}' for g in grids], fontsize=9)

# ── Right: Speedup ──
speedups = [g / p if p > 0 else 0 for g, p in zip(gmres_times, petsc_times)]
colors = ['#4CAF50' if s >= 1.0 else '#F44336' for s in speedups]
ax2.bar(range(len(grids)), speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
ax2.set_xlabel('Grid size (N × N)', fontsize=12)
ax2.set_ylabel('Speedup (GMRES / PETSc)', fontsize=12)
ax2.set_title('PETSc Speedup vs Built-in GMRES', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(grids)))
ax2.set_xticklabels([f'{g}' for g in grids], fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# Annotate bars
for i, (g, s) in enumerate(zip(grids, speedups)):
    ax2.text(i, s + 0.02, f'{s:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Plot saved to: {plot_path}")
PYEOF

echo ""
echo "  CSV saved to:  $CSV_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
