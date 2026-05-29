#!/usr/bin/env bash
#
# bench_stiffness.sh — Map the (dt, N) stiffness frontier for the implicit
# Maxwell solve, and compare preconditioners on iterations AND field-solve time.
#
# Rationale: at the production regime (dt=0.125, coarse 2D) the operator
#   A = I + (cθΔt)²∇×∇× + (θΔt·4π/V)·M
# is dominated by the identity, so it is well-conditioned and NO preconditioner
# beats unpreconditioned GMRES (PCNONE).  Stiffness — and therefore any payoff
# from a preconditioner — only appears when the curl-curl term grows, i.e. at
# large dt (α=(cθΔt)² ∝ dt²) or fine grid (k_max² ∝ N²).  This script sweeps
# both knobs and records, per solver:
#   - mean / max KSP iterations to GMREStol (the conditioning signal)
#   - total + per-cycle Field-solver time (the deployable cost signal)
#   - number of non-converged cycles
#
# Output: one tidy CSV row per (study, dt, grid, solver), written to
#   tests/test_output_stiffness/<TAG>.csv  — plot with scripts/plot_stiffness.py
#
# Usage:
#   bash tests/bench_stiffness.sh [--exe PATH] [--np N] [--cycles N]
#        [--grids "100 200"] [--dts "0.125 0.25 0.5 1.0"] [--nzc N]
#        [--topo "X Y Z"] [--tag NAME] [--solvers "L1=args;L2=args;..."]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXE="${EXE:-$PROJECT_DIR/build/iPIC3D}"
TEMPLATE="$PROJECT_DIR/inputfiles/Double_Harris_PETSc_prec.inp"
NP=4
CYCLES=12
GRIDS="100 200"
DTS="0.125 0.25 0.5 1.0"
NZC=1
TOPO=""                       # auto: square-ish XY factorization of NP
TAG="dt_sweep"
# Solver set: "label=extra-cli-args" separated by ';'.  PrecType comes from the
# template (Matrix); -prec None/Helmholtz override it, -pc_type selects the PC.
SOLVERS="PCNONE=-prec None -pc_type none;GAMG=-pc_type gamg;Helmholtz=-prec Helmholtz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exe)     EXE="$2";     shift 2 ;;
        --np)      NP="$2";      shift 2 ;;
        --cycles)  CYCLES="$2";  shift 2 ;;
        --grids)   GRIDS="$2";   shift 2 ;;
        --dts)     DTS="$2";     shift 2 ;;
        --nzc)     NZC="$2";     shift 2 ;;
        --topo)    TOPO="$2";    shift 2 ;;
        --tag)     TAG="$2";     shift 2 ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --template) TEMPLATE="$2"; shift 2 ;;
        --help|-h) sed -n '2,/^$/{ s/^# *//; p }' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

OUTDIR="$SCRIPT_DIR/test_output_stiffness"
WORK="$OUTDIR/$TAG"
mkdir -p "$WORK/inp" "$WORK/logs" "$WORK/out"
CSV="$OUTDIR/${TAG}.csv"

# Portable high-resolution timestamp (bash 5 EPOCHREALTIME, else python3)
if [[ -n "${EPOCHREALTIME:-}" ]]; then
    _now() { printf '%s' "$EPOCHREALTIME"; }
else
    _now() { python3 -c 'import time;print(time.time())'; }
fi

# MPI oversubscribe only for OpenMPI
MPIRUN_FLAGS=()
if mpirun --version 2>&1 | grep -qi "open.mpi"; then MPIRUN_FLAGS=(--oversubscribe); fi

# Topology: explicit, or square-ish factorization of NP into XLEN*YLEN (ZLEN=1 in 2D)
if [[ -n "$TOPO" ]]; then
    read -r XLEN YLEN ZLEN <<< "$TOPO"
else
    NP_XY=$NP
    if [[ "$NZC" -gt 1 ]]; then ZLEN=2; NP_XY=$(( NP / 2 )); else ZLEN=1; fi
    # largest factor <= sqrt(NP_XY)
    XLEN=1
    for (( f=1; f*f<=NP_XY; f++ )); do (( NP_XY % f == 0 )) && XLEN=$f; done
    YLEN=$(( NP_XY / XLEN ))
fi
if (( XLEN*YLEN*ZLEN != NP )); then
    echo "ERROR: topo ${XLEN}x${YLEN}x${ZLEN} != np $NP" >&2; exit 2
fi

echo "study,dt,nxc,nyc,nzc,np,topo,cycles,solver,mean_iters,max_iters,nonconv,field_total_s,field_per_cycle_ms,wall_s" > "$CSV"

echo "=================================================================="
echo " Stiffness frontier:  np=$NP  topo=${XLEN}x${YLEN}x${ZLEN}  cycles=$CYCLES"
echo "   grids=[$GRIDS]  nzc=$NZC  dts=[$DTS]"
echo "   solvers: $SOLVERS"
echo "   csv -> $CSV"
echo "=================================================================="

run_one() {
    local dt="$1" nxc="$2" nyc="$3" nzc="$4" label="$5" extra="$6"
    local tagn="${label}_${nxc}x${nyc}x${nzc}_dt${dt}"
    local inp="$WORK/inp/${tagn}.inp"
    local log="$WORK/logs/${tagn}.log"
    local sdir="$WORK/out/${tagn}"
    mkdir -p "$sdir"

    sed -E \
      -e "s|^nxc.*=.*|nxc = $nxc|" \
      -e "s|^nyc.*=.*|nyc = $nyc|" \
      -e "s|^nzc.*=.*|nzc = $nzc|" \
      -e "s|^XLEN.*=.*|XLEN = $XLEN|" \
      -e "s|^YLEN.*=.*|YLEN = $YLEN|" \
      -e "s|^ZLEN.*=.*|ZLEN = $ZLEN|" \
      -e "s|^ncycles.*=.*|ncycles = $CYCLES|" \
      -e "s|^dt .*=.*|dt = $dt|" \
      -e "s|^FieldOutputCycle.*=.*|FieldOutputCycle = 0|" \
      -e "s|^ParticlesOutputCycle.*=.*|ParticlesOutputCycle = 0|" \
      -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle = 0|" \
      -e "s|^RestartOutputCycle.*=.*|RestartOutputCycle = 0|" \
      -e "s|^PrecDiagnostics.*=.*|PrecDiagnostics = false|" \
      -e "s|^SaveDirName.*=.*|SaveDirName = $sdir|" \
      -e "s|^RestartDirName.*=.*|RestartDirName = $sdir|" \
      "$TEMPLATE" > "$inp"

    local t0 t1 wall
    t0=$(_now)
    # shellcheck disable=SC2086
    mpirun "${MPIRUN_FLAGS[@]}" -np "$NP" "$EXE" "$inp" -solver PETSc $extra > "$log" 2>&1 || true
    t1=$(_now)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    # Parse iterations and convergence (portable: grep extracts, awk reduces)
    local iters nc mean_it max_it nonconv
    iters=$(grep -oE 'PETSc KSP converged: cycle=[0-9]+ iterations=[0-9]+' "$log" \
            | grep -oE 'iterations=[0-9]+' | cut -d= -f2 || true)
    nc=$(grep -c 'PETSc KSP did NOT converge' "$log" || true); nc="${nc:-0}"
    read -r mean_it max_it nonconv < <(printf '%s\n' "$iters" \
        | awk -v nc="$nc" 'NF{s+=$1;n++; if($1>mx)mx=$1}
                           END{ if(n>0) printf "%.1f %d %d\n", s/n, mx, nc; else printf "NA NA %d\n", nc }')
    # Cumulative field-solver time (last occurrence)
    local field_total
    field_total=$(awk '/Field solver/{v=$(NF-1)} END{print (v==""?"NA":v)}' "$log")
    local fpc="NA"
    if [[ "$field_total" != "NA" ]]; then
        fpc=$(awk -v f="$field_total" -v c="$CYCLES" 'BEGIN{printf "%.2f", 1000.0*f/c}')
    fi

    printf "%s,%s,%d,%d,%d,%d,%dx%dx%d,%d,%s,%s,%s,%s,%s,%s,%s\n" \
        "$TAG" "$dt" "$nxc" "$nyc" "$nzc" "$NP" "$XLEN" "$YLEN" "$ZLEN" \
        "$CYCLES" "$label" "$mean_it" "$max_it" "$nonconv" "$field_total" "$fpc" "$wall" >> "$CSV"

    printf "  dt=%-5s %dx%dx%d  %-12s iters mean=%-6s max=%-4s nonconv=%-2s  field=%ss (%s ms/cyc)\n" \
        "$dt" "$nxc" "$nyc" "$nzc" "$label" "$mean_it" "$max_it" "$nonconv" "$field_total" "$fpc"
}

for g in $GRIDS; do
    for dt in $DTS; do
        echo "---- grid ${g}x${g}x${NZC}  dt=$dt ----"
        # Iterate solver specs (";"-separated "label=args")
        IFS=';' read -ra SPECS <<< "$SOLVERS"
        for spec in "${SPECS[@]}"; do
            label="${spec%%=*}"
            extra="${spec#*=}"
            run_one "$dt" "$g" "$g" "$NZC" "$label" "$extra"
        done
    done
done

echo ""
echo "Done. CSV: $CSV"
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
