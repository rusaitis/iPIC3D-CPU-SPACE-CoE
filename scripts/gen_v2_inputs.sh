#!/usr/bin/env bash
#
# gen_v2_inputs.sh — Generate Plan v2 stiffness-corpus input files for the
# preconditioner study (see plan-preconditioners.md §Phase 7).
#
# Produces 6 .inp variants of inputfiles/Double_Harris_PETSc_prec.inp under
# output/pc_study/v2_inputs/, one per (grid, dt, npcel, density, sheet-δ)
# combination listed in v2 Phase 7. Idempotent — safe to re-run.
#
# Usage:
#   bash scripts/gen_v2_inputs.sh [OUTDIR]
#     OUTDIR defaults to output/pc_study/v2_inputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$PROJECT_DIR/inputfiles/Double_Harris_PETSc_prec.inp"
OUTDIR="${1:-$PROJECT_DIR/output/pc_study/v2_inputs}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: source input file not found: $SRC" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# gen_input <id> <nxc> <nyc> <nzc> <Lx> <Ly> <Lz> <dt> \
#           <npcelx> <npcely> <npcelz> <rhoINIT> <delta> \
#           <XLEN> <YLEN> <ZLEN> <NpMaxNpRatio>
#
# All multi-value fields (npcel*, rhoINIT) take a 4-value space-separated
# string matching the 4 species in Double_Harris (e₁ i₁ e₂ i₂).
#
# The XLEN/YLEN/ZLEN baked in here are HINTS — tests/test.sh re-derives
# topology from --np unless --topo is given (test.sh:354-357). For the 3D
# case the user MUST pass --topo 4 2 2 because ZLEN defaults to 1.
gen_input() {
    local id="$1"
    local nxc="$2"   nyc="$3"   nzc="$4"
    local Lx="$5"    Ly="$6"    Lz="$7"
    local dt="$8"
    local npcelx="$9"  npcely="${10}"  npcelz="${11}"
    local rhoINIT="${12}"
    local delta="${13}"
    local xlen="${14}"  ylen="${15}"  zlen="${16}"
    local npmax="${17}"

    local out="$OUTDIR/${id}.inp"

    # All sed patterns anchor on `^name[[:space:]]` so e.g. `^Lx ` does not
    # accidentally match `LxLy` or similar. Multi-value lines (npcelx, rhoINIT)
    # are rewritten in their entirety; iPIC3D's parser is whitespace-tolerant.
    sed \
        -e "s|^SimulationName.*=.*|SimulationName                 = ${id}|" \
        -e "s|^SaveDirName.*=.*|SaveDirName                    = output/pc_study/v2_runs/${id}|" \
        -e "s|^RestartDirName.*=.*|RestartDirName                 = output/pc_study/v2_runs/${id}|" \
        -e "s|^dt[[:space:]].*=.*|dt                             = ${dt}|" \
        -e "s|^ncycles.*=.*|ncycles                        = 10|" \
        -e "s|^FieldOutputCycle.*=.*|FieldOutputCycle               = 0|" \
        -e "s|^ParticlesOutputCycle.*=.*|ParticlesOutputCycle           = 0|" \
        -e "s|^DiagnosticsOutputCycle.*=.*|DiagnosticsOutputCycle         = 1|" \
        -e "s|^Lx[[:space:]].*=.*|Lx                             = ${Lx}|" \
        -e "s|^Ly[[:space:]].*=.*|Ly                             = ${Ly}|" \
        -e "s|^Lz[[:space:]].*=.*|Lz                             = ${Lz}|" \
        -e "s|^nxc[[:space:]].*=.*|nxc                            = ${nxc}|" \
        -e "s|^nyc[[:space:]].*=.*|nyc                            = ${nyc}|" \
        -e "s|^nzc[[:space:]].*=.*|nzc                            = ${nzc}|" \
        -e "s|^XLEN[[:space:]].*=.*|XLEN                           = ${xlen}|" \
        -e "s|^YLEN[[:space:]].*=.*|YLEN                           = ${ylen}|" \
        -e "s|^ZLEN[[:space:]].*=.*|ZLEN                           = ${zlen}|" \
        -e "s|^npcelx[[:space:]].*=.*|npcelx              = ${npcelx}                                                       # Particles per cell along X|" \
        -e "s|^npcely[[:space:]].*=.*|npcely              = ${npcely}                                                       # Particles per cell along Y|" \
        -e "s|^npcelz[[:space:]].*=.*|npcelz              = ${npcelz}                                                       # Particles per cell along Z|" \
        -e "s|^rhoINIT[[:space:]].*=.*|rhoINIT             = ${rhoINIT}                                                       # Initial density|" \
        -e "s|^NpMaxNpRatio.*=.*|NpMaxNpRatio        = ${npmax}                                                       # Max particles allocated|" \
        -e "s|^custom_parameters.*=.*|custom_parameters              = 0.4        ${delta}|" \
        "$SRC" > "$out"

    # Compute and report alpha = (c*th*dt/dx)^2 where c=1, th=0.5, dx=Lx/nxc
    local alpha
    alpha=$(awk -v dt="$dt" -v Lx="$Lx" -v nxc="$nxc" \
        'BEGIN { dx = Lx / nxc; a = (0.5 * dt / dx); printf "%.3f", a*a }')

    printf "  %-28s  grid=%4dx%-4dx%-2d  dt=%-5s  npcel=%s  ρ=%s  δ=%-5s  α=%s\n" \
        "$id" "$nxc" "$nyc" "$nzc" "$dt" "$npcelx" "$rhoINIT" "$delta" "$alpha"
}

echo "Generating Plan v2 stiffness corpus from:"
echo "  $SRC"
echo "Output directory:"
echo "  $OUTDIR"
echo ""
echo "  ${0##*/}:"

# ── Phase 7 stiffness corpus ────────────────────────────────────────────────
# id                       nxc nyc nzc  Lx   Ly   Lz   dt    npcelx       npcely       npcelz       rhoINIT             δ     X Y Z  NpMax
gen_input DH_g200_n5_dt025        200 200 1   30.0 30.0 1.0  0.25  "5 5 5 5"    "5 5 5 5"    "1 1 1 1"    "1.0 1.0 1.0 1.0"   0.25  4 2 1   6.0
gen_input DH_g200_n2_dt025        200 200 1   30.0 30.0 1.0  0.25  "2 2 2 2"    "2 2 2 2"    "1 1 1 1"    "1.0 1.0 1.0 1.0"   0.25  4 2 1   6.0
gen_input DH_g300_n3_dt025        300 300 1   30.0 30.0 1.0  0.25  "3 3 3 3"    "3 3 3 3"    "1 1 1 1"    "1.0 1.0 1.0 1.0"   0.25  4 2 1   6.0
gen_input DH_contrast_dt05        100 100 1   30.0 30.0 1.0  0.5   "5 5 5 5"    "5 5 5 5"    "1 1 1 1"    "5.0 5.0 1.0 1.0"   0.15  4 2 1  10.0
gen_input DH_g200_contrast_dt025  200 200 1   30.0 30.0 1.0  0.25  "3 3 3 3"    "3 3 3 3"    "1 1 1 1"    "5.0 5.0 1.0 1.0"   0.15  4 2 1  10.0
gen_input DH_3D_g32_dt025          32  32 32  9.6  9.6  9.6  0.25  "2 2 2 2"    "2 2 2 2"    "2 2 2 2"    "1.0 1.0 1.0 1.0"   0.25  4 2 2   6.0
# Phase 7.5 (recovery) — 3D variant at higher dt to match the dt-driven stiffness regime
gen_input DH_3D_g32_dt05           32  32 32  9.6  9.6  9.6  0.5   "2 2 2 2"    "2 2 2 2"    "2 2 2 2"    "1.0 1.0 1.0 1.0"   0.25  2 2 2   6.0

n_files=$(find "$OUTDIR" -maxdepth 1 -name '*.inp' -type f | wc -l | tr -d ' ')
echo ""
echo "Wrote ${n_files} input files."
echo ""
echo "Next: run PCNONE characterization (Phase 7.2 in plan-preconditioners.md)."
echo ""
echo "  2D inputs (auto-topology, --np 8 enough):"
echo "    pixi run sim -- output/pc_study/v2_inputs/DH_g200_n5_dt025.inp \\"
echo "      --solver PETSc --cycles 10 -o output/pc_study/v2_phase7_DH_g200_n5_dt025_pcnone \\"
echo "      -- -pc_type none -ksp_monitor -ksp_view_singularvalues"
echo ""
echo "  3D input — must specify topology explicitly (ZLEN defaults to 1 in test.sh):"
echo "    bash tests/test.sh --input output/pc_study/v2_inputs/DH_3D_g32_dt025.inp \\"
echo "      --np 16 --topo 4 2 2 --cycles 10 --timeout 900 ..."
