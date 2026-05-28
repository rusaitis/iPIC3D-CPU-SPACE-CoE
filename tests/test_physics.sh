#!/usr/bin/env bash
# Physics validation test suite — exercises wave dispersion + instability
# growth rates across the periodic-BC halo path at np=1 and np=4.
#
# Tests:
#   A. Plane EM wave in vacuum    ω = c·k                (analyze_dispersion.py)
#   B. Shear Alfvén wave          ω = k·v_A              (analyze_dispersion.py)
#   C. Two-stream instability     γ = ω_pe/(2√2)         (analyze_growth_rate.py)
#   D. Whistler R-mode wave       ω = ω_R(k)             (analyze_dispersion.py)
#   E. Oblique shear Alfvén wave  ω = k·v_A·cos(θ_kB)    (analyze_dispersion.py)
#   F. Numerical heating          peak(EE)/KE₀ < tol     (analyze_heating.py)
#   G. Magnetic reconnection      dψ/dt ~ 0.1·B0·v_A     (analyze_reconnection.py)
#   H. Landau damping             γ_L from Z(ξ)          (analyze_landau.py)
#   I. Weibel filamentation       γ_W ≈ ω_pe·u_b/c       (analyze_growth_rate.py --field magnetic)
#
# Tests A–D run at np=1 (single-rank periodic-self halo path) and np=4
# X-decomp (cross-rank periodic halo on one axis). Test E runs at np=1 and
# np=4 with XLEN=2,YLEN=2 to stress 2-axis cross-rank halos (the regime that
# motivated Phases E.14 / E.16 / E.18). Test E uses Walen-correct δv from
# Particles3D::alfven_walen_seed so the wave excites a single forward branch
# rather than a forward+backward standing pair (essential for clean ω at
# oblique k). Test F is the CIC-vs-TSC stencil-order discriminator: a quiet
# cold Maxwellian at a grid that deeply violates the Debye criterion, where
# TSC's smoother B-spline shape function suppresses particle-grid aliasing
# (~30-40% less spurious E-field excitation than CIC at the same config).
# The unification work guarantees the halo paths produce equivalent physics;
# this suite checks the guarantee against analytical / theoretical references.
#
# Runtime (Apple silicon, 6 cores): ~30-35 min at default CIC stencil (Linear).
# Add STENCIL=TSC for n_ghost=2 / Quadratic B-spline — ~7 hours; whistler,
# twostream, reconnection dominate. Designed for nightly / pre-merge CI,
# not the inner loop. Use tests/test_energy.sh + `pixi run test` for that.

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/tests/common.sh"
EXE="${EXE:-${IPIC3D_BUILD_DIR:-$REPO/build}/iPIC3D}"
OUTROOT="${OUTROOT:-$REPO/output/test_physics}"
PYTHON="${PYTHON:-pixi run python}"

# Optional: STENCIL=TSC reruns every input with -stencil TSC (no input-file
# changes — adds 2-cell ghost halos via Quadratic B-spline). Default Linear (CIC).
STENCIL="${STENCIL:-}"
EXTRA_ARGS=""
if [[ -n "$STENCIL" ]]; then
    EXTRA_ARGS="-stencil $STENCIL"
    echo "STENCIL override: $STENCIL  (extra mpirun args: $EXTRA_ARGS)"
fi

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: iPIC3D not found at $EXE" >&2
    echo "       set EXE=... or run 'pixi run build' first" >&2
    exit 2
fi

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

cd "$REPO"

# ── Run all 18 simulations (9 tests × 2 decompositions) ──────────────────

# run_sim LABEL NP BASE.inp OUTDIR [XLEN YLEN]
# NP>1 generates the decomposition on the fly from the np=1 base via
# make_np_variant (no committed "_np4" inputs). The XLEN/YLEN per case is the
# decomposition being stress-tested: 4x1 X-only cross-rank for the 1-D waves,
# 2x2 for the cases that exercise two-axis cross-rank halos.
run_sim() {
    local label=$1 np=$2 base=$3 outdir=$4 xlen=${5:-1} ylen=${6:-1}
    local logfile="$OUTROOT/${label}.log"
    local inp="inputfiles/$base"
    rm -rf "$outdir"
    if [[ $np -gt 1 ]]; then
        local tmp="$OUTROOT/$(basename "$outdir").inp"
        make_np_variant "$inp" "$tmp" "$xlen" "$ylen" 1 "$(basename "$outdir")"
        inp="$tmp"
    fi
    echo "─── ${label}: mpirun -np ${np} $(basename "$EXE") ${inp} $EXTRA_ARGS"
    mpirun -np "$np" "$EXE" "$inp" $EXTRA_ARGS > "$logfile" 2>&1
}

run_sim plane_em_np1     1 PlaneEMWave.inp       "$REPO/data_PlaneEMWave"
run_sim plane_em_np4     4 PlaneEMWave.inp       "$REPO/data_PlaneEMWave_np4"      4 1
run_sim alfven_np1       1 AlfvenWave.inp        "$REPO/data_AlfvenWave"
run_sim alfven_np4       4 AlfvenWave.inp        "$REPO/data_AlfvenWave_np4"        4 1
run_sim twostream_np1    1 TwoStream.inp         "$REPO/data_TwoStream"
run_sim twostream_np4    4 TwoStream.inp         "$REPO/data_TwoStream_np4"         4 1
run_sim whistler_np1     1 WhistlerPacket.inp    "$REPO/data_WhistlerPacket"
run_sim whistler_np4     4 WhistlerPacket.inp    "$REPO/data_WhistlerPacket_np4"    4 1
run_sim oblique_alf_np1  1 ObliqueAlfvenWave.inp "$REPO/data_ObliqueAlfvenWave"
run_sim oblique_alf_np4  4 ObliqueAlfvenWave.inp "$REPO/data_ObliqueAlfvenWave_np4" 2 2
run_sim heating_np1      1 NumericalHeating.inp  "$REPO/data_NumericalHeating"
run_sim heating_np4      4 NumericalHeating.inp  "$REPO/data_NumericalHeating_np4"  4 1
run_sim reconnection_np1 1 Reconnection.inp      "$REPO/data_Reconnection"
run_sim reconnection_np4 4 Reconnection.inp      "$REPO/data_Reconnection_np4"      2 2
run_sim landau_np1       1 LangmuirWave.inp      "$REPO/data_LangmuirWave"
run_sim landau_np4       4 LangmuirWave.inp      "$REPO/data_LangmuirWave_np4"      4 1
run_sim weibel_np1       1 Weibel.inp            "$REPO/data_Weibel"
run_sim weibel_np4       4 Weibel.inp            "$REPO/data_Weibel_np4"            2 2

# ── Run analyzers, capture pass/fail ──────────────────────────────────────

# Theory-agreement plots (--save-plot) and propagation movies are best-effort
# add-ons: failures here do NOT change the test verdict.
OUTPLOTS="$OUTROOT/plots"
OUTMOVIES="$OUTROOT/movies"
mkdir -p "$OUTPLOTS" "$OUTMOVIES"

declare -a RESULTS
fail=0

check() {
    local label=$1 cmd=$2
    local out
    # All analyzers accept --save-plot; appending here keeps call sites tidy.
    cmd="$cmd --save-plot $OUTPLOTS/${label}.png"
    if out=$(eval "$cmd" 2>&1); then
        local tag="PASS"
    else
        local tag="FAIL"
        fail=1
    fi
    # Extract the headline number for the summary (different per analyzer):
    #   dispersion → |Δω/ω|,  growth → |Δγ/γ|,  heating → |EE|/KE₀,
    #   reconnection → "peak / (B0·v_A)"
    # `|| true` keeps a non-matching grep from triggering set-e+pipefail.
    local rel
    rel=$(echo "$out" | grep -E '\|Δω/ω\||\|Δγ/γ\||\|EE\|/KE|peak / \(B0' | sed -E 's/.*: *([0-9.e+-]+).*/\1/' | head -1 || true)
    RESULTS+=("$tag|$label|$rel")
    echo "  [$tag]  $label  $rel"
}

# Generate a propagation movie (best-effort — never fails the suite).
movie() {
    local label=$1 outdir=$2 inp=$3 field=$4 component=$5
    $PYTHON scripts/movie_physics_test.py "$outdir" \
        --inp "inputfiles/$inp" \
        --field "$field" --component "$component" \
        -o "$OUTMOVIES/${label}.mp4" \
        > "$OUTMOVIES/${label}.log" 2>&1 || \
        echo "  WARNING: movie generation for ${label} failed (see $OUTMOVIES/${label}.log)"
}

check "plane_em_np1"  "$PYTHON scripts/analyze_dispersion.py data_PlaneEMWave --inp inputfiles/PlaneEMWave.inp --field E --component y --expected 'c*k' --tol 0.02"
check "plane_em_np4"  "$PYTHON scripts/analyze_dispersion.py data_PlaneEMWave_np4 --inp inputfiles/PlaneEMWave.inp --field E --component y --expected 'c*k' --tol 0.02"
check "alfven_np1"    "$PYTHON scripts/analyze_dispersion.py data_AlfvenWave --inp inputfiles/AlfvenWave.inp --field B --component y --method phase --phase-fit-points 5 --expected 'k*v_A' --tol 0.05"
check "alfven_np4"    "$PYTHON scripts/analyze_dispersion.py data_AlfvenWave_np4 --inp inputfiles/AlfvenWave.inp --field B --component y --method phase --phase-fit-points 5 --expected 'k*v_A' --tol 0.05"
check "twostream_np1" "$PYTHON scripts/analyze_growth_rate.py data_TwoStream --inp inputfiles/TwoStream.inp --expected 'wpe/(2*sqrt(2))' --tol 0.20"
check "twostream_np4" "$PYTHON scripts/analyze_growth_rate.py data_TwoStream_np4 --inp inputfiles/TwoStream.inp --expected 'wpe/(2*sqrt(2))' --tol 0.20"
check "whistler_np1"  "$PYTHON scripts/analyze_dispersion.py data_WhistlerPacket --inp inputfiles/WhistlerPacket.inp --field B --component y --expected 'k**2*c**2*omega_ce/(omega_pe**2 + k**2*c**2)' --tol 0.10"
check "whistler_np4"  "$PYTHON scripts/analyze_dispersion.py data_WhistlerPacket_np4 --inp inputfiles/WhistlerPacket.inp --field B --component y --expected 'k**2*c**2*omega_ce/(omega_pe**2 + k**2*c**2)' --tol 0.10"
check "oblique_alf_np1" "$PYTHON scripts/analyze_dispersion.py data_ObliqueAlfvenWave --inp inputfiles/ObliqueAlfvenWave.inp --field B --component z --method phase --phase-fit-points 5 --expected 'k*v_A*cos_kB' --tol 0.05"
check "oblique_alf_np4" "$PYTHON scripts/analyze_dispersion.py data_ObliqueAlfvenWave_np4 --inp inputfiles/ObliqueAlfvenWave.inp --field B --component z --method phase --phase-fit-points 5 --expected 'k*v_A*cos_kB' --tol 0.05"
# Heating: --compare overlays the np=1 curve onto the np=4 plot for visual cross-check
check "heating_np1"    "$PYTHON scripts/analyze_heating.py data_NumericalHeating --tol 0.05"
check "heating_np4"    "$PYTHON scripts/analyze_heating.py data_NumericalHeating_np4 --tol 0.05 --compare data_NumericalHeating --compare-label np=1"
# Reconnection: peak dψ/dt / (B0·v_A) ∈ [0.05, 0.30] — GEM-challenge consensus band
check "reconnection_np1" "$PYTHON scripts/analyze_reconnection.py data_Reconnection --inp inputfiles/Reconnection.inp"
check "reconnection_np4" "$PYTHON scripts/analyze_reconnection.py data_Reconnection_np4 --inp inputfiles/Reconnection.inp"
# Landau damping: γ_L from full Z-function dispersion, kλ_D = 0.4 → γ_L/ω_pe ≈ -0.066
check "landau_np1"       "$PYTHON scripts/analyze_landau.py data_LangmuirWave --inp inputfiles/LangmuirWave.inp --tol 0.20"
check "landau_np4"       "$PYTHON scripts/analyze_landau.py data_LangmuirWave_np4 --inp inputfiles/LangmuirWave.inp --tol 0.20"
# Weibel filamentation: γ_W(k_y) at the fundamental mode k_y = 2π/Ly. Cold-fluid
# Weibel γ_W = k·u_b·ω_pe/√(k²c²+ω_pe²); the integrated B-energy rate is dominated
# by the lowest unstable mode (highest k modes saturate first via filament merging).
check "weibel_np1"       "$PYTHON scripts/analyze_growth_rate.py data_Weibel --inp inputfiles/Weibel.inp --field magnetic --expected 'k_y*u_b*omega_pe/sqrt(k_y**2*c**2 + omega_pe**2)' --tol 0.20"
check "weibel_np4"       "$PYTHON scripts/analyze_growth_rate.py data_Weibel_np4 --inp inputfiles/Weibel.inp --field magnetic --expected 'k_y*u_b*omega_pe/sqrt(k_y**2*c**2 + omega_pe**2)' --tol 0.20"

# ── Movies (only tests with FieldOutputCycle > 0) ────────────────────────
echo "Generating propagation movies (best-effort) → $OUTMOVIES/"
movie "plane_em_np1"     data_PlaneEMWave         PlaneEMWave.inp        E y
movie "plane_em_np4"     data_PlaneEMWave_np4     PlaneEMWave.inp    E y
movie "alfven_np1"       data_AlfvenWave          AlfvenWave.inp         B y
movie "alfven_np4"       data_AlfvenWave_np4      AlfvenWave.inp     B y
movie "whistler_np1"     data_WhistlerPacket      WhistlerPacket.inp     B y
movie "whistler_np4"     data_WhistlerPacket_np4  WhistlerPacket.inp B y
movie "oblique_alf_np1"  data_ObliqueAlfvenWave      ObliqueAlfvenWave.inp     B z
movie "oblique_alf_np4"  data_ObliqueAlfvenWave_np4  ObliqueAlfvenWave.inp B z
# Reconnection: 2D xy slice of B_y reveals the islands and X-line collapse.
movie "reconnection_np1" data_Reconnection           Reconnection.inp          B y
movie "reconnection_np4" data_Reconnection_np4       Reconnection.inp      B y
# Landau: 1D Hovmöller of E_x — see initial cosine seed Landau-damp away.
movie "landau_np1"       data_LangmuirWave           LangmuirWave.inp          E x
movie "landau_np4"       data_LangmuirWave_np4       LangmuirWave.inp      E x
# Weibel: 2D pcolor of B_z — filament formation + merging.
movie "weibel_np1"       data_Weibel                 Weibel.inp                B z
movie "weibel_np4"       data_Weibel_np4             Weibel.inp            B z
# TwoStream + Heating: newly enabled field output, 1D E_x movies.
movie "twostream_np1"    data_TwoStream              TwoStream.inp             E x
movie "twostream_np4"    data_TwoStream_np4          TwoStream.inp         E x
movie "heating_np1"      data_NumericalHeating       NumericalHeating.inp      E x
movie "heating_np4"      data_NumericalHeating_np4   NumericalHeating.inp  E x

# ── Summary table ────────────────────────────────────────────────────────

echo
echo "Physics Validation Summary"
echo "──────────────────────────────────────────────────────"
printf "  %-6s  %-18s  %s\n" "STATUS" "TEST" "|Δ|/expected"
echo "──────────────────────────────────────────────────────"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r tag label rel <<< "$row"
    printf "  [%s]  %-18s  %s\n" "$tag" "$label" "$rel"
done
echo "──────────────────────────────────────────────────────"
if [[ $fail -eq 0 ]]; then
    echo "  all PASS"
else
    echo "  FAILED — see logs in $OUTROOT/"
fi
n_plots=$(find "$OUTPLOTS" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l | tr -d ' ' || true)
n_movies=$(find "$OUTMOVIES" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ' || true)
echo "  artifacts: $n_plots plots → $OUTPLOTS,  $n_movies movies → $OUTMOVIES"
echo

exit $fail
