#!/usr/bin/env bash
# Physics validation test suite — exercises wave dispersion + instability
# growth rates across the periodic-BC halo path at np=1 and np=4.
#
# Tests:
#   A. Plane EM wave in vacuum   ω = c·k          (analyze_dispersion.py)
#   B. Shear Alfvén wave         ω = k·v_A        (analyze_dispersion.py)
#   C. Two-stream instability    γ = ω_pe/(2√2)   (analyze_growth_rate.py)
#   D. Whistler R-mode wave      ω = ω_R(k)       (analyze_dispersion.py)
#
# Each test runs once at np=1 (single-rank periodic-self halo path) and once
# at np=4 X-decomposition (cross-rank periodic halo path). The unification
# work guarantees these paths produce equivalent physics; this suite checks
# the guarantee against analytical references.
#
# Runtime: ~17 min on Apple silicon. Designed for nightly / pre-merge CI,
# not the inner loop. Use tests/test_energy.sh + `pixi run test` for that.

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${EXE:-${IPIC3D_BUILD_DIR:-$REPO/build}/iPIC3D}"
OUTROOT="${OUTROOT:-$REPO/output/test_physics}"
PYTHON="${PYTHON:-pixi run python}"

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: iPIC3D not found at $EXE" >&2
    echo "       set EXE=... or run 'pixi run build' first" >&2
    exit 2
fi

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

cd "$REPO"

# ── Run all 8 simulations (4 tests × 2 decompositions) ───────────────────

run_sim() {
    local label=$1 np=$2 inp=$3 outdir=$4
    local logfile="$OUTROOT/${label}.log"
    echo "─── ${label}: mpirun -np ${np} $(basename "$EXE") inputfiles/$inp"
    rm -rf "$outdir"
    mpirun -np "$np" "$EXE" "inputfiles/$inp" > "$logfile" 2>&1
}

run_sim plane_em_np1     1 PlaneEMWave.inp       "$REPO/data_PlaneEMWave"
run_sim plane_em_np4     4 PlaneEMWave_np4.inp   "$REPO/data_PlaneEMWave_np4"
run_sim alfven_np1       1 AlfvenWave.inp        "$REPO/data_AlfvenWave"
run_sim alfven_np4       4 AlfvenWave_np4.inp    "$REPO/data_AlfvenWave_np4"
run_sim twostream_np1    1 TwoStream.inp         "$REPO/data_TwoStream"
run_sim twostream_np4    4 TwoStream_np4.inp     "$REPO/data_TwoStream_np4"
run_sim whistler_np1     1 WhistlerPacket.inp    "$REPO/data_WhistlerPacket"
run_sim whistler_np4     4 WhistlerPacket_np4.inp "$REPO/data_WhistlerPacket_np4"

# ── Run analyzers, capture pass/fail ──────────────────────────────────────

declare -a RESULTS
fail=0

check() {
    local label=$1 cmd=$2
    local out
    if out=$(eval "$cmd" 2>&1); then
        local tag="PASS"
    else
        local tag="FAIL"
        fail=1
    fi
    # Extract |Δω/ω| or |Δγ/γ| value (after the colon, before "tol =")
    local rel
    rel=$(echo "$out" | grep -E '\|Δω/ω\||\|Δγ/γ\|' | sed -E 's/.*: *([0-9.e+-]+).*/\1/' | head -1)
    RESULTS+=("$tag|$label|$rel")
}

check "plane_em_np1"  "$PYTHON scripts/analyze_dispersion.py data_PlaneEMWave --inp inputfiles/PlaneEMWave.inp --field E --component y --expected 'c*k' --tol 0.02"
check "plane_em_np4"  "$PYTHON scripts/analyze_dispersion.py data_PlaneEMWave_np4 --inp inputfiles/PlaneEMWave_np4.inp --field E --component y --expected 'c*k' --tol 0.02"
check "alfven_np1"    "$PYTHON scripts/analyze_dispersion.py data_AlfvenWave --inp inputfiles/AlfvenWave.inp --field B --component y --expected 'k*v_A' --tol 0.05"
check "alfven_np4"    "$PYTHON scripts/analyze_dispersion.py data_AlfvenWave_np4 --inp inputfiles/AlfvenWave_np4.inp --field B --component y --expected 'k*v_A' --tol 0.05"
check "twostream_np1" "$PYTHON scripts/analyze_growth_rate.py data_TwoStream --inp inputfiles/TwoStream.inp --expected 'wpe/(2*sqrt(2))' --tol 0.20"
check "twostream_np4" "$PYTHON scripts/analyze_growth_rate.py data_TwoStream_np4 --inp inputfiles/TwoStream_np4.inp --expected 'wpe/(2*sqrt(2))' --tol 0.20"
check "whistler_np1"  "$PYTHON scripts/analyze_dispersion.py data_WhistlerPacket --inp inputfiles/WhistlerPacket.inp --field B --component y --expected 'k**2*c**2*omega_ce/(omega_pe**2 + k**2*c**2)' --tol 0.10"
check "whistler_np4"  "$PYTHON scripts/analyze_dispersion.py data_WhistlerPacket_np4 --inp inputfiles/WhistlerPacket_np4.inp --field B --component y --expected 'k**2*c**2*omega_ce/(omega_pe**2 + k**2*c**2)' --tol 0.10"

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
echo

exit $fail
