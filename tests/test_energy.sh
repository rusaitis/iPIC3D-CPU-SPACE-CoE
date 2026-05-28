#!/usr/bin/env bash
# Energy-conservation regression test (30 cycles).
#
# Three cases:
#   gem       Double_Harris,  Smooth=1, num_smoothings=4   np=1 (production-default path)
#   uniform   Maxwellian,     Smooth=0                     np=1 (unsmoothed energy floor)
#   gem_np4   Double_Harris,  Smooth=1, EpsilonRepro=1     np=4 (MPI halo + cross-rank path)
#
# Measured baselines on a clean tree (Apple silicon, default flags):
#   gem:     |dE/E0| = 6.48e-15
#   uniform: |dE/E0| = 5.49e-16
#   gem_np4: |dE/E0| = 1.27e-16
#
# Tolerances are ~15-80x the measured baseline. Any regression that pushes drift
# past these will be caught; ULP-class FP fluctuations across builds will not.

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/tests/common.sh"
EXE="${EXE:-${IPIC3D_BUILD_DIR:-$REPO/build}/iPIC3D}"
OUTDIR="${OUTDIR:-$REPO/output/test_energy}"
GEM_TOL="${GEM_TOL:-1e-13}"
UNIFORM_TOL="${UNIFORM_TOL:-1e-14}"
GEM_NP4_TOL="${GEM_NP4_TOL:-1e-14}"

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: iPIC3D not found at $EXE" >&2
    echo "       set EXE=... or run 'pixi run build' first" >&2
    exit 2
fi

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

cd "$REPO"

# run_case CASE NP BASE.inp [XLEN YLEN] [KEY=VAL ...]
# For NP>1 the decomposition is generated on the fly from the np=1 base (see
# make_np_variant in common.sh) — no committed "_np4" inputs. Output is forced
# to output/test_energy/<case> so the check below finds it.
run_case() {
    local case=$1 np=$2 base=$3 xlen=${4:-1} ylen=${5:-1}
    shift $(( $# < 5 ? $# : 5 ))   # remaining args = extra KEY=VAL overrides
    local logfile="$OUTDIR/${case}.log"
    local inp="inputfiles/$base"
    if [[ $np -gt 1 ]]; then
        local tmp="$OUTDIR/${case}.inp"
        make_np_variant "$inp" "$tmp" "$xlen" "$ylen" 1 "output/test_energy/${case}" "$@"
        inp="$tmp"
    fi
    echo "─── ${case}: mpirun -np ${np} $(basename "$EXE") ${inp}"
    mpirun -np "$np" "$EXE" "$inp" > "$logfile" 2>&1
}

run_case gem     1 ci_energy_gem.inp
run_case uniform 1 ci_energy_uniform.inp
run_case gem_np4 4 ci_energy_gem.inp 2 2 EpsilonReproducibility=1

# Inline check: read final |dE/E0| from each ConservedQuantities.txt and compare to tol.
python3 - "$REPO/scripts" \
    "$OUTDIR/gem"     "$GEM_TOL" \
    "$OUTDIR/uniform" "$UNIFORM_TOL" \
    "$OUTDIR/gem_np4" "$GEM_NP4_TOL" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[1])
from plot_utils import parse_conserved_quantities
sys.argv = sys.argv[:1] + sys.argv[2:]

cases = [(sys.argv[i], float(sys.argv[i+1])) for i in (1, 3, 5)]
print()
print("Energy Conservation Regression")
print("─" * 56)
fail = 0
for outdir, tol in cases:
    label = os.path.basename(outdir)
    cq = parse_conserved_quantities(outdir)
    if not cq:
        print(f"  [FAIL] {label:<10s}  no ConservedQuantities.txt in {outdir}")
        fail = 1
        continue
    e0 = cq['total_energy'][0]
    de = cq['delta_energy'][-1]
    ratio = abs(de / e0) if abs(e0) > 0 else float('inf')
    ok = ratio < tol
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fail = 1
    print(f"  [{tag}] {label:<10s}  |dE/E0| = {ratio:.3e}   tol = {tol:.0e}")
print("─" * 56)
print(f"  {'all PASS' if fail == 0 else 'FAILED'}")
print()
sys.exit(fail)
PY
