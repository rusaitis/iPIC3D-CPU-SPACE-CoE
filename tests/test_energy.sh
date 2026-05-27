#!/usr/bin/env bash
#
# Energy-conservation regression test for the ECSIM field solver (CIC).
#
# Runs the Double_Harris current sheet at np=1 and np=4 for 30 cycles and checks
# that the total energy is conserved to near machine precision. The test is
# meaningful only when run multi-threaded: the bug it guards against is a race in
# the OpenMP moment-gather (add_Rho/add_Jxh/.../add_Mass), which silently loses
# deposit contributions and breaks energy conservation at high thread counts.
#
# Usage:   tests/test_energy.sh
# Env:     IPIC3D_EXE   path to the iPIC3D executable   (default: build/iPIC3D)
#          MPIRUN       launcher                         (default: mpirun)
#          OMP_NUM_THREADS  thread count to test with    (default: 4)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${IPIC3D_EXE:-$REPO/build/iPIC3D}"
MPIRUN="${MPIRUN:-mpirun}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Final-cycle |dE/E0| tolerance per case (discretization floor is ~1e-14).
TOL_NP1="1e-12"
TOL_NP4="1e-12"

[[ -x "$EXE" ]] || { echo "ERROR: executable not found at $EXE"; echo "Build first (see FIX-ENERGY.md), or set IPIC3D_EXE."; exit 2; }

# parse_ratio <run_dir> -> prints |dE/E0| at the final cycle of ConservedQuantities.txt
parse_ratio() {
  python3 - "$1" <<'PY'
import sys, os
path = os.path.join(sys.argv[1], "ConservedQuantities.txt")
rows = [l.split() for l in open(path) if l.strip() and l.strip()[0].isdigit()]
last = rows[-1]
total, delta = float(last[10]), float(last[11])   # cols: total_energy, delta_energy
print(f"{abs(delta)/abs(total):.6e} {last[0]}")
PY
}

run_case() {  # name  np  inputfile  tol
  local name="$1" np="$2" inp="$3" tol="$4"
  local out="$REPO/output/test_energy/$name"
  rm -rf "$out"; mkdir -p "$out"
  echo ">>> $name : np=$np, OMP_NUM_THREADS=$OMP_NUM_THREADS"
  "$MPIRUN" -np "$np" "$EXE" "$REPO/inputfiles/$inp" > "$out/run.log" 2>&1 \
    || { echo "    FAIL: simulation crashed (see $out/run.log)"; return 1; }
  read -r ratio cyc < <(parse_ratio "$out")
  if python3 -c "import sys; sys.exit(0 if $ratio <= $tol else 1)"; then
    echo "    PASS: |dE/E0| = $ratio at cycle $cyc  (tol $tol)"; return 0
  else
    echo "    FAIL: |dE/E0| = $ratio at cycle $cyc  (tol $tol)"; return 1
  fi
}

echo "=== ECSIM energy-conservation regression ==="
echo "    exe = $EXE"
rc=0
run_case gem     1 ci_energy_gem.inp     "$TOL_NP1" || rc=1
run_case gem_np4 4 ci_energy_gem_np4.inp "$TOL_NP4" || rc=1
echo "============================================="
[[ $rc -eq 0 ]] && echo "ALL ENERGY TESTS PASSED" || echo "ENERGY TESTS FAILED"
exit $rc
