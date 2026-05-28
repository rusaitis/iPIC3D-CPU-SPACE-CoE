#!/usr/bin/env bash
# Long-run energy-conservation sweep: uniform plasma and Double Harris (the
# periodic magnetic-reconnection problem), each at np=1 and np=4, OMP-threaded
# to exercise the moment-gather race the atomic-update fix closes.
# Redirects all sim output to per-case logs; prints only a summary table.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="/opt/homebrew/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
EXE="${IPIC3D_EXE:-build-fix/iPIC3D}"
LOGDIR=output/test_energy_long
mkdir -p "$LOGDIR"

run() {  # name np inputfile
  local name="$1" np="$2" inp="$3"
  echo ">>> $name (np=$np, OMP=$OMP_NUM_THREADS)"
  rm -rf "$LOGDIR/$name"
  mpirun -np "$np" "$EXE" "inputfiles/$inp" > "$LOGDIR/$name.log" 2>&1 \
    && echo "    done" || echo "    CRASH (see $LOGDIR/$name.log)"
}

run uniform        1 ci_energy_uniform.inp
run uniform_np4    4 ci_energy_uniform_np4.inp
run harris         1 ci_energy_harris.inp
run harris_np4     4 ci_energy_harris_np4.inp
echo "ALL RUNS COMPLETE"
