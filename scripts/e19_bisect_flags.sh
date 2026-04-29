#!/usr/bin/env bash
# Phase E.19 bisect: for each flag, run two reps with that flag explicitly OFF
# (overriding EpsilonReproducibility=1) and report cyc-1 absdiff in col 2.
set -euo pipefail
INP=inputfiles/phaseE19_repro.inp
WORK=/tmp/e19_bisect
mkdir -p "$WORK"

run_pair() {
  local label="$1"
  local extra="$2"
  local d="$WORK/$label"
  mkdir -p "$d"
  printf '%s\n' "$(<"$INP")" "$extra" > "$d/inp.inp"
  for r in 1 2; do
    sed -e "s|SimulationName *= *.*|SimulationName = e19_${label}|" \
        -e "s|SaveDirName *= *.*|SaveDirName = output/e19_${label}|" \
        "$d/inp.inp" > "$d/inp_r${r}.inp"
    OMP_NUM_THREADS=1 mpirun -np 4 build/iPIC3D "$d/inp_r${r}.inp" > "$d/run_r${r}.log" 2>&1
    cp "output/e19_${label}/ConservedQuantities.txt" "$d/CQ_r${r}.txt"
  done
  awk 'NR==FNR{a[$1]=$0; next} ($1==1){
    split(a[$1], aa); diff = aa[2] - $2; if(diff<0) diff=-diff;
    printf "  cyc1 col2 absdiff = %.4e\n", diff;
  }' "$d/CQ_r1.txt" "$d/CQ_r2.txt"
}

flags="DeterministicMPIReductions DeterministicThreadMoments DeterministicParticleComm KahanParticleSums KahanGather KahanFieldEnergy KahanHalo"
echo "== baseline (all flags via EpsilonReproducibility=1) =="
run_pair "baseline" ""
for f in $flags; do
  echo "== ${f}=0 =="
  run_pair "no_${f}" "${f} = 0"
done
