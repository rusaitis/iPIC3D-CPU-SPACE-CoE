#!/usr/bin/env bash
# Phase 8 multi-run measurement harness
# Usage: ./scripts/phase8_bench.sh <input.inp> <label> [nruns]
# Outputs TSV: label run dE1 dE10
set -euo pipefail

INPUT="${1:?usage: $0 input.inp label [nruns]}"
LABEL="${2:?need label}"
NRUNS="${3:-5}"

# Parse SaveDirName from input file
SAVEDIR=$(grep -E "^SaveDirName" "$INPUT" | awk -F= '{gsub(/ /,"",$2); print $2}')
if [ -z "$SAVEDIR" ]; then
  echo "ERROR: Could not parse SaveDirName from $INPUT" >&2
  exit 1
fi

for run in $(seq 1 "$NRUNS"); do
  rm -rf "$SAVEDIR"
  mpirun -np 4 build/iPIC3D "$INPUT" > /tmp/phase8_run.log 2>&1 || { echo "ERROR run=$run"; tail -20 /tmp/phase8_run.log; continue; }

  # Parse ConservedQuantities.txt: column 12 (0-indexed: 11) is VI = dE
  # Columns: I=1 II=2 IIa=3 ... VI=12 VII=13
  DE1=$(awk 'NR>0 && $1==1 {print $12; exit}' "$SAVEDIR/ConservedQuantities.txt")
  DE10=$(awk 'NR>0 && $1==10 {print $12; exit}' "$SAVEDIR/ConservedQuantities.txt")
  printf "%s\t%d\t%s\t%s\n" "$LABEL" "$run" "$DE1" "$DE10"
done
