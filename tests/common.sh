#!/usr/bin/env bash
#
# common.sh — shared helper functions for test.sh
#
# Source this file after PROJECT_DIR is set:
#   source "$SCRIPT_DIR/common.sh"
#

# Validate that $2 is a positive integer; print error and exit 2 otherwise.
# Usage: validate_pos_int --flag-name value
validate_pos_int() {
    local flag="$1" val="$2"
    [[ "$val" =~ ^[0-9]+$ && "$val" -gt 0 ]] \
        || { echo "Error: $flag requires a positive integer" >&2; exit 2; }
}

# Validate that $2 is a non-negative integer; print error and exit 2 otherwise.
# Usage: validate_nonneg_int --flag-name value
validate_nonneg_int() {
    local flag="$1" val="$2"
    [[ "$val" =~ ^[0-9]+$ ]] \
        || { echo "Error: $flag requires a non-negative integer" >&2; exit 2; }
}

# Compute the most-square XY factorization of NP.
# Echoes "XLEN YLEN" (XLEN >= YLEN, i.e. XLEN is the larger factor).
compute_topology() {
    local np="$1"
    local ylen xlen
    ylen=$("${PYTHON:-python3}" -c "
import math
n = $np
best = (1, n)
for i in range(1, int(math.sqrt(n)) + 1):
    if n % i == 0:
        best = (i, n // i)
print(best[0])
")
    xlen=$(( np / ylen ))
    echo "$xlen $ylen"
}

# Warn if NP exceeds available logical CPUs.
# Usage: warn_oversubscription NP
warn_oversubscription() {
    local np="$1"
    local avail
    avail=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 0)
    if [[ $avail -gt 0 && $np -gt $avail ]]; then
        local _yw=$'\033[33m' _rs=$'\033[0m'
        [[ ! -t 1 ]] && { _yw=''; _rs=''; }
        echo "${_yw}WARNING:${_rs} --np $np exceeds $avail logical CPUs (oversubscription may degrade performance or fail)."
        echo "         Consider --np N with N <= $avail."
        echo ""
    fi
}

# Generate a parallel-decomposition .inp on the fly from an np=1 base input,
# so we don't carry duplicate "_np4" files. Substitutes the MPI topology and
# output-dir keys, then appends any extra KEY=VAL overrides (e.g. to flip
# EpsilonReproducibility on for the cross-rank energy case).
# Usage: make_np_variant BASE OUT XLEN YLEN ZLEN OUT_DIRNAME [KEY=VAL ...]
make_np_variant() {
    local base=$1 out=$2 x=$3 y=$4 z=$5 dir=$6; shift 6
    sed -E \
        -e "s|^([[:space:]]*XLEN[[:space:]]*=).*|\1 ${x}|" \
        -e "s|^([[:space:]]*YLEN[[:space:]]*=).*|\1 ${y}|" \
        -e "s|^([[:space:]]*ZLEN[[:space:]]*=).*|\1 ${z}|" \
        -e "s|^([[:space:]]*SimulationName[[:space:]]*=).*|\1 $(basename "$dir")|" \
        -e "s|^([[:space:]]*SaveDirName[[:space:]]*=).*|\1 ${dir}|" \
        -e "s|^([[:space:]]*RestartDirName[[:space:]]*=).*|\1 ${dir}|" \
        "$base" > "$out"
    local kv
    for kv in "$@"; do
        printf '%s = %s\n' "${kv%%=*}" "${kv#*=}" >> "$out"
    done
}
