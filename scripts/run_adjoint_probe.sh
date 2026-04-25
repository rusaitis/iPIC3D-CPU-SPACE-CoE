#!/usr/bin/env bash
# Step 34b driver: runs the self-adjointness probe across six flag variants.
#
# Each variant overrides one flag on the AdjointProbe.inp base (all fixes on,
# lap_graddiv, ECS smoothing). Writes the probe line + tag to
# output/adjoint_probe/results.txt.
set -u

BASE=$(cd "$(dirname "$0")/.." && pwd)
IN=$BASE/inputfiles/AdjointProbe.inp
EXE=$BASE/build/iPIC3D
WORK=${ADJOINT_PROBE_WORK:-/tmp/adjoint_probe}
LOG=$WORK/results.txt

mkdir -p "$WORK"
: > "$LOG"

run_variant () {
    local tag=$1
    local tmp_in=$WORK/AdjointProbe_${tag}.inp
    cp "$IN" "$tmp_in"
    shift
    for kv in "$@"; do
        local key=${kv%%=*}
        local val=${kv#*=}
        # replace line starting with the key (word-boundary tolerant)
        # first look for the exact key= line
        if grep -qE "^${key}[[:space:]]*=" "$tmp_in"; then
            sed -i.bak -E "s|^(${key}[[:space:]]*=).*|\1 ${val}|" "$tmp_in"
        else
            echo "${key} = ${val}" >> "$tmp_in"
        fi
    done
    rm -f "$tmp_in.bak"

    local run_log=$WORK/run_${tag}.log
    (cd "$BASE" && mpirun -np 1 "$EXE" "$tmp_in" >"$run_log" 2>&1) \
        && echo "# ${tag}" >> "$LOG" \
        && grep -E "adjoint-probe|duality-probe" "$run_log" >> "$LOG" \
        || { echo "# ${tag} FAILED (see ${run_log})" >> "$LOG"; return 1; }
}

# All-fixes-on baseline (offset-by-one halo + unify + lap_graddiv now hard-coded)
run_variant A_all_fixes_on
# Toggle the remaining togglable correctness flag
run_variant B_no_symm        SymmetrizeMaxwellImage=false
# Operator choice
run_variant C_curl_curl      MaxwellOperator=curl_curl
# Both correctness toggles off
run_variant D_curl_no_symm   MaxwellOperator=curl_curl SymmetrizeMaxwellImage=false

cat "$LOG"
