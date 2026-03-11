#!/usr/bin/env bash
#
# make_petsc_movie.sh — Generate mp4 movies of GMRES vs PETSc field comparison.
#
# Loops over all common output cycles, generates a PNG per cycle via
# plot_petsc_visual_comparison.py, then stitches them into an mp4 with ffmpeg.
#
# Usage:
#   bash tests/make_petsc_movie.sh [--gmres DIR] [--petsc DIR ...] [--fps N] [--output-dir DIR]
#
# If --gmres/--petsc are omitted, auto-discovers from tests/test_petsc_output/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/test_petsc_output"

# ── Defaults ──────────────────────────────────────────────────────────────
GMRES_DIR=""
PETSC_DIRS=()
FPS=5
OUTPUT_DIR=""

# ── Argument parsing ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gmres|--ref) GMRES_DIR="$2";       shift 2 ;;
        --petsc|--test) PETSC_DIRS+=("$2"); shift 2 ;;
        --fps)      FPS="$2";               shift 2 ;;
        --output-dir) OUTPUT_DIR="$2";      shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--gmres|--ref DIR] [--petsc|--test DIR ...] [--fps N] [--output-dir DIR]"
            echo ""
            echo "  --gmres, --ref DIR   Reference run directory (auto-discovered if omitted)"
            echo "  --petsc, --test DIR  Test run directory (repeatable; auto-discovered if omitted)"
            echo "  --fps N              Frames per second (default: 5)"
            echo "  --output-dir DIR     Directory for output mp4 (default: same as ref parent dir)"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-discovery ────────────────────────────────────────────────────────
if [[ -z "$GMRES_DIR" ]]; then
    candidates=( "$DEFAULT_OUTPUT_DIR"/GMRES_* )
    for c in "${candidates[@]}"; do
        if [[ -d "$c" ]]; then GMRES_DIR="$c"; break; fi
    done
    if [[ -z "$GMRES_DIR" ]]; then
        echo "ERROR: No GMRES directory found in $DEFAULT_OUTPUT_DIR/. Use --gmres." >&2
        exit 1
    fi
    echo "  Auto-discovered GMRES dir: $GMRES_DIR"
fi

if [[ ${#PETSC_DIRS[@]} -eq 0 ]]; then
    for c in "$DEFAULT_OUTPUT_DIR"/PETSc_*; do
        if [[ -d "$c" ]]; then
            PETSC_DIRS+=("$c")
            echo "  Auto-discovered PETSc dir: $c"
        fi
    done
    if [[ ${#PETSC_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: No PETSc directories found in $DEFAULT_OUTPUT_DIR/. Use --petsc." >&2
        exit 1
    fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$GMRES_DIR")}"

# ── Check dependencies ───────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is required but not found. Install with: brew install ffmpeg" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/plot_petsc_visual_comparison.py" ]]; then
    echo "ERROR: plot_petsc_visual_comparison.py not found in $SCRIPT_DIR/" >&2
    exit 1
fi

# ── Discover common cycles ───────────────────────────────────────────────
# Extract cycle numbers from Fields_NNNNN directories
discover_cycles() {
    local dir="$1"
    for d in "$dir"/Fields_*; do
        basename "$d" 2>/dev/null | sed -n 's/^Fields_0*\([0-9]\)/\1/p'
    done | sort -n
}

gmres_cycles=()
while IFS= read -r c; do
    gmres_cycles+=("$c")
done < <(discover_cycles "$GMRES_DIR")

if [[ ${#gmres_cycles[@]} -eq 0 ]]; then
    echo "ERROR: No Fields_* directories found in $GMRES_DIR" >&2
    exit 1
fi

# Find cycles common to GMRES and all PETSc dirs
common_cycles=("${gmres_cycles[@]}")
for pdir in "${PETSC_DIRS[@]}"; do
    pcycles=()
    while IFS= read -r c; do
        pcycles+=("$c")
    done < <(discover_cycles "$pdir")
    # Intersect: keep only cycles present in both lists
    filtered=()
    for c in "${common_cycles[@]}"; do
        for pc in "${pcycles[@]}"; do
            if [[ "$c" == "$pc" ]]; then
                filtered+=("$c")
                break
            fi
        done
    done
    common_cycles=("${filtered[@]}")
done

if [[ ${#common_cycles[@]} -eq 0 ]]; then
    echo "ERROR: No common output cycles found across run directories." >&2
    exit 1
fi

last_idx=$(( ${#common_cycles[@]} - 1 ))
echo "  Found ${#common_cycles[@]} common cycles: ${common_cycles[0]} .. ${common_cycles[$last_idx]}"

# ── Build petsc args ─────────────────────────────────────────────────────
petsc_args=()
for pdir in "${PETSC_DIRS[@]}"; do
    petsc_args+=(--petsc "$pdir")
done

# ── Verify Python dependencies before looping ────────────────────────────
echo "  Checking Python dependencies..."
dep_check=$(python3 -c "import numpy, matplotlib, h5py; print('ok')" 2>&1) || true
if [[ "$dep_check" != "ok" ]]; then
    echo "  ERROR: Missing Python dependencies for visual comparison."
    echo "  Install with: pip3 install numpy matplotlib h5py"
    echo "  $dep_check"
    exit 1
fi

# ── Pre-compute color bounds from last cycle ─────────────────────────────
last_cycle="${common_cycles[$last_idx]}"
echo "  Pre-computing color bounds from cycle $last_cycle..."
bounds=$(python3 "$SCRIPT_DIR/plot_petsc_visual_comparison.py" \
    --gmres "$GMRES_DIR" "${petsc_args[@]}" --cycle "$last_cycle" --print-bounds)

vmax_Bx=$(echo "$bounds" | sed -n 's/^vmax_Bx=//p')
vmax_Bx_diff=$(echo "$bounds" | sed -n 's/^vmax_Bx_diff=//p')
vmax_Ez=$(echo "$bounds" | sed -n 's/^vmax_Ez=//p')
vmax_Ez_diff=$(echo "$bounds" | sed -n 's/^vmax_Ez_diff=//p')

echo "    vmax_Bx=$vmax_Bx  vmax_Bx_diff=$vmax_Bx_diff"
echo "    vmax_Ez=$vmax_Ez  vmax_Ez_diff=$vmax_Ez_diff"

# ── Generate PNGs for each cycle ─────────────────────────────────────────
echo "  Generating comparison frames..."
for cycle in "${common_cycles[@]}"; do
    printf "    Cycle %05d\n" "$cycle"
    python3 "$SCRIPT_DIR/plot_petsc_visual_comparison.py" \
        --gmres "$GMRES_DIR" "${petsc_args[@]}" --cycle "$cycle" \
        --vmax-Bx "$vmax_Bx" --vmax-Bx-diff "$vmax_Bx_diff" \
        --vmax-Ez "$vmax_Ez" --vmax-Ez-diff "$vmax_Ez_diff" || {
        echo "    WARNING: Failed to generate frame for cycle $cycle"
    }
done

# ── Stitch PNGs into mp4 for each PETSc solver ──────────────────────────
for pdir in "${PETSC_DIRS[@]}"; do
    label="$(basename "$pdir")"
    pattern="$OUTPUT_DIR/visual_comparison_${label}_cycle*.png"

    # Check that at least some frames exist
    frame_count=$(ls $pattern 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$frame_count" -eq 0 ]]; then
        echo "  WARNING: No frames found for $label, skipping movie."
        continue
    fi

    output_mp4="$OUTPUT_DIR/visual_comparison_${label}.mp4"
    echo "  Stitching $frame_count frames into $output_mp4 (${FPS} fps)..."

    ffmpeg -framerate "$FPS" -pattern_type glob \
        -i "$pattern" \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -c:v libx264 -pix_fmt yuv420p -y \
        "$output_mp4" 2>/dev/null || {
        echo "  WARNING: ffmpeg failed for $label"
        continue
    }

    echo "  Saved: $output_mp4"
done

echo "  Movie generation complete."
