#!/usr/bin/env bash
#
# make_movie.sh — Generate mp4 movies of solver field comparison.
#
# Delegates frame generation to plot_field_comparison.py --all-cycles,
# then stitches PNGs into mp4 with ffmpeg.
#
# Usage:
#   bash tests/make_movie.sh [--ref DIR] [--test DIR ...] [--fps N] [OPTIONS]
#
# If --ref/--test are omitted, auto-discovers from tests/test_output/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/test_output"

PYTHON="${PYTHON:-python3}"
REF_DIR=""
TEST_DIRS=()
FPS=5
OUTPUT_DIR=""
PASSTHROUGH=()   # forwarded to plot_field_comparison.py

# ── Argument parsing ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gmres|--ref)   REF_DIR="$2";          shift 2 ;;
        --petsc|--test)  TEST_DIRS+=("$2");      shift 2 ;;
        --fps)           FPS="$2";               shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";        shift 2 ;;
        --fields)        PASSTHROUGH+=(--fields "$2");    shift 2 ;;
        --diff-mode)     PASSTHROUGH+=(--diff-mode "$2"); shift 2 ;;
        --light)         PASSTHROUGH+=(--light);          shift ;;
        --help|-h)
            cat <<'HELP'
Usage: make_movie.sh [--ref DIR] [--test DIR ...] [--fps N] [OPTIONS]

  --ref DIR              Reference run directory (auto-discovered if omitted)
  --test DIR             Test run directory (repeatable; auto-discovered if omitted)
  --fps N                Frames per second (default: 5)
  --output-dir DIR       Directory for output mp4 (default: parent of ref dir)
  --fields SPEC          Field components to compare, e.g. Bx,By,Ez (default: Bx,Ez)
  --diff-mode MODE       percent or absolute (default: percent)
  --light                Light plot theme

Legacy aliases: --gmres (= --ref), --petsc (= --test)
HELP
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Check ffmpeg early ───────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is required but not found. Install with: brew install ffmpeg" >&2
    exit 1
fi

# ── Auto-discovery when --ref/--test omitted ─────────────────────────────
if [[ -z "$REF_DIR" && ${#TEST_DIRS[@]} -eq 0 ]]; then
    # Find reference dir: check ref_solver.txt, then fall back to GMRES_*
    if [[ -f "$DEFAULT_OUTPUT_DIR/ref_solver.txt" ]]; then
        ref_label=$(< "$DEFAULT_OUTPUT_DIR/ref_solver.txt")
        for c in "$DEFAULT_OUTPUT_DIR"/"${ref_label}"_*; do
            if [[ -d "$c" ]]; then REF_DIR="$c"; break; fi
        done
    fi
    if [[ -z "$REF_DIR" ]]; then
        for c in "$DEFAULT_OUTPUT_DIR"/GMRES_*; do
            if [[ -d "$c" ]]; then REF_DIR="$c"; break; fi
        done
    fi
    if [[ -z "$REF_DIR" ]]; then
        echo "ERROR: No reference directory found in $DEFAULT_OUTPUT_DIR/. Use --ref." >&2
        exit 1
    fi
    echo "  Auto-discovered ref dir: $REF_DIR"

    # Find test dirs: everything with Fields_* subdirs that isn't the ref
    ref_base="$(basename "$REF_DIR")"
    for c in "$DEFAULT_OUTPUT_DIR"/*/; do
        [[ -d "$c" ]] || continue
        base="$(basename "$c")"
        [[ "$base" == "$ref_base" ]] && continue
        has_fields=false
        for f in "$c"/Fields_*; do
            if [[ -d "$f" ]]; then has_fields=true; break; fi
        done
        if $has_fields; then
            TEST_DIRS+=("${c%/}")
            echo "  Auto-discovered test dir: ${c%/}"
        fi
    done
    if [[ ${#TEST_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: No test directories found in $DEFAULT_OUTPUT_DIR/. Use --test." >&2
        exit 1
    fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$REF_DIR")}"

# ── Generate frames via plot_field_comparison.py --all-cycles ────────────
plot_args=(--all-cycles)
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && plot_args+=("${PASSTHROUGH[@]}")
[[ -n "$REF_DIR" ]] && plot_args+=(--ref "$REF_DIR")
for d in "${TEST_DIRS[@]}"; do plot_args+=(--test "$d"); done
[[ -n "$OUTPUT_DIR" ]] && plot_args+=(--output-dir "$OUTPUT_DIR")

echo "  Generating comparison frames..."
"$PYTHON" "$SCRIPT_DIR/plot_field_comparison.py" "${plot_args[@]}"

# ── Stitch PNGs into mp4 for each test solver ───────────────────────────
for tdir in "${TEST_DIRS[@]}"; do
    label="$(basename "$tdir")"
    pattern="$OUTPUT_DIR/visual_comparison_${label}_cycle*.png"

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
