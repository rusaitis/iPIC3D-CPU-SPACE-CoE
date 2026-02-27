#!/usr/bin/env bash
set -euo pipefail

# build.sh — Configure and build iPIC3D
#
# Usage:
#   ./build.sh [build_dir] [build_type] [options...]
#
# Examples:
#   ./build.sh                                          # defaults: build/ Release
#   ./build.sh build Release --ninja
#   ./build.sh build RelWithDebInfo --clean
#   ./build.sh --petsc                                  # enable PETSc solver
#   ./build.sh --hdf5-prefix /opt/hdf5                  # explicit HDF5 path
#
# Build types:
#   Debug | Release | RelWithDebInfo | MinSizeRel
#
# Options:
#   --petsc                Enable PETSc solver (USE_PETSC=ON)
#   --clean                Delete build dir before configuring
#   --ninja                Use Ninja generator (requires: ninja on PATH)
#   --hdf5-prefix PATH     Explicit HDF5 prefix (skips auto-detection)
#   --hdf5-formula NAME    Homebrew formula for HDF5 (default: hdf5-mpi)
#   --omp-prefix PATH      Explicit libomp prefix (macOS only — AppleClang needs this
#                            to find OpenMP; on Linux, GCC/Clang find it automatically)
#   --no-openmp-hints      Do not pass OpenMP hint variables to CMake
#   --petsc-prefix PATH    Explicit PETSc prefix (skips auto-detection)
#   --mpi-explicit         Resolve CC/CXX to full paths via `command -v`
#   --force-phdf5          Add -DHDF5_IS_PARALLEL=TRUE (compat for older CMakeLists)
#   --cmp0074 {auto,off}   CMake CMP0074 policy (default: off). Controls whether *_ROOT
#                            variables (e.g. HDF5_ROOT) are used by find_package(). CMake
#                            3.13+ defaults to NEW (enabled), so this is rarely needed.
#   -h, --help             Show this help
#
# Environment overrides:
#   HDF5_PREFIX, OMP_PREFIX, PETSC_PREFIX   Same as the --*-prefix flags
#   CC, CXX                                Compilers (default: mpicc/mpicxx)
#
# Dependency resolution order (HDF5, PETSc):
#   1. Explicit --*-prefix flag or environment variable
#   2. Homebrew (macOS)
#   3. pkg-config
#   4. Let CMake search system paths (Linux/HPC with modules)
#
# Manual CMake (without this script):
#
#   Linux (HDF5 + MPI in system paths):
#     cmake -S . -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
#     cmake --build build --parallel
#
#   Linux with PETSc:
#     cmake -S . -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
#       -DUSE_PETSC=ON
#     cmake --build build --parallel
#
#   macOS (Homebrew):
#     cmake -S . -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
#       -DHDF5_ROOT=$(brew --prefix hdf5-mpi) \
#       -DOpenMP_CXX_LIB_NAMES=omp \
#       -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
#       -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
#     cmake --build build --parallel
# END_HELP

# ─────────────────────────────────────────────────────────────────
show_help() {
  sed -n '/^# build\.sh/,/^# END_HELP/{
    /^# END_HELP/q
    s/^# \{0,1\}//; p
  }' "$0"
}

# ─────────────────────────── defaults ───────────────────────────
BUILD_DIR="build"
BUILD_TYPE="Release"
USE_NINJA=0
MPI_EXPLICIT=0
FORCE_PHDF5=0
DO_CLEAN=0
HDF5_FORMULA="hdf5-mpi"
HDF5_PREFIX_FLAG=""
OMP_PREFIX_FLAG=""
PETSC_PREFIX_FLAG=""
OPENMP_HINTS=1
CMP0074_MODE="off"
USE_PETSC=0

# ─────────────────────────── parse args ─────────────────────────
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --petsc)           USE_PETSC=1; shift ;;
    --clean)           DO_CLEAN=1; shift ;;
    --ninja)           USE_NINJA=1; shift ;;
    --mpi-explicit)    MPI_EXPLICIT=1; shift ;;
    --force-phdf5)     FORCE_PHDF5=1; shift ;;
    --no-openmp-hints) OPENMP_HINTS=0; shift ;;
    --hdf5-formula)
      [[ $# -ge 2 ]] || { echo "Error: --hdf5-formula requires a value" >&2; exit 2; }
      HDF5_FORMULA="$2"; shift 2 ;;
    --hdf5-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --hdf5-prefix requires a path" >&2; exit 2; }
      HDF5_PREFIX_FLAG="$2"; shift 2 ;;
    --omp-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --omp-prefix requires a path" >&2; exit 2; }
      OMP_PREFIX_FLAG="$2"; shift 2 ;;
    --petsc-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --petsc-prefix requires a path" >&2; exit 2; }
      PETSC_PREFIX_FLAG="$2"; shift 2 ;;
    --cmp0074)
      [[ $# -ge 2 ]] || { echo "Error: --cmp0074 requires 'auto' or 'off'" >&2; exit 2; }
      CMP0074_MODE="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    --) shift; break ;;
    -*)
      echo "Error: unknown option: $1" >&2
      echo "Run: $0 --help" >&2
      exit 2 ;;
    *)
      positional+=("$1"); shift ;;
  esac
done

if [[ ${#positional[@]} -ge 1 ]]; then BUILD_DIR="${positional[0]}"; fi
if [[ ${#positional[@]} -ge 2 ]]; then BUILD_TYPE="${positional[1]}"; fi

if [[ -z "$BUILD_DIR" || "$BUILD_DIR" == -* ]]; then
  echo "Error: invalid build_dir '$BUILD_DIR'" >&2
  exit 2
fi

# ─────────────────────────── helpers ────────────────────────────
brew_prefix() {
  command -v brew >/dev/null 2>&1 || return 1
  brew --prefix "$1" 2>/dev/null
}

pkg_config_prefix() {
  command -v pkg-config >/dev/null 2>&1 || return 1
  pkg-config --variable=prefix "$1" 2>/dev/null
}

OS="$(uname -s)"

# ─────────────────────────── resolve HDF5 ───────────────────────
# Priority: flag > env > Homebrew > let CMake find it
HDF5_PREFIX="${HDF5_PREFIX_FLAG:-${HDF5_PREFIX:-}}"

if [[ -z "$HDF5_PREFIX" ]]; then
  HDF5_PREFIX="$(brew_prefix "$HDF5_FORMULA")" || HDF5_PREFIX=""
fi

# On Linux, HDF5 is often in system paths or loaded via modules.
# If we still don't have a prefix, let CMake's find_package handle it.
if [[ -n "$HDF5_PREFIX" ]]; then
  echo "HDF5:      $HDF5_PREFIX"
else
  echo "HDF5:      (auto — letting CMake find it)"
fi

# ─────────────────────────── resolve OpenMP ─────────────────────
# Only needed on macOS (AppleClang requires explicit libomp paths).
# On Linux, GCC/Clang find OpenMP automatically with -fopenmp.
OMP_PREFIX="${OMP_PREFIX_FLAG:-${OMP_PREFIX:-}}"

if [[ -z "$OMP_PREFIX" ]]; then
  OMP_PREFIX="$(brew_prefix libomp)" || OMP_PREFIX=""
fi

if [[ -n "$OMP_PREFIX" ]]; then
  echo "OpenMP:    $OMP_PREFIX"
else
  echo "OpenMP:    (auto)"
fi

# ─────────────────────────── resolve PETSc ──────────────────────
PETSC_PREFIX="${PETSC_PREFIX_FLAG:-${PETSC_PREFIX:-}}"

if [[ "$USE_PETSC" -eq 1 ]]; then
  if [[ -z "$PETSC_PREFIX" ]]; then
    PETSC_PREFIX="$(brew_prefix petsc)" \
      || PETSC_PREFIX="$(pkg_config_prefix PETSc)" \
      || PETSC_PREFIX=""
  fi

  if [[ -n "$PETSC_PREFIX" ]]; then
    echo "PETSc:     $PETSC_PREFIX"
    export PKG_CONFIG_PATH="${PETSC_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  else
    echo "PETSc:     (auto — letting CMake/pkg-config find it)"
  fi
fi

# ─────────────────────────── compilers ──────────────────────────
CC="${CC:-mpicc}"
CXX="${CXX:-mpicxx}"
if [[ "$MPI_EXPLICIT" -eq 1 ]]; then
  CC="$(command -v "$CC" || true)"
  CXX="$(command -v "$CXX" || true)"
  [[ -n "$CC" && -n "$CXX" ]] || { echo "Error: compilers not found on PATH" >&2; exit 1; }
fi

# ─────────────────────────── generator ──────────────────────────
declare -a GENERATOR_ARGS=()
if [[ "$USE_NINJA" -eq 1 ]]; then
  command -v ninja >/dev/null 2>&1 || { echo "Error: --ninja requested but 'ninja' not found" >&2; exit 1; }
  GENERATOR_ARGS=(-G Ninja)
fi

# ─────────────────────────── summary ────────────────────────────
echo "Build dir: $BUILD_DIR"
echo "Type:      $BUILD_TYPE"
echo "Generator: ${GENERATOR_ARGS[*]:-default}"
echo "CC:        $CC"
echo "CXX:       $CXX"

# ─────────────────────────── clean ──────────────────────────────
if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "Clean:     removing '$BUILD_DIR'"
  rm -rf -- "$BUILD_DIR"
fi
mkdir -p -- "$BUILD_DIR"

# ─────────────────────────── CMake args ─────────────────────────
declare -a CMAKE_ARGS=()
if [[ ${#GENERATOR_ARGS[@]} -gt 0 ]]; then
  CMAKE_ARGS+=("${GENERATOR_ARGS[@]}")
fi
CMAKE_ARGS+=(
  -S . -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_C_COMPILER="${CC}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DMPI_C_COMPILER="${CC}"
  -DMPI_CXX_COMPILER="${CXX}"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# Add prefix path only if we have something to set
PREFIX_PATH=""
if [[ -n "$HDF5_PREFIX" ]]; then PREFIX_PATH="$HDF5_PREFIX"; fi
if [[ -n "$OMP_PREFIX" ]]; then PREFIX_PATH="${PREFIX_PATH:+${PREFIX_PATH};}${OMP_PREFIX}"; fi
if [[ -n "$PREFIX_PATH" ]]; then
  CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${PREFIX_PATH}")
fi

if [[ -n "$HDF5_PREFIX" ]]; then
  CMAKE_ARGS+=(-DHDF5_ROOT="${HDF5_PREFIX}")
fi

# CMP0074: make *_ROOT variables effective on older CMake policy settings
if [[ "$CMP0074_MODE" == "auto" ]]; then
  CMAKE_ARGS+=(-DCMAKE_POLICY_DEFAULT_CMP0074=NEW)
elif [[ "$CMP0074_MODE" != "off" ]]; then
  echo "Error: --cmp0074 must be 'auto' or 'off'" >&2; exit 2
fi

if [[ "$FORCE_PHDF5" -eq 1 ]]; then
  CMAKE_ARGS+=(-DHDF5_IS_PARALLEL=TRUE)
fi

# OpenMP hints (macOS only — AppleClang needs explicit libomp paths)
if [[ "$OPENMP_HINTS" -eq 1 && -n "$OMP_PREFIX" && "$OS" == "Darwin" ]]; then
  CMAKE_ARGS+=(
    -DOpenMP_CXX_LIB_NAMES=omp
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
    -DOpenMP_omp_LIBRARY="${OMP_PREFIX}/lib/libomp.dylib"
  )
fi

# PETSc
if [[ "$USE_PETSC" -eq 1 ]]; then
  CMAKE_ARGS+=(-DUSE_PETSC=ON)
fi

# ─────────────────────────── build ──────────────────────────────

# Print the full cmake command so users can copy/paste it later
echo ""
echo "CMake command:"
printf "  cmake"
for arg in "${CMAKE_ARGS[@]}"; do
  case "$arg" in
    -*)
      # Quote args that contain spaces or semicolons
      if [[ "$arg" == *" "* || "$arg" == *";"* ]]; then
        printf " \\\\\n    \"%s\"" "$arg"
      else
        printf " \\\\\n    %s" "$arg"
      fi ;;
    *)
      if [[ "$arg" == *" "* || "$arg" == *";"* ]]; then
        printf " \"%s\"" "$arg"
      else
        printf " %s" "$arg"
      fi ;;
  esac
done
echo ""
echo ""

cmake "${CMAKE_ARGS[@]}"

BUILD_CMD="cmake --build $BUILD_DIR --parallel"
echo ""
echo "Build command:"
echo "  $BUILD_CMD"
echo ""

$BUILD_CMD
