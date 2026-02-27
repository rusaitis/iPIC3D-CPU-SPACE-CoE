#!/usr/bin/env bash
set -euo pipefail

# configure-build-portable.sh
#
# Usage:
#   ./configure-build-portable.sh [build_dir] [build_type] [options...]
#
# Examples:
#   ./configure-build-portable.sh
#   ./configure-build-portable.sh build Release --ninja
#   ./configure-build-portable.sh build RelWithDebInfo --clean
#   ./configure-build-portable.sh build Release --hdf5-prefix /opt/hdf5 --omp-prefix /opt/libomp
#   ./configure-build-portable.sh build Release --hdf5-formula hdf5
#
# Build types:
#   Debug | Release | RelWithDebInfo | MinSizeRel
#
# Options:
#   --ninja                Use Ninja generator (requires: ninja on PATH)
#   --mpi-explicit         Resolve CC/CXX to full paths via `command -v`
#   --force-phdf5          Add -DHDF5_IS_PARALLEL=TRUE (compat for older CMakeLists)
#   --clean                Delete build dir before configuring
#   --hdf5-formula NAME    Homebrew formula to use for HDF5 (default: hdf5-mpi)
#   --hdf5-prefix PATH     Explicit HDF5 prefix (no Homebrew required)
#   --omp-prefix PATH      Explicit libomp prefix (optional)
#   --no-openmp-hints      Do not pass any OpenMP hint variables to CMake
#   --cmp0074 {auto,off}   Set CMP0074 policy default to NEW (auto) or skip it (off). Default: auto
#   --petsc                Enable PETSc solver (USE_PETSC=ON). PETSc must be installed (Homebrew or PETSC_PREFIX)
#   -h, --help             Show help
#
# Environment overrides:
#   HDF5_PREFIX, OMP_PREFIX     Same as flags above
#   CC, CXX                     Compilers (default: mpicc/mpicxx)
#
# Notes:
# - This script avoids fragile line continuations and is Bash 3.x + `set -u` friendly.
# - It tries to work without Homebrew if you provide --hdf5-prefix (or HDF5_PREFIX).
# - OpenMP hints are only applied by default on macOS when OMP_PREFIX is known.

show_help() { sed -n '1,140p' "$0"; }

BUILD_DIR="build"
BUILD_TYPE="Release"
USE_NINJA=0
MPI_EXPLICIT=0
FORCE_PHDF5=0
DO_CLEAN=0
HDF5_FORMULA="hdf5-mpi"
HDF5_PREFIX_FLAG=""
OMP_PREFIX_FLAG=""
OPENMP_HINTS=1
CMP0074_MODE="off"  # auto|off  (CMake 3.13+ defaults CMP0074 to NEW already)
USE_PETSC=0

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ninja) USE_NINJA=1; shift ;;
    --mpi-explicit) MPI_EXPLICIT=1; shift ;;
    --force-phdf5) FORCE_PHDF5=1; shift ;;
    --clean) DO_CLEAN=1; shift ;;
    --hdf5-formula)
      [[ $# -ge 2 ]] || { echo "Error: --hdf5-formula requires a value" >&2; exit 2; }
      HDF5_FORMULA="$2"; shift 2 ;;
    --hdf5-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --hdf5-prefix requires a path" >&2; exit 2; }
      HDF5_PREFIX_FLAG="$2"; shift 2 ;;
    --omp-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --omp-prefix requires a path" >&2; exit 2; }
      OMP_PREFIX_FLAG="$2"; shift 2 ;;
    --no-openmp-hints) OPENMP_HINTS=0; shift ;;
    --cmp0074)
      [[ $# -ge 2 ]] || { echo "Error: --cmp0074 requires 'auto' or 'off'" >&2; exit 2; }
      CMP0074_MODE="$2"; shift 2 ;;
    --petsc) USE_PETSC=1; shift ;;
    -h|--help) show_help; exit 0 ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
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

brew_prefix() {
  local formula="$1"
  command -v brew >/dev/null 2>&1 || return 1
  brew --prefix "$formula"
}

# Resolve prefixes: flags > env > brew > empty
HDF5_PREFIX="${HDF5_PREFIX_FLAG:-${HDF5_PREFIX:-}}"
OMP_PREFIX="${OMP_PREFIX_FLAG:-${OMP_PREFIX:-}}"

if [[ -z "$HDF5_PREFIX" ]]; then
  if HDF5_PREFIX="$(brew_prefix "$HDF5_FORMULA")"; then
    :
  else
    echo "Error: HDF5 prefix not set." >&2
    echo "  Provide --hdf5-prefix PATH (or set HDF5_PREFIX), or install via Homebrew." >&2
    exit 2
  fi
fi

if [[ -z "$OMP_PREFIX" ]]; then
  # Optional: only try brew if available. Otherwise leave empty.
  if OMP_PREFIX="$(brew_prefix libomp)"; then
    :
  else
    OMP_PREFIX=""
  fi
fi

echo "HDF5:      $HDF5_PREFIX"
if [[ -n "$OMP_PREFIX" ]]; then
  echo "OpenMP:    $OMP_PREFIX"
else
  echo "OpenMP:    (unset)"
fi
echo "Build dir: $BUILD_DIR"
echo "Type:      $BUILD_TYPE"

# Generator args (prepend later)
declare -a GENERATOR_ARGS
GENERATOR_ARGS=()
if [[ "$USE_NINJA" -eq 1 ]]; then
  command -v ninja >/dev/null 2>&1 || { echo "Error: --ninja requested but 'ninja' not found on PATH" >&2; exit 1; }
  GENERATOR_ARGS=(-G Ninja)
  echo "Generator: Ninja"
else
  echo "Generator: default"
fi

# Compilers
CC="${CC:-mpicc}"
CXX="${CXX:-mpicxx}"
if [[ "$MPI_EXPLICIT" -eq 1 ]]; then
  CC="$(command -v "$CC" || true)"
  CXX="$(command -v "$CXX" || true)"
  [[ -n "$CC" && -n "$CXX" ]] || { echo "Error: compilers not found on PATH (CC=$CC, CXX=$CXX)" >&2; exit 1; }
fi
echo "MPI CC:    $CC"
echo "MPI CXX:   $CXX"

if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "Clean:     removing '$BUILD_DIR'"
  rm -rf -- "$BUILD_DIR"
fi
mkdir -p -- "$BUILD_DIR"

# Base CMake args
declare -a CMAKE_ARGS
CMAKE_ARGS=(
  -S . -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_C_COMPILER="${CC}"
  -DCMAKE_CXX_COMPILER="${CXX}"
  -DCMAKE_PREFIX_PATH="${HDF5_PREFIX}${OMP_PREFIX:+;${OMP_PREFIX}}"
  -DMPI_C_COMPILER="${CC}"
  -DMPI_CXX_COMPILER="${CXX}"
  -DHDF5_ROOT="${HDF5_PREFIX}"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# CMP0074: only needed to make *_ROOT variables effective on older policy settings.
if [[ "$CMP0074_MODE" == "auto" ]]; then
  CMAKE_ARGS+=(-DCMAKE_POLICY_DEFAULT_CMP0074=NEW)
elif [[ "$CMP0074_MODE" == "off" ]]; then
  :
else
  echo "Error: --cmp0074 must be 'auto' or 'off'" >&2
  exit 2
fi

# Compatibility lever for older CMakeLists that rely on HDF5_IS_PARALLEL
if [[ "$FORCE_PHDF5" -eq 1 ]]; then
  echo "PHDF5:     forcing HDF5_IS_PARALLEL=TRUE (compat)"
  CMAKE_ARGS+=(-DHDF5_IS_PARALLEL=TRUE)
fi

# OpenMP hints:
# - On macOS, AppleClang often needs explicit libomp include/lib paths.
# - On Linux/HPC toolchains, these hints can be unnecessary or wrong.
if [[ "$OPENMP_HINTS" -eq 1 && -n "$OMP_PREFIX" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    CMAKE_ARGS+=(
      -DOpenMP_CXX_LIB_NAMES=omp
      -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${OMP_PREFIX}/include"
      -DOpenMP_omp_LIBRARY="${OMP_PREFIX}/lib/libomp.dylib"
    )
  fi
fi

# PETSc
if [[ "$USE_PETSC" -eq 1 ]]; then
  PETSC_PREFIX="${PETSC_PREFIX:-}"
  if [[ -z "$PETSC_PREFIX" ]]; then
    if PETSC_PREFIX="$(brew_prefix petsc)"; then
      :
    else
      echo "Error: PETSc not found. Install via Homebrew or set PETSC_PREFIX." >&2
      exit 2
    fi
  fi
  echo "PETSc:     $PETSC_PREFIX"
  export PKG_CONFIG_PATH="${PETSC_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  CMAKE_ARGS+=(-DUSE_PETSC=ON)
fi

# Prepend generator args safely (Bash 3.x + nounset friendly)
if [[ ${#GENERATOR_ARGS[@]} -gt 0 ]]; then
  CMAKE_ARGS=("${GENERATOR_ARGS[@]}" "${CMAKE_ARGS[@]}")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --parallel
