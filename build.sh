#!/usr/bin/env bash
set -euo pipefail

# build.sh — Configure and build iPIC3D
#
# Usage:
#   ./build.sh [build_dir] [build_type] [options...]
#
#   build_dir   Output directory for build files (default: build)
#   build_type  CMake build type: Debug | Release | RelWithDebInfo | MinSizeRel
#               (default: Release)
#
# Examples:
#   ./build.sh                                          # build/ Release
#   ./build.sh build-debug Debug                        # custom dir and build type
#   ./build.sh build RelWithDebInfo --clean             # clean rebuild for profiling
#   ./build.sh --petsc                                  # enable PETSc solver
#   ./build.sh --hdf5-prefix /opt/hdf5                  # explicit HDF5 path
#
# Build types (standard CMake, controls compiler flags):
#   Release        -O3, no debug info         — production runs
#   Debug          -O0, full debug info (-g)  — debugging with gdb/lldb
#   RelWithDebInfo -O2, full debug info (-g)  — profiling
#   MinSizeRel     -Os, no debug info         — minimise binary size
#
# Common options:
#   --petsc                Enable PETSc solver (USE_PETSC=ON)
#   --clean                Delete build dir before configuring
#   --jobs N               Limit parallel build jobs (default: all cores; useful on
#                            shared HPC login nodes where using all cores is discouraged)
#   --hdf5-prefix PATH     Explicit HDF5 install prefix (skips auto-detection)
#   --petsc-prefix PATH    Explicit PETSc install prefix (skips auto-detection)
#   -h, --help             Show this help
#
# Advanced options (rarely needed):
#   --llvm                 (macOS) Use Homebrew LLVM clang instead of AppleClang;
#                            implies --no-openmp-hints (LLVM finds OpenMP natively).
#                            Auto-detects OpenMPI vs MPICH.
#   --ninja                Use Ninja generator instead of Make (requires ninja)
#   --omp-prefix PATH      (macOS) Explicit libomp prefix — AppleClang needs this
#                            to find OpenMP; on Linux, GCC/Clang find it automatically
#   --no-openmp-hints      (macOS) Do not pass OpenMP hint variables to CMake
#   --mpi-explicit         Resolve CC/CXX to absolute paths before passing to CMake;
#                            use if CMake's FindMPI cannot identify the MPI installation
#                            from the wrapper name alone (some HPC module environments)
#   --configure-only       Run CMake configure step only; do not build
#   --force-phdf5          Add -DHDF5_IS_PARALLEL=TRUE (for older CMakeLists.txt files)
#
# Environment overrides (alternative to --*-prefix flags):
#   HDF5_PREFIX, OMP_PREFIX, PETSC_PREFIX
#   CC, CXX   Compilers (default: mpicc/mpicxx)
#
# Dependency resolution order (HDF5, PETSc):
#   1. Explicit --*-prefix flag or environment variable
#   2. Homebrew (macOS)
#   3. pkg-config
#   4. CMake system path search (Linux/HPC with environment modules)
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
#   macOS (Homebrew, without LLVM installed):
#     cmake -S . -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
#       -DHDF5_ROOT=$(brew --prefix hdf5-mpi) \
#       -DOpenMP_CXX_LIB_NAMES=omp \
#       -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
#       -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
#     cmake --build build --parallel
# END_HELP

# -----------------------------------------------------------------
show_help() {
  sed -n '/^# build\.sh/,/^# END_HELP/{
    /^# END_HELP/q
    s/^# \{0,1\}//; p
  }' "$0"
}

# --------------------------- defaults ---------------------------
BUILD_DIR="build"
BUILD_TYPE="Release"
USE_NINJA=0
MPI_EXPLICIT=0
FORCE_PHDF5=0
DO_CLEAN=0
HDF5_PREFIX_FLAG=""
OMP_PREFIX_FLAG=""
PETSC_PREFIX_FLAG=""
OPENMP_HINTS=1
USE_LLVM=0
USE_PETSC=0
JOBS=""
CONFIGURE_ONLY=0

# --------------------------- parse args -------------------------
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --petsc)           USE_PETSC=1; shift ;;
    --llvm)            USE_LLVM=1; shift ;;
    --configure-only)  CONFIGURE_ONLY=1; shift ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "Error: --jobs requires a number" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "Error: --jobs requires a positive integer" >&2; exit 2; }
      JOBS="$2"; shift 2 ;;
    --clean)           DO_CLEAN=1; shift ;;
    --ninja)           USE_NINJA=1; shift ;;
    --mpi-explicit)    MPI_EXPLICIT=1; shift ;;
    --force-phdf5)     FORCE_PHDF5=1; shift ;;
    --no-openmp-hints) OPENMP_HINTS=0; shift ;;
    --hdf5-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --hdf5-prefix requires a path" >&2; exit 2; }
      HDF5_PREFIX_FLAG="$2"; shift 2 ;;
    --omp-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --omp-prefix requires a path" >&2; exit 2; }
      OMP_PREFIX_FLAG="$2"; shift 2 ;;
    --petsc-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --petsc-prefix requires a path" >&2; exit 2; }
      PETSC_PREFIX_FLAG="$2"; shift 2 ;;
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

case "$BUILD_TYPE" in
  Release|Debug|RelWithDebInfo|MinSizeRel) ;;
  *) echo "Warning: unrecognised build type '$BUILD_TYPE' (expected Release|Debug|RelWithDebInfo|MinSizeRel)" >&2 ;;
esac

# --------------------------- helpers ----------------------------
brew_prefix() {
  command -v brew >/dev/null 2>&1 || return 1
  brew --prefix "$1" 2>/dev/null
}

pkg_config_prefix() {
  command -v pkg-config >/dev/null 2>&1 || return 1
  pkg-config --variable=prefix "$1" 2>/dev/null
}

# Print a cmake command array in copy-pasteable format
print_cmake_cmd() {
  printf "  cmake"
  for arg in "$@"; do
    case "$arg" in
      -*)
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
}

OS="$(uname -s)"

# --------------------------- resolve HDF5 -----------------------
# Priority: flag > env > Homebrew > let CMake find it
HDF5_PREFIX="${HDF5_PREFIX_FLAG:-${HDF5_PREFIX:-}}"

if [[ -z "$HDF5_PREFIX" ]]; then
  # Try parallel formula first, fall back to serial — only one can be installed at a time
  HDF5_PREFIX="$(brew_prefix hdf5-mpi)" \
    || HDF5_PREFIX="$(brew_prefix hdf5)" \
    || HDF5_PREFIX=""
fi

# On Linux, HDF5 is often in system paths or loaded via modules.
# If we still don't have a prefix, let CMake's find_package handle it.
if [[ -n "$HDF5_PREFIX" ]]; then
  echo "HDF5:      $HDF5_PREFIX"
else
  echo "HDF5:      (auto — letting CMake find it)"
fi

# --------------------------- resolve OpenMP ---------------------
# AppleClang on macOS requires explicit libomp paths; GCC/Clang on Linux
# find OpenMP automatically via -fopenmp, so we only probe on macOS.
OMP_PREFIX="${OMP_PREFIX_FLAG:-${OMP_PREFIX:-}}"

if [[ "$OS" == "Darwin" && -z "$OMP_PREFIX" ]]; then
  OMP_PREFIX="$(brew_prefix libomp)" || OMP_PREFIX=""
fi

if [[ -n "$OMP_PREFIX" ]]; then
  echo "OpenMP:    $OMP_PREFIX"
else
  echo "OpenMP:    (auto)"
fi

# --------------------------- LLVM (macOS) -----------------------
if [[ "$USE_LLVM" -eq 1 ]]; then
  [[ "$OS" == "Darwin" ]] || { echo "Error: --llvm is macOS-only (uses Homebrew LLVM)" >&2; exit 1; }
  LLVM_PREFIX="$(brew_prefix llvm)" \
    || { echo "Error: LLVM not found. Install with: brew install llvm" >&2; exit 1; }
  # ompi_info ships with OpenMPI but not MPICH/MVAPICH2 — reliable detector
  if command -v ompi_info >/dev/null 2>&1; then
    export OMPI_CC="${LLVM_PREFIX}/bin/clang"
    export OMPI_CXX="${LLVM_PREFIX}/bin/clang++"
    echo "LLVM:      ${LLVM_PREFIX} (OpenMPI)"
  else
    export MPICH_CC="${LLVM_PREFIX}/bin/clang"
    export MPICH_CXX="${LLVM_PREFIX}/bin/clang++"
    echo "LLVM:      ${LLVM_PREFIX} (MPICH)"
  fi
  OPENMP_HINTS=0  # LLVM clang finds OpenMP natively; no -Xpreprocessor workaround needed
fi

# --------------------------- resolve PETSc ----------------------
PETSC_PREFIX="${PETSC_PREFIX_FLAG:-${PETSC_PREFIX:-}}"

if [[ "$USE_PETSC" -eq 1 ]]; then
  grep -q "option(USE_PETSC" CMakeLists.txt 2>/dev/null \
    || { echo "Error: --petsc requested but USE_PETSC is not defined in CMakeLists.txt" >&2
         echo "       PETSc support may not be available in this version of the code." >&2
         exit 1; }
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

# --------------------------- compilers --------------------------
CC="${CC:-mpicc}"
CXX="${CXX:-mpicxx}"
if [[ "$MPI_EXPLICIT" -eq 1 ]]; then
  CC="$(command -v "$CC" || true)"
  CXX="$(command -v "$CXX" || true)"
  [[ -n "$CC" && -n "$CXX" ]] || { echo "Error: compilers not found on PATH" >&2; exit 1; }
fi

# --------------------------- summary ----------------------------
echo "Build dir: $BUILD_DIR"
echo "Type:      $BUILD_TYPE"
if [[ "$USE_NINJA" -eq 1 ]]; then
  echo "Generator: Ninja"
else
  echo "Generator: default (Make)"
fi
echo "CC:        $CC"
echo "CXX:       $CXX"

# --------------------------- clean ------------------------------
if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "Clean:     removing '$BUILD_DIR'"
  rm -rf -- "$BUILD_DIR"
fi
mkdir -p -- "$BUILD_DIR"

# --------------------------- CMake args -------------------------
CMAKE_ARGS=()

if [[ "$USE_NINJA" -eq 1 ]]; then
  command -v ninja >/dev/null 2>&1 || { echo "Error: --ninja requested but 'ninja' not found" >&2; exit 1; }
  CMAKE_ARGS+=(-G Ninja)
fi

CMAKE_ARGS+=(
  -S . -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_C_COMPILER="${CC}"     # explicit MPI wrappers; often auto-detected but avoids
  -DCMAKE_CXX_COMPILER="${CXX}"  # ambiguity when multiple compilers are on PATH
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

# --------------------------- build ------------------------------
echo ""
echo "Configure:"
print_cmake_cmd "${CMAKE_ARGS[@]}"
echo ""
cmake "${CMAKE_ARGS[@]}"

if [[ "$CONFIGURE_ONLY" -eq 1 ]]; then
  echo ""
  echo "Configure-only: skipping build step."
  exit 0
fi

BUILD_PARALLEL=(--parallel)
[[ -n "$JOBS" ]] && BUILD_PARALLEL+=("$JOBS")

echo ""
echo "Build:"
echo "  cmake --build $BUILD_DIR ${BUILD_PARALLEL[*]}"
echo ""
cmake --build "$BUILD_DIR" "${BUILD_PARALLEL[@]}"
