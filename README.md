# iPIC3D

iPIC3D is an implicit particle-in-cell (PIC) code for 3D plasma simulations. This version includes the energy-conserving semi-implicit method (ECSIM).

**This branch (`with-petsc`)** adds optional PETSc support as an alternative field solver to the built-in GMRES, selectable at runtime via the `-solver PETSc` flag.

## Requirements

- C++17 compiler (GCC, Clang, or AppleClang)
- CMake (>= 3.13)
- MPI (OpenMPI or MPICH)
- HDF5 (enabled by default; parallel HDF5 is auto-detected if available)
- pkg-config (required for PETSc discovery)
- PETSc (optional, for the PETSc solver)
- OpenMP (optional, for thread-level parallelism)
- VTK / ParaView Catalyst (optional, for in-situ visualization)

## Installation

1. Clone the repository and checkout the `with-petsc` branch:
```shell
git clone https://github.com/rusaitis/iPIC3D-CPU-SPACE-CoE.git
cd iPIC3D-CPU-SPACE-CoE
git checkout with-petsc
```

2. Build with the provided script:
```shell
./build.sh                # default build (GMRES only)
./build.sh --petsc        # build with PETSc support
./build.sh --clean --petsc  # clean rebuild with PETSc
```

For all available options and manual CMake instructions:
```shell
./build.sh --help
```

### Quick start with pixi

[Pixi](https://pixi.sh) manages all C++ and Python dependencies automatically via conda-forge — no manual setup needed:

```shell
pixi run build                     # configure + build (Release, GMRES only)
pixi run build-petsc               # build with PETSc support
pixi run test-smoke                # run a short GEM2D simulation
pixi run test-petsc                # compare GMRES vs PETSc (script defaults)
```

Tasks that exist in only one environment (`build-petsc`, `test-petsc`) are resolved automatically by pixi — no `-e petsc` flag needed.

**Passing custom arguments** — extra arguments after `--` are forwarded to the underlying command:

```shell
pixi run test-petsc -- --np 8 --cycles 20
pixi run test-petsc -- --np 4 --cycles 50 --grid 200 200
pixi run test-petsc -- --grid-min 50 --grid-max 400   # scaling mode
```

**Running arbitrary commands** in the pixi environment (still needs `-e`):

```shell
pixi run -e petsc -- mpirun -np 4 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc
```

**Tip:** To avoid typing `-e petsc` for arbitrary commands, add a shell alias:

```shell
alias pxp='pixi run -e petsc'      # in ~/.zshrc
pxp -- mpirun -np 4 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc
```

Available environments: `default` (build + Python), `petsc` (+ PETSc), `build-only` (no Python), `py` (Python only).

Use `pixi shell -e petsc` to drop into an interactive shell with all dependencies available (useful for Debug builds or custom CMake flags).

3. Run a simulation:
```shell
# np = XLEN x YLEN x ZLEN (set in the input file)
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris.inp

# With PETSc solver (requires --petsc build):
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc

# PETSc options can be passed directly on the command line:
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc \
  -ksp_type bcgs -ksp_max_it 1000 -ksp_monitor
```

PETSc runtime flags are passed through to PETSc automatically. Some useful ones:

| Flag | Description |
|------|-------------|
| `-ksp_type <type>` | Krylov method: `gmres` (default), `fgmres`, `bcgs`, `tfqmr` |
| `-ksp_gmres_restart <n>` | GMRES restart parameter (default: 30) |
| `-ksp_max_it <n>` | Maximum iterations |
| `-ksp_rtol <tol>` | Relative tolerance |
| `-ksp_monitor` | Print residual norm at each iteration |
| `-ksp_view` | Print solver configuration after solve |

## Testing the PETSc solver

Two test scripts compare the built-in GMRES and PETSc solvers (requires a `--petsc` build):

**Multi-solver comparison** — runs GMRES and several PETSc Krylov methods, reports convergence and timing:
```shell
./tests/test_petsc.sh                       # defaults: 8 procs, 10 cycles
./tests/test_petsc.sh --np 4 --cycles 20    # custom
./tests/test_petsc.sh --np 8 --cycles 50 --grid 200 200 --topo 4 2
```

**Scaling study** — use `--grid-min` / `--grid-max` to sweep grid sizes:
```shell
./tests/test_petsc.sh --grid-min 50 --grid-max 400              # defaults: 8 procs, 20 cycles
./tests/test_petsc.sh --grid-min 50 --grid-max 400 --cycles 10  # faster run
```

Results are saved to `tests/test_petsc_output/` (CSV, HDF5 fields, plots).

### Test recipes

Three common test scenarios:

**1. Simple GMRES vs PETSc comparison** — single grid, two solvers:
```shell
./tests/test_petsc.sh --grid 100 100 --cycles 100 --solvers GMRES,PETSc_gmres
```

**2. Multi-solver grid sweep** — all PETSc Krylov methods across grid sizes:
```shell
./tests/test_petsc.sh --all-solvers --grid-min 50 --grid-max 150 --grid-step 50 --cycles 100
```

**3. Extended run with field output** — longer simulation for convergence analysis:
```shell
./tests/test_petsc.sh --grid 150 150 --cycles 1000 --solvers GMRES,PETSc_bcgs \
  --field-output 10 --name long-run
```

`DiagnosticsOutputCycle` is hardcoded to 1 — `ConservedQuantities.txt` is always written every cycle (cheap). Use `--field-output N` to control the expensive HDF5 field snapshots.

### Plotting results

After a test run, generate plots from the output directory:
```shell
pixi run plot-timing                 # solver timing comparison
pixi run plot-fields                 # 2D field heatmaps with difference panels
pixi run plot-energy                 # energy conservation
pixi run plot-l2                     # L2 error accumulation over cycles
```

All plotting commands accept `--light` for a light theme (default is dark). Pass extra arguments after `--`:
```shell
pixi run plot-timing -- --sort --loglog
pixi run plot-fields -- --fields Bx,By,Ez --all-cycles
pixi run plot-energy -- --output custom_energy.png
```

Or run the scripts directly:
```shell
python3 tests/plot_timing.py tests/test_petsc_output/timing_results.csv
python3 tests/plot_energy.py tests/test_petsc_output/timing_results.csv --light
```

## Python postprocessing

Postprocessing scripts in `postprocessing_tools/python/` require numpy, matplotlib, h5py, and mpi4py.

**With pixi** (recommended): dependencies are already included in the default environment.

**Without pixi** (e.g., on HPC with system Python):
```shell
pip install -e .                   # core deps (numpy, matplotlib, h5py, mpi4py)
pip install -e ".[vtk]"            # + scipy, tables, pyevtk for VTK export
```

## OpenMP

Hybrid OpenMP + MPI parallelism is under development for ECSIM/RelSIM. To disable OpenMP threading:

```shell
export OMP_NUM_THREADS=1
```

# Acknowledgements and Citations
This version of iPIC3D (with the implicit moment method) has been developed by Prof Stefano Markidis and his team. The energy conserving semi-implicit method (ECSIM) and relativistic semi-implicit method (RelSIM) have been implemented by Dr Pranab J Deka and Prof Fabio Bacchini.

If you use this iPIC3D code, please cite:

1. The link to version of iPIC3D: https://github.com/Pranab-JD/iPIC3D-CPU-SPACE-CoE

2. Stefano Markidis, Giovanni Lapenta, and Rizwan-uddin (2010), *Multi-scale simulations of plasma with iPIC3D*, Mathematics and Computers in Simulation, 80, 7, 1509-1519 [[DOI]](https://doi.org/10.1016/j.matcom.2009.08.038)

3. Giovanni Lapenta (2017), *Exactly energy conserving semi-implicit particle in cell formulation*, Journal of Computational Physics, 334, 349
[[DOI]](http://dx.doi.org/10.1016/j.jcp.2017.01.002)

4. Fabio Bacchini (2023), *RelSIM: A Relativistic Semi-implicit Method for Particle-in-cell Simulations*, The Astrophysical Journal Supplement Series, 268:60 [[DOI]](https://doi.org/10.3847/1538-4365/acefba)
