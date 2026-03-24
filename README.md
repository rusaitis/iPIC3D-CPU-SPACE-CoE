# iPIC3D

iPIC3D is an implicit particle-in-cell (PIC) code for 3D plasma simulations. This version includes the energy-conserving semi-implicit method (ECSIM).

**This branch (`with-petsc-matrix`)** extends the PETSc field solver with an explicit preconditioner matrix. Starting from a block-diagonal approximation, the preconditioner evolved through a full 27-point stencil incorporating curl-curl coupling and near-null space information, followed by benchmarking against the unpreconditioned solver. Code review confirmed MPI/HPC correctness; systematic testing across preconditioner types (Jacobi, SOR, ASM, AMG) is the next step.

## Quickstart (pixi)

```shell
git clone https://github.com/rusaitis/iPIC3D-CPU-SPACE-CoE.git && cd iPIC3D-CPU-SPACE-CoE
git checkout with-petsc
pixi run build          # downloads all deps (incl. PETSc) + builds (~2 min first time)
pixi run test-smoke     # 3-cycle sanity check
pixi run test           # full GMRES vs PETSc comparison
```

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

### 1. Clone the repository

```shell
git clone https://github.com/rusaitis/iPIC3D-CPU-SPACE-CoE.git
cd iPIC3D-CPU-SPACE-CoE
git checkout with-petsc
```

### 2. Build

#### Option A: pixi (recommended — manages all dependencies)

[Pixi](https://pixi.sh) automatically downloads and manages all dependencies — including PETSc — on first run. No manual setup needed.

```shell
pixi run build                     # configure + build (with PETSc)
pixi run test-smoke                # run a short GEM2D simulation
pixi run test                      # compare GMRES vs PETSc
```

**Passing custom arguments** — extra arguments after `--` are forwarded to the underlying command:

```shell
pixi run test -- --np 8 --cycles 20
pixi run test -- --np 4 --cycles 50 --grid 200 200
pixi run test -- --grid-min 50 --grid-max 400   # scaling mode
```

**Running arbitrary commands** in the pixi environment:

```shell
pixi run -- mpirun -np 4 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc
```

Available environments: `default` (build with PETSc + Python), `build-only` (no Python), `py` (Python only).

Use `pixi shell` to drop into an interactive shell with all dependencies available (useful for Debug builds or custom CMake flags).

#### Option B: build.sh (HPC clusters / system toolchains only)

> Prefer `pixi run build` for local development. Use `build.sh` only on HPC
> systems where pixi is unavailable or you need system-provided MPI/compilers.
> Note: `build.sh` does not include PETSc by default — use `--petsc` if needed.

```shell
./build.sh                # default build (GMRES only)
./build.sh --petsc        # build with PETSc support
./build.sh --clean --petsc  # clean rebuild with PETSc
```

For all available options and manual CMake instructions:
```shell
./build.sh --help
```

### 3. Run a simulation

The `pixi run sim` command builds (if needed), runs, and logs automatically:

```shell
pixi run sim inputfiles/Double_Harris.inp                    # build + run (np auto-detected)
pixi run sim inputfiles/Double_Harris.inp --solver PETSc     # PETSc solver
pixi run sim inputfiles/Double_Harris.inp -np 4 --cycles 50  # override procs and cycles
pixi run sim inputfiles/Double_Harris.inp -o output/my_run   # custom output dir
pixi run sim inputfiles/Double_Harris.inp --solver PETSc --diag  # PETSc diagnostics mode
pixi run sim inputfiles/Double_Harris.inp --dry-run          # show command without running
```

**Restart** — pass an output directory instead of an input file:
```shell
pixi run sim output/data_reconnection                        # restart from checkpoint
pixi run sim output/data_reconnection --cycles 500           # extend to 500 cycles
pixi run sim output/data_reconnection -o output/petsc_run --solver PETSc  # restart with different solver
```

Output is logged to `<output_dir>/run.log` by default (use `--no-log` to disable). After the run, a per-cycle timing profile (`profile_run.csv`) is auto-extracted from the log.

**Direct mpirun** (without sim.sh):
```shell
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris.inp
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
| `-ksp_view_eigenvalues` | Print Ritz eigenvalues after each solve |
| `-ksp_view_singularvalues` | Print extreme singular values and condition number |
| `-pc_type <type>` | Preconditioner: `none` (default), `gamg`, `hypre`, `ilu`, `jacobi` |

### Preconditioner diagnostics

The `--diag` flag enables all PETSc diagnostic output in one command:

```bash
# Run with full PETSc diagnostics (PrecMatrix, KSP monitor, eigenvalues, auto-logged)
pixi run sim inputfiles/Double_Harris.inp --solver PETSc --diag

# Plot KSP diagnostics (iterations, convergence, Ritz eigenvalues, condition number)
pixi run plot-ksp -- output/data_reconnection/run.log

# Visualize the preconditioner matrix P (sparsity, block norms, diagonal structure)
pixi run plot-prec -- output/data_reconnection/Double_Harris_P_cycle0.bin
```

Input file parameters for preconditioner matrix:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PrecMatrix` | `false` | Assemble explicit 27-point stencil preconditioner matrix |
| `PrecDiagnostics` | `false` | Print matrix norms on cycle 0 and dump P to binary file |

Both scripts support `--light` for light theme and `-o FILE` for explicit output path.

## Testing

The test script auto-detects PETSc from the build and runs the appropriate solvers.

**Multi-solver comparison** — runs GMRES and PETSc Krylov methods (when available), reports convergence and timing:
```shell
pixi run test                                                    # defaults: 8 procs, 10 cycles
pixi run test -- --np 4 --cycles 20                              # custom
pixi run test -- --np 8 --cycles 50 --grid 200 200 --topo 4 2

# or manually:
./tests/test.sh                       # defaults: 8 procs, 10 cycles
./tests/test.sh --np 4 --cycles 20    # custom
```

**Scaling study** — use `--grid-min` / `--grid-max` to sweep grid sizes:
```shell
pixi run test -- --grid-min 50 --grid-max 400              # defaults: 8 procs, 20 cycles
pixi run test -- --grid-min 50 --grid-max 400 --cycles 10  # faster run
```

Results are saved to `tests/test_output/` (CSV, HDF5 fields, plots).

### Test recipes

Three common test scenarios:

**1. Simple GMRES vs PETSc comparison** — single grid, two solvers:
```shell
pixi run test -- --grid 100 100 --cycles 100 --solvers GMRES,PETSc_gmres
```

**2. Multi-solver grid sweep** — all PETSc Krylov methods across grid sizes, 3 averages:
```shell
pixi run test -- --all-solvers --grid-min 50 --grid-max 150 --grid-step 50 \
  --cycles 100 --field-output 10 --avg 3
```

**3. Extended run with field output** — longer simulation for convergence analysis:
```shell
pixi run test -- --grid 150 150 --cycles 1000 --solvers GMRES,PETSc_bcgs \
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
pixi run movie                       # stitch comparison into mp4 (requires ffmpeg)
```

All plotting commands accept `--light` for a light theme (default is dark). Pass extra arguments after `--`:
```shell
pixi run plot-timing -- --sort --loglog
pixi run plot-fields -- --fields Bx,By,Ez --all-cycles
pixi run plot-energy -- --output custom_energy.png
```

Or run the scripts directly:
```shell
python3 scripts/plot_timing.py tests/test_output/results.csv
python3 scripts/plot_energy.py tests/test_output/results.csv --light
```

### Validation and run summary

After a simulation or test run, check results programmatically:

```shell
pixi run validate -- tests/test_output/              # pass/fail: energy, convergence, L2
pixi run validate -- output/data_reconnection/       # works on sim output too
pixi run summarize -- output/data_reconnection/      # CLI summary: timing, energy, convergence
```

### Movie generation

Generate mp4 movies of field evolution across cycles. Requires `ffmpeg` and test data with field output (`--field-output N`).

```shell
# Via test.sh (generates movie at the end):
pixi run test -- --cycles 50 --field-output 5 --movie

# Standalone (after test data exists):
pixi run movie
pixi run movie -- --fields Bx,By,Ez --fps 10 --light

# Explicit directories:
bash tests/make_movie.sh --ref tests/test_output/GMRES_20x20x1 \
    --test tests/test_output/PETSc_gmres_20x20x1
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
