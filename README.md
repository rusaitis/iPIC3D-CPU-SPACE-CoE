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

**Scaling study** — sweeps grid sizes from 50x50 to 400x400, generates a CSV and plot:
```shell
./tests/test_petsc_scaling.sh               # defaults: 8 procs, 20 cycles
./tests/test_petsc_scaling.sh --cycles 10   # faster run
```

Results are saved to `tests/scaling_results.csv` and `tests/scaling_plot.png`.

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
