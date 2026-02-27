# PETSc Integration Notes

## Overview

PETSc is an optional alternative to the built-in GMRES field solver. It uses the same
matrix-free `MaxwellImage()` callback via PETSc's `MatShell`, so the physics is identical —
only the Krylov solver implementation differs.

**Files:**
- `solvers/PETSC.cpp` + `include/PETSC.h` — PETSc solver wrapper
- `fields/EMfields3D.cpp` — solver dispatch (`calculateE()`)
- `main/iPIC3Dlib.cpp` — PetscSolver creation, SolverType validation
- `include/Collective.h` / `main/Collective.cpp` — SolverType parameter + `-solver` flag

**Build:** `./build.sh --petsc` or `cmake -DUSE_PETSC=ON`
**Run:** `mpirun -np 8 build/iPIC3D inputfiles/Double_Harris.inp -solver PETSc`

## Design Decisions

### Matrix-free approach (MatShell)
The implicit Maxwell system is never assembled as a matrix. Both the built-in GMRES and
PETSc use `EMfields3D::MaxwellImage(output, input)` as the A*x product. PETSc wraps this
in a `MatShell` with the `PetscMaxwellMatMult` callback. A `const_cast` is needed because
`MaxwellImage`'s legacy signature takes `double*` (not `const double*`), but it only reads
the input.

### Zero-copy Vec reuse
PETSc Vec wrappers (`petsc_x_`, `petsc_b_`) are created once in the constructor with no
underlying array (`nullptr`). Each `solve()` call uses `VecPlaceArray` / `VecResetArray`
to temporarily point them at the caller's arrays. This avoids per-solve `VecCreate` /
`VecDestroy` overhead.

### PetscInit / MPI ordering
`PetscInitialize` is called before `MPIdata::init()`, and `PetscFinalize` before
`MPI_Finalize`. `MPIdata.cpp` guards `MPI_Init` / `MPI_Finalize` against double-calls
since PETSc already initializes MPI internally.

### Error handling
- In the callback (returns `PetscErrorCode`): `PetscCall()` propagates errors to PETSc's
  error handler.
- In constructor / destructor / solve (return `void`): `PetscCallAbort()` prints a full
  PETSc error traceback and aborts MPI. Previously, all error codes were silently discarded.
- Requires PETSc >= 3.17 (2022) for `PetscCall` / `PetscCallAbort` / `PETSC_SUCCESS`.

### Initial guess
`KSPSetInitialGuessNonzero(PETSC_TRUE)` — the solution vector from the previous cycle is
reused as the starting guess, same as the built-in GMRES.

## Convergence: Built-in GMRES vs PETSc

Both solvers receive the same `GMREStol` from the input file, but interpret it slightly
differently:

**Built-in GMRES** (`GMRES.cpp`):
- `rho_tol = initial_error * tol` — relative to the residual at **each restart**
- `initial_error` is recomputed at the start of every restart cycle (line 90 is inside the
  outer loop), so the convergence bar resets with each restart

**PETSc KSP**:
- `rtol` — converges when `||r_k|| / ||r_0|| < rtol`, where `r_0` is the residual at the
  start of `KSPSolve` (not per-restart)

**In practice**: With `GMREStol = 1E-15` (ECSIM energy conservation), both drive the
residual to machine precision and the difference is negligible. For looser tolerances,
PETSc could converge slightly differently because it doesn't reset the reference residual
at each restart?

## Solver Configuration

GMRES restart is hardcoded at 20 in both solvers (not configurable from input file):
- Built-in: literal `20` passed at the call site in `EMfields3D.cpp:2616`
- PETSc: `KSPGMRESSetRestart(ksp_, 20)` in the constructor

`KSPSetFromOptions()` allows runtime override of PETSc settings without recompiling:
```
mpirun -np 8 build/iPIC3D input.inp -solver PETSc -ksp_type bcgs
mpirun -np 8 build/iPIC3D input.inp -solver PETSc -ksp_gmres_restart 40 -ksp_monitor
```

## Performance Notes

Tested on Double_Harris (200x200, 50 cycles, 8 MPI ranks, Apple M2 Pro):
- PETSc GMRES: ~1.3x faster than built-in GMRES
- PETSc BiCGStab (`-ksp_type bcgs`): ~2.7x faster (fewer iterations, but 2 mat-vec products each)
- Scaling advantage increases with grid size

## Future Improvements

- Make GMRES restart configurable from input file (e.g. `GMRESRestart = 20`)
- Add `const` to `MaxwellImage`'s input parameter to eliminate `const_cast`
- Consider preconditioners
