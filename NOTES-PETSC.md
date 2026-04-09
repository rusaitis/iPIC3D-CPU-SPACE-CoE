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

GMRES restart was originally hardcoded at 20 in both solvers; **PETSc was bumped to 40 on 2026-04-09** after empirical sweeps showed it consistently faster across the stiffness corpus (see `plan-preconditioners.md` §Phase 9b, ~14-20% speedup at dt≥0.75 with no penalty up to dt=1.0). Built-in GMRES stayed at 20 because it's slower per matvec — bumping its restart didn't pay off as cleanly.
- Built-in: literal `20` at `EMfields3D.cpp:2643`
- PETSc: `KSPGMRESSetRestart(ksp_, 40)` at `solvers/PETSC.cpp:188`

`KSPSetFromOptions()` allows runtime override of PETSc settings without recompiling:
```
mpirun -np 8 build/iPIC3D input.inp -solver PETSc -ksp_type bcgs
mpirun -np 8 build/iPIC3D input.inp -solver PETSc -ksp_gmres_restart 50 -ksp_monitor
```

### Recommended PETSc preconditioner for ECSIM Maxwell

**None.** Plan v2 of the preconditioner study (`plan-preconditioners.md`) ran a comprehensive sweep across 5 stiffness regimes × 8-10 preconditioner variants and found zero PCs that beat unpreconditioned GMRES in wall-clock time. The explicit `P` matrix (`assembleP()`, `solvers/PETSC.cpp:436`) omits the smoothing operator `S` from the implicit Maxwell operator, which makes `P` a poor enough approximation of `A` that any PC built from it scatters the spectrum instead of clustering it.

The single piece of HYPRE BoomerAMG that was *not* tested (because Homebrew's PETSc is built without `--with-hypre`) remains the only theoretically promising approach. To enable it, install HYPRE and rebuild PETSc with `--with-hypre-dir=$(brew --prefix hypre)`.

## Performance Notes

2. PETSc GMRES and FGMRES are the clear winners, ~10–16% faster

  ┌────────────────┬───────────────────────────────────┐
  │     Solver     │ Typical speedup vs built-in GMRES │
  ├────────────────┼───────────────────────────────────┤
  │ PETSc (GMRES)  │ ~11%                              │
  ├────────────────┼───────────────────────────────────┤
  │ PETSc_fgmres   │ ~12%                              │
  ├────────────────┼───────────────────────────────────┤
  │ PETSc_bcgs     │ ~5%                               │
  ├────────────────┼───────────────────────────────────┤
  │ PETSc_tfqmr    │ ~3%                               │
  ├────────────────┼───────────────────────────────────┤
  │ Built-in GMRES │ baseline                          │
  └────────────────┴───────────────────────────────────┘

PETSc GMRES and FGMRES perform almost identically (without a preconditioner, flexible GMRES degenerates to standard GMRES). The speedup over the built-in GMRES is likely from PETSc's optimized BLAS dot-product and norm operations compared to iPIC3D's hand-rolled MPI_Allreduce per Krylov iteration?

PETSc GMRES gives a real but modest ~10-15% speedup. Next step: preconditioning? (e.g., multigrid or ILU via PETSc -pc_type).

## Block-Diagonal Preconditioner Matrix

### What it is

An optional explicit matrix `P` passed to PETSc as `KSPSetOperators(ksp, A_shell, P)`.
The operator `A` stays matrix-free (MatShell); PETSc builds preconditioners from `P`.

### What's in P

Each interior node (i,j,k) contributes a 3x3 diagonal block:

```
P[i][j][k] = I_3x3
  + (c·th·dt)² · diag(+0.5/dy²+0.5/dz², +0.5/dx²+0.5/dz², +0.5/dx²+0.5/dy²)
  + dt·th·4π·invVOL · M_center_3x3[i][j][k]
```

- **Identity** — from the `tempX[i][j][k]` self-term in MaxwellImage
- **Curl-curl diagonal** — self-coupling from composed curlN2C→curlC2N (positive, +0.5/h²)
- **Mass matrix center block** — the g=0 stencil entry (3x3, varies per node and per cycle)

**Omitted:** off-diagonal stencil entries (neighbors), energy-conserving smoothing filter.

### How to enable

In the input file:
```
SolverType = PETSc
PrecMatrix = true
```

Default is `PrecMatrix = false` (current behavior, PCNONE).

### How to test different preconditioners

```bash
# Default (PETSc chooses based on P matrix type)
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris_PETSc_prec.inp -ksp_monitor

# Jacobi (diagonal of P)
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris_PETSc_prec.inp -pc_type jacobi -ksp_monitor

# Block Jacobi with ILU on each block
mpirun -np 8 build/iPIC3D inputfiles/Double_Harris_PETSc_prec.inp -pc_type bjacobi -sub_pc_type ilu -ksp_monitor
```

### Code paths

- `PrecMatrix = false`: `KSPSetOperators(ksp, A, A)` — MatShell only, PCNONE (existing behavior)
- `PrecMatrix = true`: `KSPSetOperators(ksp, A, P)` — MatShell operator + explicit P for preconditioner.
  **Default PC is PCNONE** (set programmatically) because P omits the smoothing filter
  applied 4× in `MaxwellImage`, making ILU(P) a poor preconditioner that increases iterations.
  Users opt in to AMG via runtime `-pc_type gamg` flags (`KSPSetFromOptions` overrides the default).
- `assembleP()` is called in the constructor (unconditionally, since `-pc_type` isn't known yet)
  and again before each `KSPSolve` — but **only when the active PC is not PCNONE** (checked via
  `PetscObjectTypeCompare`), avoiding the expensive 27-block assembly when it won't be used.
- P is MATMPIAIJ with block size 3, preallocated for up to 81 nonzeros per row (27 blocks × 3 DOFs)

## Future Improvements

- Make GMRES restart configurable from input file (e.g. `GMRESRestart = 20`)
- Add `const` to `MaxwellImage`'s input parameter to eliminate `const_cast`
- Full sparse matrix (27-point stencil + full mass matrix) for GAMG/HYPRE preconditioners
