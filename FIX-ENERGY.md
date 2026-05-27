# Energy-conservation fix for the ECSIM field solver (CIC)

## TL;DR

The semi-implicit ECSIM scheme conserves total energy to machine precision, but
on `main` that property silently breaks when the code is run **multi-threaded**.
The cause is a data race in the OpenMP moment-gather: the per-particle deposit
helpers do an unguarded `+=` into shared grid nodes. The fix is a one-concept
change — add `#pragma omp atomic update` to those helpers. It is confined to a
single header, `include/EMfields3D.h`.

```
include/EMfields3D.h | add_Rho, add_Jxh, add_Jyh, add_Jzh, add_Mass
```

Single-threaded runs are unaffected (already correct); the fix has no effect on
results except to make the multi-threaded gather deterministic-enough to conserve
energy.

## The bug

`Particles3D::computeMoments` deposits charge/current/mass-matrix moments from
particles to grid nodes inside `#pragma omp parallel for` over particles.
Neighbouring particles owned by different threads scatter into the same grid
nodes via the inline helpers in `include/EMfields3D.h`:

```cpp
inline void EMfields3D::add_Rho(double weight[8], int X, int Y, int Z, int is) {
    for (int i...) for (int j...) for (int k...)
        rhons[is][X-i][Y-j][Z-k] += weight[...] * invVOL;   // unguarded RMW
}
```

The read-modify-write `+=` races: concurrent updates to the same node lose
contributions. The deposited moments are then slightly wrong in a
thread-schedule-dependent way, and the resulting source term no longer matches
the discrete operator the ECSIM energy theorem relies on. Energy drifts.

## The fix

Guard each accumulation with `#pragma omp atomic update` in `add_Rho`,
`add_Jxh`, `add_Jyh`, `add_Jzh`, and `add_Mass`. No interface or call-site
changes; no new build options or input parameters.

## Evidence (why this is the whole fix for CIC energy)

Measured on the Double_Harris current sheet (`inputfiles/ci_energy_*.inp`),
final-cycle `|dE/E0|`:

| Config                                   | unmodified `main` | with the atomic fix |
|------------------------------------------|-------------------|---------------------|
| np=1, OMP=1, 32² / 128² / 256×128        | ~1e-14 (already ε)| ~1e-14              |
| np=4, OMP=1                              | 3.8e-16 (already ε)| 3.8e-16            |
| np=1, all cores (8 threads), 32²          | **4.9e-6**        | 6.5e-15             |
| np=4, all cores (8 threads), 32²          | **5.6e-5**        | 5.1e-16             |

So `main` already conserves energy to machine precision single-threaded; the
race is the only mechanism that breaks it, and the atomic guard restores ε at any
thread count. A bisect over candidate field-operator changes (a self-adjoint
`lap_graddiv` Maxwell operator, a periodic node-halo off-by-one correction in
`Com3DNonblk.cpp`, and a Krylov periodic-duplicate-DOF projection) found **none**
of them move `|dE/E0|` for CIC energy conservation — they were originally
measured against race-contaminated baselines. They remain relevant for discrete-
operator self-adjointness, cross-code bit-matching, and higher-order (TSC) shape
functions, but are out of scope for this minimal branch.

## Build & test

```bash
# Build (standard iPIC3D CMake; no PETSc/extra deps required)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Verify energy conservation at np=1 and np=4 (runs multi-threaded to exercise the race)
OMP_NUM_THREADS=4 IPIC3D_EXE=build/iPIC3D tests/test_energy.sh
```

`tests/test_energy.sh` runs `inputfiles/ci_energy_gem.inp` (np=1) and
`inputfiles/ci_energy_gem_np4.inp` (np=4) for 30 cycles and checks the final
`|dE/E0|` from `ConservedQuantities.txt` against a 1e-12 tolerance. On `main`
(without the fix) the multi-threaded cases fail at ~1e-5; with the fix they pass
at machine precision.
