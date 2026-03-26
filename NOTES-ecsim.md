# ECSIM: Energy Conserving Semi-Implicit Method in iPIC3D

Reference document for iPIC3D's default field-particle coupling algorithm.

## 1. Overview

ECSIM is the **sole** field-particle coupling method in iPIC3D — it is hardcoded,
not toggle-selectable. The only runtime choice is the **linear solver backend**
(GMRES vs PETSc) used to invert the implicit field equation that ECSIM produces.

A relativistic variant, **RelSIM** (Relativistic Semi-Implicit Method), shares
the same algorithmic structure but accounts for the Lorentz factor γ in the
rotation matrix and velocity push. It is selected with the `Relativistic`
flag in the input file.

### Key references

- **Lapenta (2017)** — *Exactly energy conserving semi-implicit particle in cell
  formulation*, J. Comput. Phys. 334, 349–366. The foundational ECSIM paper.
- **Markidis & Lapenta (2011)** — *Multi-scale simulations of plasma with iPIC3D*,
  Math. Comput. Simul. 80(7), 1509–1519.
- **Bacchini et al. (2023)** — *A relativistic semi-implicit method for
  magnetized plasmas*, ApJ 268, 60. RelSIM formulation.

## 2. Why ECSIM conserves energy

Standard implicit PIC codes evaluate the current density at the old time level
J^n, which introduces an artificial dissipation term — the discrete energy
exchange between fields and particles does not cancel exactly, and the total
energy drifts.

ECSIM fixes this with two coupled modifications:

1. **Half-step implicit current J_h**: Instead of J^n, ECSIM gathers a current
   density that is evaluated at the half time step using the Boris-rotated
   velocity. This J_h is the natural "midpoint" current consistent with the
   particle trajectory.

2. **Mass matrix M**: The implicit field equation acquires an extra term
   `θΔt·4π·M·E` where M encodes the coupling between the electric field and
   the particle response. This mass matrix arises from linearizing the
   implicit current — it represents ∂J_h/∂E, the sensitivity of the current
   to the field being solved for.

Together, these two ingredients ensure that the discrete energy exchange
(field → particle via E·J) exactly cancels to **machine precision** when
θ = 0.5 (Crank-Nicolson). No predictor-corrector iteration is needed — a
single pass per cycle suffices (`Collective.h:416`: "mover predictor correction
iteration (not needed for ECSIM)").

The solver tolerance must be ≤ 1e-15 (`GMREStol`) for strict energy
conservation; looser tolerances introduce solver-residual-level energy drift.

## 3. The three ECSIM ingredients

### 3a. Mass matrix M

The mass matrix is a **9-component tensor** (Mxx, Mxy, Mxz, Myx, Myy, Myz,
Mzx, Mzy, Mzz) stored on each grid node with a **14-element stencil**
(`NE_MASS = 14`, defined at `EMfields3D.cpp:59`).

Storage (`EMfields3D.cpp:151–159`):
```cpp
Mxx (NE_MASS, nxn, nyn, nzn),
Myy (NE_MASS, nxn, nyn, nzn),
...
Mzy (NE_MASS, nxn, nyn, nzn),
```

The stencil is defined by the `NeighbouringNodes` class
(`include/Neighbouring_Nodes.h`):
- Index `g = 0`: center node (0, 0, 0)
- Indices `g = 1..3`: face neighbors (+1,0,0), (0,+1,0), (0,0,+1)
- Indices `g = 4..9`: edge neighbors (±1,±1,0), (±1,0,±1), (0,±1,±1)
- Indices `g = 10..12`: mixed-sign corner neighbors
- Index `g = 13`: (+1,+1,+1) corner

Each forward offset at `+NeNo(g)` implies a backward offset at `−NeNo(g)` via
symmetry, so 14 stored groups cover all **27 neighbors** in the 3×3×3 cube.

**Assembly** happens per-particle in `computeMoments()` (`Particles3D.cpp:1698–1735`):

1. Build the Boris **rotation matrix α** (`Particles3D.cpp:1642–1653`):
   ```
   Ω = (q·Δt)/(2mc) · B / γ
   denom = 1 / ((1 + Ω²) · γ)
   α[i][j] = (δ_ij + Ω_i·Ω_j ± ε_ijk·Ω_k) · denom
   ```

2. For each pair of cell corners (i, j, k) and (i2, j2, k2) that a particle
   straddles, compute:
   ```cpp
   qww = q * q_dt_2mc * weights[index1] * weights[index2];
   value[a][b] = alpha[b][a] * qww;   // note the transpose!
   ```

3. Accumulate into `EMf->add_Mass(value, ni, nj, nk, n_node)` for 8 cell
   corners × 14 neighbor nodes.

### 3b. Implicit current J_h

Gathered in the same `computeMoments()` loop (`Particles3D.cpp:1655–1691`):

```cpp
qau = q * (alpha[0][0]*(u_n + dt/2*Fxl) + alpha[0][1]*(v_n + ...) + ...)
qav = q * (alpha[1][0]*(u_n + ...) + ...)
qaw = q * (alpha[2][0]*(u_n + ...) + ...)
```

This is α·v^n — the velocity rotated by the Boris α-matrix into the half-step
frame. It replaces the standard J^n with the implicit midpoint current J_h.

The rotated charge-velocity products `(qau, qav, qaw)` are deposited onto the
8 cell corners via trilinear shape function weights:
```cpp
for (int ii = 0; ii < 8; ii++)
    temp[ii] = qau * weights[ii];
EMf->add_Jxh(temp, ix, iy, iz, ns);   // lines 1678–1691
```

### 3c. θ-decentering parameter

- Default: `th = 0.5` (`Collective.cpp:107`)
- θ = 0.5: **Crank-Nicolson** — 2nd-order temporal accuracy + exact energy conservation
- 0.5 < θ ≤ 1: stable but introduces numerical dissipation (1st-order accurate)
- θ < 0.5: unstable — do not use

## 4. How ECSIM enters the field solver

### 4a. MaxwellImage (LHS operator) — `EMfields3D.cpp:2797–2875`

> **Note on code comments**: The source has the labels swapped — `MaxwellSource`
> is marked `//? LHS` (line 2680) and `MaxwellImage` is marked `//? RHS`
> (line 2757). This document uses the mathematically standard convention:
> Image = LHS operator A·E, Source = RHS vector b.

The implicit operator applied to the trial field E:
```
A·E = E + (cθΔt)²·∇×(∇×E) + θΔt·4π·invVOL·S[M·S[E]]
```

Where S[·] denotes the energy-conserving smoothing operator. In code:

| Step | Code | Line(s) |
|------|------|---------|
| Krylov → physical space | `solver2phys(tempX, tempY, tempZ, vector, ...)` | 2809 |
| Communicate ghost nodes | `communicateNodeBC_old(...)` | 2811–2813 |
| Curl N→C | `curlN2C(tempXC, ..., tempX, ...)` | 2816 |
| Communicate cell centers | `communicateCenterBC(...)` | 2818–2820 |
| Curl C→N | `curlC2N(imageX, ..., tempXC, ...)` | 2822 |
| Identity + curl-curl | `imageX = tempX + factor * imageX` | 2830 |
| Smooth E before M·E | `energy_conserve_smooth(tempX, ...)` | 2836 |
| Mass matrix × vector | `mass_matrix_times_vector(&MEx, ..., tempX, ..., i, j, k)` | 2845 |
| Scale | `temp2X = dt*th*FourPI*MEx` | 2847 |
| Smooth M·E after | `energy_conserve_smooth(temp2X, ...)` | 2853 |
| Divide by cell volume | `temp2X *= invVOL` | 2859 |
| Add mass term | `imageX += temp2X` | 2868 |
| Physical → Krylov space | `phys2solver(im, imageX, ...)` | 2874 |

The factor `(cθΔt)²` appears at line 2825: `double factor = c*th*dt*c*th*dt`.

### 4b. MaxwellSource (RHS) — `EMfields3D.cpp:2681–2755`

The source term for the implicit solve:
```
b = E^n + θΔt · [c·(∇×B^n) − 4π·J_h·invVOL]
```

| Step | Code | Line(s) |
|------|------|---------|
| Communicate B ghost cells | `communicateCenterBC(Bxc, ...)` | 2688–2690 |
| ∇×B on nodes | `curlC2N(temp2X, ..., Bxc, ...)` | 2693 |
| Smooth J_h | `energy_conserve_smooth(Jxh, Jyh, Jzh, ...)` | 2716 |
| Source assembly | `tempX = th*dt*(c*temp2X - FourPI*Jx_tot*invVOL)` | 2731–2733 |
| Add E^n | `addscale(1.0, tempX, Ex, ...)` | 2737–2739 |
| Physical → Krylov space | `phys2solver(bkrylov, tempX, ...)` | 2754 |

### 4c. Energy-conserving smoothing

`energy_conserve_smooth()` (`EMfields3D.cpp:3003–3014`) dispatches to
`energy_conserve_smooth_direction()` (`EMfields3D.cpp:2960–3001`) which applies
a **27-point box filter** with weights:

| Neighbors | Weight | Count |
|-----------|--------|-------|
| Center | 8 | 1 |
| Faces (±x, ±y, ±z) | 4 | 6 |
| Edges | 2 | 12 |
| Corners | 1 | 8 |

Total: 8 + 24 + 24 + 8 = 64. The prefactor `0.015625 = 1/64` normalizes the
filter (line 2983).

Applied **symmetrically** to:
- E before M·E (line 2836)
- M·E result after multiplication (line 2853)
- J_h in the source (line 2716)
- E^θ before Faraday (line 2934)

The symmetry (smoothing both E and M·E, and J_h on the source side) is
essential — asymmetric smoothing would break energy conservation.

Controlled by:
- `Smooth` (bool, default 0) — enable/disable
- `SmoothCycle` (int, default 1) — apply every N cycles
- `num_smoothings` (int, default 0) — passes per application

**Critical implementation note**: must use `communicateNodeBC_old` (line 2974).
The `//!` comment at that line warns that new communication routines cause
energy growth.

## 5. The mass_matrix_times_vector function

`EMfields3D.cpp:2261–2297` — computes M·v at node (i, j, k):

```cpp
// Center (g = 0): standard 3×3 matrix-vector
resX  = vectX[i][j][k]*Mxx[0][i][j][k]
      + vectY[i][j][k]*Myx[0][i][j][k]
      + vectZ[i][j][k]*Mzx[0][i][j][k];
// (similarly for resY, resZ)

// Neighbors (g = 1..13):
for (int g = 1; g < NE_MASS; g++) {
    // Forward offset: (i1,j1,k1) = (i,j,k) + NeNo(g)
    resX += vectX[i1][j1][k1]*Mxx[g][i][j][k] + ...

    // Backward offset: (i2,j2,k2) = (i,j,k) - NeNo(g)
    // Uses M stored at the *neighbor* node (exploiting symmetry)
    resX += vectX[i2][j2][k2]*Mxx[g][i2][j2][k2] + ...
}
```

The forward direction uses M stored at the current node (i,j,k). The backward
direction uses M stored at the neighbor node (i2,j2,k2). This exploits the
symmetry of the mass matrix: M[g] at node n encodes coupling to the neighbor
at `n + offset(g)`, so the reverse coupling (`n − offset(g)`) is M[g] stored
at that neighbor.

## 6. Simulation loop (ECSIM flow)

From `iPIC3D.cpp:65–97` and `iPIC3Dlib.cpp`:

```
Init → WriteOutput(0) → for each cycle:
  CalculateMoments()       ← gather ρ, J_h, M from particles     (iPIC3Dlib.cpp:404)
    └─ computeMoments()    ← per-species loop                    (Particles3D.cpp:1471)
  ComputeEMFields(i)       ← solve A·E^θ = b, then update B      (iPIC3Dlib.cpp:542)
    ├─ calculateE()        ← GMRES or PETSc                      (EMfields3D.cpp:2568)
    └─ calculateB()        ← Faraday's law                       (EMfields3D.cpp:2917)
  ParticlesMover()          ← advance particles                   (iPIC3Dlib.cpp:615)
    ├─ ECSIM_velocity()    ← Boris push with E^θ, B^n            (Particles3D.cpp:993)
    └─ ECSIM_position()    ← advance x with v^{n+1}              (Particles3D.cpp:1277)
  WriteOutput(i)
→ Finalize
```

**Key difference from standard implicit PIC**: No predictor-corrector loop.
The old `mover_PC()` iterated `NiterMover` times (predictor-corrector); ECSIM
needs only **one pass** because the mass matrix implicitly encodes the particle
response.

## 7. ECSIM_velocity in detail

`Particles3D.cpp:993–1090`

For each particle:

1. **Interpolate E, B** to particle position via trilinear weights on 8 cell
   corners (lines 1031–1061):
   ```cpp
   grid->get_safe_cell_and_weights(x_old, y_old, z_old, cx, cy, cz, weights);
   sampled_field[i] += weights_c * field_components_c[i];
   ```

2. **Half E-kick** (line 1066):
   ```
   u_temp = u_n + qdto2mc * Exl
   ```

3. **Boris rotation** (lines 1070–1081):
   ```
   Ω = qdto2mc · B                         [gyration vector]
   denom = 1 / (1 + Ω²)                    [denominator]
   uavg = (u_temp + v_temp×Ω_z − w_temp×Ω_y + (u_temp·Ω)·Ω_x) · denom
   ```
   This is the standard Boris rotation written in the compact
   `v + v×Ω + (v·Ω)Ω` form.

4. **New velocity** (line 1084):
   ```
   v^{n+1} = 2·v_avg − v^n
   ```

The constant `qdto2mc = 0.5 * dt * qom / c` (line 1005) — the `/c` is the
Gaussian-CGS Lorentz force normalization.

**Note**: `ECSIM_velocity` uses E^θ (the solved-for theta-weighted field) and
B^n (old-time magnetic field). The fields are packed into `fieldForPcls` by
`set_fieldForPcls()` before the mover is called.

### RelSIM variant

`RelSIM_velocity()` (`Particles3D.cpp:1093`) follows the same structure but
divides Ω by the Lorentz factor γ and includes the `denom /= γ` factor. Two
sub-variants exist for computing γ: `"Boris"` (uses u_temp to estimate γ) and
`"Lapenta_Markidis"` (solves a quartic polynomial for γ, following Bacchini
2023).

## 8. ECSIM_position

`Particles3D.cpp:1277`

Advances positions using the newly computed velocities:
```
x^{n+1} = x^n + v^{n+1} · Δt / γ
```
(γ = 1 in the non-relativistic case.)

Includes **charge-conserving density update**: the residual divergence of E is
corrected by depositing charge via `getResDiv()` (line 1310) to enforce Gauss's
law. This is necessary because ECSIM inherently conserves *energy* but not
*charge* — the charge correction is applied separately.

## 9. E^θ recovery

After the solver returns E^θ (stored as `Exth, Eyth, Ezth`), the full-step
field E^{n+1} is extracted (`EMfields3D.cpp:2639–2642`):

```cpp
E^{n+1} = (1/θ)·E^θ − ((1−θ)/θ)·E^n
```

This is the inverse of E^θ = θ·E^{n+1} + (1−θ)·E^n. E^{n+1} then becomes
the E^n for the next cycle.

## 10. Key parameters

| Parameter | Default | Where | Effect |
|-----------|---------|-------|--------|
| `th` | 0.5 | `Collective.cpp:107` | θ-decentering (0.5 = Crank-Nicolson, energy conserving) |
| `Smooth` | 0 (off) | `Collective.cpp:109` | Enable energy-conserving smoothing |
| `SmoothCycle` | 1 | `Collective.cpp:110` | Apply smoothing every N cycles |
| `num_smoothings` | 0 | `Collective.cpp:111` | Number of smoothing passes per application |
| `GMREStol` | 1e-15 | input file | Solver tolerance (≤1e-15 for strict energy conservation) |
| `Relativistic` | false | input file | Use RelSIM instead of ECSIM for velocities |
| `PoissonMAres` | 0.01 | `Collective.cpp:155` | Charge-conserving position correction residual threshold (0.0 to disable) |
| `NiterMover` | — | `Collective.h:417` | Legacy predictor-corrector count (unused by ECSIM) |

## 11. Key source files

| File | What it does for ECSIM |
|------|------------------------|
| `particles/Particles3D.cpp` | `computeMoments()` (ρ, J_h, M assembly), `ECSIM_velocity()`, `ECSIM_position()` |
| `fields/EMfields3D.cpp` | `calculateE()` (solver dispatch), `MaxwellImage()` (LHS), `MaxwellSource()` (RHS), `calculateB()`, `mass_matrix_times_vector()`, `energy_conserve_smooth()` |
| `include/Neighbouring_Nodes.h` | 14-group stencil for mass matrix neighbor indices |
| `include/Collective.h` | `th`, `Smooth`, `SmoothCycle`, `NiterMover` parameters |
| `main/Collective.cpp` | Parameter parsing and defaults |
| `main/iPIC3Dlib.cpp` | `CalculateMoments()`, `ComputeEMFields()`, `ParticlesMover()` — the cycle orchestration |
| `iPIC3D.cpp` | Main simulation loop |
| `solvers/GMRES.cpp` | Built-in Krylov solver |
| `solvers/PETSC.cpp` | PETSc MatShell wrapper (calls same `MaxwellImage`) |

## 12. Why energy conservation may not be exact in practice

ECSIM conserves energy to machine precision in theory (θ = 0.5, solver tolerance
≤ 1e-15). In practice, the default `Double_Harris.inp` shows ~0.03% energy
deviation after 2000 cycles (~1.4e-05 relative drift at 200 cycles).

An empirical study (data in `output/energy_study/`) tested all plausible
culprits and found that **floating-point rounding noise — primarily from MPI
communication — is the dominant source**, not algorithmic defects in smoothing
or charge correction.

### 12a. Empirical findings

**Phase 1 — Factorial test** (200 cycles, 100×100 grid, 8 MPI processes):

| Case | Smooth | PoissonMAres | Mean GMRES iters | |ΔE/E₀| |
|------|--------|-------------|------------------|---------|
| A | ON (4 passes) | 0.01 (default) | 29.3 | 1.37e-05 |
| B | OFF | 0.01 (default) | 43.1 | 2.11e-05 |
| C | ON (4 passes) | 0.0 | 29.3 | 1.59e-05 |
| D | OFF | 0.0 | 43.2 | 2.63e-05 |

Turning smoothing OFF makes drift **worse**, not better. Case D (both off) is
the worst. This contradicts the earlier hypothesis that smoothing or charge
correction cause the drift.

**Phase 2 — Noise source isolation**:

| Test | Key variable | |ΔE/E₀| | Factor |
|------|-------------|---------|--------|
| np=1 (serial), 100 cycles | MPI eliminated | 2.61e-08 | 185× better |
| np=8 (baseline), 100 cycles | MPI present | 4.84e-06 | reference |
| npcelx=3, 200 cycles | 9 particles/cell | 1.59e-05 | |
| npcelx=5, 200 cycles | 25 particles/cell | 1.37e-05 | |
| npcelx=10, 200 cycles | 100 particles/cell | 9.32e-06 | 1.5× better |
| GMREStol=1e-8 | 15 iters/cycle | 1.84e-05 | |
| GMREStol=1e-12 | 24 iters/cycle | 1.66e-05 | |
| GMREStol=1e-15 | 29 iters/cycle | 1.54e-05 | |

Energy drift growth rate: ~N^2.4 (superlinear), consistent with compounding
FP noise rather than a fixed per-cycle systematic error.

### 12b. Dominant source: MPI communication FP noise

Going from np=8 to np=1 reduces drift by **185×**. This implicates:

1. **MPI_Allreduce non-associativity.** The built-in GMRES batches k+2 local
   dot products into a single `MPI_Allreduce(MPI_IN_PLACE, y, k+2, MPI_SUM)`
   call per Krylov iteration (`GMRES.cpp:124`). The MPI reduction tree order
   is implementation-dependent and may vary between calls, producing different
   FP rounding each time.

2. **Ghost cell exchange values.** Communicated ghost cells are computed on a
   different rank with potentially different FP rounding order. The smoothing
   stencil, curl operators, and mass matrix multiply all read ghost values,
   propagating cross-rank FP inconsistencies into the solution.

3. **Energy measurement MPI_SUM.** `get_E_field_energy()`, `get_B_field_energy()`,
   and `get_kinetic_energy()` each compute local sums then `MPI_Allreduce` with
   `MPI_SUM`. The reduction itself introduces ~1–5 ULPs of non-determinism per
   call, adding measurement noise on top of the physics drift.

### 12c. Secondary source: naive dot product summation

All dot products in the codebase use naive sequential summation with no
compensated (Kahan) or pairwise accumulation:

```cpp
// utility/Basic.cpp:40
double dot(const double *vect1, const double *vect2, int n) {
  double result = 0;
  for (int i = 0; i < n; i++)
    result += vect1[i] * vect2[i];   // no compensation
  return (result);
}
```

For Krylov vectors of length 3×98×98×1 ≈ 28,812 (the 100×100 grid interior),
each dot product accumulates ~28k terms. The last ~15 bits of each addition are
lost, giving ~1e-12 relative error per dot product. Over ~30 inner iterations
per cycle, this is ~3e-11 per cycle, compounding to ~3e-9 at 100 cycles — which
is close to the observed np=1 serial drift of 2.6e-8.

The same applies to the `normP()` function used for residual norms. The GMRES
convergence check at each restart (`initial_error = normP(r, ...)` at line 90)
recomputes the residual norm with this noisy summation, so the solver may declare
convergence at a slightly different point each time.

### 12d. Energy measurement noise

Even if the fields were computed exactly, the energy *measurement* introduces
its own FP noise:

- **Field energy** (`EMfields3D.cpp:6104`): sequential loop over interior nodes
  accumulating `0.5 * dx * dy * dz * (Ex^2 + Ey^2 + Ez^2) / 4π` — naive sum
  of ~10k terms per rank, then `MPI_Allreduce`.
- **Kinetic energy** (`Particles3Dcomm.cpp:1416`): sequential loop over all
  particles accumulating `0.5 * (q/qom) * (u^2 + v^2 + w^2)` — naive sum of
  ~250k terms per rank, then `MPI_Allreduce`.

The measurement noise is likely smaller than the solver noise (it happens once
per cycle vs. ~30× in the Krylov loop) but contributes to the total.

### 12e. Why smoothing helps conservation

Counter-intuitively, `Smooth = 1` **reduces** energy drift (1.37e-05 vs 2.63e-05).

The smoothing operator applied symmetrically inside `MaxwellImage()` (lines
2836, 2853) and `MaxwellSource()` (line 2716) is part of the ECSIM formulation
— it does not break the energy proof. The in-place smoothing of E^θ in
`calculateB()` (line 2934) happens after the curl for the B update is already
stored, so the field paths remain self-consistent.

The conservation benefit comes from **improved operator conditioning**:
smoothing suppresses high-frequency modes in the mass matrix term, reducing
the GMRES iteration count from ~43 (unsmoothed) to ~29 (smoothed). Fewer
iterations means fewer FP-noisy dot products and MPI reductions per cycle.

### 12f. Smoothing performance notes

Smoothing adds measurable cost inside the Krylov loop. Each
`energy_conserve_smooth()` call dispatches to
`energy_conserve_smooth_direction()` three times (X, Y, Z), and each direction
performs `1 + num_smoothings` MPI ghost exchanges (`communicateNodeBC_old`).
`MaxwellImage()` calls smoothing twice (lines 2836, 2853), so every Krylov
iteration adds **6 × (1 + num_smoothings)** MPI exchanges. Over ~30 GMRES
iterations that is ~180 extra MPI barriers per cycle. Additionally,
`energy_conserve_smooth_direction()` heap-allocates a temp array (`newArr3` /
`delArr3`) on every call — allocator churn inside the hot loop. Disabling
smoothing removes all of this but increases iteration count by ~47%.

**Performance optimization opportunities** (if smoothing stays enabled):

1. **Pre-allocate workspace array.** `energy_conserve_smooth_direction()` calls
   `newArr3` / `delArr3` on every invocation to create a temporary buffer
   (`EMfields3D.cpp:2966, 2999`). With 2 smooth calls per `MaxwellImage()`
   iteration × 3 directions × ~30 GMRES iterations, that is ~180 heap
   alloc/free cycles per field solve. The class already declares a
   `smooth_temp` member (`EMfields3D.cpp:146`) that is never used — wiring it
   in as the workspace would eliminate all allocator churn from the hot loop.

2. **Skip the redundant ghost exchange.** The first `energy_conserve_smooth()`
   in `MaxwellImage()` (line 2836) communicates ghost nodes for `tempX/Y/Z`,
   but those arrays were already exchanged three lines earlier
   (`communicateNodeBC_old`, lines 2811–2813). Skipping the duplicate exchange
   inside that first smooth call would save 3 × (1 + `num_smoothings`) MPI
   barriers per Krylov iteration.

3. **Longer-term ideas.**
   - *Batched MPI*: fuse the three per-component `communicateNodeBC_old` calls
     into a single message to reduce latency (would require a new comm routine).
   - *Pre-allocate comm buffers*: `communicateNodeBC_old` packs/unpacks into
     fresh buffers each time; persistent buffers would cut allocator traffic
     further.
   - *OpenMP in the stencil loop*: the 27-point averaging in
     `energy_conserve_smooth_direction()` is embarrassingly parallel over
     interior nodes — an `#pragma omp parallel for collapse(2)` on the i-j
     loops would scale with thread count at the cost of a barrier per pass.

### 12g. Charge conservation correction (PoissonMAres)

`PoissonMAres` defaults to 0.01 (`Collective.cpp:155`) and is not overridden
in `Double_Harris.inp`. This adds position corrections in `ECSIM_position()`
(lines 1423–1437) derived from the residual ∇·E − 4πρ.

The empirical effect on energy conservation is **small** (~15% increase in drift
when disabled: 1.37e-05 → 1.59e-05 with smoothing on). It has no effect on
GMRES iteration count. The slight benefit likely comes from keeping the charge
distribution closer to the Gauss-law-consistent state, which reduces noise in
the moment gathering.

**Performance note**: Negligible. One divergence computation on the grid plus a
few extra FMA ops per particle, once per cycle.

### 12h. Summary

| Factor | Impact on |ΔE/E₀| | Mechanism |
|--------|----------------------|-----------|
| MPI process count | **185×** (np=1 vs np=8) | MPI_Allreduce non-associativity, ghost cell FP inconsistency |
| Particle count | ~1.5× (npc=5 vs 10) | Smoother moments → less communicated noise |
| Smoothing | ~1.9× (on vs off) | Better conditioning → fewer GMRES iters |
| Solver tolerance | ~1.2× (1e-8 vs 1e-15) | Tighter solve marginally helps |
| Charge correction | ~1.15× (on vs off) | Small secondary effect |

Even with optimal algorithmic settings (th=0.5, GMREStol=1e-15, Smooth=1), the
energy drift at np=8 is ~1.4e-05 at 200 cycles. In serial (np=1), the floor
drops to ~2.6e-08 — still well above machine precision (~1e-14), likely due to
naive summation in the GMRES dot products and energy calculation.

**Achieving machine-precision conservation would require** compensated (Kahan)
summation in `dot()`, `dotP()`, `normP()` (`utility/Basic.cpp`), and the energy
calculation functions. This is a targeted change — the rest of the ECSIM
algorithm is correctly energy-conserving.
