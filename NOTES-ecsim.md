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
- **Lapenta (2023)** — *Advances in the Implementation of the Exactly Energy
  Conserving Semi-Implicit (ECsim) Particle-in-Cell Method*, Physics 5(1), 72.
  Acknowledges that exact conservation holds only for NGP interpolation.
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
Mzx, Mzy, Mzz) stored on each grid node with a runtime-sized stencil
controlled by the `StencilOrder` input parameter (see `EMfields3D::ne_mass_`,
declared in `include/EMfields3D.h`):

| StencilOrder | Shape function       | Particle support | NE_MASS |
|--------------|----------------------|------------------|---------|
| Linear (default) | Trilinear / CIC  | 8 nodes (2³)     | 14      |
| Quadratic        | Quadratic B-spline / TSC | 27 nodes (3³) | 63 |

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
| `StencilOrder` | "Linear" | input file | Particle/grid shape function: "Linear" (CIC, default) or "Quadratic" (TSC) — see §13 |
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

An empirical study (data in `output/energy_study/`) systematically tested all
plausible culprits: smoothing, charge conservation correction, MPI communication,
solver tolerance, particle count, GMRES restart size, and Kahan summation.

### 12a. Root cause: particle cell crossings

**ECSIM conserves energy to machine precision (~1e-14) as long as no particles
cross cell boundaries.** The energy proof requires that each particle's
interpolation stencil (which 8 grid nodes it touches) is the same when moments
are gathered and when the particle is pushed. When a particle moves to a new
cell, its stencil changes, breaking the exact field-particle energy cancellation.

Per-cycle energy change at np=1 (serial, Smooth=0, PoissonMAres=0, GMREStol=1e-15):

```
cycle   per-cycle |ΔE|    what happens
─────   ──────────────     ────────────
 1-20   1e-15 to 1e-14    machine precision — all particles in birth cells
   21   1.75e-14           slightly elevated
   23   6.02e-09           first fast electron crosses a cell boundary
 24-25  1e-16              back to machine precision — no crossings
   27   1.28e-13           another crossing
 28-30  1e-15              machine precision again
   31   5.95e-11           more crossings as flows develop
   36   2.69e-10           crossings becoming more frequent
   40   1.23e-10           ...
```

**Between cell-crossing events, energy conservation returns to machine
precision.** This proves the ECSIM algorithm itself is perfectly
energy-conserving — the drift is entirely from discrete cell-crossing events.

**Timescale check:** electron thermal velocity u_th,e = 0.06, dt = 0.125,
dx = 0.3. A ~3σ electron (v ≈ 0.18) moves 0.0225/cycle, crossing a cell after
~13 cycles. With the perturbation-driven flows (amplitude 0.4 × B0) on top,
the first crossing at cycle 23 is consistent with the fastest tail electrons.

**Confirmation via thermal velocity scan** (np=1, Smooth=0, PoissonMAres=0):

| Config            | u_th,e | Perturbation | First spike | Machine-precision cycles |
|-------------------|--------|-------------|-------------|------------------------|
| Baseline          | 0.06   | 0.4         | cycle 18    | 18                     |
| Half u_th         | 0.03   | 0.4         | cycle 59    | 59                     |
| Quarter u_th      | 0.015  | 0.4         | cycle 57    | 57 (flow floor)        |
| Quarter + low pert| 0.015  | 0.01        | cycle 87    | 87                     |

Halving u_th delays onset from 18→59 (thermal electrons cross later). Quartering
with the same perturbation gives no further delay (57≈59) — perturbation-driven
E×B flows now dominate the crossing timescale. Reducing perturbation amplitude
(0.4→0.01) breaks through this floor (57→87). All configurations maintain
|ΔE| ~ 1e-15 between spikes.

This is a fundamental limitation of grid-based PIC methods with moving
particles. Lapenta (2017) proves conservation under the assumption that
interpolation weights are computed at x^n and held constant; the cell-crossing
consequence is not discussed there. Lapenta (2023, "Advances in the
Implementation of ECsim", *Physics* 5(1), 72) explicitly states: *"Equation (45)
is exact only for the NGP scheme where it still leads to exact energy
conservation. When other orders of interpolation are used, energy conservation
is lost."* For CIC (bilinear, used by iPIC3D), the per-crossing energy error
is a geometric inconsistency that cannot be fixed by numerical precision
improvements.

### 12b. What does NOT cause the drift

**Smoothing (`Smooth` parameter).** Turning smoothing OFF makes drift *worse*
(2.63e-05 vs 1.37e-05), not better. Smoothing improves operator conditioning
(29 GMRES iters vs 43), which reduces MPI noise amplification. The smoothing
operator applied symmetrically in `MaxwellImage()` and `MaxwellSource()` is part
of the ECSIM formulation and does not break the energy proof.

**Charge correction (`PoissonMAres`).** Disabling it has a small effect (~15%
more drift: 1.37e-05 → 1.59e-05). The position correction is outside the ECSIM
variational framework, but its contribution is secondary.

**GMRES dot product precision.** Kahan (compensated) summation in `dot()`,
`dotP()`, `norm2()`, `normP()` produces **bit-identical results** through cycle
20 — the summation noise does not affect the solver output at these early stages.
The drift onset at cycle ~23 is the same regardless of summation method.

**GMRES restart size.** Tested restart=20 (default), 40, and 100 at np=1. The
restart value changes the *character* of the drift (discrete spikes vs gradual
leak) but not the ultimate magnitude (~3-8e-08 at cycle 100):

| Restart | Mean iters | Pattern | |ΔE/E₀| at 100 cycles |
|---------|-----------|---------|----------------------|
| 20 | 38.4 | Flat until cycle 23, then sharp spike | 2.44e-07 |
| 40 | 25.6 | Earlier gradual onset at cycle 20 | 1.01e-07 |
| 100 | 52.0 | Gradual onset at cycle 20 | 1.36e-07 |

### 12c. MPI amplification (secondary effect)

While cell crossings are the root cause, MPI communication amplifies the drift
by **185×** (np=1 gives 2.6e-08 vs np=8 gives 4.8e-06 at 100 cycles).

1. **MPI_Allreduce non-associativity.** GMRES batches dot products into
   `MPI_Allreduce(MPI_IN_PLACE, y, k+2, MPI_SUM)` per iteration
   (`GMRES.cpp:124`). The reduction tree order varies, introducing FP noise.

2. **Ghost cell exchange.** Values computed on different ranks with different
   FP rounding propagate through curl, smoothing, and mass matrix operations.

3. **Energy measurement.** `get_E_field_energy()`, `get_B_field_energy()`, and
   `get_kinetic_energy()` use `MPI_Allreduce(MPI_SUM)`, adding measurement
   noise.

### 12d. Empirical data

**Phase 1 — Factorial test** (200 cycles, 100×100 grid, np=8):

| Case | Smooth | PoissonMAres | Mean iters | |ΔE/E₀| |
|------|--------|-------------|-----------|---------|
| A | ON (4 passes) | 0.01 | 29.3 | 1.37e-05 |
| B | OFF | 0.01 | 43.1 | 2.11e-05 |
| C | ON (4 passes) | 0.0 | 29.3 | 1.59e-05 |
| D | OFF | 0.0 | 43.2 | 2.63e-05 |

**Phase 2 — Noise source isolation**:

| Test | Key variable | |ΔE/E₀| | Factor |
|------|-------------|---------|--------|
| np=1 (serial), 100 cycles | MPI eliminated | 2.61e-08 | 185× better |
| np=8 (baseline), 100 cycles | MPI present | 4.84e-06 | reference |
| npcelx=10, 200 cycles | 100 particles/cell | 9.32e-06 | 1.5× better |
| GMREStol=1e-8 | 15 iters/cycle | 1.84e-05 | |
| GMREStol=1e-15 | 29 iters/cycle | 1.54e-05 | |

**Phase 3 — Serial noise floor investigation** (np=1, 100 cycles):

| Test | Change | |ΔE/E₀| at 100 cycles | Effect |
|------|--------|---------------------|--------|
| Naive summation (baseline) | — | 2.44e-07 | reference |
| Kahan summation in dot/normP | Compensated FP sums | 1.29e-07 | ~2× (butterfly noise) |
| GMRES restart=40 | Fewer restarts | 1.01e-07 | ~2× |
| GMRES restart=100 | No restarts | 1.36e-07 | ~2× |
| main branch (no PETSc) | Clean codebase | 2.44e-07 | identical |

All serial variants give ~1e-07 at 100 cycles, confirming cell crossings are
the sole root cause. The ~2× variations are butterfly-effect noise from
different code paths diverging chaotically after cycle ~23.

Verified on the `main` branch (no PETSc changes): first 10 cycles are
bit-identical to `with-petsc-matrix`, confirming the PETSc integration
introduced no numerical changes to the GMRES path.

### 12e. Smoothing performance notes

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

### 12f. Mitigation strategies

The cell-crossing energy error is a fundamental geometric limitation of
grid-based PIC with finite-width shape functions. It cannot be eliminated, but
several approaches can reduce its magnitude.

**Why the per-crossing error is O(dt²).** The ECSIM operator couples fields to
particles through θ·dt·M·E (mass matrix term) and the implicit current J_h. A
crossing changes the interpolation weights by O(v·dt/dx), and the energy error
from the stencil mismatch scales as the product: O(θ·dt) × O(v·dt/dx) = O(dt²/dx).

**Empirical dt test** (np=1, Smooth=0, PoissonMAres=0): comparing dt=0.125 (100
cycles) with dt=0.0625 (200 cycles) over the same physical time (T=12.5):

| dt | Cycles | First spike | At T=12.5 |ΔE/E₀| |
|----|--------|------------|-----------|
| 0.125 | 100 | T=2.25, 4.0e-11 | 2.61e-08 |
| 0.0625 | 200 | T=3.25, 2.3e-11 | 1.15e-07 |

The first crossing spike is ~2× smaller with half dt, but the total drift at the
same physical time is actually *higher* in this realization. In a chaotic plasma,
different dt values produce different particle trajectories — the O(dt²) scaling
applies to the expectation over many crossings, not to single realizations.

**Practical approaches, ranked by feasibility:**

1. **Keep smoothing enabled** (no code change). `Smooth = 1` with 4 passes
   reduces drift by ~1.9× by improving operator conditioning. This is already
   the default and is the most practical mitigation.

2. **Use more particles per cell** (no code change). Increasing from 25 to 100
   particles/cell reduces drift by ~1.5×. Each particle's crossing contributes
   a smaller fraction to the total moment error.

3. **Higher-order shape functions** (code change). Replacing bilinear (CIC,
   current iPIC3D) with quadratic (TSC) interpolation would smooth the weight
   transition at cell boundaries. TSC shape functions have a continuous first
   derivative, so the stencil change is less abrupt. Trade-off: the stencil
   grows from 8 to 27 nodes, increasing moment gathering cost by ~3×, and the
   mass matrix becomes denser. This is the most promising code-level fix.

   **Future validation idea:** implement TSC as a compile-time or runtime option
   and run Double Harris reconnection with both CIC and TSC. The macroscopic
   physics (reconnection rate, X-point geometry, island growth, outflow jets)
   should be robust to the shape function choice — if they differ significantly,
   it flags a resolution issue where CIC results are noise-dominated. This would
   also quantify the energy conservation improvement from the smoother stencil
   transition. The implementation is localized to particle gather
   (`computeMoments()` in `Particles3D.cpp`), field interpolation
   (`set_fieldForPcls()`), and mass matrix assembly (`add_Mass()` in
   `EMfields3D.h`) — well-scoped but touches performance-critical code.

4. **Smaller dt** (no code change, slower). Reducing dt reduces per-crossing
   error as O(dt²), but requires proportionally more cycles for the same physics
   time. Net effect is problem-dependent due to chaotic trajectory divergence.

5. **Sub-stepping at crossings** (major code change). Detect cell crossings,
   split the timestep, and recompute moments at intermediate positions. This
   breaks ECSIM's single-pass design and introduces per-particle branching —
   essentially reverting to a predictor-corrector loop for crossing particles.

6. **Iterative moment correction** (major code change). After the particle push,
   re-gather moments at the new positions and re-solve. This is exactly the
   predictor-corrector iteration that ECSIM was designed to avoid. It would
   guarantee consistent moments but at 2–5× the cost of a single pass.

**Context: comparison with standard implicit PIC.** Despite the cell-crossing
limitation, ECSIM is significantly better than non-energy-conserving implicit
PIC. Standard implicit PIC uses J^n (old-time current), which introduces a
systematic O(dt) dissipation *every cycle* regardless of cell crossings. ECSIM
eliminates this systematic term entirely — its only energy error comes from
the discrete cell-crossing events, which are individually O(dt²) and occur
stochastically. For typical PIC parameters, ECSIM's drift (~0.01–0.1% over
thousands of cycles) is orders of magnitude better than the systematic
dissipation in non-energy-conserving schemes.

### 12g. Summary

| Factor | Impact | Mechanism |
|--------|--------|-----------|
| **Cell crossings** | **Root cause** | Interpolation stencil change breaks ECSIM energy proof |
| MPI communication | 185× amplifier | Non-associative reductions, ghost cell FP inconsistency |
| Smoothing | 1.9× (helps!) | Better conditioning → fewer iters → less MPI noise |
| Particle count | ~1.5× | Smoother moments → less noise propagated |
| Solver tolerance | ~1.2× | Tighter solve marginally helps |
| Charge correction | ~1.15× | Small secondary effect |
| Kahan summation | No effect | Bit-identical to naive for first 20 cycles |
| GMRES restart size | No effect | Changes drift character, not magnitude |

The ~0.03% drift in the default Double Harris simulation is an inherent property
of grid-based PIC with moving particles, not a numerical deficiency. ECSIM
delivers exact conservation between cell-crossing events — which is the
theoretical best for this class of methods.

## 13. Optional higher-order shape function (TSC, opt-in)

§12 traces the residual energy drift to the C⁰ discontinuity of trilinear
(CIC) cell crossings. The natural follow-up is whether a smoother shape
function reduces the drift. iPIC3D supports an optional **quadratic B-spline
(TSC)** path behind the input flag `StencilOrder = Quadratic` (default
`Linear`). The default Linear path is byte-identical to the legacy code.

### 13a. Geometry

| Quantity | Linear (CIC) | Quadratic (TSC) |
|----------|--------------|-----------------|
| Particle support per axis | 2 nodes | 3 nodes |
| Particle support (3D) | 8 nodes | 27 nodes |
| Mass-matrix product cube | 3×3×3 (27) | 5×5×5 (125) |
| `NE_MASS` (forward halves + center) | 14 | 63 |
| Mass-matrix memory (×9 components) | reference | ~4.5× |

The TSC weight, per axis with fractional offset `s ∈ [-½, ½]` from the
**nearest** node:

```
w_left   = ½ · (½ - s)²
w_center = ¾  -  s²
w_right  = ½ · (½ + s)²
```

These satisfy a partition of unity (Σ w = 1) and the same gather/scatter
helper (`Grid3DCU::get_nearest_node_and_weights_tsc`) is used in both
`Particles3D::computeMoments` and `Particles3D::ECSIM_velocity` to keep the
shape function consistent across the implicit step.

### 13b. Implementation surface

| File | TSC additions |
|------|---------------|
| `include/Collective.h` / `main/Collective.cpp` | `StencilOrder` flag, validates `PrecType=Matrix` is not combined with `Quadratic`. |
| `include/Neighbouring_Nodes.h` | Constructor-parameterised: `NeighbouringNodes(order)` builds either the 14-entry or 63-entry table; both share `getX/Y/Z(g)` API. |
| `include/Grid3DCU.h` | `get_weights_tsc(weights[27], sx, sy, sz)` and `get_nearest_node_and_weights_tsc(...)`. |
| `include/EMfields3D.h` | `add_Rho_node`/`add_Jxh_node`/`add_Jyh_node`/`add_Jzh_node` single-node deposit helpers (the 8-corner `add_Rho`/`add_Jxh` API is too rigid for 27-node TSC scatter). New `const int stencil_order_, ne_mass_` declared early so they initialize before the mass-matrix arrays. |
| `fields/EMfields3D.cpp` | `Mxx ... Mzz` allocated with `ne_mass_`; `mass_matrix_times_vector` does in-bounds checks before partner reads (out-of-range "wing" contributions are skipped at boundary nodes). The MaxwellImage M·E loop bounds remain `[1, nxn-2]`. |
| `particles/Particles3D.cpp` | `computeMoments` and `ECSIM_velocity` branch on `stencil_order_`: the TSC path uses the 27-node gather, the 27-node ρ/Jₕ scatter, and a 27×63 mass-matrix assembly. Linear path is unchanged. |

### 13c. Caveats and known limitations (v1)

1. **Energy conservation is not at machine precision for TSC.** The default
   Double Harris smoke test on a 60×60 grid shows TSC drift around `~3e-4`
   per cycle at `dt = 0.125`, vs. `~1e-9` for Linear. The drift scales
   approximately as `dt²` per cycle (vs. effectively `dt^∞` for Linear),
   confirming that the TSC path is a *consistent* discretization but does
   not satisfy ECSIM's exact discrete cancellation. Two probable causes:
   (a) the wider mass-matrix cube needs **2 ghost cell layers** to fully
   close the boundary contributions; iPIC3D currently exchanges only one,
   so the outermost interior nodes drop their wing contributions in
   `mass_matrix_times_vector`. (b) the per-rank mass matrix is not
   communicated across MPI boundaries — a pre-existing limitation that
   bites harder when the support widens from 8 to 27 nodes.
2. **`PrecType = Matrix` is unsupported.** `solvers/PETSC.cpp::assembleP()`
   hardcodes a 3×3×3 neighbour loop and 14-group lookup. With
   `StencilOrder = Quadratic`, `Collective.cpp` aborts early with a clear
   error message. Use `PrecType = None` (matrix-free GMRES or matrix-free
   PETSc) for the TSC experiments.
3. **Memory footprint.** The 9 mass-matrix arrays grow ~4.5× when
   `StencilOrder = Quadratic`. Default Linear is unchanged.

### 13d. How to test

```bash
# Default Linear (byte-identical baseline regression)
pixi run test

# TSC smoke test
mkdir -p /tmp/ipic3d_tsc
mpirun -np 4 build/iPIC3D inputfiles/ci_smoke_tsc.inp > tsc.log 2>&1
tail /tmp/ipic3d_tsc/ConservedQuantities.txt
```

`inputfiles/ci_smoke_tsc.inp` is a copy of `ci_smoke.inp` with
`StencilOrder = Quadratic` and `PrecType = None`. Compare its
`ConservedQuantities.txt` to a Linear run on the same problem to see the
relative drift.
