# Block-Diagonal Preconditioner Matrix — Phase 2

## What problem are we solving?

The implicit Maxwell field solver in iPIC3D solves a linear system at every PIC cycle:

```
A · E = b
```

where **A** is the *implicit Maxwell operator* and **E** is the electric field vector (all three components at every interior grid node). `A` is never assembled as a matrix — it is applied through the function `MaxwellImage()`, which computes **y = A·x** for any input vector **x**. Both the built-in GMRES and the PETSc solver use this matrix-free product.

## Key Concepts

These terms appear throughout this document. If you're comfortable with iterative solvers, skip ahead.

- **Linear system A·x = b** — "Solving" means finding the x that satisfies this equation. Direct solvers (Gaussian elimination) are too expensive for large 3D problems, so we use *iterative* methods that start from a guess and improve it until the residual ‖A·x − b‖ is small enough.

- **Stencil** — The pattern of grid neighbors that contribute to a computation at each node. A "27-point stencil" means the node itself plus all 26 neighbors in its 3×3×3 cube.

- **Matrix-free method** — Computing A·x without ever storing the matrix A. Instead, the physics equations are evaluated directly (`MaxwellImage`). This saves memory (A for a 100³ grid would have ~10⁹ entries) but means preconditioners that need to inspect A's entries (like ILU) must work with an *approximation* P ≈ A.

- **Condition number (κ)** — Ratio of A's largest to smallest eigenvalue. κ ≈ 1 means the system is "easy" (like solving x = b); κ ≫ 1 means "stiff" — some error modes decay much faster than others, requiring many iterations. Our operator currently has κ ≈ few, which is why unpreconditioned GMRES works well.

- **Preconditioner** — An approximate inverse M⁻¹ ≈ A⁻¹ applied so the solver works on M⁻¹A·x = M⁻¹b instead. A good M⁻¹ clusters eigenvalues near 1, reducing κ(M⁻¹A). Trade-off: each iteration becomes more expensive (cost of applying M⁻¹), but fewer iterations are needed. Only worthwhile when the iteration reduction outweighs the per-iteration overhead.

- **GMRES (Generalized Minimal Residual)** — Builds an orthogonal basis of the Krylov subspace {b, Ab, A²b, ...} and finds the best approximation within it. Restarted GMRES(m) caps the basis at m vectors (here m=20) to limit memory, then restarts from the current best solution.

- **Near-null space** — Vectors x for which A·x ≈ 0. For curl-curl operators: constant fields have zero curl, so [1,0,0], [0,1,0], [0,0,1] are near-null vectors. AMG must be told about these explicitly to build coarse grids that preserve these "invisible" modes — otherwise it has no way to correct errors in the constant-field components.

- **Operator stiffness** — When different error modes decay at very different rates during iteration. A stiff operator has some modes that converge in a few iterations while others barely move, dragging out the total solve. Preconditioners aim to equalize these rates.

## What does the operator A look like?

### Where does this come from?

The implicit θ-scheme discretizes Maxwell's equations as:

```
Faraday:  (B^{n+1} - B^n) / Δt = -c · ∇×E_θ
Ampère:   (E^{n+1} - E^n) / Δt =  c · ∇×B_θ - 4π · J
```

where E_θ = θ·E^{n+1} + (1−θ)·E^n is the time-centered field (θ = 0.5 gives Crank-Nicolson). Eliminating B^{n+1} from Faraday and substituting B_θ = B^n − θ·c·Δt·∇×E_θ into Ampère yields a single equation for E_θ:

```
E_θ + (c·θ·Δt)²·∇×∇×E_θ + Δt·θ·4π·invVOL·S[M·S[E_θ]] = E^n + θ·Δt·(c·∇×B^n − 4π·J_h·invVOL)
```

This is the linear system **A·E_θ = b** solved by `calculateE()`. Each term has a physical meaning:

| Term | Code | Physics |
|------|------|---------|
| **E_θ** (identity) | self-copy in `MaxwellImage` | "The field remembers itself" — time-stepping self-coupling |
| **(c·θ·Δt)²·∇×∇×E_θ** | `curlN2C` → `curlC2N` | Electromagnetic wave propagation — couples E to spatial neighbors via curl |
| **Δt·θ·4π·invVOL·S[M·S[E_θ]]** | smoothing → `mass_matrix_times_vector` → smoothing | Implicit plasma response — how particles shield/amplify the field. Stronger in denser regions |
| **b** (RHS) | `MaxwellSource` | Known quantities from the current time step: E^n, ∇×B^n, and explicit current J_h |

After solving for E_θ, the full time step is recovered (`EMfields3D.cpp:2640`): **E^{n+1} = (1/θ)·E_θ − ((1−θ)/θ)·E^n**.

### Operator structure

Reading `MaxwellImage()` (`EMfields3D.cpp:2800–2875`), the operator has three terms:

```
A·E = E + (c·θ·Δt)² · ∇×∇×E + Δt·θ·4π·invVOL · S[M·S[E]]
```

| Term | Physics | Stencil |
|------|---------|---------|
| **E** (identity) | Time-stepping self-term | Diagonal — only couples node to itself |
| **(c·θ·Δt)² · ∇×∇×E** | Curl-curl from Faraday+Ampère | 27-point (composed `curlN2C` → `curlC2N`, each a 4-point average on 8 cells) |
| **Δt·θ·4π·invVOL · S[M·S[E]]** | Implicit current (mass matrix from ECSIM moments, double-smoothed) | 27-point (NE_MASS=14 neighbor groups, ±symmetry → 3×3×3 cube) |

**S** denotes `energy_conserve_smooth()` — a 27-point box filter (weights 8:4:2:1 for center:face:edge:corner) that preserves the energy-conserving property of ECSIM. It is applied twice: once to E before the mass matrix multiply, and once to the result after.

Each node thus couples to its full 27-node cube (itself + 26 neighbors), with each coupling being a **3×3 block** (Ex↔Ey↔Ez cross-coupling from the mass matrix off-diagonals). The full operator A is a sparse block matrix with block size 3 and a 27-point stencil.

## What we built (Phase 2): block-diagonal P

We assembled an explicit *approximation* P of A that captures only the **self-coupling at each node** — the 3×3 diagonal block:

```
P[i,j,k] = I₃ + (c·θ·Δt)² · diag(curl-curl self) + Δt·θ·4π·invVOL · M_center[i,j,k]
```

Specifically:
- **Identity I₃** — from the `tempX[i][j][k]` self-term
- **Curl-curl diagonal** — the composed `curlN2C`·`curlC2N` stencil contributes `+0.5/h²` per transverse direction to each field component (traced through `Grid3DCU.cpp`, 4-point averages of (0.25)² × 8 terms)
- **Mass matrix center block** — the 3×3 `M[g=0][i][j][k]` tensor, which encodes the local plasma response (density, temperature, anisotropy). Changes every cycle as particles move.

**Omitted:** all 26 neighbor blocks (curl-curl off-diagonals + mass matrix neighbor stencils) and the smoothing filter contribution.

### Why block-diagonal?

It's the simplest non-trivial preconditioner to implement and verify. The diagonal blocks are available locally (no MPI communication), are easy to index (one 3×3 block per DOF triad), and have a clear physical interpretation: they capture how each node's E-field responds to itself.

## Code changes

| File | Change |
|------|--------|
| `solvers/PETSC.cpp` | `assembleP()` method: builds `MATMPIAIJ` (block size 3) with 27-point stencil per node. Called in constructor and before each `KSPSolve` (mass matrix is cycle-dependent). `solve()` calls `assembleP()` when `usePrecMatrix_=true`. |
| `include/PETSC.h` | New members: `usePrecMatrix_`, `P_`, `emf_`, `globalBlockOffset_`, cached grid/physics constants. Constructor takes `bool usePrecMatrix`. |
| `include/Collective.h` | `bool PrecMatrix` member + `getPrecMatrix()` getter |
| `main/Collective.cpp` | Parse `PrecMatrix = true/false` from input file (default: false) |
| `include/EMfields3D.h` | 9 mass matrix getters (`getMxx()` etc.) + 8 physics parameter getters (`get_dx()` etc.), all `#ifdef USE_PETSC` guarded |
| `main/iPIC3Dlib.cpp` | Pass `col->getPrecMatrix()` to `PetscSolver` constructor |
| `inputfiles/Double_Harris_PETSc_prec.inp` | New input file with `PrecMatrix = true` |

## Test results

All configurations converge correctly. The block-diagonal P does not reduce iteration count at this problem size:

| Configuration | Avg iterations | Notes |
|---|---|---|
| Baseline (no P, PCNONE) | 26 | Constant across all cycles |
| P + default PC | ~30–35 | Converges, slightly more iterations |
| P + `-pc_type jacobi` | ~31 | Similar |
| P + `-pc_type bjacobi -sub_pc_type ilu` | ~30 | ILU on diagonal blocks ≈ exact diagonal solve, but no neighbor info to exploit |

**Why no improvement?** The block-diagonal approximation only captures the self-coupling at each node. ILU and Jacobi applied to a block-diagonal matrix can at most invert the 3×3 blocks exactly — but the 3×3 blocks are already close to identity (dominated by the I₃ term). The off-diagonal neighbor coupling, which actually causes GMRES to need multiple iterations, is entirely absent from P. The preconditioner application has a cost but provides almost no spectral improvement.

This was expected and validates the infrastructure. The real payoff comes from Phase 3.

## Phase 3: Full Stencil Assembly for Algebraic Multigrid

### Goal

Assemble the **full** sparse operator A (not just the diagonal blocks) so that PETSc's algebraic multigrid preconditioners (GAMG, HYPRE/BoomerAMG) can build a proper coarse-grid hierarchy. These preconditioners need the actual sparsity pattern and coupling strengths to construct effective restriction/prolongation operators and coarse-level solves.

### What needs to be assembled

The full operator A has a 27-point stencil with 3×3 blocks. For each interior node (i,j,k), there are up to 27 block entries connecting it to its neighbors:

1. **Curl-curl stencil** — The composed `curlN2C`·`curlC2N` operator. Each curl uses 4-point averaging over the 8 cells sharing a node, so the composed operator couples node (i,j,k) to all 27 nodes in its 3×3×3 cube. The off-diagonal entries encode how the curl of E at one node depends on E at neighboring nodes. These coefficients are **constant** (depend only on grid spacing), so they can be precomputed once.

2. **Mass matrix neighbor blocks** — `mass_matrix_times_vector()` loops over `g = 1..13` (NE_MASS=14 groups), each contributing coupling to nodes at offsets `±NeNo(g)`. Each coupling is a 3×3 block (Mxx, Mxy, ..., Mzz). These are **cycle-dependent** (change as particles move) and require reassembly every cycle — same as the current `assembleP()` but with 27× more blocks.

3. **Energy-conserving smoothing filter** — Applied twice in `MaxwellImage()` (once before mass matrix multiply, once after). This is a 27-point box filter with weights 8:4:2:1 (center:face:edge:corner), normalized by 1/64. Composing the smoothing with the mass matrix expands the stencil further. The smoothing is the trickiest part: it effectively convolves the mass matrix stencil with the filter stencil.

### Implementation approach

**MATMPIBAIJ with 27-block preallocation:**
```cpp
// 27 diagonal blocks (self + 26 neighbors within this rank's subdomain)
// Off-process blocks for nodes near MPI boundaries (ghost layer)
MatMPIBAIJSetPreallocation(P_, 3, 27, nullptr, 27, nullptr);
```

**Stencil assembly strategy:**

- **Curl-curl** — Compute once in the constructor. Work out the composed stencil analytically: for the Ex component, the curl-curl couples Ex to itself and to Ey, Ez at specific offsets. The 4-point averaging in each curl step produces fractional coefficients (multiples of 0.25² = 0.0625). This is a one-time derivation — the stencil coefficients depend only on `invdx`, `invdy`, `invdz`.

- **Mass matrix** — Loop over all 27 stencil nodes (g=0..13, ±), extract the 3×3 block `M_αβ[g][i][j][k]`, multiply by `scale = Δt·θ·4π·invVOL`, and insert into the matrix at the appropriate global block row/column. Must be reassembled every cycle (same as current Phase 2, but more entries).

- **Smoothing filter** — Two options:
  1. **Ignore it** in P. The smoothing modifies coupling strengths but doesn't change the stencil footprint. The preconditioner will be approximate anyway — omitting smoothing preserves the correct sparsity pattern and dominant coupling strengths. GAMG/HYPRE should still build a good hierarchy from the unsmoothed operator.
  2. **Include it** by convolving the mass matrix blocks with the smoothing stencil. This is algebraically correct but significantly more complex (each mass matrix entry fans out to 27 neighbors through the first smoothing, then the result fans out again through the second). This would require scratch storage and careful index arithmetic.

  **Recommendation:** Start without smoothing. If GAMG convergence is good, keep it simple. Add smoothing only if iteration counts suggest P is too far from the true A.

- **MPI ghost data** — Nodes near subdomain boundaries couple to ghost nodes owned by neighbor ranks. The mass matrix values at ghost nodes are already communicated via `communicateGhostP2G()`. Use `MatSetValuesBlocked()` with global indices — PETSc handles the off-process insertion and communication during `MatAssemblyBegin/End`.

### Expected payoff

Algebraic multigrid (GAMG or HYPRE BoomerAMG) with the full stencil should give **mesh-independent convergence**: iteration count stays bounded as the grid is refined. This is the standard result for elliptic-like operators (which the implicit Maxwell system approximates). For a 100×100 grid, unpreconditioned GMRES takes 26 iterations; for a 500×500 grid this would grow significantly. AMG should keep it at O(10) iterations regardless of resolution.

The per-cycle cost of assembling and applying the preconditioner will be higher than the block-diagonal version, but the reduction in Krylov iterations (and the associated `MaxwellImage` calls, each of which requires MPI communication) should more than compensate at scale.

### Risks

- **Smoothing composition** — Getting the stencil coefficients exactly right when the smoothing filter is included will require careful algebraic derivation and testing. Start without it.
- **Assembly cost** — 27 blocks per node (each 3×3 = 9 values), reassembled every cycle. For a 100³ grid: ~24M scalar insertions per cycle. PETSc handles this efficiently with `MatSetValuesBlocked`, but it's worth profiling.
- **Memory** — Full stencil: 27 × 9 doubles per node ≈ 1944 bytes/node. For a 100³ grid: ~1.8 GB total. Non-trivial but manageable on HPC nodes. The block-diagonal version uses 27× less.
- **GAMG setup cost** — AMG hierarchy construction is expensive but amortized over cycles if the sparsity pattern doesn't change (only values change). PETSc supports `KSPSetReusePreconditioner()` and `SNESSetLagPreconditioner()` for this — we could rebuild P every N cycles instead of every cycle.

### Phase 3 results (100×100, 8 MPI, Double Harris)

Phase 3 was implemented: `assembleP()` now inserts all 27 neighbor blocks of the mass matrix stencil (g=0..13, ± offsets). The matrix type was changed from `MATMPIBAIJ` to `MATMPIAIJ` with block size 3, since GAMG and HYPRE require AIJ format. Ghost→global index mapping via `nodeToGlobalBlock()` handles MPI boundary nodes and periodic BCs.

Smoothing filter was omitted from P (as recommended).

| Preconditioner | Iters | Notes |
|---|---|---|
| None (PCNONE) | 26 | Baseline, unchanged |
| HYPRE BoomerAMG | 28–32 | No improvement over baseline |
| GAMG | 250–500 | Catastrophic — no near-null space |
| ILU (bjacobi/ilu) | 28–34 | Mass matrix near-identity at this resolution |

**Why no improvement?** Two critical pieces were missing:

1. **No near-null space** — GAMG's smoothed aggregation and HYPRE's nodal coarsening need to know the 3 constant null-space vectors (one per E-component) to build effective coarse grids. Without this, GAMG's aggregation was essentially random, producing terrible coarse-grid operators. Physically: a spatially uniform electric field has zero curl everywhere, so the curl-curl part of the operator cannot "see" it. These uniform fields are the near-null vectors. AMG must know about them to build coarse grids that preserve these invisible modes — otherwise the coarse-grid correction has no way to fix errors in the constant-field component.

2. **Missing curl-curl off-diagonals** — P only contained the curl-curl *diagonal* (`ccDiagEx/Ey/Ez`). The full composed `curlC2N(curlN2C(E))` stencil has 27-point same-component neighbor coupling (discrete transverse Laplacian) AND cross-component coupling (discrete mixed derivatives like ∂²Ey/∂x∂y). These constant coefficients were absent.

3. **Near-identity structure** — At 100×100 resolution, the curl-curl diagonal is ~0.024 vs I=1.0 and the mass matrix entries are O(0.01). The operator is already well-conditioned (κ ≈ few), so any simple preconditioner (ILU, Jacobi) applied to near-identity blocks cannot beat unpreconditioned GMRES.

## Phase 4: Curl-Curl Stencil + Near-Null Space for AMG

### Goal

Add the two missing pieces to make AMG preconditioners effective:
1. **Full curl-curl off-diagonals** — precomputed analytically from tensor-product structure
2. **Near-null space vectors** — 3 constant vectors (one per E-component) attached to P

### Curl-curl stencil: tensor-product derivation

The composed `curlC2N(curlN2C(E))` has a clean tensor-product structure on a uniform grid. Three weight vectors fully determine all 243 stencil entries (3 output components × 3 input components × 27 offsets):

- `w_avg = [1, 2, 1]` — averaging in non-differentiated direction
- `w_lap = [1, -2, 1]` — second derivative
- `w_der = [-1, 0, 1]` — first derivative (for mixed terms)

The 1/16 prefactor comes from (0.25)² across the two curl operations, each averaging over 4 cell-face corners.

**Same-component (a==b): negative transverse Laplacian**

```
CC[a][a] = -1/16 · Σ_{q≠a} w_avg ⊗ w_lap/dq² ⊗ w_avg
```

For example, Ex→Ex: `-1/16 · (w_avg(x) ⊗ w_lap(y)/dy² ⊗ w_avg(z) + w_avg(x) ⊗ w_avg(y) ⊗ w_lap(z)/dz²)`

**Cross-component (a≠b): mixed second derivative**

```
CC[a][b] = +1/16 · w_der(a)/da ⊗ w_der(b)/db ⊗ w_avg(r)
```

For example, Ey→Ex: `+1/16 · w_der(x)/dx ⊗ w_der(y)/dy ⊗ w_avg(z)`

**Verification (asserted at construction time):**
- Diagonal matches existing: `CC[0][0][1][1][1] = 0.5·(1/dy² + 1/dz²)` (= old `ccDiagEx`)
- Cross-component self-coupling is zero: `CC[0][1][1][1][1] = 0` (because `w_der[1] = 0`)
- Row sums are zero: `Σ CC[a][b][di][dj][dk] = 0` for all (a,b) — curl-curl of constant = 0

### Offset→mass-group lookup table

The 14 NeighbouringNodes groups (g=0..13) cover all 27 offsets: g=0 is the center, g=1..13 each contribute a forward (+offset) and backward (-offset) neighbor. A lookup table `massGroupLookup_[oi+1][oj+1][ok+1]` maps each offset to its group index and whether to use the center node's mass matrix (forward) or the neighbor node's (backward):

```
Forward:  M[g][i][j][k]       — mass matrix stored at the row node
Backward: M[g][i+oi][j+oj][k+ok] — mass matrix stored at the column node
```

### Unified assembly loop

The previous `assembleP()` had a split structure: diagonal block first, then a separate g=1..13 loop for ± neighbors. Phase 4 replaces this with a single triple-nested offset loop `(oi, oj, ok) ∈ {-1,0,+1}³` that combines all three contributions into each 3×3 block:

```
for each offset (oi, oj, ok):
    block[a][b]  = cthdt² · CC[a][b][oi+1][oj+1][ok+1]    // 1. curl-curl
    block[a][b] += δ_{ab} · δ_{offset=0}                    // 2. identity
    block[a][b] += scale · M_ab[g][source_node]              // 3. mass matrix
    → MatSetValuesBlocked(P, grow, gcol, block)
```

### Near-null space

Three normalized PETSc Vecs, one per E-component. The DOF ordering is interleaved (`phys2solver` packs as `[Ex,Ey,Ez, Ex,Ey,Ez, ...]` per node), so vector `c` has 1.0 at every index ≡ c (mod 3). These are near-null vectors of the curl-curl part (curl of constant = 0); the mass matrix breaks exact nullity, hence "near."

Attached to P via `MatSetNearNullSpace(P_, nsp_)`. This is what GAMG needs for smoothed aggregation coarsening and HYPRE needs for nodal/unknown-based coarsening.

### Code changes

| File | Change |
|------|--------|
| `include/PETSC.h` | Added `curlCurlStencil_[3][3][3][3][3]`, `MassGroupEntry` struct + `massGroupLookup_[3][3][3]`, `MatNullSpace nsp_`, new methods `computeCurlCurlStencil()`, `buildMassGroupLookup()`, `setupNearNullSpace()` |
| `solvers/PETSC.cpp` | 4 new methods + restructured `assembleP()` with unified 27-offset loop. Stencil assertions in constructor. `nsp_` destroyed in destructor. |

No changes to EMfields3D, Collective, iPIC3Dlib, or input files.

### Phase 4 results (100×100, 8 MPI, Double Harris)

| Preconditioner | Phase 3 | Phase 4 | Change |
|---|---|---|---|
| None (PCNONE) | 26 | 26 | Baseline unchanged |
| GAMG | 250–500 | **22–23** | Near-null space fixed coarsening |
| Default (ILU) | 28–34 | 27–28 | Curl-curl off-diags help slightly |

GAMG convergence went from catastrophic to **better than unpreconditioned baseline**. Steady 22–23 iterations across all cycles with clean residual decay (~0.2× per iteration).

### Design decisions

1. **Analytical tensor-product for curl-curl** — Cleaner than numerical extraction (which would require disabling smoothing/mass-matrix in MaxwellImage). The three weight vectors compactly determine all 243 entries. Correctness verified by three independent checks (diagonal match, zero cross-coupling, zero row sums).

2. **Near-null space on P, not A** — `MatSetNearNullSpace(P_, ...)` tells the PC (GAMG/HYPRE) about P's null space. Since A is a shell matrix, attaching null space to A doesn't help the PC setup.

3. **Smoothing filter still omitted** — S⁴·M·S⁴ would create a radius-9 stencil (19³ = 6859 points per node). Impractical. The raw mass matrix in P provides correct coupling topology; AMG's coarsening still works well.

## Phase 5: Benchmark Results and PCNONE Default Fix

### Benchmark setup

Two studies run via `tests/bench_preconditioners.sh` (Double Harris, 20 cycles, 8 MPI, `PrecMatrix = true`):

1. **Preconditioner sweep** — fixed 100×100 grid, all PC types
2. **Mesh scaling** — 52×52 → 300×300, GMRES vs PETSc (PCNONE) vs GAMG

### Sweep results (100×100, 20 cycles, pre-fix)

These results were collected *before* the PCNONE default fix, so `PETSc_gmres` was running with PETSc's default BJACOBI+ILU on the approximate P.

| Solver | Field time (s) | Avg iters | vs GMRES |
|--------|---------------|-----------|----------|
| **PCNONE** | **5.61** | **26.0** | **1.19×** |
| Jacobi | 6.06 | 31.2 | 1.10× |
| GMRES (built-in) | 6.65 | 28.0 | baseline |
| PETSc_gmres (ILU) | 6.65 | 28.8 | 1.00× |
| HYPRE | 7.92 | 31.4 | 0.84× |
| FieldSplit | 8.14 | 30.8 | 0.82× |
| HYPRE_nodal | 10.52 | 29.4 | 0.63× |
| GAMG_tuned | 24.46 | 21.0 | 0.27× |
| GAMG | 26.60 | 23.0 | 0.25× |

**Key finding:** PCNONE (explicit `-pc_type none`) was the fastest PETSc configuration — 19% faster than built-in GMRES — while every preconditioner *increased* wall time. PETSc_gmres with default ILU(P) lost all of PETSc's per-iteration advantage by (a) paying ILU application cost and (b) adding ~3 iterations (28.8 vs 26.0).

### Scaling results (52→300, 20 cycles, pre-fix)

| Grid | GMRES (s) | iters | PCNONE (s) | iters | PETSc/ILU (s) | iters | GAMG (s) | iters |
|------|-----------|-------|------------|-------|---------------|-------|----------|-------|
| 52×52 | 1.23 | 26.6 | 1.15 | 24.2 | 1.65 | 26.9 | 42.66 | **194** |
| 100×100 | 4.76 | 28.0 | 4.59 | 26.0 | 5.83 | 28.8 | 21.27 | 23.0 |
| 152×152 | 10.53 | 28.8 | 10.38 | 26.9 | 13.70 | 31.8 | 44.11 | 21.8 |
| 200×200 | 17.91 | 28.9 | 16.71 | 27.1 | 23.08 | 33.5 | 57.11 | 17.1 |
| 252×252 | 27.50 | 29.6 | 25.59 | 28.0 | 34.81 | 34.5 | 105.35 | 20.9 |
| 300×300 | 39.19 | 29.9 | 36.85 | 28.6 | 51.03 | **35.9** | 188.32 | 19.1 |

**Observations:**

1. **PETSc with ILU(P) gets progressively worse** — at 300×300 it's 30% *slower* than built-in GMRES (51s vs 39s) with 35.9 iters vs 29.9. Since P omits the smoothing filter applied 4× in `MaxwellImage`, ILU(P) is an increasingly inaccurate preconditioner as the off-diagonal curl-curl terms grow relative to identity.

2. **PCNONE tracks built-in GMRES** but is consistently ~6-10% faster — PETSc's optimized BLAS orthogonalization pays off when not burdened by preconditioner overhead.

3. **GAMG iterations are mesh-independent** (17-23 for grids ≥100) — AMG theory works. But each V-cycle costs 4-5× a bare GMRES iteration (a V-cycle restricts the residual to progressively coarser grids, solves cheaply at the coarsest, then interpolates corrections back — touching every level costs several MatVec equivalents), so wall time is 3-5× worse at every resolution tested. The 52×52 outlier (194 iters) suggests the near-null space vectors don't help GAMG at very coarse grids where the operator has too few DOFs for meaningful aggregation.

4. **Unpreconditioned iterations are nearly flat** — GMRES goes from 26.6 to 29.9 across a 6× refinement. The operator is still well-conditioned.

### Root cause: the operator is near-identity

Operationally, "near-identity" means the solver is asked to find x in (I + ε·B)·x = b where ε < 1. This is almost as easy as solving x = b — all eigenvalues are clustered near 1, so GMRES converges in few iterations regardless of grid size. No preconditioner can improve on "already fast."

The dimensionless curl-curl strength is `(c·θ·Δt)²/h²`:

| Grid | h = L/nxc | (c·θ·Δt)²/h² | Character |
|------|-----------|---------------|-----------|
| 52×52 | 0.577 | 0.012 | ~identity |
| 100×100 | 0.300 | 0.043 | ~identity |
| 200×200 | 0.150 | 0.174 | slightly stiff |
| 300×300 | 0.100 | 0.391 | moderate |
| **480×480** | **0.0625** | **1.0** | **crossover** |

With `c=1, θ=0.5, dt=0.125`: `(c·θ·Δt)² = 0.00391`. The curl-curl term only reaches O(1) at ~480 cells. Below that, the identity dominates, condition number κ ≈ few, and no preconditioner can recoup its overhead.

The mass matrix contributes `dt·θ·4π·invVOL · M ≈ 0.01–0.05` at these parameters — also small compared to I. During the first 20 cycles, the plasma hasn't moved significantly, so M stays near its initial (relatively uniform) state.

### Fix: default to PCNONE when `PrecMatrix = true`

**Problem:** With `PrecMatrix = true`, PETSc defaulted to BJACOBI+ILU on P. Since P omits the smoothing filter, ILU(P) actively hurt convergence — adding 3-10 iterations and erasing PETSc's per-iteration speedup.

**Fix (solvers/PETSC.cpp):**
1. After `KSPSetOperators(ksp_, A_, P_)`, explicitly set `PCSetType(pc, PCNONE)` as the programmatic default. Runtime `-pc_type gamg` flags still override via `KSPSetFromOptions`.
2. In `solve()`, skip the per-cycle `assembleP()` when the active PC is PCNONE (checked via `PetscObjectTypeCompare`). No point assembling a 27-block stencil matrix that nobody reads.

The constructor's initial `assembleP()` stays unconditional — it runs before `KSPSetFromOptions`, so the final PC type isn't known yet (user might pass `-pc_type gamg`).

**After fix:** `PETSc_gmres` with `PrecMatrix = true` behaves identically to PCNONE — 26 iterations, ~10-15% faster than built-in GMRES — while still allowing `-pc_type gamg` opt-in.

## Preconditioner Reference

Summary of every preconditioner tested, what it does, and why it did (or didn't) help.

| PC | What it does | Good for | Our result |
|----|-------------|----------|------------|
| **PCNONE** | No preconditioning — each GMRES iteration costs one `MaxwellImage` + orthogonalization | Already well-conditioned systems (κ ≈ few) | **Best overall:** 26 iters, fastest wall time |
| **Jacobi** | Scales each equation by 1/(diagonal of A). Cheapest possible PC. | Diagonally dominant matrices where the diagonal varies significantly | +5 iters. Diagonal ≈ 1 everywhere (identity-dominated), so Jacobi ≈ doing nothing |
| **Block Jacobi + ILU** | Each MPI rank does an incomplete LU factorization of its local block. ILU ≈ Gaussian elimination but drops fill-in to stay sparse. PETSc's default when an explicit matrix is available. | General sparse systems with moderate κ | +3–10 iters. P omits smoothing filter, so ILU(P) is inaccurate; gets worse at finer grids |
| **GAMG** | Algebraic multigrid (smoothed aggregation): builds a hierarchy of coarser grids, solves cheaply at the coarsest, interpolates corrections back up (V-cycle). | Elliptic/Laplacian-like operators at large scale — gives mesh-independent iteration counts | 22–23 iters (with near-null space) — fewer than baseline, but each V-cycle costs ~4–5× a bare MatVec, so net slower at current sizes |
| **HYPRE/BoomerAMG** | Another AMG variant using classical Ruge-Stüben coarsening (point-based). Nodal variant groups unknowns by grid node for vector fields. | Similar to GAMG; nodal variant better for vector-valued problems | 29–32 iters; slower than PCNONE |
| **FieldSplit** | Splits Ex/Ey/Ez into separate blocks, applies a sub-PC to each. Like block Gauss-Seidel across field components. | Block-structured systems where components are weakly coupled | +3 iters. Mass matrix cross-terms (Mxy, Mxz, ...) couple all three components, defeating the split |

**V-cycle explained:** A V-cycle *restricts* (downsamples) the residual to progressively coarser grids, solves at the coarsest level (cheap — few DOFs), then *interpolates* (upsamples) the correction back through each level. The "V" shape comes from going down then up the grid hierarchy. Each pass costs roughly as much as several matrix-vector products on the fine grid.

## When Would Preconditioners Actually Help?

The benchmarks above test a regime where preconditioning can't win: 20 cycles from initialization, near-identity operator, moderate resolution. Here's where the physics changes enough to make the operator genuinely stiff.

### 1. During active reconnection (cycles 400–2000)

The Double Harris setup drives visible reconnection starting at ~400 cycles (50 ω_pi⁻¹). By cycle 800–1500:
- **Density gradients sharpen** — current sheet thins, density piles up at X-points and separatrices. The mass matrix entries `M[g][i][j][k]` near the X-point grow significantly as local density increases.
- **Strong currents** — the off-diagonal mass matrix terms (Mxy, Mxz, etc.) grow from near-zero to values comparable to the diagonal, coupling Ex↔Ey↔Ez at the reconnection site.
- **Non-uniform conditioning** — the operator becomes easy in the inflow regions (still near-identity) but stiff near the X-point. This spatial non-uniformity is exactly what AMG handles well.

**Test:** Run `pixi run reconnection` (2000 cycles, 100×100), save a checkpoint at cycle 1000, then benchmark solver performance at that state. Compare iteration counts and per-iteration cost against fresh initialization.

### 2. Higher resolution (500×500+)

At 500×500, `(c·θ·Δt)²/h² ≈ 1.1` — the curl-curl term matches the identity. Unpreconditioned GMRES iterations should grow noticeably. At 1000×1000 it reaches ~4.3; by that point AMG's bounded iteration count (17-23) becomes a clear win even with expensive V-cycles.

**Test:** Extend the scaling study to 500 and 750 (may need more MPI ranks or a cluster). Watch for the crossover where GAMG wall time drops below GMRES.

### 3. Larger time steps

Increasing `dt` directly increases `(c·θ·Δt)²/h²`. With `dt = 0.5` (4× current), the crossover drops to ~120 cells. Larger time steps are physically meaningful — implicit PIC can take `dt ≫ ω_pe⁻¹` — and this is precisely the regime where implicit solvers justify their cost.

**Test:** Run the scaling study with `dt = 0.25` and `dt = 0.5` to see if the GAMG crossover shifts to accessible grid sizes.

### 4. Combined stress test: reconnection + fine grid + large dt

The most realistic scenario: 250×250 grid, dt=0.25, 1000 cycles. By cycle 500, the reconnection is well developed and the operator has both stiff curl-curl terms (`(c·θ·Δt)²/h² ≈ 0.43`) and large, non-uniform mass matrix entries. This is where AMG should start earning its keep.

### 5. 3D problems

The current benchmarks are 2.5D (nzc=1). In 3D, the problem size grows as N³, MPI communication costs increase, and the ratio of computation to communication shifts. AMG's ability to reduce iteration count becomes more valuable when each `MaxwellImage` call (which requires ghost exchanges) is more expensive.

## Practical Suggestions for Future Benchmarking

### Runtime PETSc flags

```bash
# Best current configuration (unpreconditioned PETSc GMRES)
mpirun -np 8 build/iPIC3D input.inp -solver PETSc

# Opt in to GAMG (only worthwhile when operator is stiff)
mpirun -np 8 build/iPIC3D input.inp -solver PETSc -pc_type gamg

# GAMG with tuning (slightly fewer iters)
-pc_type gamg -pc_gamg_threshold -1.0 -pc_gamg_agg_nsmooths 3 -mg_levels_ksp_max_it 2

# HYPRE BoomerAMG with nodal coarsening
-pc_type hypre -pc_hypre_type boomeramg \
  -pc_hypre_boomeramg_nodal_coarsen 1 \
  -pc_hypre_boomeramg_vec_interp_variant 2

# FieldSplit: split Ex/Ey/Ez, AMG each scalar block
-pc_type fieldsplit -pc_fieldsplit_block_size 3 \
  -pc_fieldsplit_type symmetric_multiplicative \
  -fieldsplit_pc_type hypre
```

### Preconditioner lag

Since the mass matrix changes every cycle but the sparsity pattern doesn't, rebuilding P every N cycles instead of every cycle could amortize assembly cost. PETSc supports this via `KSPSetReusePreconditioner()` or manual flagging. Worth trying N=5 or N=10 during reconnection runs where the mass matrix evolves slowly between cycles.

### Include smoothing filter in P

P currently omits the energy-conserving smoothing filter (applied 4× in `MaxwellImage`). This means P underestimates the coupling between neighbors. Including even one level of smoothing (convolving the mass matrix stencil with the 27-point box filter) would make P closer to the true operator A, potentially improving AMG convergence enough to offset the extra assembly cost. The full S⁴ composition would produce an impractically large stencil (radius-9), but S¹ would only expand from 27 to 125 points — still tractable.
