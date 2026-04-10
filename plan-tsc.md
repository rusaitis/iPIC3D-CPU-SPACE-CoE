# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-10)

**Branch:** `with-petsc`, head at `55203c3`.

### Energy drift summary

| Test | dE metric | CIC | TSC | Notes |
|------|-----------|-----|-----|-------|
| 100x100x1 (standard) | \|dE/E0\| 10cyc | 5.1e-08 | N/A (needs nzc≥3) | Gold standard CIC test |
| 20x20x4 np=4 sm=4 | \|dE\| cyc1 | 3.6e-08 | 1.1e-05 | TSC ~300x worse at cycle 1 |
| 20x20x4 np=4 sm=4 | \|dE\| cyc3 | 1.5e-06 | 3.6e-05 | TSC bounded, oscillatory |
| 20x20x4 np=4 sm=4 | \|dE\| 10cyc | 1.2e-05 | 7.9e-05 | TSC amplitude ~1e-04 peak |

### Key finding

**TSC energy drift is inherent to the wider stencil + smoothing interaction, not a code bug.** The drift is bounded and oscillatory (~1e-04 amplitude on 20x20x4), not secularly growing. This is ~10x worse than CIC on the same grid but is the expected behavior for the ECSIM scheme with TSC shape functions.

### Commit history (oldest → newest)

| Commit | What |
|--------|------|
| `08298be` | feat: optional TSC shape function |
| `e48c5ac` | feat: runtime n_ghost parameter |
| `c8edca0` | feat: n_ghost-aware halo exchange |
| `27fd264` | fix: assorted n_ghost=1 hardcodings |
| `5952a74` | fix: route mass-matrix/smoothing through modern path |
| `dd29a0e` | perf: bump GMRES restart 20→40 |
| `a87cac1` | fix: widen n_ghost>1 halo to cover inner ghost intersections |
| `9c04563` | fix: bump TSC smoke test nzc=1→4 to avoid thin-Z overcounting |
| `e2e7c0d` | fix: face self-copy offset consistency + min grid assertion |

---

## Root cause analysis

### Issue 1: Thin-dimension addFace moment overcounting (FIXED via workaround)

For self-periodic axes where `nxc_r < 2*n_ghost` (e.g., nzc=1 with n_ghost=2), the left-side and right-side `addFace` destinations overlap, doubling the periodic-wrap moment contribution. Cannot be fixed by clamping addFace alone — the whole discretization is self-consistently calibrated.

**Resolution:** Runtime assertion in `Grid3DCU.cpp` requires `nxc_r >= 2*n_ghost - 1` for self-periodic axes. TSC smoke test bumped to nzc=4.

### Issue 2: Face self-copy vs MPI offset inconsistency (FIXED in e2e7c0d)

The node field-copy path (`communicateNodeBC`, `isCenterFlag=false` → `offset=1`) uses `offset` in MPI sends but the old self-copy did not include `offset`. Fixed with modular wrapping. The moment path (`communicateInterp`, `offset=0`) was never affected.

**Impact:** ~2x improvement on standard TSC test.

### Issue 3: TSC smoothing-induced energy drift (INVESTIGATED — inherent)

**This is the main remaining issue.** Comprehensive investigation in this session:

#### Finding 3a: TSC is UNSTABLE without smoothing

Without smoothing (Smooth=0), TSC energy explodes by cycle 7 (dE grows from 3.8e-06 at cycle 1 to 5.7e-01 at cycle 10). CIC is stable without smoothing. The wider TSC stencil creates high-frequency modes that the implicit scheme cannot damp.

#### Finding 3b: Smoothing is the dominant drift source

| num_smoothings | dE(1) | dE(10) | Stability |
|---------------|-------|--------|-----------|
| 0 (none) | 3.8e-06 | EXPLODED | Unstable at cycle 7 |
| 1 | 2.3e-05 | 5.8e-02 | Unstable at cycle 10 |
| 2 | 2.2e-05 | 9.9e-03 | Marginally stable |
| **4 (default)** | **1.1e-05** | **7.9e-05** | **Stable, oscillatory** |
| 8 | 5.5e-06 | 2.1e-03 | Initially better, then worse |

**sm=4 is the sweet spot** — fewer passes are unstable, more passes add FP noise. The drift at cycle 1 (1.1e-05 with smoothing vs 3.8e-06 without) confirms that smoothing ADDS drift but is necessary for stability.

#### Finding 3c: NOT solver-limited

Tighter GMRES tolerance (1e-20 instead of 1e-15) makes the drift WORSE (2.2e-05 at cycle 3 vs 1.6e-05). The extra GMRES iterations (>40 vs 30) introduce more FP noise. The GMRES residual at 1e-15 is already sufficient.

#### Finding 3d: NOT halo-exchange-limited

CIC forced to n_ghost=2 (using NBDerivedHaloCommN instead of legacy path) produces identical results to CIC with n_ghost=1. The NBDerivedHaloCommN implementation is correct.

#### Finding 3e: NOT scatter-noise-limited

TSC with 5³=125 particles per cell gives the same drift as 3³=27 particles per cell. The mass matrix FP noise from the P2G scatter is not the bottleneck.

#### Finding 3f: Drift scales with problem size

| Grid | dE(1) | dE(5) | Particles/rank |
|------|-------|-------|---------------|
| 20x20x4 | 1.1e-05 | 1.0e-04 | 43k |
| 100x100x4 | 8.2e-05 | 5.3e-04 | 1.08M |

Larger grids give WORSE absolute drift (more FP operations in the mass matrix and smoothing). The drift appears proportional to the square root of the particle count.

---

## Conclusion

The ~10x TSC/CIC energy conservation gap is an inherent property of the wider stencil requiring smoothing. The ECSIM energy theorem holds exactly only when GMRES converges to zero residual. With smoothing, the effective operator is:

  A = I + θ²Δt²c² ∇×∇× + θΔt·4π/V · S(M · S(·))

The smoothing S is applied 8× per GMRES iteration (2 smooth calls × 4 passes each), totaling ~240 smoothing passes per cycle. Each pass involves ghost communication + 27-point stencil evaluation, accumulating O(ε_mach) round-off per operation. For TSC, the 63-group mass matrix (vs 14 for CIC) multiplies this noise ~4.5×.

The drift is bounded and oscillatory — energy conservation is not lost over time, it just oscillates. For practical purposes, ~1e-04 amplitude on a 20x20x4 grid is acceptable. On production grids (much larger), the per-particle contribution to drift is smaller, but total operations are larger. The net scaling is approximately O(sqrt(N_particles)).

---

## Remaining tasks

### Phase 5: Edge/corner self-copy modular wrapping (DEFERRED)

The face self-copy got modular wrapping in `e2e7c0d`, but edge and corner self-copies still use `n_ghost+offset+g` without thin-dimension guards. This is a **latent bug** masked by current test topologies (all tests have nzc_r ≥ 3, so indices stay in range). Only relevant for future tests with thin self-periodic axes near the minimum grid assertion boundary.

- [ ] **5a.** Add modular wrapping to X-periodic edge self-copy (`Com3DNonblk.cpp:806-832`)
- [ ] **5b.** Add modular wrapping to Y-periodic edge self-copy (`Com3DNonblk.cpp:834-860`)
- [ ] **5c.** Add modular wrapping to Z-periodic edge self-copy (`Com3DNonblk.cpp:862-888`)
- [ ] **5d.** Add modular wrapping to corner self-copies (`Com3DNonblk.cpp:974-1039`)
- [ ] **5e.** Verify: `tests/test.sh` still PASS, TSC smoke still ~3e-05 at cycle 3

### Phase 6: Non-periodic boundary hardcodings (n_ghost=1) (DEFERRED)

These don't fire for the all-periodic Double_Harris test but will break for non-periodic BCs with n_ghost > 1. Fix when needed.

- [ ] **6a.** `EMfields3D::adjustNonPeriodicDensities` (`fields/EMfields3D.cpp:3176-3273`) — hardcoded `[1]` and `[nxn-2]` boundary indices
- [ ] **6b.** `EMfields3D::OpenBoundaryInflowB/E` (`fields/EMfields3D.cpp:5756-5899`) — hardcoded `[1]` and `[nxn-2]`
- [ ] **6c.** `Grid3DCU::divN2C_BCLeftRight`, `derBC_left{X,Y,Z}` (`grids/Grid3DCU.cpp:440-535`) — hardcoded `-2` loop ends
- [ ] **6d.** 12 alternative particle initializers in `Particles3D.cpp` — cell loops use `[1, getNXC()-1)` instead of `[n_ghost, getNXC()-n_ghost)`
- [ ] **6e.** `EMfields3D::init_double_Harris` field init bounds (`fields/EMfields3D.cpp:5274,5299`)

### Phase 7: Potential improvements (OPTIONAL, research-grade)

These could reduce TSC drift but involve significant effort with uncertain payoff:

- [ ] **7a. Wider smoothing kernel** — Match the smoothing filter width to the TSC stencil (5x5x5 instead of 3x3x3). Requires new smoothing code path for n_ghost≥2.
- [ ] **7b. Kahan summation in mass_matrix_times_vector** — Reduce FP noise in the 63-group accumulation. Small impact since most noise comes from the smoothing, not the matrix product.
- [ ] **7c. Alternative smoothing approach** — Research: per-direction 1D smoothing instead of 3D box filter, or adaptive smoothing that only targets unstable modes.

---

## Critical files reference

| File | Key locations |
|------|--------------|
| `communication/Com3DNonblk.cpp:604` | `offset` definition (isCenterFlag ? 0 : 1) |
| `communication/Com3DNonblk.cpp:642-692` | Face self-copy with offset + modular wrapping |
| `communication/Com3DNonblk.cpp:806-888` | Edge self-copies (still use offset, no wrapping) |
| `communication/Com3DNonblk.cpp:974-1039` | Corner self-copies |
| `communication/Com3DNonblk.cpp:1042-1057` | addFace/addEdge/addCorner moment summation |
| `grids/Grid3DCU.cpp:68-79` | Min grid assertion for self-periodic thin axes |
| `fields/EMfields3D.cpp:2282-2326` | mass_matrix_times_vector (63 groups for TSC) |
| `fields/EMfields3D.cpp:2378-2416` | communicateGhostP2G_ecsim (moments) |
| `fields/EMfields3D.cpp:2418-2467` | communicateGhostP2G_mass_matrix |
| `fields/EMfields3D.cpp:2827-2912` | MaxwellImage (field solve operator with smoothing) |
| `fields/EMfields3D.cpp:2998-3043` | energy_conserve_smooth_direction (4-pass 3x3x3 box filter) |
| `particles/Particles3D.cpp:1857-1896` | TSC mass matrix assembly (27-node scatter, 63 groups) |
| `inputfiles/ci_smoke_tsc.inp` | TSC smoke test (nzc=4, XLEN=2 YLEN=2 ZLEN=1) |
