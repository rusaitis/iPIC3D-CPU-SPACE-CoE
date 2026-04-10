# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-10)

**Branch:** `with-petsc`, head at `e2e7c0d`.

### Energy drift summary

| Test | Before fixes | After a87cac1 | After e2e7c0d | Target |
|------|-------------|---------------|---------------|--------|
| CIC n_ghost=1 100x100x1 | ~7e-08 | ~7e-08 | ~6e-09 | ~1e-07 |
| TSC 20x20x1 (nzc=1) | ~1.5e-02/cyc | ~3.6e-02/cyc | N/A (blocked by assertion) | ~1e-07 |
| TSC 20x20x4 (nzc=4) | — | ~2e-06 @ cyc3 | ~1e-06 @ cyc3 | ~1e-07 |
| CIC 20x20x4 (nzc=4) | — | — | ~1.6e-04 @ 10cyc | — |
| CIC 100x100x4 (nzc=4) | — | — | ~2.5e-05/cyc | — |

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

**Critical finding:** The inconsistency only affects the **node field-copy path** (`communicateNodeBC`, `communicateNode_P`) where `isCenterFlag=false` → `offset=1`. The **moment summation path** (`communicateInterp`) sets `isCenterFlag=true` → `offset=0`, making the old and new formulas identical. So moment P2G communication was never affected.

The node field-copy path is used by:
- `energy_conserve_smooth_direction` (smoothing during ECSIM field solve)
- `communicateNode_P` (post-moment ghost population)
- `communicateNodeBC` in `MaxwellImage` (GMRES matrix-vector product)

**Fix:** Face self-copy in `NBDerivedHaloCommN` now uses `n_ghost+offset+g` source depth (matching MPI sends), with modular wrapping for thin dimensions (`stride = n - 2*n_ghost - offset`). For offset=0 (moment path), this is algebraically identical to the old formula — no behavior change. For offset=1 (node path), ghost values now match what MPI exchange would produce.

**Impact:** ~2x improvement on standard TSC test (2.15e-06 → 1.06e-06 at cycle 3).

### Issue 3: Remaining TSC vs CIC gap (~10-100x)

On the same grid, TSC drifts more than CIC. On 20x20x4: TSC ~1e-06 at cycle 3 vs CIC ~5e-07 at cycle 1 (comparable early-cycle). The gap widens with more cycles as cell-crossing events accumulate.

Possible remaining causes:
- 63 mass matrix components (vs 14 for CIC) → 4.5x more FP operations in communication and summation
- Wider TSC stencil → more arithmetic per operator application → more round-off
- Edge/corner self-copies still use `n_ghost+offset+g` without modular wrapping (latent thin-dim bug, masked by current test topologies)

GMRES converges to ~9e-16 in both cases, so solver convergence is not the issue.

---

## Remaining work

### Phase 4: Investigate TSC/CIC gap further (only if ~1e-06 isn't acceptable)

**Step 4a: Mass matrix symmetry check**
Add diagnostic in `EMfields3D::communicateGhostP2G_mass_matrix()` (after communicateInterp + communicateNode_P) to check `|M[g] - M[g']|` for symmetric partner pairs. If max asymmetry > 1e-14, the communication introduces asymmetry.
- File: `fields/EMfields3D.cpp:2418-2467`

**Step 4b: Forced n_ghost=2 CIC vs TSC comparison**
Use `IPIC3D_FORCE_NGHOST` scaffold (temporarily add to `Grid3DCU.cpp:43`) to test CIC with n_ghost=2 on the same grid. Isolates halo exchange effects (shared) from TSC-specific stencil effects.
- Note: scaffold crashed previously for np=1. Use np=4+ topologies.

**Step 4c: Solver residual isolation**
Run TSC at `GMREStol=1e-20` to check if drift is solver-residual-dominated or scheme-intrinsic.

### Phase 6: Secondary cleanups (deferred)

These are real n_ghost=1 hardcodings that don't fire for periodic Double_Harris but should be fixed before non-periodic runs with n_ghost > 1:

- `EMfields3D::adjustNonPeriodicDensities` (L3176-3273)
- `EMfields3D::OpenBoundaryInflowB/E` (L5756-5899)
- `Grid3DCU::divN2C_BCLeftRight` etc. (L440-535)
- 12 alternative particle initializers in `Particles3D.cpp`
- Edge/corner self-copy modular wrapping (thin-dim safety, currently masked)

---

## Critical files reference

| File | Key locations |
|------|--------------|
| `communication/Com3DNonblk.cpp:604` | `offset` definition (isCenterFlag ? 0 : 1) |
| `communication/Com3DNonblk.cpp:642-692` | Face self-copy with offset + modular wrapping |
| `communication/Com3DNonblk.cpp:797-886` | Edge self-copies (still use offset, no wrapping) |
| `communication/Com3DNonblk.cpp:950-1035` | Corner self-copies |
| `communication/Com3DNonblk.cpp:1037-1060` | addFace/addEdge/addCorner moment summation |
| `communication/Com3DNonblk.cpp:1268-1330` | addFace implementation |
| `grids/Grid3DCU.cpp:68-79` | Min grid assertion for self-periodic thin axes |
| `fields/EMfields3D.cpp:2378-2416` | communicateGhostP2G_ecsim (moments) |
| `fields/EMfields3D.cpp:2418-2467` | communicateGhostP2G_mass_matrix |
| `fields/EMfields3D.cpp:2827-2912` | MaxwellImage (field solve operator) |
| `fields/EMfields3D.cpp:2998-3043` | energy_conserve_smooth_direction |
| `inputfiles/ci_smoke_tsc.inp` | TSC smoke test (nzc=4, XLEN=2 YLEN=2 ZLEN=1) |
| `inputfiles/ci_smoke_tsc_selfperiodic.inp` | Self-periodic X test (XLEN=1 YLEN=2 ZLEN=2) |
