# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-10)

**Branch:** `with-petsc`, head at `5b421ee`.

### Energy drift summary

| Test | dE metric | CIC | TSC | Notes |
|------|-----------|-----|-----|-------|
| 100x100x1 (standard) | \|dE/E0\| 10cyc | 5.1e-08 | N/A (needs nzc≥3) | Gold standard CIC test |
| 20x20x4 np=4 sm=4 | \|dE\| cyc1 | 3.6e-08 | 1.1e-05 | TSC ~300x worse at cycle 1 |
| 20x20x4 np=4 sm=4 | \|dE\| cyc3 | 1.5e-06 | 3.6e-05 | TSC bounded, oscillatory |
| 20x20x4 np=4 sm=4 | \|dE\| 10cyc | 1.2e-05 | 7.9e-05 | TSC peak ~1e-04 |

---

## What we know

### Fixed bugs

1. **Thin-dimension addFace overcounting** (commit `9c04563`) — For self-periodic axes where `nxc_r < 2*n_ghost`, left/right addFace destinations overlap. Workaround: runtime assertion requiring `nxc_r >= 2*n_ghost - 1`.

2. **Face self-copy offset inconsistency** (commit `e2e7c0d`) — Node field-copy path (`offset=1`) was missing the offset in face self-copies. Fixed with modular wrapping. Moment path (`offset=0`) was never affected. ~2x improvement.

### What has been ruled out as the drift source

| Hypothesis | Test | Result |
|-----------|------|--------|
| Halo exchange bug | CIC forced to n_ghost=2 (NBDerivedHaloCommN) | Identical to CIC n_ghost=1 |
| Solver residual | GMREStol=1e-20 (residual ~1e-17) | Drift WORSE — more iterations = more FP noise |
| P2G scatter noise | TSC with 5³ ppc vs 3³ ppc | No change |

### What we know about the drift

1. **TSC is unstable without smoothing.** Energy explodes by cycle 7. CIC is stable without smoothing. The wider TSC stencil creates modes that the implicit scheme alone cannot damp.

2. **Smoothing is both the stabilizer and the dominant drift source.** The smoothing filter suppresses the instability but introduces energy drift. Sweep results (20x20x4, 3 cycles):

   | num_smoothings | dE(1) | dE(10) | Stability |
   |---------------|-------|--------|-----------|
   | 0 | 3.8e-06 | EXPLODED | Unstable cycle 7 |
   | 1 | 2.3e-05 | 5.8e-02 | Unstable cycle 10 |
   | 2 | 2.2e-05 | 9.9e-03 | Marginal |
   | **4** | **1.1e-05** | **7.9e-05** | **Stable, oscillatory** |
   | 8 | 5.5e-06 | 2.1e-03 | Drifts after cycle 5 |

3. **sm=4 is the current sweet spot.** The drift is bounded and oscillatory (not secular).

4. **Drift scales with problem size.** 100x100x4 is worse than 20x20x4 in absolute dE (more total FP operations).

---

## What we don't know

The investigation so far has narrowed the drift to the **smoothing ↔ mass-matrix interaction**, but hasn't pinpointed the exact mechanism. The ECSIM energy theorem says energy is exactly conserved when the implicit operator A is self-adjoint and the GMRES solve is exact. For TSC, something breaks this — but what?

Possible mechanisms (not yet tested):

- **Operator asymmetry**: Is MaxwellImage actually non-self-adjoint in floating-point? By how much?
- **Smoothing asymmetry**: Is the discrete smoothing operator S non-self-adjoint due to ghost communication at n_ghost=2?
- **Self-copy convention mismatch**: The legacy n_ghost=1 path uses `ghost = boundary` (ignoring offset) for self-copies. NBDerivedHaloCommN uses `ghost = MPI-consistent` (with offset). Does this matter?
- **RHS/operator inconsistency**: MaxwellSource smooths J once. MaxwellImage smooths E twice per iteration. Are these "the same" smoothing in practice?
- **Theoretical limitation**: Does published ECSIM literature predict this drift level for TSC?

---

## Recommended investigation steps

### Phase 4: Operator symmetry measurement (HIGH PRIORITY)

This is the most direct diagnostic. If the operator A is self-adjoint to machine precision, the drift must come from elsewhere (particle push, moment gather ordering, etc.). If A has measurable asymmetry, we've found the smoking gun.

- [ ] **4a. Operator symmetry test** — After the first moment gather (cycle 1, before GMRES), pick two random Krylov-space vectors u, v. Compute `δ = |<u, A*v> - <v, A*u>| / (||u|| × ||v||)`. If δ >> ε_mach (~2e-16), the operator has non-trivial asymmetry. Add as a temporary diagnostic in `calculateE()` (`fields/EMfields3D.cpp:2586`), call `MaxwellImage` twice with test vectors and compute the inner products.

- [ ] **4b. Smoothing symmetry test** — Isolate the smoothing operator: apply `energy_conserve_smooth_direction` to two random vectors and check `<u, S*v> - <v, S*u>`. This tells us if the smoothing alone is asymmetric (ghost communication issue) or if asymmetry comes from the S·M·S chain.

- [ ] **4c. Component isolation** — Temporarily disable the mass-matrix term in MaxwellImage (skip lines 2876-2900 in `EMfields3D.cpp`), leaving only the curl-curl term. Run TSC with smoothing. If drift vanishes → asymmetry is in S·M·S. If drift persists → asymmetry is in S alone or the curl operators.

### Phase 5: Self-copy convention experiment (MEDIUM PRIORITY)

The legacy n_ghost=1 path uses `vector[0] = vector[nx-2]` (boundary value, no offset) for face self-copies, while NBDerivedHaloCommN uses `vector[0] = vector[nx-1-n_ghost-offset-g]` (MPI-consistent). For a self-periodic single-rank axis, the boundary value IS the periodic image of itself, making the legacy self-copy use a "wrong" but SYMMETRIC value. The MPI-consistent self-copy uses the correct value but may introduce asymmetry between left and right ghost layers.

- [ ] **5a. Test offset=0 for node-path self-copy** — In `NBDerivedHaloCommN` (`Com3DNonblk.cpp:649-690`), temporarily set `offset_selfcopy = 0` for the face self-copy only (keep MPI sends using real offset). Run TSC smoke test. If drift improves, the self-copy convention matters.

- [ ] **5b. Compare 1-rank vs 4-rank** — Run the same 20x20x4 TSC test on 1 rank (XLEN=1 YLEN=1 ZLEN=1, all self-periodic, all self-copies) vs 4 ranks (XLEN=2 YLEN=2 ZLEN=1, mixed MPI+self-copy). Different drift → self-copy convention matters. Same drift → convention doesn't matter.

### Phase 6: RHS/operator smoothing consistency (MEDIUM PRIORITY)

In MaxwellSource, J is smoothed once (line 2746) with ghost values from `communicateGhostP2G_ecsim` (which called `communicateNode_P`). In MaxwellImage, E is smoothed with ghost values from `communicateNodeBC` (called at the top of MaxwellImage, line 2846). These are different communication contexts. If the ghost values differ (e.g., because communicateNode_P fills ghosts at slightly different precision than communicateNodeBC), the energy theorem's assumption that the same S appears on both sides is violated.

- [ ] **6a. Ghost value audit** — After communicateNode_P in `communicateGhostP2G_ecsim`, and after communicateNodeBC at the top of MaxwellImage, dump the Z-axis ghost values for a single node line. Compare — they should be identical for the same field data.

- [ ] **6b. Separate RHS smoothing test** — Skip the smoothing in MaxwellSource (comment out line 2746) AND skip it in MaxwellImage (set Smooth=0). The operator becomes unsmoothed on both sides — theoretically exactly energy-conserving. For CIC this should give dE ~ ε_mach. For TSC it will be unstable, but compare dE(1) (before instability kicks in) — if it's at ε_mach level, the smoothing is truly the only source.

### Phase 7: Theoretical calibration (LOW PRIORITY but informative)

- [ ] **7a. Literature check** — Search for ECSIM + TSC/quadratic shape function results in Markidis & Lapenta (2011), Deka & Bacchini papers, and Vu & Brackbill (1992, energy-conserving smoothing). Do they report energy drift with higher-order shapes?

- [ ] **7b. Spectral analysis** — Save per-cycle dE for 50+ cycles. Fourier-transform to find dominant oscillation frequency. If it's at or near the grid Nyquist, the smoothing isn't fully suppressing the problematic TSC mode.

- [ ] **7c. Reference CIC+smoothing comparison** — Run CIC on 20x20x4 with num_smoothings=4 (current default). The CIC drift with smoothing should be essentially the same as without (since CIC doesn't need smoothing for stability). If CIC drift increases with more smoothing, it quantifies the smoothing FP noise floor independent of TSC.

### Phase 8: Targeted fixes (only after Phase 4 identifies the mechanism)

- [ ] **8a. If operator asymmetry from S·M·S** → try Kahan summation in mass_matrix_times_vector, or symmetrize the result: `temp2 = (S·M·S(E) + S·M^T·S(E)) / 2`
- [ ] **8b. If smoothing asymmetry from ghost comm** → align self-copy convention between legacy and NBDerivedHaloCommN paths
- [ ] **8c. If RHS/operator inconsistency** → ensure identical ghost-fill path for both contexts
- [ ] **8d. If theoretical limitation** → document expected drift level, adjust the CI smoke test tolerance, move on

### Phase 9: Edge/corner self-copy modular wrapping (DEFERRED)

Latent bug: edge and corner self-copies in NBDerivedHaloCommN still lack thin-dimension modular wrapping. Masked by current tests (all have nzc_r ≥ 3).

- [ ] **9a–d.** Add modular wrapping to edge/corner self-copies (`Com3DNonblk.cpp:806-888, 974-1039`)
- [ ] **9e.** Verify tests still pass

### Phase 10: Non-periodic boundary hardcodings (DEFERRED)

Won't fire for the all-periodic Double_Harris test. Fix when non-periodic BCs are needed with n_ghost > 1.

- [ ] **10a.** `adjustNonPeriodicDensities` — hardcoded `[1]` / `[nxn-2]`
- [ ] **10b.** `OpenBoundaryInflowB/E` — hardcoded `[1]` / `[nxn-2]`
- [ ] **10c.** `divN2C_BCLeftRight` / `derBC_left*` — hardcoded `-2` loop ends
- [ ] **10d.** Particle initializers — `[1, getNXC()-1)` instead of `[n_ghost, getNXC()-n_ghost)`
- [ ] **10e.** `init_double_harris` field init bounds

---

## Critical files reference

| File | Key locations |
|------|--------------|
| `communication/Com3DNonblk.cpp:604` | `offset` definition (isCenterFlag ? 0 : 1) |
| `communication/Com3DNonblk.cpp:642-692` | Face self-copy with offset + modular wrapping |
| `communication/Com3DNonblk.cpp:806-888` | Edge self-copies (offset, no wrapping) |
| `communication/Com3DNonblk.cpp:974-1039` | Corner self-copies |
| `communication/Com3DNonblk.cpp:1042-1057` | addFace/addEdge/addCorner moment summation |
| `communication/Com3DNonblk.cpp:127-134` | Legacy n_ghost=1 self-copy (NO offset!) |
| `grids/Grid3DCU.cpp:68-79` | Min grid assertion for self-periodic thin axes |
| `fields/EMfields3D.cpp:2282-2326` | mass_matrix_times_vector (63 groups for TSC) |
| `fields/EMfields3D.cpp:2378-2416` | communicateGhostP2G_ecsim (moments) |
| `fields/EMfields3D.cpp:2418-2467` | communicateGhostP2G_mass_matrix |
| `fields/EMfields3D.cpp:2711-2785` | MaxwellSource (RHS: smooths J at line 2746) |
| `fields/EMfields3D.cpp:2827-2912` | MaxwellImage (operator: smooths E at 2871, M*E at 2891) |
| `fields/EMfields3D.cpp:2998-3043` | energy_conserve_smooth_direction |
| `particles/Particles3D.cpp:1857-1896` | TSC mass matrix assembly |
| `inputfiles/ci_smoke_tsc.inp` | TSC smoke test (nzc=4, sm=4) |

## Commit history

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
| `5b421ee` | docs: comprehensive TSC energy conservation investigation results |
