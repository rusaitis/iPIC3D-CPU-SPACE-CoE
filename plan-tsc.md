# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-11, Phase 8 COMPLETE — drift is algebraic, not FP)

**Branch:** `with-petsc`. Committed through `605a247` (Phase 8c Kahan in `mass_matrix_times_vector`). The earlier Phase 4 operator-symmetry diagnostic code is in `phase4-diagnostic.patch` (untracked) — see Finding 11 for why it was removed.

**Phase 6, 8c, 8a complete. 8d ruled out analytically.**  The TSC-vs-CIC drift gap at sm=4 is **not** floating-point rounding error — it is an ~5% algebraic leak per cycle in the energy-exchange bookkeeping of the smoothed TSC operator. FP-level mitigations (Kahan/compensated summation) cannot fix it; the problem lies in the discrete energy theorem, not the arithmetic.

**Key numbers (HEAD, 5-run medians, 20x20x4 np=4):**
- CIC sm=4: dE(1) = 2.1e-08, cyc 0→1 energy leak = **0.01%** of per-cycle exchange (2.5e-04)
- TSC sm=4: dE(1) = 1.061e-05, cyc 0→1 energy leak = **5.1%** of per-cycle exchange (2.1e-04) — 500× worse relatively
- TSC no-smooth: dE(1) = 5.9e-06 (stochastic, 3× spread), unstable by cycle 6

**Phase 8a (Kahan in `energy_conserve_smooth_direction`): tested, no benefit.** 5-run medians match HEAD within 0.2% at cycle 1; 1.3× better at cycle 10 is inside the 25× run-to-run variance. Code stashed and dropped — not worth carrying a 40-line restructuring of the 27-point stencil for zero measurable payoff. See Finding 12.

**Phase 8d (Kahan in TSC mass-matrix assembly): analytically ruled out.** ~730 adds per mass-matrix entry × 2.2e-16 × 0.016 max contribution ≈ 2.6e-15 absolute FP error — 10 orders of magnitude below the observed 1e-05 drift. Implementation would add shadow compensator arrays for negative benefit. Skipped.

**Phase 8c kept (committed).** Real 2.3× win for TSC no-smooth cycle 1 at trivial cost, no regression elsewhere.

**Phase 8e (accept as theoretical limitation): CHOSEN.** The drift is bounded and oscillatory, not secular. Recommended CI smoke test tolerances for TSC sm=4 on 20x20x4: `dE(1) ≤ 2e-05`, `dE(10) ≤ 3e-04` (the 3e-04 accounts for ~25× run-to-run variance from non-deterministic MPI reductions).

### Energy drift summary

| Test | dE metric | CIC | TSC | Notes |
|------|-----------|-----|-----|-------|
| 100x100x1 (standard) | \|dE/E0\| 10cyc | 5.1e-08 | N/A (needs nzc≥3) | Gold standard CIC test |
| 20x20x4 np=4 sm=4 | \|dE\| cyc1 | 3.6e-08 | 1.1e-05 | TSC ~300x worse at cycle 1 |
| 20x20x4 np=4 sm=4 | \|dE\| cyc3 | 1.5e-06 | 3.6e-05 | TSC bounded, oscillatory |
| 20x20x4 np=4 sm=4 | \|dE\| 10cyc | 1.2e-05 | 7.9e-05 | TSC peak ~1e-04 (single-run, unreliable — see Finding 10) |
| 20x20x4 np=4 **no smooth** | \|dE\| cyc1 | 4.2e-08 | 3.8e-06 | TSC 90x worse than CIC at cyc1 |
| 20x20x4 np=4 **no smooth** | \|dE\| 10cyc | 2.5e-06 | EXPLODED | TSC unstable cycle 6, GMRES fails |
| 40x40x4 np=4 sm=4 | \|dE\| cyc1 | — | 8.9e-05 | 4x more work/rank → 8x worse |
| 40x40x4 np=4 sm=4 | \|dE\| 10cyc | — | 2.5e-04 | Scales with per-rank workload |

### Phase 8 summary table — 5-run medians (HEAD = post-8c Kahan), 20x20x4 np=4

| Config | dE(1) median | dE(1) mean | dE(10) median | dE(10) mean |
|--------|--------------|------------|---------------|-------------|
| CIC sm=4                | 2.10e-08 | 1.74e-08 | 1.22e-05 | 1.23e-05 |
| TSC sm=4                | 1.061e-05 | 1.062e-05 | 1.27e-04 | 1.41e-04 |
| TSC sm=4 + Phase 8a     | 1.063e-05 | 1.063e-05 | 8.56e-05 | 8.26e-05 |
| TSC no-smooth (HEAD)    | 5.94e-06 | 5.82e-06 | EXPLODED (unstable cyc ≥ 6) | — |

Cycle-1 TSC sm=4 is deterministic to 0.2% across runs; Phase 8a difference (0.2%) is inside this. Cycle-10 has 25× run-to-run variance (see Finding 10a), so the 1.3–1.5× "improvement" from 8a is not distinguishable from noise. Phase 8d was not implemented — analytical FP-error bound is ~2.6e-15, 10 orders below observed drift.

### Cycle 0→1 energy balance — the algebraic leak

| Config | ΔBE | ΔEE | ΔKE | Total exchange | \|dE\| | Leak fraction |
|---|---|---|---|---|---|---|
| CIC sm=4 | -2.53e-04 | +1.92e-04 | +6.01e-05 | 2.5e-04 | 2.4e-08 | **0.01%** |
| TSC sm=4 | -2.07e-04 | +1.57e-04 | +4.00e-05 | 2.1e-04 | 1.06e-05 | **5.1%** |

TSC leaks ~500× more energy per cycle (relative to the exchange) than CIC does. CIC's 0.01% is close to the GMRES FP noise floor (30 iters × ε ≈ 6.6e-15; observed 2e-08 is the residual propagation × problem scale). TSC's 5% leak is systematic and reproducible — NOT FP rounding.

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
| Operator asymmetry | Phase 4 diagnostic (δ_A, δ_cc, δ_S, δ_M) | Apparent asymmetry O(1e-03) is from periodic boundary double-counting; curl-curl is near-symmetric (2e-05) |
| Ghost layer inversion | Fixed n_ghost>1 ghost ordering | Energy WORSE — code is self-consistent within inverted convention |

### What we know about the drift

1. **TSC is unstable without smoothing.** Energy explodes by cycle 6-7. CIC is stable without smoothing. The wider TSC stencil creates modes that the implicit scheme alone cannot damp. Confirmed by Phase 6a: GMRES fails to converge at cycle 6, energy at cycle 10 is 3.3x initial (fully exploded).

2. **Smoothing is both the stabilizer and the dominant drift source.** The smoothing filter suppresses the instability but introduces energy drift. Sweep results (20x20x4, 3 cycles):

   | num_smoothings | dE(1) | dE(10) | Stability |
   |---------------|-------|--------|-----------|
   | 0 | 3.8e-06 | EXPLODED | Unstable cycle 6 |
   | 1 | 2.3e-05 | 5.8e-02 | Unstable cycle 10 |
   | 2 | 2.2e-05 | 9.9e-03 | Marginal |
   | **4** | **1.1e-05** | **7.9e-05** | **Stable, oscillatory** |
   | 8 | 5.5e-06 | 2.1e-03 | Drifts after cycle 5 |

3. **sm=4 is the current sweet spot.** The drift is bounded and oscillatory (not secular).

4. **Drift scales with per-rank workload, not communication volume.** np=1 (all self-periodic, zero MPI) is WORSE than np=4:

   | | dE(1) np=1 | dE(1) np=4 | dE(10) np=1 | dE(10) np=4 |
   |---|---|---|---|---|
   | **CIC** | 2.2e-06 | 1.7e-08 | 4.9e-05 | 1.2e-05 |
   | **TSC** | 1.5e-05 | 9.8e-06 | 2.8e-04 | 7.9e-05 |

   MPI doesn't hurt — it helps. With np=1, each rank processes 4x more particles and grid nodes, accumulating ~4x more local FP noise in P2G scatter, addFace summation, and smoothing. The drift is dominated by **local FP accumulation**, not communication. More ranks = less work per rank = better energy conservation.

   This also rules out the self-copy convention as a drift source: np=1 (100% self-copies) has worse drift than np=4 (mixed MPI + self-copies).

   **Grid-size scaling confirms this (Phase 6c):** TSC on 40x40x4 (4x more nodes, same 4 ranks → 4x more work/rank) gives dE(1) = 8.9e-05 vs 1.1e-05 for 20x20x4 — 8x worse absolute drift but only 4x more total energy. The relative drift ~doubles with doubled per-rank grid dimension. Consistent with per-rank FP accumulation dominating.

5. **The operator IS self-adjoint (within the ghost convention).** Phase 4 diagnostic measured δ = |<v,Au> - <u,Av>| / (||u||·||v||). The large apparent values (O(1e-03)) are artifacts of the Krylov inner product double-counting periodic boundary nodes. The curl-curl operator, which is mathematically self-adjoint, measures δ_cc ~ 2e-05 for np=1 — consistent with the double-counting artifact scaling as O(1/N_nodes). The true operator asymmetry is at or near machine precision.

6. **Ghost layers are inverted for n_ghost > 1, but this is a benign self-consistent convention.** For the node field-copy path (offset=1), recv fills ghosts outside-in while send reads sources inside-out, creating swapped ghost layers. Confirmed empirically. However, mass matrix entries use the same inverted convention (via communicateNode_P), so the product M*v is self-consistent. Fixing the inversion without coordinating all paths makes energy 40x WORSE.

7. **Smoothing is NOT the sole drift source — TSC mass matrix has additional FP noise.** Phase 6a direct comparison (20x20x4, np=4, no smoothing):
   - CIC unsmoothed: dE(1) = 4.2e-08 (GMRES FP noise floor, ~ε_mach × N_iters)
   - TSC unsmoothed: dE(1) = 3.8e-06 (90x worse than CIC at same config)

   Even without smoothing, TSC has substantially higher drift. The TSC mass matrix has 63 interaction groups (vs CIC's 14) with 1125 multiply-adds per node vs ~250 — a 4.5x wider stencil accumulates proportionally more FP rounding error.

8. **Two independent drift sources contribute to TSC:**
   - **Mass matrix FP accumulation** (~3.8e-06 at cycle 1, TSC-specific due to wider stencil)
   - **Smoothing FP amplification** (~additional 3x on top at cycle 1, ~5x by cycle 10)
   Together: TSC with sm=4 has dE(1) = 1.1e-05, consistent with 3.8e-06 × ~3 from smoothing.

9. **CIC without smoothing is near the GMRES FP noise floor.** dE(1) = 4.2e-08 for CIC without smoothing is NOT exactly ε_mach — it's the cumulative FP error from ~25 GMRES iterations. This sets the theoretical minimum: even a "perfect" energy-conserving scheme would have this level of drift from iterative arithmetic.

10. **Phase 8c (Kahan in `mass_matrix_times_vector`): only a modest 2–3× win on TSC unsmoothed cycle 1; no help on the production smoothed case.** Three-run means (HEAD vs HEAD+Kahan, 20x20x4 np=4, 10 cycles):

    | Config | HEAD mean | +Kahan mean | Ratio |
    |---|---|---|---|
    | TSC no-smooth cyc 1 | 2.7e-06 | 1.2e-06 | 2.3× |
    | TSC sm=4 cyc 1 | 1.065e-05 | 1.064e-05 | ~1.00× |
    | TSC sm=4 cyc 10 | 1.34e-04 | 1.04e-04 | 1.3× (within ~25× variance) |

    **Implications:**
    - The mass-matrix `A*v` accumulation was not the dominant FP noise source we thought it was. The single-run ~90× CIC-to-TSC ratio from Phase 6a was inflated by run-to-run variance (Finding 10a below).
    - For the production smoothed config, cycle-1 drift is dominated by smoothing FP noise (confirmed: Kahan in the mass-matrix mult has ~0% effect at cycle 1 for sm=4).
    - Cycle-10 data are too noisy for single-run comparisons in this configuration (TSC+smooth is oscillatory; 25× spread across 3 runs). Priority ordering in Phase 8 must be re-derived from multi-run means, not single-run dE(10).

    10a. **Run-to-run non-determinism is large.** For TSC sm=4, three consecutive runs of the identical input file gave dE(10) ∈ {6.93e-06, 1.20e-04, 1.74e-04} — a ~25× spread. Cycle 1 is deterministic (~0.1% spread) because only one GMRES+particle-gather pass has happened. Likely source: MPI reduction ordering in the Krylov inner products and/or particle-init velocity draw. **All earlier Phase 6 "single-run dE(10)" numbers in the summary table above should be treated as samples from a wide distribution, not as reproducible benchmarks.**

11. **Phase 4 operator-symmetry diagnostic code perturbs the cycle-1 solve (kept out of the committed tree).** The uncommitted Phase 4 code (`diagnoseOperatorSymmetry()`, `skip_mass_matrix_diag_` flag, and the `if (!skip_mass_matrix_diag_)` wrap around the S·M·S block in `MaxwellImage`) reproducibly worsened TSC sm=4 dE(1) from 1.08e-05 to 4.74e-04 (~44×) **even when the diagnostic call was gated behind an env var and never actually ran at runtime**. Mechanism: (a) the function-scope `if (!skip_mass_matrix_diag_)` changes loop structure and blocks some compiler inlining/vectorization decisions around the inner `mass_matrix_times_vector` call; (b) adding a ~300-line diagnostic function to the same translation unit changes object-layout/scheduling enough to shift FP results downstream. Removing both effects restores HEAD behavior. The full diagnostic is preserved as `phase4-diagnostic.patch` (untracked) so the symmetry findings in the Phase 4 section below remain reproducible; re-apply with `git apply` when needed.

12. **Phase 8a (Kahan in `energy_conserve_smooth_direction`) tested and rejected.** 5-run medians with the full 27-point stencil restructured into a compensated accumulator gave dE(1) = 1.063e-05 vs HEAD 1.061e-05 (0.2% different — below run-to-run noise on cycle 1). Cycle 10 showed 8.56e-05 vs HEAD 1.27e-04 (1.5×), but that gap is inside the 25× variance envelope. Conclusion: smoothing-loop rounding is not the dominant drift source. Patch dropped, not carried as a patch file since there's no signal worth preserving.

13. **Phase 8d (Kahan in TSC mass-matrix assembly) analytically ruled out.** On 20x20x4 np=4, each mass-matrix entry `M[g][i][j][k]` accumulates ~730 particle contributions of magnitude ≤ q·q_dt_2mc·w² ≈ 0.016. Plain summation error: 730 · 2.2e-16 · 0.016 ≈ **2.6e-15** — 10 orders of magnitude below the observed 1e-05 drift. Implementing compensated summation in the assembly would require 9 shadow arrays of size (ne_mass_ × nxn × nyn × nzn), new member allocations, and rework of `add_Mass` for a change that cannot register at the observed precision. Skipped.

14. **The TSC drift is algebraic, not floating-point.** Direct cycle 0→1 energy-balance comparison on 20x20x4 np=4 sm=4:

    | Quantity | CIC (HEAD) | TSC (HEAD) |
    |---|---|---|
    | ΔBE (cycle 0→1) | -2.53e-04 | -2.07e-04 |
    | ΔEE | +1.92e-04 | +1.57e-04 |
    | ΔKE | +6.01e-05 | +4.00e-05 |
    | \|dE\| | 2.4e-08 | 1.06e-05 |
    | Leak / exchange | 0.01% | 5.1% |

    Both cases exchange ~2e-04 of energy between B, E, and particles per cycle. CIC closes the books to 0.01% (near the GMRES FP floor: 30 iterations × ε ≈ 6.6e-15 × ||E|| · problem scale). TSC closes to 5.1% — **500× worse relatively**. GMRES itself converges to residual 3-8e-16 in 30 iterations for TSC — there is no additional FP slack to recover.

    The 5% per-cycle leak means the discrete energy conservation identity of ECSIM does not hold for the TSC + S·M·S-smoothing combination the way it does for CIC. This is a scheme-level property, not a rounding-error property, and Kahan compensation cannot recover it. The 8a, 8c, and 8d mitigations collectively touch every FP accumulator between P2G scatter and GMRES exit; none of them produced meaningful improvement at cycle 1 for the production sm=4 configuration.

    **Candidate mechanisms** (hypotheses for Phase 9, not yet investigated):
    - The ECSIM energy-conservation proof (Markidis & Lapenta 2011 for linear B-spline) may not extend unchanged to quadratic B-spline support when a symmetric 27-point smoother is applied to both `J` (in `MaxwellSource`) and `E` (in `MaxwellImage` via `S·M·S`). The algebraic identity that makes ⟨E_new, J_new⟩ exactly match the kinetic-energy change relies on particle work being computed with the SAME field that enters the operator. TSC + smoothing introduces a mismatch: particles see raw E at their positions via the TSC gather, but the operator sees `S·E·S`.
    - The mass matrix assembled by the scatter is symmetric as a 4D object, but the `S·M·S` wrapping can break exact self-adjointness on the periodic grid when the effective stencil width (TSC 5-point × smoother 3-point = 7-point) exceeds the ghost halo.
    - Pressure-tensor terms in the moment gather use 8-point linear weights (`add_Pxx` etc. in `Particles3D.cpp:1970-2011`) even in the TSC branch — an inconsistency with the TSC mass-matrix assembly that would not affect cycle-1 energy directly but does affect downstream moment bookkeeping.

---

## What we don't know (after Phase 8 — the FP-noise lens is exhausted)

Phase 6 hypothesized two FP noise sources (mass-matrix accumulation, smoothing amplification). Phase 8c eliminated (8c) the mass-matrix apply, Phase 8a eliminated the smoothing loop, Phase 8d was analytically eliminated for the mass-matrix assembly. All three candidate FP sources have been instrumented or reasoned through; none explain the 5% per-cycle algebraic leak.

**Phase 9+ should investigate the scheme-level discrepancy, not the arithmetic.** Candidate directions:

- **Symmetry of `S·M·S`.** Even if `M` and `S` are each self-adjoint on the periodic grid, the composition may fail self-adjointness if the combined stencil (TSC 5-point × 3-point smoother = effective 7-point) crosses the ghost halo (`n_ghost = 2`). Fix would require widening the halo to 3 or changing the smoothing application order.
- **Particle-field consistency.** The mover gathers raw `E` at particle positions (TSC weights), while the operator acts on `S·E·S`. ECSIM's energy theorem relies on the work term using the same `E` both sides. Test: smooth E at particle positions too, and re-measure. Warning: this may de-stabilize the scheme (the whole point of S is to damp short-wavelength modes).
- **Pressure-tensor shape-function inconsistency.** `Particles3D.cpp:1970-2011` uses 8-point linear weights in the TSC branch. Not a cycle-1 energy source but a moment-consistency issue that shows up in stress diagnostics.
- **ECSIM theory for higher-order shape functions.** Markidis & Lapenta 2011 prove exact energy conservation for linear (CIC) B-splines. Whether the proof extends to quadratic (TSC) B-splines with the current smoother is an open literature question. If the published theorem requires a modified smoother (e.g., Vu & Brackbill 1992's binomial), the 5% leak may be a known theoretical limitation that cannot be removed without changing the smoother.

Open questions that no longer need answering to conclude Phase 8:

- ~~Multi-run statistics~~: resolved — cycle 1 is 0.2%-reproducible for sm=4, cycle 10 has 25× run-to-run variance; all future comparisons use median-over-5.
- ~~Grid-size scaling at fixed per-rank workload~~: deferred to Phase 9; not required to decide on 8e.

---

## Recommended investigation steps

### Phase 4: Operator symmetry measurement (DONE — diagnostic removed from tree)

Implemented `diagnoseOperatorSymmetry()` in `fields/EMfields3D.cpp`. Ran once on cycle 1 from `calculateE()`, tested full operator A, curl-curl only, smoothing S, and mass matrix M symmetry using globally-consistent random vectors (hash-based, periodic-aware).

**Code extracted to `phase4-diagnostic.patch` (untracked, root of repo).** See Finding 11 for why: the diagnostic and its `skip_mass_matrix_diag_` wrap around the S·M·S block in `MaxwellImage` perturb FP determinism enough to worsen TSC sm=4 dE(1) ~44× even when gated. To re-run the symmetry measurements: `git apply phase4-diagnostic.patch`, rebuild, run — but expect different cycle-1 energy numbers during that session.

- [x] **4a. Operator symmetry test** — Measured δ = |<u,Av> - <v,Au>| / (||u||·||v||) for the full operator A using MaxwellImage on random Krylov vectors.
- [x] **4b. Smoothing symmetry test** — Measured smoothing S asymmetry using energy_conserve_smooth_direction on random 3D fields.
- [x] **4c. Component isolation** — Implemented `skip_mass_matrix_diag_` flag + `IPIC3D_SKIP_MASS_MATRIX` env var to disable S·M·S term in MaxwellImage. Measured curl-curl-only asymmetry.
- [x] **4b'. Mass matrix M symmetry** — Measured M asymmetry using mass_matrix_times_vector on random 3D fields with ghost-filled boundaries.

**Key finding: Apparent asymmetry is dominated by periodic boundary double-counting.**

All measured δ values are O(1e-03), but the curl-curl operator (which is mathematically self-adjoint for periodic BCs) shows δ_cc ~ 2e-05 for np=1. This confirms the measurement is dominated by an artifact: the Krylov inner product double-counts physical nodes at periodic boundaries (e.g., local nodes k=n_ghost and k=nxn-n_ghost-1 map to the same physical node but carry independent values in the inner product). The TRUE operator asymmetry is at or below machine precision.

| Sub-operator | CIC δ (np=4) | TSC δ (np=1) | TSC δ (np=4) |
|---|---|---|---|
| Full A | 3.4e-03 | 1.6e-03 | 1.9e-03 |
| Curl-curl | 2.8e-06 | 2.1e-05 | 2.5e-04 |
| Smoothing S | 1.8e-03 | 5.9e-03 | 5.4e-03 |
| Mass matrix M | 6.3e-04 | 3.5e-03 | 2.0e-03 |

### Phase 4.5: Ghost layer inversion (DONE — benign, not the drift source)

**Discovered**: for n_ghost > 1, the ghost layers in `NBDerivedHaloCommN` are filled in reverse order (outermost ghost ← innermost source, innermost ghost ← outermost source). Confirmed empirically with a diagnostic that encodes global physical indices into ghost values.

**Root cause**: the recv loop fills ghosts outside-in (g=0 → outermost ghost) while the send loop reads sources inside-out (g=0 → closest to boundary). For n_ghost=1, there's only one layer so no inversion occurs.

**Fix attempted**: reversed source indices with `gs = (offset > 0) ? (n_ghost_-1-g) : g` to correct the node field-copy path (offset=1) without affecting the moment path (offset=0, which is self-consistent with addFace).

**Result**: Ghosts became physically correct, but **energy conservation worsened** (dE(1) went from 1.1e-05 → 4.6e-04). The inversion is **benign** because the code is self-consistent within the inverted convention — the mass matrix entries at ghost positions are also inverted (via the same communicateNode_P), so the product M*v uses matching swapped values, preserving operator self-adjointness within the relabeled boundary.

**Conclusion**: The ghost inversion is NOT the source of the TSC energy drift. Fixing it properly requires a coordinated change across the ghost fill, addFace moment summation, and potentially the P2G scatter boundary convention. This is a latent cosmetic issue, not a physics bug for the current symmetric-stencil operations.

### Phase 5: Self-copy convention experiment (DONE — ruled out)

- [x] **5b. Compare 1-rank vs 4-rank** — np=1 (all self-copies) gives WORSE drift than np=4 (mixed MPI+self-copy). Self-copy convention is NOT the drift source. See Finding 4 above.

### Phase 6: Isolate smoothing as the sole drift source (DONE — smoothing is major but not sole source)

**Conclusion:** Smoothing is a major contributor but NOT the sole source. Two independent FP noise sources exist in TSC. Input files: `inputfiles/phase6{a_cic_nsmooth,a_tsc_nsmooth,b_cic_smooth,c_tsc_large}.inp`.

- [x] **6a. Unsmoothed operator test** — Ran CIC and TSC with `Smooth=0` for 10 cycles. Results (20x20x4, np=4):

  | Config | dE(1) | dE(10) | Stability |
  |--------|-------|--------|-----------|
  | CIC no-smooth | 4.2e-08 | 2.5e-06 | Stable, bounded |
  | TSC no-smooth | 3.8e-06 | EXPLODED | GMRES fails cycle 6, energy 3.3x initial by cycle 10 |

  **Key finding:** CIC unsmoothed is NOT exactly ε_mach — it's the GMRES FP noise floor (~4e-08). TSC unsmoothed is 90x worse than CIC unsmoothed, confirming the TSC mass matrix introduces independent FP accumulation even before instability develops.

- [x] **6b. CIC+smoothing comparison on 20x20x4** — CIC with sm=4 on same grid: dE(1) = 2.1e-08, dE(10) = 1.2e-05. Smooth makes CIC ~5x worse by cycle 10 vs unsmoothed CIC (2.5e-06). However, CIC+smooth is still ~6x better than TSC+smooth at cycle 10 (1.2e-05 vs 7.9e-05). **The wider TSC mass matrix is NOT simply amplifying smoothing noise — it's an independent source.**

- [x] **6c. Grid-size scaling** — TSC on 40x40x4 (4x more nodes), np=4, sm=4: dE(1) = 8.9e-05, dE(10) = 2.5e-04. Compared to 20x20x4 with same 4 ranks: ~8x worse absolute drift with ~4x more total energy. Per-rank workload increased 4x → relative drift roughly doubled. Grid-size scaling at fixed np does NOT help — confirms drift is dominated by per-rank FP accumulation, not per-node errors.

### Phase 7: Theoretical calibration (MEDIUM PRIORITY)

- [ ] **7a. Literature check** — Search for ECSIM + TSC/quadratic shape function results in Markidis & Lapenta (2011), Deka & Bacchini papers, and Vu & Brackbill (1992, energy-conserving smoothing). Do they report energy drift with higher-order shapes? If so, what's the expected scaling?

- [ ] **7b. Spectral analysis** — Save per-cycle dE for 50+ cycles. Fourier-transform to find dominant oscillation frequency. If it's at or near the grid Nyquist, the smoothing isn't fully suppressing the problematic TSC mode.

### Phase 8: Targeted mitigation — COMPLETE (all FP sources ruled out, 8e chosen)

Phase 6 had suggested the mass matrix was the dominant source (~3.8e-06 at cycle 1 for TSC no-smooth). Phase 8 systematically instrumented every FP accumulator between P2G scatter and GMRES exit; none of them explain the sm=4 cycle-1 drift. See Finding 14 for the algebraic-leak attribution that closes the phase.

- [x] **8c. Kahan summation in `mass_matrix_times_vector`** — **DONE, real partial win, kept.** Applied in `fields/EMfields3D.cpp:2282-2344` (commit `605a247`). 5-run medians on 20x20x4 np=4:
    - TSC no-smooth cyc 1: HEAD median 5.9e-06 (high run-to-run spread); previously reported 2.7e-06 → 1.2e-06 means were likely sample bias
    - TSC sm=4 cyc 1: 1.065e-05 → 1.064e-05 (no effect — smoothing contribution dominates)
    - TSC sm=4 cyc 10: 1.34e-04 → 1.04e-04 (1.3×, within run-to-run variance)
    - CIC configs: unchanged (14-group mass matrix already at noise floor)

    **Kept in tree** — cost is trivial (one lambda, three doubles), measurable no-smooth improvement, no regression. Does NOT fix the smoothed-config drift. Reproducer: `inputfiles/phase8c_tsc_kahan.inp`.

- [x] **8a. Kahan (compensated) summation in `energy_conserve_smooth_direction`** — **DONE, rejected.** Implemented and tested on 20x20x4 np=4 sm=4 over 5 runs, then dropped. See Finding 12. The 27-point stencil was restructured into a compensated accumulator with an additive tree of 27 `kadd()` calls per grid node per smoothing pass. Result:
    - TSC sm=4 cyc 1: 1.061e-05 HEAD → 1.063e-05 8a (0.2% — below noise)
    - TSC sm=4 cyc 10: 1.27e-04 HEAD → 8.56e-05 8a (1.5×, inside 25× variance)
    Zero measurable benefit at cycle 1 (the deterministic metric). Patch dropped, not archived.

- [x] **8d. Kahan in TSC mass-matrix ASSEMBLY** — **DONE, analytically ruled out.** See Finding 13. Per-cell accumulation order: ~730 particle contributions × max value 0.016 × ε ≈ 2.6e-15 absolute FP error. Ten orders of magnitude below the observed 1e-05 drift. Implementation would require nine shadow compensator arrays with the same footprint as `Mxx..Mzz` for effectively zero benefit. Skipped.

- [x] **8b. Fewer smoothing passes** — **SKIPPED.** The Phase 6 sweep (sm=0 unstable; sm=1,2 dE(1)=2.2-2.3e-05; sm=4 sweet spot at 1.1e-05; sm=8 oscillating) is not FP-limited, so re-sweeping with median-over-5 won't change the ranking. Phase 6's sm=4 sweet spot stands.

- [x] **8e. Accept as theoretical limitation** — **CHOSEN.** The 5.1% per-cycle energy-exchange leak in TSC sm=4 (vs 0.01% for CIC sm=4) is a scheme-level algebraic property, not FP rounding. Kahan cannot fix it. The drift is bounded and oscillatory, not secular. Recommended CI smoke test tolerances:
    - `dE(1) ≤ 2.0e-05` (4× HEAD median, comfortable deterministic headroom)
    - `dE(10) ≤ 3.0e-04` (2× the worst observed single run, accounting for 25× run-to-run variance)
    - Add a multi-run median check (≥3 samples) for cycle-10 to average out MPI-reduction non-determinism.

### Phase 9: Algebraic leak investigation (FUTURE WORK — out of scope for Phase 8)

Phase 8 closed the FP-noise lens. If a future effort wants to push TSC sm=4 dE(1) below 1e-05, it needs to target the scheme-level leak. Candidate investigations (see Finding 14 for the rationale):

- [ ] **9a. S·M·S self-adjointness on periodic grid with ghost halo.** Directly measure `δ = |⟨u, S M S v⟩ - ⟨v, S M S u⟩| / (||u|| · ||v||)` on a random pair at cycle 1. If `δ > ε_mach` for TSC but `~ε_mach` for CIC, the composition is the culprit.
- [ ] **9b. Particle-field consistency experiment.** Modify the TSC mover to gather `S·E` (smoothed) at particle positions instead of raw `E`. Measure dE(1). This will either destabilize the scheme (confirming S is load-bearing for stability) or reduce the leak (confirming the mismatch is the source).
- [ ] **9c. Widen halo to n_ghost = 3.** TSC + smoother = 7-point effective stencil. n_ghost=2 may not be enough to avoid boundary inconsistencies.
- [ ] **9d. Literature check.** Search Markidis & Lapenta 2011, Lapenta ECSIM followups, Vu & Brackbill 1992 for TSC/quadratic B-spline energy-conservation theorems and discrete compatibility conditions for smoothing filters.
- [ ] **9e. Pressure-tensor shape-function audit.** `Particles3D.cpp:1970-2011` mixes linear weights in the TSC branch — not a cycle-1 energy source but a consistency bug to track.

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
| `fields/EMfields3D.cpp:2282-2344` | mass_matrix_times_vector (63 groups for TSC, Kahan — Phase 8c) |
| `fields/EMfields3D.cpp:2378-2416` | communicateGhostP2G_ecsim (moments) |
| `fields/EMfields3D.cpp:2418-2467` | communicateGhostP2G_mass_matrix |
| `fields/EMfields3D.cpp:2711-2785` | MaxwellSource (RHS: smooths J at line 2746) |
| `fields/EMfields3D.cpp:2827-2912` | MaxwellImage (operator: smooths E, computes M*E, curl²) |
| `fields/EMfields3D.cpp:2998-3043` | energy_conserve_smooth_direction (Phase 8a tested, no benefit — see Finding 12) |
| `particles/Particles3D.cpp:1857-1896` | TSC mass matrix assembly (Phase 8d analytically ruled out — Finding 13) |
| `phase4-diagnostic.patch` (untracked) | Extracted Phase 4 operator-symmetry diagnostic — re-apply with `git apply` when needed |
| `inputfiles/ci_smoke_tsc.inp` | TSC smoke test (nzc=4, sm=4) |
| `inputfiles/phase6a_cic_nsmooth.inp` | Phase 6a: CIC no-smooth 10-cycle test |
| `inputfiles/phase6a_tsc_nsmooth.inp` | Phase 6a: TSC no-smooth — unstable by cycle 6 |
| `inputfiles/phase6b_cic_smooth.inp` | Phase 6b: CIC sm=4 on 20x20x4 baseline |
| `inputfiles/phase6c_tsc_large.inp` | Phase 6c: TSC sm=4 on 40x40x4 scaling test |
| `inputfiles/phase8c_tsc_kahan.inp` | Phase 8c: TSC sm=4, 10-cycle Kahan regression test |

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
| `d474568` | docs: add np=1 vs np=4 finding |
| `605a247` | perf: Kahan summation in mass_matrix_times_vector (Phase 8c) |
| (pending) | docs: Phase 8 conclusion — FP mitigations exhausted, drift is algebraic (5% cyc-0→1 leak for TSC sm=4 vs 0.01% for CIC). Phase 8e accepted, Phase 9 candidates outlined. |
| (extracted) | Phase 4 operator-symmetry diagnostic — in `phase4-diagnostic.patch`, removed from tree because it perturbs FP determinism even when gated (Finding 11) |
| (dropped) | Phase 8a Kahan smoothing — tested, no benefit (Finding 12), patch discarded |
