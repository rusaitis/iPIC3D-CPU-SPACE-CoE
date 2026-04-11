# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-11, Phase 9 COMPLETE — algebraic leak localised, but no easy fix)

**Branch:** `with-petsc`. Committed through `a7c6576` (Phase 8 closure). Phase 9a-c complete this session; no code changes kept from Phase 9 (all experiments reverted after measurement). Phase 4 operator-symmetry diagnostic is in `phase4-diagnostic.patch` (untracked), and was re-applied transiently during Phase 9a to add an S·M·S composition test.

**Phase 9 summary:** The ~5% per-cycle TSC energy leak (vs 0.01% for CIC) is NOT from (a) S·M·S composition asymmetry, (b) a particle-field raw-E/smoothed-operator mismatch, or (c) the num_smoothings setting being suboptimal. The current sm=4 arrangement is at a local optimum — any direction moved (more/less smoothing, raw vs smoothed particle-facing E, extra or fewer passes) makes things strictly worse. The leak is therefore an inherent property of the **ECSIM + TSC + (27-point 8-4-2-1 smoother) combination**, not a bug in any specific implementation choice.

**Key numbers after Phase 9 (5-run medians, 20x20x4 np=4):**
- CIC sm=4: dE(1) = 2.1e-08, cyc 0→1 leak = 0.01% of exchange
- TSC sm=4: dE(1) = 1.06e-05, cyc 0→1 leak = 5.1% of exchange (baseline holds)
- TSC sm=8: dE(1) = **4.83e-06** (2.2× better at cyc 1), dE(10) = **2.25e-03** (20× worse)
- TSC sm=16: dE(1) = **5.87e-06** (1.8× better), dE(10) = **2.39e-03** (20× worse)
- TSC sm=4 + skip Exth smoothing: dE(1) = **9.16e-05** (9× worse), dE(10) = **0.47** (catastrophic — 4300× worse)
- TSC sm=4 + extra Exth smoothing: dE(1) = **2.62e-05** (2.5× worse), dE(10) = **2.34e-03** (20× worse)

**Phase 9a (SMS self-adjointness): ruled out.** Measured `δ = |⟨u, SMS v⟩ - ⟨v, SMS u⟩|/(||u||·||v||)` directly with globally-consistent random vectors. For TSC sm=4: `δ_SMS = 2.25e-03`, actually *smaller* than `δ_M = 2.37e-03` alone — the composition S·M·S does NOT introduce asymmetry beyond the individual factors, if anything S slightly dampens M's boundary artifact. All δ values (2.2e-03 for TSC, 6.9e-04 for CIC) are dominated by periodic-boundary inner-product double-counting, not true operator asymmetry. See Finding 15.

**Phase 9b (particle-field consistency): ruled out.** Discovered that `calculateB()` already smoothes `Exth` in place (`fields/EMfields3D.cpp:2991`) BEFORE `set_fieldForPcls()` copies it — so particles already see `S·E`, matching the operator's S·M·S internal convention. Tested two variations via env-gated bypasses, both catastrophic:
- Skip the in-place smoothing (particles see raw E) → dE(1) = 9.16e-05 (9× worse), dE(10) = 0.47 (exploded)
- Apply extra smoothing (particles see S²·E) → dE(1) = 2.62e-05 (2.5× worse), dE(10) = 2.34e-03
The current setup is near-optimal — the particle-facing smoothing IS load-bearing for energy closure, just not sufficient to fully close the 5% leak. See Finding 16.

**Phase 9c (n_ghost=3 / smoothing sweep): reframed and ruled out.** The n_ghost=3 hypothesis was dropped (grid/comm code is hard-coded to `assert(n_ghost <= 2)` in `grids/Grid3DCU.cpp:45`, and expanding would be a multi-day refactor for a speculative hypothesis). Instead tested num_smoothings ∈ {8, 16} with 5-run stats, confirming Phase 6's sweep ranking: higher sm counts trade cyc-1 wins for cyc-10 secular drift, and sm=4 remains the optimum for any realistic simulation length. See Finding 17.

**Phase 9 conclusion: Phase 8e (accept as theoretical limitation) is confirmed.** The 5% per-cycle leak has been localised to the **S·M·S-operator / smoothed-J-RHS / smoothed-particle-E combination** as a whole — no single factor in that chain is the cause, and the overall arrangement is at a local optimum. A true fix would require reformulating the discrete energy identity for TSC + smoothing, likely by adopting a different smoother (e.g., Vu & Brackbill 1992 binomial) or a different coupling between the mover and the operator. That's Phase 10+ territory; see `## What we don't know` section for candidate directions.

Recommended CI smoke test tolerances for TSC sm=4 on 20x20x4 (unchanged from Phase 8): `dE(1) ≤ 2e-05`, `dE(10) ≤ 3e-04` (with multi-run median ≥ 3).

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

    **Candidate mechanisms investigated in Phase 9:**
    - ~~S·M·S self-adjointness fails on periodic grid with ghost halo~~ — **ruled out in Finding 15.**
    - ~~Particle-field consistency mismatch (raw E at particles vs smoothed in operator)~~ — **discovered the premise was wrong.** See Finding 16: `calculateB()` already smoothes `Exth` in-place, so particles ALREADY see `S·E` matching the operator. Tested skip/extra variants and both are catastrophic.
    - Pressure-tensor shape-function inconsistency — not investigated in Phase 9, remains an open consistency issue but is not a cycle-1 energy source.

15. **Phase 9a: S·M·S composition is NOT more asymmetric than M alone.** Extended Phase 4 diagnostic with a direct `δ_SMS = |⟨u, SMS v⟩ - ⟨v, SMS u⟩| / (||u||·||v||)` measurement at cycle 1 using globally-consistent random vectors (hash-seeded, periodic-aware). Results on 20x20x4 np=4:

    | δ | CIC sm=4 | TSC sm=4 |
    |---|---|---|
    | δ_A (full operator) | 1.39e-04 | 1.70e-03 |
    | δ_cc (curl-curl only) | 1.39e-04 | 1.80e-03 |
    | δ_S (smoothing alone) | 4.11e-04 | 5.46e-03 |
    | δ_M (mass matrix alone) | 1.39e-03 | 2.37e-03 |
    | **δ_SMS (composition)** | **6.87e-04** | **2.25e-03** |

    The SMS composition is actually SLIGHTLY SMALLER than δ_M alone (for both CIC and TSC) — the outer S dampens M's boundary artifact rather than amplifying it. All δ values are dominated by periodic-boundary inner-product double-counting (Finding 5's interpretation stands), not true operator asymmetry. Critically, TSC's 500× energy drift ratio does NOT track the 1.7–13× symmetry ratios. **Operator asymmetry is NOT the source of the algebraic leak.** Phase 4 diagnostic reverted after measurement (Finding 11 side effect).

16. **Phase 9b: particle-field smoothing is load-bearing AND the current setup is near-optimal.** Originally hypothesized that particles see raw E while the operator uses smoothed E (a mismatch). Reading `calculateB()` revealed the premise is wrong: `fields/EMfields3D.cpp:2991` smooths `Exth` in place BEFORE `set_fieldForPcls()` copies it to the mover. Particles ALREADY see `S·E`, matching the operator's S·M·S internal convention.

    Tested two env-gated variations on the particle-facing E (`IPIC3D_PHASE9B_SKIP_ETH_SMOOTH`, `IPIC3D_PHASE9B_EXTRA_ETH_SMOOTH`), 5 runs each:

    | Variation | dE(1) median | dE(10) median | Comment |
    |---|---|---|---|
    | HEAD (particles see S·E) | 1.06e-05 | 1.3e-04 | baseline |
    | Skip Exth smooth (raw E at particles) | 9.16e-05 | **0.47** | catastrophic — energy explodes |
    | Extra smoothing (S²·E at particles) | 2.62e-05 | 2.34e-03 | 2.5× worse at cyc 1, 20× at cyc 10 |

    Both directions make things strictly worse. The current single-pass post-solve smoothing is at a local optimum for particle-facing E. The 5% leak is NOT a "particle-field smoothing count mismatch" — the counts are already consistent between the operator and the mover. Source reverted after measurement.

17. **Phase 9c reframed: num_smoothings sweep (instead of n_ghost=3).** The n_ghost=3 hypothesis was dropped as too invasive (`grids/Grid3DCU.cpp:45` hard-asserts `n_ghost <= 2` and expanding would be a multi-day comms refactor for a speculative hypothesis). Instead, re-ran Phase 6's num_smoothings sweep with 5-run medians to double-check the sm=4 sweet spot:

    | num_smoothings | dE(1) median | dE(10) median | Notes |
    |---|---|---|---|
    | 4 (HEAD) | 1.06e-05 | 1.3e-04 | bounded, oscillatory |
    | 8 | **4.83e-06** | 2.25e-03 | 2.2× better at cyc 1, 20× worse at cyc 10 (secular drift) |
    | 16 | 5.87e-06 | 2.39e-03 | same pattern |

    Confirms Phase 6's non-monotonic shape. High smoothing over-filters: lower cyc-1 drift because less dynamics, higher cyc-10 drift because the filter damps physical modes that would otherwise be energy-conserving. sm=4 is the optimum for any simulation length ≥ ~8 cycles. Input files `inputfiles/phase9_tsc_sm{8,16}.inp` kept for reproducibility.

## What we don't know (after Phase 9 — scheme-level candidates also exhausted)

Phase 9 targeted the three "scheme-level" candidates Phase 8 couldn't distinguish with FP arguments alone. All three are now ruled out:

- ~~**S·M·S composition asymmetry**~~ — Phase 9a (Finding 15): δ_SMS is actually smaller than δ_M alone. Not the source.
- ~~**Particle-field raw/smoothed mismatch**~~ — Phase 9b (Finding 16): particles already see S·E matching the operator. Current setup at local optimum; both skip and extra smoothing are strictly worse.
- ~~**n_ghost=3 halo insufficient**~~ — Phase 9c reframed (Finding 17): n_ghost=3 dropped as too invasive; smoothing sweep shows sm=4 is the genuine optimum (sm=8,16 trade cyc-1 for cyc-10).

**Remaining candidate directions (Phase 10+ territory, NOT investigated):**

- **Phase 10a — Literature review on ECSIM + higher-order shape functions + smoothing (NEXT STEP, see detailed brief below).**
- **Alternative smoothing kernels.** The current smoother is a fixed 27-point (8,4,2,1)-weighted average. Candidates to test: (a) a binomial (1,2,1) separable kernel applied N times, (b) a spectral filter that exactly preserves a subspace of interest, (c) no smoothing on the operator but one-shot smoothing on the solve output. Any of these would change the discrete energy balance; at least one should be tested before concluding the leak is an unavoidable theoretical limit. **Do 10a first** — the right kernel choice is likely already published.
- **Mover / operator basis change.** Currently the solve variable E is "raw" (unsmoothed) and the particles see `S·E`. Alternative: change the solve variable to `E' = S·E` (or `S^2·E`), re-derive the operator in that basis, and let the particles see `E'` directly. This removes the S-conjugation from the operator at the cost of a more complex RHS.
- **Pressure-tensor shape-function audit.** `Particles3D.cpp:1970-2011` mixes 8-point linear weights into the TSC branch. Not a cycle-1 energy source (the mass matrix dominates) but a consistency bug that shows up downstream.

---

### Phase 10a: literature review brief (for a deep-research session)

**The central question.** Does the ECSIM (Energy-Conserving Semi-Implicit Method, Lapenta 2017 / Markidis & Lapenta 2011) discrete energy-conservation theorem continue to hold *exactly* when the particle shape function is quadratic (TSC / B-spline order 2) instead of linear (CIC / B-spline order 1), **and** a spatial smoothing filter is applied symmetrically to both the current density on the RHS and the mass matrix on the LHS of the implicit Maxwell solve? If not, what modifications to the filter or the coupling are required to restore exact (or improved) energy conservation?

**Why this matters in one paragraph.** iPIC3D's TSC implementation conserves energy to 0.01% per cycle with CIC shape functions and drops to a bounded-but-annoying 5% per-cycle energy-exchange leak with TSC + the legacy 27-point (8-4-2-1) smoother at `num_smoothings = 4`. Phases 4–9 of this investigation have ruled out every FP-rounding explanation (Kahan in all three hot loops had ≤1.3× effect) and every single-factor scheme-level explanation (operator asymmetry, particle-field mismatch, smoothing pass count). The leak is therefore a property of the particular ECSIM + TSC + symmetric filter combination. It is plausible that a different filter (e.g. a binomial kernel) or a different application pattern (e.g. filter only the RHS, not the operator) recovers the energy identity. Before writing more code, we want to know which of these directions is already answered in the literature.

**Primary sources to examine (ordered by expected relevance).**

1. **Lapenta, G. (2017). "Exactly energy conserving semi-implicit particle in cell formulation." *J. Comput. Phys.* 334, 349–366.** The foundational ECSIM paper as implemented in iPIC3D. Check:
    - Is the energy-conservation proof written for a *generic* B-spline of order p, or is it specialised to order 1?
    - What assumptions does the proof make about smoothing filters (if any)? Specifically, is there a compatibility condition relating the filter to the shape function?
    - Does §3 or §4 show numerical experiments with shape functions higher than linear?

2. **Markidis, S. & Lapenta, G. (2011). "The energy conserving particle-in-cell method." *J. Comput. Phys.* 230, 7037–7052.** The earlier, slightly different, energy-conserving formulation (pre-ECSIM). Check:
    - How does the 2011 scheme differ from Lapenta 2017 in its treatment of J and the mass matrix? Does the 2011 version extend more naturally to TSC?
    - §2-3: the discrete energy identity derivation. Which step requires linearity of the shape function?
    - Do they discuss smoothing filters? If yes, what compatibility conditions?

3. **Chen, G., Chacón, L., & Barnes, D. C. (2011). "An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm." *J. Comput. Phys.* 230, 7018–7036.** A different implicit family (fully-implicit, nonlinear Jacobian-free Newton-Krylov), also energy-conserving. Check:
    - Do they use higher-order B-splines? What shape function(s)?
    - Do they need a smoothing filter at all? If not, what stability mechanism replaces it?
    - §4: discrete conservation proof — how general is it in the shape-function order?

4. **Chacón, L. & Chen, G. (2016). "A curvilinear, fully implicit, conservative electromagnetic PIC algorithm in multiple dimensions." *J. Comput. Phys.* 316, 578–597.** The electromagnetic (Maxwell-Vlasov) follow-up to Chen et al. 2011. Check:
    - Same shape-function question.
    - §3-4: charge/energy conservation proofs for the EM case.
    - Any remarks on smoothing/filtering in §5 implementation notes.

5. **Vu, H. X. & Brackbill, J. U. (1992). "CELEST1D: An implicit, fully kinetic model for low-frequency, electromagnetic plasma simulation." *Comput. Phys. Commun.* 69, 253–276.** Classic paper on implicit PIC with filtering. Check:
    - The "energy-conserving binomial filter" construction — what kernel do they use, and what is the compatibility condition with the mass matrix?
    - Do they prove energy conservation holds with the filter applied, and under what assumptions on shape-function order?
    - Is the filter applied once to J, or twice (RHS + LHS like iPIC3D does)?

6. **Birdsall, C. K. & Langdon, A. B. (1991, reissued 2004). *Plasma Physics via Computer Simulation.* CRC/Taylor & Francis.** The PIC textbook. Check:
    - Chapter 14 (or equivalent) on digital filtering in PIC: which filters are "energy-safe" and why? What are the trade-offs between filter width and energy drift?
    - Any discussion of higher-order shape functions: are they paired with specific filters?

7. **Deka, P. J. & Bacchini, F. (ECSIM / RelSIM papers, 2023–2025).** These authors implemented the ECSIM and RelSIM code paths in iPIC3D (per `CLAUDE.md`). Check recent arXiv/journal papers by either author for:
    - Any report on TSC/quadratic shape functions in ECSIM. Do they observe the same 5% leak?
    - Any discussion of smoother choice or filter compatibility.
    - Do they recommend n_ghost ≥ 3 or a specific smoothing pattern for higher-order shapes?

8. **Stanier, A., Chacón, L., & Chen, G. (2019). "A fully implicit, conservative, non-linear, electromagnetic hybrid particle-ion/fluid-electron algorithm." *J. Comput. Phys.* 376, 597–616.** Different regime (hybrid) but same authors as #3/4 and same conservation philosophy. Check whether they use TSC and what filter.

**Specific sub-questions the review should answer.**

1. **Is there a published ECSIM or close-relative scheme that explicitly demonstrates exact energy conservation with TSC (quadratic B-spline) shape functions?** If yes: what filter do they use, and what are the discrete compatibility conditions? If no: is there a theoretical argument for why linear shape functions are "special" in the ECSIM energy-conservation proof?

2. **Is the symmetric sandwich `S·M·S` (smoother on both sides of the mass matrix) a *standard* choice, or is it idiosyncratic to iPIC3D?** Alternatives to check in the literature: `S·M` only (smooth the input, not the output), `M·S²` (both filters collapsed on one side), filter the RHS `J` only and leave the operator raw, "binomial compensation" schemes where a high-pass correction is added after filtering.

3. **What filter kernels are known to commute (or nearly commute) with the discrete curl operator and the mass matrix?** Specifically: (a) is the (8,4,2,1) 27-point tensor-product kernel iPIC3D uses actually the right filter for TSC shape functions, or is it a legacy CIC-era choice? (b) Vu & Brackbill's binomial (1,2,1)^n construction — does it have a proof attached, and for which shape-function order?

4. **What is the expected scaling of the energy drift with filter width?** If a filter of width `w` introduces an O(w^p) drift for some p that depends on the shape-function order, we can set expectations numerically and decide whether 5% at sm=4 is "close to theoretical minimum" or "far off the best known result."

5. **Is the right fix a different filter, a different application pattern, or a different formulation of the ECSIM equations entirely** (e.g., solve for a filtered field `E' = S·E` as the primary variable)? The three alternatives each have very different implementation costs.

**Search terms / keywords to plug into Google Scholar, arXiv, ADS, and NASA ADS:**

- `"energy conserving" "particle in cell" "B-spline"`
- `"ECSIM" (TSC OR quadratic OR "higher order")`
- `"implicit PIC" smoothing filter "energy conservation"`
- `"binomial filter" "particle in cell" compensation`
- `Lapenta ECSIM quadratic shape`
- `Markidis Lapenta mass matrix`
- `"S·M·S" OR "sandwich filter" PIC`
- `Chacón Chen Barnes conservative PIC shape function`
- `"charge conserving" "energy conserving" "shape function" implicit PIC`

**What "success" looks like for Phase 10a.** A short (≈ 1-page) written summary that answers:
  (i) whether any published scheme conserves energy exactly for TSC (yes/no + citation),
  (ii) if yes, what modifications they make (filter, application pattern, formulation) relative to what iPIC3D does,
  (iii) if no, what the published best-known drift is and whether 5% is within that envelope,
  (iv) the top-1 or top-2 most promising code experiments implied by the literature (e.g. "switch to a separable binomial filter applied only to J" or "move the smoother to act only on the primary solve variable"), ranked by implementation cost.

With that in hand we can decide between: (A) implement the best published fix; (B) propose a novel smoother-compatibility condition and derive it analytically; or (C) conclude the 5% drift is theoretically optimal for this scheme class and finalise Phase 8e's "accept as limitation" stance with a citation.

---

Open questions from Phase 8 that Phase 9 also did not resolve:

- ~~Operator asymmetry at cycle 1~~: **RESOLVED** — Finding 15 confirms the true asymmetry is at machine precision; measured δ values are boundary-inner-product artifacts.
- ~~Multi-run statistics~~: resolved earlier.
- ~~Grid-size scaling at fixed per-rank workload~~: not attempted; Phase 8 findings (per-rank work, not grid size, drives drift) are consistent with the algebraic-leak picture and don't need further data.

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

### Phase 9: Algebraic leak investigation — COMPLETE (all three targeted candidates ruled out)

Phase 8 closed the FP-noise lens; Phase 9 targeted three scheme-level candidates. Results:

- [x] **9a. S·M·S self-adjointness on periodic grid with ghost halo.** **DONE, ruled out.** Re-applied Phase 4 diagnostic with a new composition test (added ~120 lines to `diagnoseOperatorSymmetry`), ran with `IPIC3D_OPERATOR_DIAG=1`. δ_SMS for TSC sm=4 is 2.25e-03 (smaller than δ_M = 2.37e-03 alone); for CIC is 6.87e-04 (smaller than δ_M = 1.39e-03 alone). The composition does NOT add asymmetry. All δ values are dominated by periodic-boundary inner-product double-counting, not true operator asymmetry. See Finding 15. Diagnostic code reverted after measurement.
- [x] **9b. Particle-field consistency experiment.** **DONE, premise refuted.** Discovered that `calculateB()` already smoothes `Exth` in place before `set_fieldForPcls()` copies it to the mover — particles ALREADY see `S·E`, not raw E. Tested both variations (skip Exth smooth, extra Exth smooth) via env-gates on 5-run TSC sm=4. Both strictly worse: skip → dE(1) 9× worse + dE(10) explodes to 0.47; extra → dE(1) 2.5× worse + dE(10) 20× worse. The current arrangement is at a local optimum and the particle-facing smoothing IS load-bearing for energy closure. See Finding 16. Source reverted after measurement.
- [x] **9c. Widen halo to n_ghost = 3** → reframed as **smoothing sweep with multi-run medians.** n_ghost=3 dropped as too invasive (`grids/Grid3DCU.cpp:45` hard-asserts `n_ghost <= 2`; expanding would be a multi-day comms refactor for a speculative hypothesis). Instead ran 5-run medians at sm={8, 16} to double-check the Phase 6 sweep. Confirmed: sm=4 is the genuine optimum. sm=8 gives dE(1) = 4.83e-06 (2.2× better) but dE(10) = 2.25e-03 (20× worse secular drift); sm=16 similar. See Finding 17. Input files `phase9_tsc_sm{8,16}.inp` kept.
- [ ] **9d → 10a. Literature check.** NOT STARTED — **promoted to Phase 10a as the next concrete action.** See the "Phase 10a: literature review brief" subsection in "What we don't know" above for the full research brief with primary sources, sub-questions, search keywords, and success criteria.
- [ ] **9e. Pressure-tensor shape-function audit.** NOT STARTED — known consistency bug (`Particles3D.cpp:1970-2011` uses linear weights in TSC branch), not a cycle-1 energy source, punted to Phase 10+.

**Phase 9 conclusion:** The 5% per-cycle algebraic leak is a property of the **(ECSIM + TSC + 27-point 8-4-2-1 smoother)** combination as a whole. Every individually-tweakable factor (operator symmetry, particle-facing smoothing count, smoothing pass count) is already at a local optimum. A fix would require reformulating the scheme or replacing the smoother — that is Phase 10 territory and requires a literature review first. Phase 8e (accept as theoretical limitation) stands, now with stronger evidence.

### Phase 9.x: Edge/corner self-copy modular wrapping (DEFERRED)

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
| `inputfiles/phase9_tsc_sm8.inp` | Phase 9c: TSC with num_smoothings=8 (cyc-1 winner, cyc-10 loser) |
| `inputfiles/phase9_tsc_sm16.inp` | Phase 9c: TSC with num_smoothings=16 (same pattern) |

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
| `a7c6576` | docs: Phase 8 conclusion — FP mitigations exhausted, drift is algebraic |
| (pending) | docs: Phase 9 conclusion — SMS symmetry, particle-field consistency, smoothing count all ruled out as leak sources. Phase 8e holds. |
| (extracted) | Phase 4 operator-symmetry diagnostic — in `phase4-diagnostic.patch`, removed from tree because it perturbs FP determinism even when gated (Finding 11). Re-applied transiently in Phase 9a with a new δ_SMS test, then reverted. |
| (dropped) | Phase 8a Kahan smoothing — tested, no benefit (Finding 12), patch discarded |
| (dropped) | Phase 9b env-gated Exth smoothing skip/extra — tested (Finding 16), both catastrophic, reverted |
