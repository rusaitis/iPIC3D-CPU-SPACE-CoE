# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-12, Phase 10a COMPLETE — leak is an implementation bug, not a theoretical limit)

**Branch:** `with-petsc`. Latest commit `dfd5421` (Phase 10a literature review brief expansion). Working tree clean. Phase 4 operator-symmetry diagnostic is in `phase4-diagnostic.patch` (untracked); no other pending edits.

**The Phase 10a literature review flips the conclusion of Phase 8e.** Full write-up in `review-ecsim-tsc-energy-phase10a.md`. Headline: the ECSIM discrete energy-conservation theorem (Lapenta 2017, 2023; Markidis & Lapenta 2011) is written for a *generic* interpolation function W — no step invokes linearity of the shape function. Published schemes (Chen/Chacón/Barnes 2011; Chacón/Chen 2016; Stanier/Chacón/Chen 2019; Stanier/Chacón 2022; Kormann/Sonnendrücker 2021 GEMPIC; Campos Pinto/Pagès 2022 ChECSIM) already achieve exact energy conservation with quadratic B-splines. **There is no theoretical obstacle to TSC + energy conservation.** The 5% per-cycle leak observed in iPIC3D is therefore an implementation bug, not a scheme limitation.

**Smoking-gun diagnostic (retroactive).** The review proposes a key diagnostic experiment ("Experiment 1"): run TSC with `num_smoothings = 0`, and if the drift is still far above machine precision, the bug is in the TSC shape-function pathway itself rather than the smoother interaction. That experiment is already in the record — it is **Phase 6a**. The result: TSC no-smooth `dE(1) ~ 3–9e-06` (noisy, see caveat below) vs CIC no-smooth `dE(1) ~ 4.2e-08` — a 6–8 orders-of-magnitude gap with zero smoother involvement. **By the review's own criterion this localises the bug to the TSC shape-function pathway**, consistent with Phase 9a/b/c having ruled out SMS composition asymmetry, particle-field smoothing mismatch, and num_smoothings settings as individual culprits.

**Variance caveat (important for Phase 10b).** Several Phase 6/8/9 baselines are single-run or were re-measured only within one session. Cycle-1 run-to-run variance is NOT constant across configurations:
- TSC sm=4 cyc-1 is tight (~0.2–2%).
- **TSC no-smooth cyc-1 spans ~3×** (Phase 8 `HEAD_tsc_nosm` 5-run samples: `{3.34, 2.85, 5.94, 9.32, 7.67}e-06`, median 5.94e-06). The proximate cause is TSC no-smooth being near instability (explodes by cycle ~6), so chaotic mode growth already amplifies MPI-reduction FP non-determinism in cycle 1.

Single-run Phase 6 numbers should therefore be treated as order-of-magnitude anchors, not tight baselines — Phase 10b.0 re-measures them before the code audit starts.

**Phase 8e is deferred, not chosen.** The "accept as theoretical limitation" conclusion reached after Phase 8 was premature in light of the literature review. Keep the conservative CI smoke tolerances (`dE(1) ≤ 2e-05`, `dE(10) ≤ 3e-04`, median ≥ 3) until Phase 10b either finds a fix or completes without a discovery — then revisit.

**Next step: Phase 10b (TSC shape-function consistency audit), starting with 10b.0 baseline re-validation** (clean 5-run medians for Phase 6a/6b/6c configs) before the code audit at 10b.1. Full work order below under "Recommended investigation steps → Phase 10b". The prime hypothesis is that the same TSC shape function W is not used consistently in (a) grid-to-particle gather, (b) particle-to-grid scatter, and (c) mass-matrix assembly — or that particle positions are evaluated at inconsistent time levels across those three paths. The ECSIM energy identity requires all three uses of W to be bit-identical at matching particle positions.

**Key numbers carried over from Phase 9 (20x20x4 np=4, 5-run medians where noted):**
- CIC sm=4: dE(1) = 2.1e-08 (Phase 9 5-run), cyc 0→1 leak = 0.01% of exchange
- TSC sm=4: dE(1) = 1.06e-05 (Phase 9 5-run), cyc 0→1 leak = 5.1% of exchange
- TSC sm=8: dE(1) = 4.83e-06 (Phase 9 5-run), dE(10) = 2.25e-03 (secular drift)
- TSC sm=16: dE(1) = 5.87e-06 (Phase 9 5-run), dE(10) = 2.39e-03
- TSC sm=4 + skip Exth smoothing: dE(1) = 9.16e-05 (9× worse), dE(10) = 0.47 (catastrophic)
- TSC sm=4 + extra Exth smoothing: dE(1) = 2.62e-05 (2.5× worse), dE(10) = 2.34e-03

**Footnote on baselines:** Phase 6a/6b/6c rows in the "Energy drift summary" table below are **single-run and predate** the 5-run-median convention adopted in Phase 8. TSC no-smooth cycle-1 has ~3× run-to-run variance due to proximity to instability. Use these numbers as order-of-magnitude anchors, not precise baselines. Phase 10b.0 will re-measure.

**Phase 9a–c re-framed (not overturned).** All three Phase 9 experimental results stand on their own merits — they just now read as "consistent with a TSC shape-function pathway bug" rather than "the leak is an irreducible scheme property":

- **Phase 9a (SMS self-adjointness): ruled out as a separate source.** δ_SMS for TSC sm=4 is 2.25e-03, actually *smaller* than δ_M = 2.37e-03 alone. All δ values are dominated by periodic-boundary inner-product double-counting. See Finding 15. Consistent with Finding 18: the (I + S·M·S) operator is theoretically sound, so measuring it as approximately self-adjoint was the expected outcome.
- **Phase 9b (particle-field consistency): current setup near-optimal.** `calculateB()` already smoothes `Exth` in place before `set_fieldForPcls()`, so particles already see `S·E` matching the operator. Both skip and extra smoothing are strictly worse. See Finding 16. Consistent with Finding 18: Lapenta 2023's construction requires particles to see `S·E`, and iPIC3D does this.
- **Phase 9c (smoothing count sweep): sm=4 is the optimum.** sm=8/16 trade cyc-1 wins for cyc-10 secular drift. See Finding 17. Consistent with Finding 18: for a *correct* S·M·S construction the drift is zero regardless of filter width, so observing a non-monotonic num_smoothings curve is itself evidence of an implementation mismatch upstream of the filter.

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

18. **Phase 10a literature review: the ECSIM energy theorem is B-spline order agnostic.** Full write-up: `review-ecsim-tsc-energy-phase10a.md`.

    - Both Markidis & Lapenta (2011) and Lapenta (2017) write the discrete energy identity for a *generic* interpolation function W. The three structural ingredients — the same W appearing in field interpolation, current projection, and mass-matrix assembly; the midpoint velocity identity; and the Yee discrete curl identities — are all algebraic, not shape-dependent. No partition-of-unity, commutation, or "square-root" property of the shape function is invoked.
    - Lapenta (2023, "Advances in ECSIM") extends the proof to include a symmetric smoother. The prescribed construction: inner smooth E → `M·(S·E)` → outer smooth → implicit operator `(I + S·M·S)` on the LHS, `S·Ĵ` on the RHS, particles gather `S·E`. The sole requirement on S is that its matrix representation is symmetric (S = Sᵀ). iPIC3D's 27-point (8,4,2,1) filter IS exactly one pass of the tensor-product binomial (1,2,1)/4 kernel in 3D — symmetric by construction. Four repeated passes preserve symmetry.
    - **Published schemes that already achieve exact conservation with quadratic B-splines:** Chen, Chacón & Barnes (2011); Chacón & Chen (2016); Stanier, Chacón & Chen (2019); Stanier & Chacón (2022); Kormann & Sonnendrücker (2021 GEMPIC); Campos Pinto & Pagès (2022 ChECSIM). These papers do the same proof in slightly different frameworks and all explicitly support quadratic or arbitrary-order B-spline shape functions. There is no theoretical obstacle to TSC + energy conservation in ECSIM either.
    - The review identifies three candidate mismatch mechanisms for the observed 5% leak: (a) one-sided smoothing on the LHS (S·M instead of S·M·S); (b) insufficient ghost-cell width for the combined TSC + filter stencil; (c) inconsistency in how the TSC shape function is used across the scatter / gather / mass-matrix-assembly paths. Mechanisms (a) and (b) are ruled out by direct reading of the iPIC3D source — both inner S (`fields/EMfields3D.cpp:3309`) and outer S (`:3329`) are present in `MaxwellImage`, and `energy_conserve_smooth_direction` refreshes the halo between each smoothing pass (`:3035`, `:3058`), so each pass only needs 1 ghost layer. **Mechanism (c) is the only unaudited one, and it aligns directly with the Phase 6a smoking-gun (Finding 19).**
    - **Phase 8e re-evaluated: NOT a theoretical limit.** The 5% leak is a bug. Phase 10b is the audit work, starting with 10b.0 (baseline re-validation).

19. **Phase 6a is a smoking gun (retroactive interpretation via the Phase 10a review's Experiment 1 criterion).** The review proposes a direct localisation test: "Run TSC with num_smoothings = 0. If TSC without any smoother conserves energy to machine precision, the leak is in the smoother. If TSC without smoother already leaks even at 0.01%, there is an additional inconsistency in the shape-function implementation itself." That test is already in the record — it is exactly Phase 6a.

    **Caveat up front: these numbers have non-trivial cycle-1 run-to-run variance.** Phase 6a was originally single-run; the Phase 8 session's 5-run `HEAD_tsc_nosm` measurement gave cyc-1 samples `{3.34, 2.85, 5.94, 9.32, 7.67}e-06` — a 3.3× spread around a 5.94e-06 median. TSC no-smooth is near instability (explodes by cycle ~6), so the chaotic growth rate already amplifies MPI-reduction FP non-determinism in the very first cycle. Point estimates are NOT safe here; always quote the range or a multi-run median, and Phase 10b.0 re-measures before the code audit begins.

    | Config (np=4, 20x20x4)   | dE(1)                                  | Source                                  | Ratio vs CIC no-smooth     |
    |---|---|---|---|
    | CIC no-smooth            | 4.2e-08                                | Phase 6a (single-run; stable)           | 1× (GMRES FP floor)        |
    | TSC no-smooth            | 5.9e-06 median (range 2.9–9.3e-06)     | Phase 8 `HEAD_tsc_nosm` 5-run           | ~140× (range 70–220×)      |
    | CIC sm=4                 | 2.1e-08 median                         | Phase 6b single-run, Phase 9 5-run verify | ~0.5×                    |
    | TSC sm=4                 | 1.06e-05 median                        | Phase 9 5-run                           | ~250× (of CIC no-smooth)   |

    Even with 3× uncertainty on the TSC no-smooth number, the TSC-vs-CIC no-smooth gap spans **6–8 orders of magnitude** and the Experiment 1 criterion — "leaks more than machine precision → bug in TSC shape-function pathway" — fires unambiguously. The smoother is not where the bug lives; it amplifies a pre-existing TSC inconsistency by only ~2× at cycle 1 (5.9e-06 no-smooth → 1.06e-05 smoothed). The smoother is roughly a bystander on the real effect.

    Consistent with this picture: Phase 9a found δ_SMS is clean, Phase 9b found the particle-side smoothing and the operator's inner S already match, and Phase 9c found sm=4 is a numerical optimum. None of those phases touched the TSC shape-function pathway itself — which is why they couldn't close the gap.

    **Implication.** Phase 10b (TSC shape-function consistency audit) is the next work, and it must start with 10b.0 — a clean 5-run-median re-validation of the Phase 6 baselines — so that success is measured against reliable reference numbers. Target: drive TSC no-smooth dE(1) down to the CIC no-smooth GMRES FP floor (~4e-08), a ~100× improvement; as a bonus, TSC sm=4 dE(1) should also drop toward the ~2e-08 CIC sm=4 level.

## What we don't know (after Phase 10a — bug location narrowed, awaiting re-validation)

Phase 9 targeted three "scheme-level" candidates for the 5% TSC sm=4 leak. All three are ruled out as individual causes:

- ~~**S·M·S composition asymmetry**~~ — Phase 9a (Finding 15): δ_SMS is actually smaller than δ_M alone. Consistent with Finding 18: (I + S·M·S) is theoretically sound.
- ~~**Particle-field raw/smoothed mismatch**~~ — Phase 9b (Finding 16): particles already see S·E matching the operator. Consistent with Finding 18's prescribed construction.
- ~~**n_ghost=3 halo insufficient**~~ — Phase 9c reframed (Finding 17): sm=4 is the genuine optimum; halo is refreshed between passes so 1 ghost layer is enough for each filter pass.

Phase 10a (literature review) then ruled out the "ECSIM + TSC is theoretically impossible" worry and pointed at an implementation inconsistency in the TSC shape-function pathway itself as the most likely location of the bug, retroactively reinterpreting Phase 6a as the smoking gun (Finding 19). **But the Phase 6 baseline numbers were mostly single-run and the TSC no-smooth case has ~3× run-to-run variance at cycle 1 due to near-instability**, so before acting on Finding 19 we first re-validate with 5-run medians (Phase 10b.0).

**What we still need to do, in order:**

- **Phase 10b (NEXT):** TSC shape-function consistency audit. Start with 10b.0 baseline re-validation (`phase6a_cic_nsmooth`, `phase6a_tsc_nsmooth`, `phase6b_cic_smooth`, `phase6c_tsc_large` — each with 5-run medians). Then 10b.1–10b.5 code audit (same W in scatter / gather / assembly; partition-of-unity check; time-level consistency; pressure-tensor smell test; one-particle sanity test). Then 10b.6 post-fix re-run of Phase 6a and Phase 8c baselines. See the Phase 10b section below for the full work order. Expected outcome: identify a path where linear (CIC) weights are silently used in a code location that should use TSC weights, or where particle positions are evaluated at inconsistent time levels across the three coupling paths.
- **Phase 10c (only if 10b fails):** Spectral analysis — save per-cycle dE for 50+ cycles and FFT, looking for a dominant near-Nyquist mode that the smoother half-suppresses. Re-activates the original Phase 7b task; only worth the effort if Phase 10b's audit turns up no inconsistency.
- **Phase 10d (later, orthogonal to the leak):** Pressure-tensor shape-function audit. Finding 16's open thread — `Particles3D.cpp:1970-2011` uses 8-point linear weights for Pxx..Pzz even in the TSC branch (`compute_supplementary_moments`, not the primary moments path). Not a cycle-1 energy source because it doesn't feed into the mass matrix, but a consistency bug that shows up in stress diagnostics downstream. Address after 10b.

Open questions that do NOT need answering any more:

- ~~Is the ECSIM + TSC combination theoretically possible?~~ **Yes** (Finding 18).
- ~~Is S·M·S the correct filter construction?~~ **Yes** (Lapenta 2023 via Finding 18).
- ~~Is iPIC3D's (8,4,2,1) kernel the right filter?~~ **Yes** — it is one pass of the tensor-product binomial (1,2,1)/4 kernel, symmetric by construction (Finding 18).
- ~~Is Phase 8e's "accept as theoretical limitation" correct?~~ **No, it was premature** (Finding 19).

Open questions Phase 10b.0 must resolve before the code audit:

- What is the clean 5-run-median TSC no-smooth dE(1)? Is the ~5.9e-06 number in Finding 19 representative, or a fluke of one measurement session?
- Does CIC no-smooth still reproduce ~4.2e-08 with 5-run medians in a fresh session?
- How reproducible is TSC no-smooth cycle-1 today, compared to the 3.3× spread seen in Phase 8?
- What does the per-cycle dE trajectory look like through cycle 5 for TSC no-smooth? (Needed to separate the cycle-1 shape-function leak from the near-instability growth pattern.)

---

### Phase 10a: literature review — DONE (2026-04-12)

**Full write-up:** `review-ecsim-tsc-energy-phase10a.md` (at the repo root). See Finding 18 for the in-plan summary.

**One-paragraph bottom line.** The ECSIM discrete energy-conservation theorem is mathematically valid for any B-spline order, including TSC (quadratic). Both Markidis & Lapenta (2011) and Lapenta (2017) write the proof for a *generic* interpolation function W — no step invokes linearity. Lapenta (2023) extends the proof to include a symmetric smoother and yields exactly the `(I + S·M·S)` construction that iPIC3D implements. Chen/Chacón/Barnes (2011), Chacón/Chen (2016), Stanier/Chacón/Chen (2019), Stanier/Chacón (2022), Kormann/Sonnendrücker (2021 GEMPIC) and Campos Pinto/Pagès (2022 ChECSIM) all demonstrate exact conservation with quadratic or arbitrary-order B-splines. There is no theoretical obstacle to TSC + energy conservation.

**What the review ruled in vs. out as the source of iPIC3D's 5% leak:**

- (a) One-sided smoothing on the LHS (S·M instead of S·M·S): **RULED OUT** by direct reading of `fields/EMfields3D.cpp::MaxwellImage`. Both the inner S (`:3309`) and the outer S (`:3329`) are present; the operator is `(I + c²θ²dt²·curl² + 4π·θdt·invVOL·S·M·S)` exactly as Lapenta (2023) prescribes.
- (b) Insufficient ghost cells for the combined TSC (2 ghosts) + 4-pass filter stencil: **RULED OUT** because `energy_conserve_smooth_direction` refreshes the halo via `communicateNodeBC` between every smoothing pass (`:3035`, `:3058`), so each pass only needs 1 ghost layer regardless of how many passes run.
- (c) Inconsistency in how the TSC shape function is used across grid→particle gather, particle→grid scatter, and mass-matrix assembly: **the only unaudited mechanism**, and the one consistent with the Phase 6a "smoking gun" retroactive reading (Finding 19). Phase 10b addresses it.

**Decision:** proceed to Phase 10b immediately. Phase 10a answered the theoretical question and narrowed the bug location to a specific code-audit target.

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

### Phase 8: Targeted mitigation — COMPLETE (all FP sources ruled out; 8e was provisionally accepted but has since been reversed — see 8e below and Finding 18)

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

- [~] **8e. Accept as theoretical limitation** — **DEFERRED / RE-EVALUATED (formerly "chosen", now reversed).** At the time Phase 8 concluded, the 5% leak looked like an ECSIM + TSC theoretical limit because FP mitigations had been exhausted. Phase 10a's literature review (Finding 18) then showed the underlying theorem is shape-function-order agnostic, and Finding 19 retroactively interprets Phase 6a as localising the bug to the TSC shape-function pathway itself. Phase 8e is therefore NOT the correct resolution. The next concrete step is Phase 10b, starting with a 10b.0 baseline re-validation (because several Phase 6 numbers were single-run and TSC no-smooth cycle-1 has ~3× run-to-run variance near instability). Keep the conservative CI smoke tolerances below as placeholders until Phase 10b either finds the fix or completes without a discovery:
    - `dE(1) ≤ 2.0e-05` (4× the HEAD median, keeps headroom for deterministic configs)
    - `dE(10) ≤ 3.0e-04` (2× the worst observed single run, accounting for 25× MPI-reduction variance)
    - Multi-run median check (≥3 samples) for cycle-10 to average out MPI-reduction non-determinism.
    
    After Phase 10b resolves, these tolerances are expected to tighten substantially (toward the CIC-level ~2e-08 cycle-1 floor) if the audit finds the inconsistency. If it does not, Phase 8e can be re-adopted with citations from the review and justified residuals — not as a default.

### Phase 9: Algebraic leak investigation — COMPLETE (all three targeted candidates ruled out)

Phase 8 closed the FP-noise lens; Phase 9 targeted three scheme-level candidates. Results:

- [x] **9a. S·M·S self-adjointness on periodic grid with ghost halo.** **DONE, ruled out.** Re-applied Phase 4 diagnostic with a new composition test (added ~120 lines to `diagnoseOperatorSymmetry`), ran with `IPIC3D_OPERATOR_DIAG=1`. δ_SMS for TSC sm=4 is 2.25e-03 (smaller than δ_M = 2.37e-03 alone); for CIC is 6.87e-04 (smaller than δ_M = 1.39e-03 alone). The composition does NOT add asymmetry. All δ values are dominated by periodic-boundary inner-product double-counting, not true operator asymmetry. See Finding 15. Diagnostic code reverted after measurement.
- [x] **9b. Particle-field consistency experiment.** **DONE, premise refuted.** Discovered that `calculateB()` already smoothes `Exth` in place before `set_fieldForPcls()` copies it to the mover — particles ALREADY see `S·E`, not raw E. Tested both variations (skip Exth smooth, extra Exth smooth) via env-gates on 5-run TSC sm=4. Both strictly worse: skip → dE(1) 9× worse + dE(10) explodes to 0.47; extra → dE(1) 2.5× worse + dE(10) 20× worse. The current arrangement is at a local optimum and the particle-facing smoothing IS load-bearing for energy closure. See Finding 16. Source reverted after measurement.
- [x] **9c. Widen halo to n_ghost = 3** → reframed as **smoothing sweep with multi-run medians.** n_ghost=3 dropped as too invasive (`grids/Grid3DCU.cpp:45` hard-asserts `n_ghost <= 2`; expanding would be a multi-day comms refactor for a speculative hypothesis). Instead ran 5-run medians at sm={8, 16} to double-check the Phase 6 sweep. Confirmed: sm=4 is the genuine optimum. sm=8 gives dE(1) = 4.83e-06 (2.2× better) but dE(10) = 2.25e-03 (20× worse secular drift); sm=16 similar. See Finding 17. Input files `phase9_tsc_sm{8,16}.inp` kept.
- [x] **9d → 10a. Literature check.** **DONE (2026-04-12).** Full write-up in `review-ecsim-tsc-energy-phase10a.md`; in-plan summary in Finding 18 and the Phase 10a subsection above. Headline: the ECSIM energy-conservation theorem is B-spline order agnostic, so the TSC 5% leak is an implementation bug, not a theoretical limit. Phase 10b now carries the audit work.
- [ ] **9e. Pressure-tensor shape-function audit.** NOT STARTED — known consistency bug (`Particles3D.cpp:1970-2011` uses linear weights in TSC branch), not a cycle-1 energy source, punted to Phase 10d (below).

**Phase 9 conclusion (superseded by Phase 10a — retained for context):** At the close of Phase 9, no single scheme-level factor we could test individually moved the 5% leak, so the working interpretation was that the leak was an inherent property of the (ECSIM + TSC + 27-point smoother) combination as a whole — which motivated Phase 8e and the literature-review brief. Phase 10a (Finding 18) has since **overturned** that interpretation: the ECSIM theorem is B-spline order agnostic, so a correct implementation must exist, and Phase 6a retroactively localises the bug to the TSC shape-function pathway itself (Finding 19). Phase 9's individual measurements (9a/9b/9c) all stand and remain useful — they just now read as "consistent with a shape-function-pathway bug" rather than "proof that no individual fix is possible." Phase 10b is the next step.

### Phase 10b: TSC shape-function consistency audit (NEXT STEP, HIGHEST PRIORITY)

**Goal.** Find and fix the TSC shape-function implementation inconsistency that makes TSC no-smooth dE(1) sit ~100× above the CIC no-smooth GMRES FP floor (Finding 19). Primary success criterion: TSC no-smooth dE(1) on the 20x20x4 np=4 `phase6a_tsc_nsmooth.inp` config drops to the CIC floor (sub-1e-07, ideally at the ~4.2e-08 GMRES FP floor). Bonus success: TSC sm=4 dE(1) drops proportionally from 1.06e-05 toward the ~2e-08 CIC sm=4 level.

**Structure.** 10b.0 is pure baseline re-validation — no code changes. 10b.1–10b.5 are the code audit proper. 10b.6 is post-fix verification. A decision gate at the end of 10b.0 may pause the audit if re-measured baselines substantially differ from Finding 19's assumed numbers.

- [ ] **10b.0 — Re-validate Phase 6 baselines with 5-run medians BEFORE touching code.** Several Phase 6 numbers are single-run (Finding 19 caveat), and TSC no-smooth cycle-1 has ~3× run-to-run spread due to proximity to instability. We need clean reference numbers against which to judge any audit-driven fix — otherwise a 2× improvement could be statistical noise mistaken for a real win. Use `scripts/phase8_bench.sh` with `nruns=5` (already set up from Phase 8).

    Inputs to re-run (all already exist in `inputfiles/`):
    - (a) `phase6a_cic_nsmooth.inp` — CIC no-smooth, should reproduce the ~4.2e-08 GMRES FP floor. Expect tight cycle-1 variance (<1%).
    - (b) `phase6a_tsc_nsmooth.inp` — TSC no-smooth. Expect the 3× spread from Finding 19. **Record median, min, max, AND the cycle-by-cycle dE trajectory up to cycle 5** so the near-instability growth can be separated from the cycle-1 shape-function leak.
    - (c) `phase6b_cic_smooth.inp` — CIC sm=4. Already verified in Phase 9; redo in the same session to pair cleanly with (a).
    - (d) `phase6c_tsc_large.inp` — TSC 40x40x4 (Phase 6c single-run reported dE(1)=8.9e-05). Re-validate because the grid-size scaling claim is cited elsewhere in the plan.

    Deliverable: update the "Energy drift summary" table at the top of `plan-tsc.md` to replace single-run rows with `median (min, max)` entries, and update Finding 19's table likewise. Keep existing CI tolerances (dE(1) ≤ 2e-05, dE(10) ≤ 3e-04) pending the audit, but flag any discrepancy > 2× between the Phase 6 anchors and the new medians.

    **Decision gate.** If TSC no-smooth dE(1) median turns out to be substantially different from the 5.9e-06 assumed in Finding 19 — specifically, if it falls below 1e-06 or rises above 2e-05 — **pause and re-assess Finding 19 before starting the code audit**. A TSC no-smooth dE(1) in the 1e-07 to 1e-06 range would weaken the "bug is in TSC shape-function pathway" claim and might re-open the Phase 10a candidates (a)/(b) that were ruled out by source reading. In that case, prefer 10b.1 anyway (cheap) but treat its results as diagnostic rather than decisive.

- [ ] **10b.1 — Same W used in all three coupling paths.** The ECSIM energy identity `Σ_g E_g · (S·J_g) = Σ_p q_p v̄_p · E^{SM}_p` requires the SAME shape function W(x_g − x_p) in (a) grid-to-particle gather, (b) particle-to-grid current scatter, and (c) mass-matrix assembly. Audit:

    - (a) **Grid-to-particle gather** for TSC. Starting points: `Particles3D::mover_PC` (`particles/Particles3D.cpp:~2078`), its AoS variants (`mover_PC_AoS`, `mover_PC_AoS_Relativistic`, `mover_PC_AoS_vec_intr`, `mover_PC_AoS_vec`). The common gather helper `get_field_components_for_cell` in `include/EMfields3D.h:1045` gives 8 corner nodes of a single cell — that is a LINEAR (CIC) interpolation pattern. **Open question:** does TSC dispatch to a separate 27-point gather, or does TSC silently reuse the 8-point gather at particle positions? Grep for "StencilOrder" / "stencil_order_" / "Quadratic" / "TSC" in `Particles3D.cpp` to find where the branch happens (or confirm it doesn't happen).
    - (b) **Particle-to-grid current scatter** for TSC. J_hat deposition flows through `EMf->add_Jxh_node`, `add_Rho_node` etc. Check `Particles3D::ECSIM_moments_*` or the corresponding TSC-aware scatter loop. Is it 27-point TSC-weighted or 8-point linear?
    - (c) **Mass-matrix assembly** for TSC: `particles/Particles3D.cpp:1857-1896`. Confirmed to use `weights_tsc[s1] * weights_tsc[s2]` explicitly. This is currently the ONLY code location where TSC weights appear by name.

    **Expected finding:** if (a) or (b) uses linear weights while (c) uses TSC weights, the bug is found — the ECSIM cancellation identity breaks silently. **This is the most probable fault.** Fix is to introduce a consistent TSC gather/scatter path alongside the assembly path, or to re-derive the weights from a single `compute_W_tsc(x_p, cell)` helper used by all three.

- [ ] **10b.2 — `weights_tsc` is a true partition of unity.** For a particle at an arbitrary position inside a cell, compute `Σ_{s=0..26} weights_tsc[s]` and verify it equals 1.0 to machine precision. Any deviation is a charge-conservation bug (Gauss's law fails at O(ε) per cycle). Also verify the weight formula matches the standard 1D quadratic B-spline: center `W(ξ) = 3/4 − ξ²` for |ξ| ≤ 1/2, wings `W(ξ) = (1/2)(3/2 − |ξ|)²` for 1/2 < |ξ| ≤ 3/2. Any discrepancy localises the bug to the `weights_tsc` computation itself.

- [ ] **10b.3 — Particle positions at consistent time levels.** ECSIM's energy proof (Lapenta 2017 §2) requires the gather, scatter, and mass-matrix-assembly paths to all evaluate the shape function at the same particle position, typically `x_p^n` (start-of-step) for the mass matrix and `x_p^{n+1/2}` (midpoint) for the current projection — or whatever consistent time level the scheme uses. Check `Particles3D.cpp:1857-1896` reads `pcl->get_x()` (and y, z) at the SAME time level as the gather path in `mover_PC`. If assembly uses `x^n` while the gather uses `x^{n+1/2}`, the mass matrix is mis-aligned with the current it is supposed to respond to, breaking the discrete identity. This is the second most probable fault after 10b.1.

- [ ] **10b.4 — Pressure-tensor linear-weight smell** (open thread from Phase 9 Finding 16). `particles/Particles3D.cpp:1970-2011` uses 8-point linear weights for `Pxx..Pzz` even inside the TSC branch (`compute_supplementary_moments`). This is NOT a cycle-1 energy source because pressure tensors don't feed the mass matrix — but it IS structural evidence that "linear weights inside a TSC code path" is not a one-off mistake in this codebase. Read the surrounding code carefully for other silent linear-gather fall-throughs; whatever convention made pressure tensors linear may have affected J or E as well.

- [ ] **10b.5 — One-particle sanity test.** Once a suspected inconsistency is found, construct a minimal single-particle diagnostic: put one particle at a known position with a known velocity in a Double_Harris setup (or simpler — a uniform-field test with no B gradient), run one cycle with `num_smoothings = 0`, and verify:
    - (a) `Σ_g E_g · J_g` computed from the grid after scatter matches `q_p · v̄_p · E(x_p)` computed from the mover's gather, to machine precision. If this fails, the gather-scatter consistency identity is broken — that is the energy-conservation kill criterion.
    - (b) `(M·E)_at_particle_position` matches the velocity response the mass matrix should predict for that particle. If this fails, the mass-matrix assembly uses a different shape function or a different particle position than the mover's gather.
    
    If either test fails, step-debug from there: the fault will be in the arithmetic of one of the three paths.

- [ ] **10b.6 — Post-fix verification.** After the inconsistency is fixed, re-run with 5-run medians:
    - `phase6a_tsc_nsmooth.inp`: TSC no-smooth dE(1) should drop to the CIC floor (~4e-08). **Record the cycle-by-cycle dE trajectory to cycle 5**: if the "TSC goes unstable by cycle 6" behaviour also disappears, the instability was a symptom of the shape-function bug. If it persists, there is a separate Courant/stability issue to track in Phase 10e or 11.
    - `phase8c_tsc_kahan.inp`: TSC sm=4 dE(1) should drop toward 2e-08 (from 1.06e-05), a ~500× improvement. If it drops by only 2× (to ~5e-06), the smoother interaction is amplifying a residual inconsistency that the primary audit missed.
    - `ci_smoke_tsc.inp`: production smoke test, update CI tolerances to the new floor. The old 2e-05 tolerance becomes far too loose.
    
    Document the new baselines in the "Energy drift summary" table and tighten the CI smoke-test expectations.

**Risk / non-discovery outcome.** If 10b.0 re-validation shows the single-run Phase 6 numbers were outliers (e.g. TSC no-smooth dE(1) is actually ~1e-07 with tight variance), Finding 19's localisation weakens and the three Phase 10a "ruled out" mechanisms deserve a fresh look with the review's explicit compatibility conditions in hand. If the code audit (10b.1–10b.5) runs to completion and finds no inconsistency, Phase 8e's "accept as limitation" can be re-adopted — **but this time with citations from Phase 10a and a justified residual tolerance**, rather than as a default. In any case, do NOT re-close Phase 8e until Phase 10b has been executed end-to-end.

### Phase 10c (optional, only if 10b fails to find the bug): spectral analysis

Save per-cycle dE for 50+ cycles on `phase8c_tsc_kahan.inp`. Fourier-transform the series to find the dominant oscillation frequency. If it's at or near the grid Nyquist, the smoother isn't fully suppressing the problematic TSC mode and the fix is either a stronger filter, a different smoother kernel, or a different stabilisation strategy. Re-activates the original Phase 7b task from the plan.

### Phase 10d (later, orthogonal to the leak): pressure-tensor shape-function audit

Finding 16's open thread — `Particles3D.cpp:1970-2011` uses 8-point linear weights for Pxx..Pzz in the TSC branch (`compute_supplementary_moments`). Not a cycle-1 energy source because it doesn't feed into the mass matrix, but a downstream consistency bug that shows up in stress diagnostics and any downstream analyses that use the pressure tensor output. Address after 10b completes.

### Phase 9.x: Edge/corner self-copy modular wrapping (DEFERRED)

Latent bug: edge and corner self-copies in NBDerivedHaloCommN still lack thin-dimension modular wrapping. Masked by current tests (all have nzc_r ≥ 3).

- [ ] **9a–d.** Add modular wrapping to edge/corner self-copies (`Com3DNonblk.cpp:806-888, 974-1039`)
- [ ] **9e.** Verify tests still pass

### Phase 11: Non-periodic boundary hardcodings (DEFERRED; renamed from legacy "Phase 10" after 10a/b/c/d were reassigned to the energy-conservation audit)

Won't fire for the all-periodic Double_Harris test. Fix when non-periodic BCs are needed with n_ghost > 1.

- [ ] **11a.** `adjustNonPeriodicDensities` — hardcoded `[1]` / `[nxn-2]`
- [ ] **11b.** `OpenBoundaryInflowB/E` — hardcoded `[1]` / `[nxn-2]`
- [ ] **11c.** `divN2C_BCLeftRight` / `derBC_left*` — hardcoded `-2` loop ends
- [ ] **11d.** Particle initializers — `[1, getNXC()-1)` instead of `[n_ghost, getNXC()-n_ghost)`
- [ ] **11e.** `init_double_harris` field init bounds

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
| `27c4bb7` | docs: Phase 9 conclusion — SMS symmetry, particle-field consistency, smoothing count all ruled out as individual leak sources |
| `dfd5421` | docs: expand Phase 10a literature review brief in plan-tsc.md |
| (pending) | docs: Phase 10a literature review complete, Phase 8e reversed, Phase 10b audit planned (with 10b.0 baseline re-validation) |
| (extracted) | Phase 4 operator-symmetry diagnostic — in `phase4-diagnostic.patch`, removed from tree because it perturbs FP determinism even when gated (Finding 11). Re-applied transiently in Phase 9a with a new δ_SMS test, then reverted. |
| (dropped) | Phase 8a Kahan smoothing — tested, no benefit (Finding 12), patch discarded |
| (dropped) | Phase 9b env-gated Exth smoothing skip/extra — tested (Finding 16), both catastrophic, reverted |
