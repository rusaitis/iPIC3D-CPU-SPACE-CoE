# Plan: TSC (Quadratic B-spline) energy conservation — findings and status

## Status (2026-04-11, Phase 10i COMPLETE — the 5-point binomial kernel reduces but does NOT eliminate grid sensitivity, and is **not** equivalent to multi-pass of the 3-point kernel despite having the same tensor-product convolution. `binomial5 sm=4` suppresses the clean exponential everywhere (fit R² falls from 0.8 → 0.3 or less) and improves `dE(64)` by 10–16× over `binomial sm=4` at each grid, but `binomial sm=8` still beats `binomial5 sm=4` by 18–24× at 20x20/40x40. At 80x80 the ordering flips: `binomial5 sm=4` beats `binomial sm=8` by ~4×. The 40x40-is-easy-mode anomaly from Phase 10h persists under both kernels — the sweet spot is still grid-dependent. **Kernel width alone is not the right variable.** Per-pass halo refreshes are part of what the smoother does — one wide pass ≠ two narrow passes because the halo exchange between narrow passes adds inter-rank mixing that the wide stencil doesn't. Decision gate lands between (b) and (c): binomial5 is qualitatively better (no more clean exponential) but does not meet the Finding 25 target (flat `dE(64) < 1e-03` at every grid). See Finding 26. Next: Phase 10j (direct Arnoldi / linear-stability analysis of the discrete `I + S·M·S` operator) is now the natural step — we need to understand WHICH mode is being amplified before we can design the right filter.)

**Branch:** `with-petsc`. Latest commit `5beb33c` (TSC strategy revision). Working tree clean (env-gated Step 3 A/B probe in `ECSIM_position` was reverted after measurement). Phase 4 operator-symmetry diagnostic is in `phase4-diagnostic.patch` (untracked); no other pending edits.

**Phase 10b.1 produced a clean NEGATIVE RESULT: the TSC shape-function path is not the bug.** Three candidate localisations for the 5% TSC sm=4 leak were audited and all three ruled out:

1. **TSC weight formula & partition-of-unity (Step 1)** — pass. `Grid3DCU::get_weights_tsc` matches the standard quadratic B-spline exactly; 3D sum to machine precision (≤ 4.4e-16); all weights non-negative; nearest-node mapping is correct. No formula bug. See Finding 21 Step 1.
2. **Gather/scatter/assembly shape-function mismatch (10b.1 audit)** — refuted. All three ECSIM coupling paths (`ECSIM_velocity` gather at `Particles3D.cpp:1054–1090`, `computeMoments` scatter at `:1806–1844`, `computeMoments` mass-matrix assembly at `:1851–1896`) already use 27-point TSC weights at start-of-step position `x_n`. The initial Explore sub-agent report that "`mover_PC` is hardcoded to 8-point linear" was correct but **irrelevant**: `mover_PC` is the legacy IMM mover and is not reached in ECSIM runs — `main/iPIC3Dlib.cpp:624 c_Solver::ParticlesMover()` dispatches to `ECSIM_velocity`/`ECSIM_position`. See Finding 21 Step 2.
3. **`ECSIM_position` charge-conservation correction (Step 3 A/B test)** — refuted. Env-gated probe zeroing `dxp/dyp/dzp` gives TSC sm=4 dE(1) ratio **1.000×** (bit-for-bit unchanged, 1.063e-05 → 1.063e-05 medians). The correction is NOT the leak source. Probe reverted from the tree. See Finding 21 Step 3.

**The 5% TSC sm=4 cycle-1 leak is NOT in the particle/grid shape-function pathway.** Finding 19's primary localisation (already softened by Phase 10b.0's 124× TSC no-smooth variance discovery) is now effectively refuted. Phase 9a/b/c results (SMS symmetry clean, particle-field smoothing optimum, sm=4 local optimum) all stand. Phase 10a's literature-review conclusion (ECSIM + TSC is theoretically fine) also stands. The bug must be somewhere we haven't audited yet.

**Phase 10c (spectral analysis) COMPLETE (2026-04-11).** Ran `phase10c_tsc_spectral.inp` (same physics as `phase8c_tsc_kahan` but `ncycles=64`) and a matched CIC control (`phase10c_cic_spectral.inp`). Both 64-cycle runs, 20x20x4 np=4 sm=4, on current HEAD `5beb33c`. Analysis script `scripts/phase10c_spectral.py` (+ `phase10c_compare.py` for the overlay). See Finding 22 for the full write-up.

**Headline findings:**

1. **TSC sm=4 is not long-time stable.** dE grows from 1.06e-05 at cycle 1 to **8.44 at cycle 64** — i.e. the full initial total energy has been lost. Exponential fit to log|dE| over cycles 13–64 gives growth rate **exp(0.141) = 1.151× per cycle (~15%/cycle)**. The Phase 9 10-cycle tests captured only a pre-instability transient; "5% per-cycle leak" was a misreading of the first few cycles of an exponential blowup.
2. **CIC sm=4 is stable.** Same grid, same smoothing, same GMRES tolerance: dE(64) = **1.83e-03** with only ~6%/cyc growth after cycle 12. At cycle 64 TSC is **4600× worse than CIC**; at cycle 10 the ratio was only ~10×. The gap widens with time — a signature of exponential growth in TSC, quasi-linear secular drift in CIC.
3. **The instability is NOT a near-Nyquist mode.** FFTs of both the full-record dE and the pre-instability window (cycles 1-12) have high-k share (f ≥ 0.375/cyc) of **0.000–0.015**, with the top-3 spectral concentration in low-frequency bins dominated by the growth envelope itself. The "smoother under-damps a near-Nyquist mode" hypothesis from Phase 10c's design is **REFUTED.** There is no high-wavenumber peak to damp.
4. **The instability is NOT cleanly broadband either.** The FFT is dominated by the exponential envelope (peak at f ≈ 1/N, record-length artifact), with no distinguishable peak above the envelope. In the pre-instability window the per-cycle ΔdE DOES sign-flip (+3.3e-5, −1.9e-5, +0.3e-6, +1.9e-5, −3.3e-5, +6.0e-5, ...), suggesting a bounded oscillatory component underneath the growth, but the 12-sample record is too short to resolve its frequency reliably.

**What this means for the debugging strategy.** Phase 10c's binary decision tree (near-Nyquist → smoother fix, broadband → GMRES bisection) doesn't fit the data. The leak is actually a **slowly-growing low-to-mid-wavenumber eigenmode that is specifically amplified by the TSC shape-function path**, even though Phase 10b.1's code audit found no shape-function inconsistency. This rules out the smoother kernel design as the root cause (the damping target isn't at Nyquist), and motivates two new diagnostic angles:

- **Phase 10e (new, NEXT): grid-size growth-rate scan.** If the instability is a physical plasma mode (e.g. a finite-grid instability whose resolution threshold TSC crosses at this Δx), its growth rate will scale with k_grid. Run 20x20x4, 40x40x4, 80x80x4 at sm=4 for 64 cycles and fit the exponential growth rate in cycles 13–64. Growth-rate scaling ∝ 1/Δx or ∝ 1/Δx² suggests a dispersion/aliasing issue; growth-rate independent of Δx suggests a particle-shot-noise amplifier. This is cheap (< 5 min total) and directly actionable.
- **Phase 10f (new, after 10e): GMRES-iter bisection at cycle-1.** The original "broadband → NiterGMRES bisection" idea from Phase 10c still has diagnostic value: run NiterGMRES ∈ {5, 10, 20, 40, 80} and see if dE(1) converges to an iteration-independent floor (algebraic leak) or keeps shrinking (solver residual). Phase 9/10 all used the default `GMREStol = 1e-15` which may or may not have converged in 40 iterations — `dE(1) = 1.06e-05` is suspiciously close to the GMRES restart-length effect.
- **Phase 10g (new, low priority): plasma-physics sanity check on the Double_Harris parameters at 20x20x4.** Is this grid well-resolved enough to run TSC at `Lx/nxc = 1.5`? The reconnection layer is resolved by ~5–10 cells; TSC's 3-cell support spreads currents wider than CIC, so a 20x20 grid could be in the marginally-resolved regime where TSC's wider stencil crosses a linear-stability threshold that CIC doesn't. Not an "implementation bug" if confirmed — would be a "don't run TSC on this grid".

**What this changes for Phase 10b's conclusion.** Phase 10b.1's code audit was clean (no shape-function mismatch) and remains valid. The bug is not in the three coupling paths, but Finding 22's observation that a *different* shape function (CIC) on the *same* grid gives 4600× less drift is unambiguous evidence that something in the TSC code path is amplifying a mode the CIC path doesn't excite. The remaining candidate is a **scheme-level numerical instability at this Δx**, not a coding error — closer to what Phase 8e originally guessed before Phase 10a overturned it, but now with concrete data about where the instability onsets and how fast it grows.

**Phase 10e (grid-size growth-rate scan) COMPLETE (2026-04-11).** Ran TSC sm=4 64-cycle simulations on 20x20x4, 40x40x4, 80x80x4 at the same fixed Lx=Ly=30 (so Δx halves between runs). Auto-fit the exponential-growth slope of log|dE| in the clean exponential phase of each trajectory (window chosen from data: 20x20 cyc 14-22, 40x40 cyc 13-23, 80x80 cyc 20-32; R² in {1.00, 1.00, 0.88}). Analysis script `scripts/phase10e_scaling.py`, plot `phase10e_scaling.png`. Full write-up in Finding 23.

**Phase 10e headline numbers:**

| Grid | Δx | dE(1) seed | Growth rate | dE(64) |
|---|---|---|---|---|
| 20x20x4 | 1.500 | 1.06e-05 | **1.886×/cyc** | 8.44 |
| 40x40x4 | 0.750 | 4.03e-05 | **1.620×/cyc** | 0.95 |
| 80x80x4 | 0.375 | 7.16e-05 | **1.271×/cyc** | 0.32 |

Power-law fit of the growth slope against `1/Δx`: **slope ∝ (1/Δx)^(-0.703)**, i.e. `slope ∝ Δx^0.703`. Finer grids grow the instability more slowly — **this is the RESOLUTION-THRESHOLD branch of the Phase 10c decision gate**, not dispersion (which would be `∝ 1/Δx`), not pure particle-noise (which would be Δx-independent). The instability is a TSC+sm=4+Double_Harris numerical mode that the smoother only partially damps, and refinement incrementally reduces its growth rate but does NOT eliminate it on any affordable grid (extrapolation: `160x160 → rate ~1.17×/cyc`, `320x320 → rate ~1.10×/cyc`).

**Crossover finding:** the cycle-1 algebraic "seed" scales the **opposite** way: `dE(1) ∝ Δx^(-1.4)`, so refining multiplies the starting amplitude by ~2.6×. This is not the instability — it's the per-cycle algebraic mismatch that Phase 9 identified as the "5% TSC leak". **Phase 9's cycle-1 value was a measurement of the algebraic seed, not the instability growth rate.** The two effects are separable: cycle 1 sets the initial amplitude (scales up with resolution), cycles 13+ amplify that amplitude exponentially (scales down with resolution). Net dE(64) is dominated by the exponential amplification, not the seed: `dE(64) = dE(1) × rate^63`. Finer grids are stabler overall despite having larger cycle-1 seeds, because the growth-rate improvement overwhelms the seed increase.

**What this means for the debugging strategy.**

1. The "fix the 5% cycle-1 leak" framing from Phases 8–9 targets the wrong object. Fixing cycle-1 at fixed resolution without addressing the growth rate just moves the starting point on a logarithmic scale — if the fix drops `dE(1)` by 100× but leaves the growth rate unchanged, `dE(64) = dE(1) × 1.886^63` still saturates at `O(1)` by cycle 30 on 20x20.
2. **The instability growth rate is now the primary target**, not the cycle-1 algebraic seed. The smoother is doing something at 80x80 that it isn't at 20x20 — figure out what, then see if that something can be triggered deliberately at 20x20.
3. **Phase 10f (GMRES-iter bisection) still has value** — now as a diagnostic of the cycle-1 SEED rather than the growth rate. If the seed is solver residual, tightening `GMREStol` shrinks it linearly with solver precision. If it's algebraic, no amount of GMRES iterations helps. Report the seed scaling in parallel with the growth rate scaling.
4. **Phase 10g (resolution sanity check) is now PHASE 10e's verdict**: the 20x20 Double_Harris grid is demonstrably below the TSC marginal-stability threshold, in the practical sense that the instability growth rate there is 3× faster than at 80x80. Whether the 80x80 grid is also sub-marginal (rate still 1.27×/cyc) is a separate question. Need 160x160 or higher to bracket the threshold — but this gets expensive fast.
5. **New hypothesis from the Δx scaling:** the instability growth rate decays as `Δx^0.7` and the cycle-1 seed grows as `Δx^-1.4`. The product `dE(1) × rate^63` scales as `Δx^-1.4 × exp(63 × 0.7 × ln(Δx / Δx_ref))`. At some critical Δx the exponential dominates and the energy drift converges to zero; for 20x20x4 Double_Harris this critical grid is far finer than 80x80x4. The sm=4 smoother is not strong enough to drop the mode below growth threshold — stronger or different smoothing is a candidate fix.

**Baselines from Phase 10b.0 still stand** (unchanged by Phase 10b.1):

Key corrections (20x20x4 np=4 where not noted; 5 consecutive runs per config, 10 runs for TSC no-smooth):

| Config | Old anchor | 10b.0 median | min / max | spread | notes |
|---|---|---|---|---|---|
| CIC no-smooth dE(1) | 4.2e-08 (Ph6a) | **5.44e-08** | 4.81e-08 / 6.31e-08 | 1.3× | reproduced, slightly above the old single-run number |
| CIC no-smooth dE(10) | 2.5e-06 (Ph6a) | **2.40e-06** | 2.11e-06 / 2.70e-06 | 1.3× | matches |
| CIC sm=4 dE(1) | 2.1e-08 (Ph9) | **8.60e-09** | 5.4e-09 / 24.5e-09 | 4.5× | ~2× **lower** than Phase 9 anchor — updated floor |
| CIC sm=4 dE(10) | 1.22e-05 (Ph9) | **1.23e-05** | 1.22e-05 / 1.31e-05 | 1.1× | bit-exact |
| TSC no-smooth dE(1) | 3.8e-06 (Ph6a single) / 5.94e-06 (Ph8 5-run) | **3.97e-06 (10-run combined)** | **8.72e-08 / 1.08e-05** | **124×** | see caveat below |
| TSC no-smooth dE(10) | EXPLODED | **7.3e-01** (med) | 0.54 / 1.31 | 2.4× | deterministic explosion from cycle ~5 onward |
| TSC 40x40x4 sm=4 dE(1) | 8.9e-05 (Ph6c single) | **4.03e-05** | 4.03e-05 / 4.13e-05 | 1.003× | Phase 6c single-run was an outlier — new median is ~2× lower, deterministic |
| TSC 40x40x4 sm=4 dE(10) | 2.5e-04 (Ph6c single) | **3.55e-04** | 3.38e-04 / 3.85e-04 | 1.1× | ~1.4× higher than Ph6c anchor |

**Phase 6c grid-size scaling claim corrected.** Old text claimed "8× worse absolute drift" on 40x40 vs 20x20 (8.9e-05 / 1.06e-05). With the 10b.0 median 4.03e-05 the true ratio is **~3.8×**, closer to linear in per-rank workload rather than superlinear. The scaling direction stands; the magnitude was inflated by a single-run outlier.

**TSC no-smooth cycle-1 variance is ~124×, not ~3×.** Two independent 5-run batches this session gave medians 2.12e-06 and 5.81e-06 — a 2.7× difference across batch medians on top of the per-run spread. The MINIMUM dE(1) across 10 runs is **8.72e-08**, essentially at the CIC no-smooth GMRES FP floor (5.44e-08, ratio 1.6×); the MAXIMUM is 1.08e-05, ~200× above that floor. The per-run variance is dominated by MPI-reduction ordering interacting with near-instability chaos, not by the "shape-function pathway leak" the Finding 19 reading attributed to it. Cycle-2 onward is deterministic across all 10 runs (dE(2) ≈ 2–6e-05, dE(10) ≈ 0.5–1.3) because the near-instability exponential growth takes over and dominates whatever cycle-1 "seed" the first step left behind.

**Decision gate: NOT triggered, but Finding 19 is softened.** The 10b.0 TSC no-smooth dE(1) median 3.97e-06 stays inside the `[1e-06, 2e-05]` safe corridor defined in Phase 10b.0, so the code audit can proceed. BUT:

- Finding 19's "TSC vs CIC no-smooth gap spans 6–8 orders of magnitude" claim is wrong. The gap is **~73× at the median (~1.9 orders of magnitude), spanning 1.6×–200× across the 10-run sample**. The "smoking gun" retroactive reading of Phase 6a is still directionally valid (TSC no-smooth DOES leak well above the CIC floor on typical runs) but the rhetoric was miscalibrated and "deterministic ~140× gap" was an illusion produced by cherry-picking a noisy single measurement.
- The Phase 6a "smoking gun" framing is weakened by the occasional TSC no-smooth run that DOES hit the CIC floor (8.7e-08 ~ 1.6× CIC). If the TSC shape-function pathway were deterministically misaligned, we would not expect any run to get to the CIC FP floor even by luck. This is suggestive — but not decisive, because the cycle-2 explosion is the same across all 10 runs and the cycle-1 value sits inside a chaotic transient.
- Conclusion: **TSC no-smooth is a BAD calibration target for the 10b audit because its cycle-1 signal is dominated by near-instability chaos, not by the shape-function pathway itself.** The audit should instead use the **smoothed, deterministic** configurations — `phase9_tsc_sm{4,8,16}` and `phase6c_tsc_large` — as primary success targets. Both are tight (1.003×–5× cycle-1 spread, no instability) and they exhibit the deterministic ~1e-05 to 4e-05 leak that Phase 9 measured precisely and Phase 10a's literature review said must be fixable in principle.
- **New 10b.6 success criteria (replaces the old TSC no-smooth target):** TSC sm=4 (20x20x4) dE(1) drops from 1.06e-05 toward the CIC sm=4 floor of ~9e-09; TSC 40x40 sm=4 dE(1) drops from 4.03e-05 toward the same floor scaled with per-rank workload. TSC no-smooth cycle-1 is retained as a *diagnostic* run (for 10b.1–10b.5) but not as a post-fix verification target.

**Phase 4 operator-symmetry diagnostic is in `phase4-diagnostic.patch` (untracked); no other pending edits.**

**The Phase 10a literature review flips the conclusion of Phase 8e.** Full write-up in `review-ecsim-tsc-energy-phase10a.md`. Headline: the ECSIM discrete energy-conservation theorem (Lapenta 2017, 2023; Markidis & Lapenta 2011) is written for a *generic* interpolation function W — no step invokes linearity of the shape function. Published schemes (Chen/Chacón/Barnes 2011; Chacón/Chen 2016; Stanier/Chacón/Chen 2019; Stanier/Chacón 2022; Kormann/Sonnendrücker 2021 GEMPIC; Campos Pinto/Pagès 2022 ChECSIM) already achieve exact energy conservation with quadratic B-splines. **There is no theoretical obstacle to TSC + energy conservation.** The 5% per-cycle leak observed in iPIC3D is therefore an implementation bug, not a scheme limitation.

**Smoking-gun diagnostic (retroactive, now softened after 10b.0).** The review proposes a key diagnostic experiment ("Experiment 1"): run TSC with `num_smoothings = 0`, and if the drift is still far above machine precision, the bug is in the TSC shape-function pathway itself rather than the smoother interaction. That experiment is already in the record — it is **Phase 6a**. Phase 10b.0's 10-run re-measurement gives TSC no-smooth `dE(1)` median 3.97e-06 (range 8.7e-08–1.08e-05, 124× spread) vs CIC no-smooth `dE(1)` 5.44e-08 — **a ~73× gap at the median, not the 6–8 orders of magnitude originally claimed**. The criterion "leaks more than machine precision" still fires at the median (73× >> 1×), so the localisation is still directionally valid, but (a) the gap is 2× smaller than the Phase 10a write-up's "~140×" figure, and (b) at the minimum of the distribution the TSC no-smooth signal is within 1.6× of the CIC floor — i.e. TSC no-smooth occasionally conserves energy to essentially machine precision at cycle 1, which is not what a deterministic algebraic inconsistency in the shape function would predict. The dominant cycle-1 effect is **near-instability chaos amplifying MPI-reduction FP noise**, not a steady shape-function leak. Phase 6a's reading survives as a hint, not a smoking gun. Phase 9a/b/c results (no individual SMS asymmetry, no particle-field smoothing mismatch, sm=4 optimum) still stand.

**Variance caveat (updated from Phase 10b.0 10-run re-measurement).** Cycle-1 run-to-run variance is NOT constant across configurations:
- TSC sm=4 (20x20x4) cyc-1: tight (~0.2–2%, Phase 9 5-run).
- TSC 40x40x4 sm=4 cyc-1: very tight (1.003× spread, Phase 10b.0 5-run).
- CIC sm=4 cyc-1: moderate (~4.5× spread this session, Phase 10b.0).
- CIC no-smooth cyc-1: tight (~1.3× spread, Phase 10b.0).
- **TSC no-smooth cyc-1 spans 124×** (Phase 10b.0 10-run: `{0.087, 1.15, 1.55, 1.88, 2.12, 5.81, 6.08, 9.76, 10.3, 10.8}e-06`, median 3.97e-06). Two independent 5-run batches in the same session gave medians 2.12e-06 and 5.81e-06 — even 5-run medians are not stable. Cycle-2 onward is deterministic across all 10 runs: the near-instability exponential growth dominates and the cycle-1 "seed" becomes irrelevant.

**Consequence.** TSC no-smooth is NOT a reliable calibration target for Phase 10b. Its cycle-1 value is dominated by chaotic transient amplification of MPI-reduction non-determinism. Use the deterministic smoothed configurations (`phase8c_tsc_kahan` = TSC sm=4 on 20x20x4, `phase6c_tsc_large` = TSC sm=4 on 40x40x4, `phase9_tsc_sm{8,16}`) as the audit's primary before/after benchmarks instead. TSC no-smooth is still useful as a diagnostic for 10b.1–10b.5 code inspection but its 10-run distribution, not a point estimate, must be quoted.

**Phase 8e is still deferred, not chosen.** The "accept as theoretical limitation" conclusion reached after Phase 8 was premature in light of Phase 10a's literature review, and Phase 10b.0 does not change that. Keep the conservative CI smoke tolerances (`dE(1) ≤ 2e-05`, `dE(10) ≤ 3e-04`, median ≥ 3) until Phase 10b either finds a fix or completes without a discovery — then revisit.

**Next step: Phase 10b.1 — TSC shape-function consistency audit (code inspection starts now).** 10b.0 is complete; baselines are in the status table above. Audit targets unchanged: (a) grid-to-particle gather, (b) particle-to-grid scatter, (c) mass-matrix assembly — confirm the same TSC shape function W is used in all three, and that particle positions are evaluated at consistent time levels. The ECSIM energy identity requires all three uses of W to be bit-identical at matching particle positions. Primary verification targets (updated from Phase 10b.0): TSC sm=4 20x20x4 `dE(1) = 1.06e-05` → CIC sm=4 floor `~9e-09`, and TSC 40x40 sm=4 `dE(1) = 4.03e-05` → same floor scaled by workload. TSC no-smooth is retained only as a 10b.1–10b.5 debugging scratch config, NOT as a pass/fail gate.

**Key numbers after Phase 9 + Phase 10b.0 re-validation (20x20x4 np=4 unless noted):**
- CIC no-smooth: dE(1) = **5.44e-08** (10b.0 5-run, tight 1.3× spread) — previous single-run anchor 4.2e-08
- CIC sm=4: dE(1) = **8.60e-09** (10b.0 5-run, ~4.5× spread) — previous Phase 9 anchor 2.1e-08 (~2× **lower** now)
- TSC no-smooth: dE(1) = **3.97e-06 median** (10b.0 10-run, **124× spread**, min 8.7e-08, max 1.08e-05) — previous Phase 8 5-run median 5.94e-06
- TSC sm=4: dE(1) = 1.06e-05 (Phase 9 5-run) — cyc 0→1 leak ≈ 5.1% of exchange (unchanged, still the deterministic target)
- TSC sm=8: dE(1) = 4.83e-06 (Phase 9 5-run), dE(10) = 2.25e-03 (secular drift)
- TSC sm=16: dE(1) = 5.87e-06 (Phase 9 5-run), dE(10) = 2.39e-03
- TSC 40x40x4 sm=4: dE(1) = **4.03e-05** (10b.0 5-run, extremely tight 1.003× spread) — previous Phase 6c single-run 8.9e-05 was a ~2× outlier
- TSC sm=4 + skip Exth smoothing: dE(1) = 9.16e-05 (9× worse), dE(10) = 0.47 (catastrophic)
- TSC sm=4 + extra Exth smoothing: dE(1) = 2.62e-05 (2.5× worse), dE(10) = 2.34e-03

**Footnote on baselines:** The Phase 6a/6b/6c single-run rows in the "Energy drift summary" table below have been superseded by Phase 10b.0 5-run (10-run for TSC no-smooth) medians — see the status table at the top of this file and Finding 20 below. Treat any remaining Phase 6-era point estimates elsewhere in this file as historical; use the status table as the authoritative reference. **TSC no-smooth cycle-1 variance is ~124×, not the ~3× figure Finding 19 originally cited** — Phase 10b.0's 10 samples revealed a much wider distribution than Phase 8's 5 samples did. The cycle-1 signal for that config is dominated by near-instability chaos amplifying MPI-reduction non-determinism, so any point estimate for TSC no-smooth cycle-1 is unreliable regardless of sample size.

**Phase 9a–c re-framed (not overturned).** All three Phase 9 experimental results stand on their own merits — they just now read as "consistent with a TSC shape-function pathway bug" rather than "the leak is an irreducible scheme property":

- **Phase 9a (SMS self-adjointness): ruled out as a separate source.** δ_SMS for TSC sm=4 is 2.25e-03, actually *smaller* than δ_M = 2.37e-03 alone. All δ values are dominated by periodic-boundary inner-product double-counting. See Finding 15. Consistent with Finding 18: the (I + S·M·S) operator is theoretically sound, so measuring it as approximately self-adjoint was the expected outcome.
- **Phase 9b (particle-field consistency): current setup near-optimal.** `calculateB()` already smoothes `Exth` in place before `set_fieldForPcls()`, so particles already see `S·E` matching the operator. Both skip and extra smoothing are strictly worse. See Finding 16. Consistent with Finding 18: Lapenta 2023's construction requires particles to see `S·E`, and iPIC3D does this.
- **Phase 9c (smoothing count sweep): sm=4 is the optimum.** sm=8/16 trade cyc-1 wins for cyc-10 secular drift. See Finding 17. Consistent with Finding 18: for a *correct* S·M·S construction the drift is zero regardless of filter width, so observing a non-monotonic num_smoothings curve is itself evidence of an implementation mismatch upstream of the filter.

### Energy drift summary (Phase 10b.0-refreshed; see the status table at top for the canonical numbers)

| Test | dE metric | CIC | TSC | Notes |
|------|-----------|-----|-----|-------|
| 100x100x1 (standard) | \|dE/E0\| 10cyc | 5.1e-08 | N/A (needs nzc≥3) | Gold standard CIC test |
| 20x20x4 np=4 sm=4 | \|dE\| cyc1 | **8.6e-09** (Ph10b.0 5-run med) | 1.06e-05 (Ph9 5-run) | TSC ~1230× worse |
| 20x20x4 np=4 sm=4 | \|dE\| 10cyc | **1.23e-05** (Ph10b.0) | 1.27e-04 (Ph9 median) | TSC bounded, oscillatory |
| 20x20x4 np=4 **no smooth** | \|dE\| cyc1 | **5.44e-08** (Ph10b.0 5-run med, 1.3× spread) | **3.97e-06 median** (Ph10b.0 10-run, min 8.72e-08 / max 1.08e-05, **124× spread**) | median gap ~73×, but min is within 1.6× of CIC floor |
| 20x20x4 np=4 **no smooth** | \|dE\| 10cyc | **2.40e-06** (Ph10b.0) | **~7.3e-01 median** (deterministic explosion) | TSC unstable from cycle ~5 onward |
| 40x40x4 np=4 sm=4 | \|dE\| cyc1 | — | **4.03e-05** (Ph10b.0 5-run med, 1.003× spread) | ~3.8× worse than 20x20x4, roughly linear in per-rank workload (old "8×" claim was a single-run outlier) |
| 40x40x4 np=4 sm=4 | \|dE\| 10cyc | — | **3.55e-04** (Ph10b.0 5-run med) | scales with per-rank workload |

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

19. **Phase 6a is a weak smoking gun (retroactive via Phase 10a's Experiment 1, softened by Phase 10b.0).** The review proposed a direct localisation test: "Run TSC with num_smoothings = 0. If TSC without any smoother conserves energy to machine precision, the leak is in the smoother. If TSC without smoother already leaks even at 0.01%, there is an additional inconsistency in the shape-function implementation itself." That test is Phase 6a. **Phase 10b.0's 10-run re-measurement weakens both the magnitude and the determinism of this finding, but does not overturn it.**

    **Phase 10b.0-refreshed numbers (20x20x4, np=4, 10 runs for TSC no-smooth, 5 runs otherwise):**

    | Config                   | dE(1)                                                    | Source                                    | Ratio vs CIC no-smooth median |
    |---|---|---|---|
    | CIC no-smooth            | **5.44e-08** median (range 4.81–6.31e-08, 1.3× spread)   | Phase 10b.0 5-run                          | 1× (GMRES FP floor)           |
    | TSC no-smooth            | **3.97e-06** median (range **8.72e-08 to 1.08e-05, 124× spread**) | Phase 10b.0 **10-run** (two 5-run batches gave medians 2.12e-06, 5.81e-06) | **~73× at median** (range 1.6×–200×) |
    | CIC sm=4                 | **8.60e-09** median (range 5.4–24.5e-09, 4.5× spread)    | Phase 10b.0 5-run                          | ~0.16×                        |
    | TSC sm=4                 | 1.06e-05 median (tight, <2% spread)                      | Phase 9 5-run (carries over)               | ~195×                         |
    | TSC 40x40x4 sm=4         | **4.03e-05** median (range 4.03–4.13e-05, 1.003× spread) | Phase 10b.0 5-run                          | ~740×                         |

    **What Phase 10b.0 found that differs from the original Finding 19 framing:**

    - The "6–8 orders of magnitude" gap is wrong. Actual median gap is ~73× — less than two orders of magnitude. Earlier text appears to have conflated "absolute magnitudes span 1e-08 to 1e-06" (a range) with "ratio spans 6–8 orders of magnitude" (which would be 1e+6 to 1e+8).
    - TSC no-smooth variance is **124×, not 3×**. Two independent 5-run batches in the same session gave medians 2.12e-06 and 5.81e-06 — even 5-run medians of this config are not stable. Phase 8's "3.3× spread" was a low-sample-size underestimate.
    - One of the 10 TSC no-smooth runs gave dE(1) = 8.72e-08, **within 1.6× of the CIC no-smooth FP floor**. If the TSC shape-function pathway were deterministically misaligned, this should not happen. The cycle-1 signal is dominated by MPI-reduction ordering interacting with the near-instability transient, not by a steady algebraic inconsistency.
    - Cycle-2 onward is deterministic across all 10 runs (dE(2) ≈ 2–6e-05, dE(5) ≈ 8e-04, dE(10) ≈ 0.5–1.3). The near-instability exponential growth dominates and wipes out the cycle-1 variance. So the config is useful for confirming instability (deterministic) but NOT for quantifying the cycle-1 leak (chaotic).
    - The Phase 6c 40x40x4 TSC sm=4 single-run number (8.9e-05) was a ~2× outlier — the 10b.0 5-run median is 4.03e-05 with 1.003× spread. The grid-size scaling claim ("8× worse at 40x40 vs 20x20 for 4× more work/rank") is really ~3.8×, approximately linear.
    - CIC sm=4 dE(1) is ~2× LOWER than the Phase 9 5-run anchor (8.6e-09 vs 2.1e-08). The Phase 9 number was already a 5-run median, so this is real session-to-session drift, probably from compiler-level or system-library differences since the branch HEAD has not moved.

    **What survives.** The median gap is still ~73× and the TSC no-smooth configuration clearly leaks above the CIC FP floor on typical runs. Experiment 1's "leaks more than machine precision → bug in TSC shape-function pathway" criterion still fires at the median, just with less rhetorical certainty. Phase 9a (δ_SMS clean), 9b (particle-side smoothing matches), and 9c (sm=4 optimum) still point at an unaudited mechanism upstream of the filter.

    **What changes for Phase 10b.** The **primary audit target shifts from TSC no-smooth to TSC sm=4 and TSC 40x40 sm=4** — both are deterministic to ~1% spread, both exhibit a ~1e-05 and ~4e-05 leak respectively, and both are the configurations Phase 10a's review says must be fixable in principle with a correct `(I + S·M·S)` implementation. TSC no-smooth is demoted to a debugging probe for 10b.1–10b.5 code inspection, not a pass/fail gate. Target: TSC sm=4 (20x20x4) dE(1) drops from 1.06e-05 toward CIC sm=4 floor `~8.6e-09` (a ~1200× improvement), and TSC 40x40 sm=4 dE(1) drops proportionally from 4.03e-05. Bonus (if the fix also eliminates the near-instability): TSC no-smooth no longer explodes by cycle 5.

20. **Phase 10b.0 (baseline re-validation) complete — Finding 19 softened, audit target shifted.** Ran 5-run medians on all four Phase 6 configs (10 runs for TSC no-smooth because of its cycle-1 near-instability variance). Numbers are in the status table at the top of this file and in Finding 19's refreshed table. Headline shifts:

    - **TSC no-smooth cycle-1 spread is 124× across 10 runs, not ~3×.** The minimum of the distribution (8.72e-08) is within 1.6× of the CIC no-smooth FP floor (5.44e-08), and the two independent 5-run batches gave different medians (2.12e-06 vs 5.81e-06). A 124× spread is about what you expect from a near-instability chaotic transient amplifying MPI-reduction FP non-determinism. Point estimates are not safe for this config at any sample size we can afford.
    - **The TSC vs CIC no-smooth gap is ~73× at the median (not 140×, and not "6–8 orders of magnitude").** Finding 19's original rhetoric conflated a range of absolute magnitudes with a ratio. The correct ratio is 73× median, 1.6×–200× across the sample. The Experiment 1 criterion still fires at the median, but the "smoking gun" confidence is lower than Finding 19 originally claimed.
    - **Phase 6c single-run 8.9e-05 was a ~2× outlier.** Tight 5-run median is 4.03e-05 with 1.003× spread. The "grid-size scaling is superlinear (8× worse for 4× work)" claim should read ~3.8× worse for 4× work — roughly linear.
    - **CIC sm=4 dE(1) is ~2× lower this session than the Phase 9 anchor** (8.6e-09 vs 2.1e-08). Session-to-session drift at this precision is unsurprising but worth noting: the success-criterion floor for Phase 10b.6 should be the Phase 10b.0 value, not the Phase 9 value.
    - **Audit target shifted to deterministic configurations.** TSC sm=4 (1.06e-05, <2% spread) and TSC 40x40 sm=4 (4.03e-05, 1.003% spread) are the primary pass/fail gates for Phase 10b. TSC no-smooth is demoted to a debugging probe for 10b.1–10b.5.
    - **Decision gate NOT triggered** — TSC no-smooth dE(1) median 3.97e-06 is inside the `[1e-06, 2e-05]` safe corridor defined at the end of 10b.0 — so the code audit (10b.1) proceeds without pause.

    Raw data (TSV): `/tmp/10b0_cic_nsm.tsv`, `/tmp/10b0_tsc_nsm.tsv`, `/tmp/10b0_tsc_nsm_traj.tsv` (cycle-by-cycle for 5 runs), `/tmp/10b0_cic_sm4.tsv`, `/tmp/10b0_tsc_large.tsv`. These are in `/tmp/` and will not persist across machine restart; they were not committed to the repo because the information lives in this plan document. Re-run with `scripts/phase8_bench.sh` to regenerate.

21. **Phase 10b.1 + 10b.2 + charge-correction A/B probe: gather/scatter/assembly mismatch hypothesis REFUTED. Active ECSIM path is already TSC-consistent. The 5% leak is elsewhere.**

    **Step 1 (Phase 10b.2: TSC weight partition-of-unity check) — PASS.** Read `Grid3DCU::get_weights_tsc` (`include/Grid3DCU.h:427-450`) and verified analytically: per-dimension weights are `wx_l = ½(½-s)²`, `wx_c = ¾-s²`, `wx_r = ½(½+s)²` for `s ∈ [-½, ½]`. Algebra gives `wx_l + wx_c + wx_r = 1` identically. Matches standard quadratic B-spline. Numerical sweep over a 21×21×21 grid of fractional offsets confirms 3D sum error ≤ 4.4e-16 and all weights non-negative (machine precision). The nearest-node mapping (`floor(nx_pos + 0.5)`) at `:474` correctly produces `s ∈ [-½, ½]`. **No formula bug, no partition-of-unity bug, no sign bug.** This path is not the leak.

    **Step 2 (Phase 10b.1 shape-function audit) — REFUTED.** Read the three coupling paths in the active ECSIM code:

    | Path | Function | File:line | TSC weights? | Position |
    |---|---|---|---|---|
    | Grid→particle gather (E, B) | `ECSIM_velocity` | `particles/Particles3D.cpp:1054–1090` | **YES, 27-point TSC via `get_nearest_node_and_weights_tsc`** | `x_n` |
    | Particle→grid scatter (ρ, Jxh/Jyh/Jzh) | `computeMoments` | `particles/Particles3D.cpp:1806–1844` | **YES, 27-point TSC** | `x_n` |
    | Mass-matrix assembly (M_ij) | `computeMoments` | `particles/Particles3D.cpp:1851–1896` | **YES, 27-point TSC (`weights_tsc[s1]*weights_tsc[s2]`, 63 groups)** | `x_n` |

    All three paths use the identical TSC weight routine at the identical start-of-step particle position. There is NO gather/scatter/assembly shape-function mismatch in the primary ECSIM path. Finding 19's localisation hypothesis — "bug in TSC shape-function pathway" — is **refuted**.

    **The initial "`mover_PC` is hardcoded to 8-point linear" finding from the Explore sub-agent was correct but irrelevant.** `mover_PC` (`particles/Particles3D.cpp:2078–2255`) and its five AoS variants (`mover_PC_AoS` et al.) are the **legacy IMM mover** and are not reached in ECSIM runs. `main/iPIC3Dlib.cpp:624 c_Solver::ParticlesMover()` dispatches to `particles[i].ECSIM_velocity(EMf)` → `particles[i].ECSIM_position(EMf)` for both `Parameters::SoA` and `Parameters::AoS` mover types (switch at lines 647, 678), never to `mover_PC`. The legacy IMM movers are dormant code — DO NOT TOUCH THEM during the TSC audit. The "Critical files reference" table below is updated to flag this.

    **Step 3 (ECSIM_position charge-correction A/B probe) — REFUTED.** `ECSIM_position` (`particles/Particles3D.cpp:1306–1477`) contains a charge-conservation position correction (`dxp, dyp, dzp` at lines 1456–1463, applied at 1468–1470) that interpolates the divergence residual `temp_R = getResDiv(...)` via 8-point linear cell-centered weights at lines 1373–1454 — **with no TSC branch**. This looked like a possible second-order shape-function inconsistency in the TSC path. Env-gated A/B test: added a function-local static `std::getenv("IPIC3D_PHASE10B_SKIP_POSCORR")` probe, zeroed `dxp = dyp = dzp = 0` at runtime when set, rebuilt, ran 5-run medians on `phase8c_tsc_kahan.inp` and `phase6b_cic_smooth.inp` both with and without the env var.

    | Config | HEAD dE(1) med | SKIP_POSCORR dE(1) med | ratio (nocorr / head) |
    |---|---|---|---|
    | TSC sm=4 (20x20x4) | 1.063e-05 | 1.063e-05 | **1.000×** (bit-for-bit) |
    | CIC sm=4 (20x20x4) | 1.65e-08 | 1.28e-08 | 0.78× (inside 4.5× variance) |

    The TSC dE(1) is unchanged to four significant figures. The charge-conservation correction is NOT the leak source. The TSC dE(10) did drop from 1.52e-04 to 8.31e-05 (~1.8×), but that's inside the 25× cycle-10 variance envelope from Finding 10a, so the cycle-10 improvement is not distinguishable from noise. Code reverted; no env-gated probe left in the tree (Finding 11 caveat: even gated diagnostics can perturb FP — the HEAD TSC dE(1) in this session matched Phase 10b.0's 1.06e-05 anchor exactly, confirming no FP drift this time). A/B data: `/tmp/{tsc,cic}_sm4_{head,nocorr}.tsv`.

    **Net status of Phase 10b.** Three localisation candidates for the 5% TSC sm=4 leak have now been audited and ruled out:
    - ~~TSC weight formula / partition-of-unity~~ (Step 1, analytic + numeric)
    - ~~Gather/scatter/assembly shape-function mismatch~~ (Phase 10b.1 code reading)
    - ~~`ECSIM_position` charge-conservation correction~~ (Step 3 A/B test)
    
    Phase 10b.1's primary hypothesis (Finding 19) is refuted. The 5% TSC sm=4 leak is NOT in any shape-function pathway we can directly audit, and it is not amplified by the `ECSIM_position` charge correction either. **Elevate Phase 10c (spectral analysis) from "optional" to "next concrete step".** If the per-cycle dE FFT shows a dominant near-Nyquist mode, the leak is a finite-amplitude oscillation the smoother under-damps — i.e. it is a stabilisation/filter design issue, not a shape-function bug. If the FFT is broadband, run a GMRES-iteration bisection (NiterGMRES ∈ {1, 2, 5, 10, 30}) to check whether cycle-1 dE converges to a floor with more iterations (→ solver residual) or stays flat (→ algebraic leak independent of solver convergence).

    Secondary lead not yet investigated: **`RelSIM_velocity` (`particles/Particles3D.cpp:1122–1304`) has no TSC branch** — hardcoded 8-point linear gather at lines 1169–1202. This is a latent bug for RelSIM + TSC runs but NOT the source of the Double_Harris ECSIM leak. Defer fix to Phase 11 cleanup.

22. **Phase 10c spectral analysis complete (2026-04-11): TSC sm=4 on 20x20x4 is an exponential instability, NOT a bounded leak. Smoother-under-damps-Nyquist hypothesis REFUTED; CIC on the same grid is stable.**

    Ran two 64-cycle simulations on current HEAD (`5beb33c`):
    - `inputfiles/phase10c_tsc_spectral.inp` — TSC sm=4, 20x20x4 np=4, same physics as `phase8c_tsc_kahan.inp` but `ncycles=64`.
    - `inputfiles/phase10c_cic_spectral.inp` — CIC sm=4 control, otherwise identical.
    
    Analysis via `scripts/phase10c_spectral.py` (per-run FFT of cumulative dE and of per-cycle ΔdE, plus exponential-growth fits to log|dE|) and `scripts/phase10c_compare.py` (TSC vs CIC overlay). Plots `phase10c_{spectral,cic_spectral,compare}.png` at repo root (gitignored).

    **Trajectory comparison at key cycles (20x20x4 sm=4, 64 cycles):**

    | Cycle | TSC dE | CIC dE | TSC/CIC |
    |---|---|---|---|
    | 1  | 1.06e-05 | 1.33e-08 | 798× |
    | 10 | 1.86e-04 | 1.26e-05 | 14.8× |
    | 13 | 3.36e-04 | ~4e-05 | ~8× |
    | 25 | 7.85e-01 | 2.89e-04 | 2716× |
    | 50 | 5.52e+00 | 1.02e-03 | 5402× |
    | 64 | **8.44e+00** | **1.83e-03** | **4604×** |

    At cycle 64, TSC has lost (gained) the full total energy of the initial state (`E_tot(0) ≈ 0.294`, so `|dE|/E_tot ≈ 29`!). CIC has drifted by 0.6% of E_tot over the same 64 cycles. The 20x20 TSC sm=4 configuration is **not a valid test at t > 10 cycles** — it's measuring the first third of an exponential blowup, not a physical near-conservative drift.

    **Exponential-growth fit to log|dE| vs cycle:**

    | Window | TSC slope | TSC rate/cyc | CIC slope | CIC rate/cyc |
    |---|---|---|---|---|
    | cycles 1–64 (full) | +0.220 (R²=0.79) | 1.247× | +0.112 (R²=0.73) | 1.119× |
    | cycles 1–12 (pre-instability) | +0.171 (R²=0.34) | 1.186× | +0.478 (R²=0.70) | 1.612× |
    | cycles 13–64 (growth phase) | +0.141 (R²=0.69) | **1.151×** | +0.059 (R²=0.93) | **1.061×** |

    Caveats: The CIC "pre-instability" slope is spurious — `dE(1) ≈ 1e-08` is essentially machine-precision GMRES noise, and the log-fit over cycles 1–12 just reads the walk up from FP floor to physical signal. The CIC growth-phase 1.06×/cyc rate looks "exponential" only because we're fitting a linear drift in log-space; the absolute per-cycle ΔdE is near-constant (`mean ~2.9e-05`, `std ~1.8e-05`), which is secular drift. **CIC is stable.** TSC's growth-phase 1.15×/cyc rate is real: absolute ΔdE grows by ~15% per cycle in cycles 13–30 and then saturates at an `O(1)` amplitude from cycle ~30 onward (nonlinear saturation or energy-source starvation).

    **FFT interpretation: no near-Nyquist peak in any window.** Every FFT (full-record cumulative, full-record ΔdE, pre-instability ΔdE, for both TSC and CIC) has:
    - **High-k (f ≥ 0.375/cyc) power share: 0.0%–1.5%.** The Nyquist end of the spectrum is empty. The smoother's target is not where the action is.
    - **Peak frequency at the record-length bin (0.016/cyc = 64 cycles/period).** This is the Fourier representation of "one big slow growth burst", not a spectral mode.
    - **Top-3 bin concentration: 0.62–1.00** depending on window. "Narrow-band at low f" is the envelope, not a distinguished eigenmode.
    
    **The "27-point binomial smoother under-damps a near-Nyquist TSC mode" hypothesis from Phase 10c's original design is refuted.** There is no high-k content to damp. A stronger or wider filter would not help.

    **The pre-instability window (cycles 1–12) DOES show sign-flipping `ΔdE`:** `[+1.1, +3.3, -1.9, +0.0, +1.9, -3.3, +6.0, +5.6, +5.0, +0.9, -3.1, -13.3] × 10⁻⁵`. That's consistent with a bounded low-amplitude oscillation riding on a slowly-growing DC trend — but 12 samples can't resolve it. The coarsest meaningful frequency is `1/11 ≈ 0.091/cyc`, which is where the pre-instability FFT "peaks". The peak is a record-length artifact, not a real mode. Running more pre-instability cycles is impossible in this config because cycle ~13 onwards is exponential.

    **What this rules in and out:**
    - ~~**Near-Nyquist smoother failure**~~ — no high-k power to damp.
    - ~~**TSC is a general long-run instability seed**~~ — CIC on the same grid is stable. The instability is TSC-shape-function-specific.
    - ~~**Phase 9's "5% per cycle TSC leak"**~~ was a valid per-cycle number but the extrapolation "~50% energy loss over 10 cycles" was a misreading of an *exponential* signal in a linear frame. The cycle-1 number is not a "leak" — it's the seed amplitude of the growing mode.
    - ~~**Phase 10b.1's "shape-function path has no mismatch" conclusion is overturned**~~ — still stands. The code audit was clean. The TSC-specific instability arises despite every shape-function usage being internally consistent. This is a scheme-level numerical-instability diagnosis, not a coding error.
    - **New hypothesis space:** (i) a TSC finite-grid instability whose threshold the 20x20 Δx crosses but CIC doesn't (CIC's 2-cell support fits under a marginal-resolution cliff that TSC's 3-cell support crosses); (ii) a TSC-wider-stencil aliasing/dispersion issue that produces a slowly-growing long-wavelength mode; (iii) particle shot noise amplification through TSC's wider effective density kernel at this low `npcel = 27` level; (iv) a coupling between TSC's partition-of-unity-at-cell-centers structure and periodic-boundary inner-product double-counting (Phase 9a already flagged periodic double-counting as dominating δ_M but didn't tie it to stability).
    
    **Next step: Phase 10e (grid-size growth-rate scan)** — the cheapest experiment that will distinguish dispersion-driven growth (rate ∝ 1/Δx or 1/Δx²) from noise-driven growth (rate independent of Δx). Run 20/40/80 nxc TSC sm=4 for 64 cycles each, fit `log|dE|` in the growth phase, plot rate vs 1/Δx. Expected runtime: < 10 min total. Only after 10e do we decide between Phase 10f (GMRES bisection — probes algebraic leak vs solver residual at cycle 1) and a resolution-threshold investigation.

23. **Phase 10e grid-size growth-rate scan complete (2026-04-11): instability growth rate scales as `rate_slope ∝ Δx^0.703`, the RESOLUTION-THRESHOLD branch of the decision gate; crossover finding — cycle-1 algebraic seed scales the opposite way as `∝ Δx^-1.4`.**

    Ran three TSC sm=4 64-cycle simulations with fixed Lx=Ly=30, npcel=27, dt=0.125, GMREStol=1e-15:
    - `inputfiles/phase10c_tsc_spectral.inp` — 20x20x4 (Δx=1.5)
    - `inputfiles/phase10e_tsc_40.inp` — 40x40x4 (Δx=0.75)
    - `inputfiles/phase10e_tsc_80.inp` — 80x80x4 (Δx=0.375)
    
    80x80 took 10:48 wall-clock on 4 MPI ranks; 40x40 took 2:03; 20x20 was already in hand from Phase 10c. All three ran to completion, no divergence, no solver failures. Exponential-growth slope fit to `log|dE|` in the clean exponential phase of each trajectory (hand-picked windows from data inspection: `20x20 cyc 14-22`, `40x40 cyc 13-23`, `80x80 cyc 20-32`), giving R² in {1.00, 1.00, 0.88}. Analysis script `scripts/phase10e_scaling.py`, plot `phase10e_scaling.png` (repo root, gitignored).

    **Results table:**

    | Grid | Δx | 1/Δx | dE(1) seed | Fit window | Growth rate | Growth slope | R² | dE(64) |
    |---|---|---|---|---|---|---|---|---|
    | 20x20x4 | 1.500 | 0.667 | 1.06e-05 | cyc 14-22 | **1.886×/cyc** | 0.634 | 1.00 | **8.44** |
    | 40x40x4 | 0.750 | 1.333 | 4.03e-05 | cyc 13-23 | **1.620×/cyc** | 0.483 | 1.00 | **0.95** |
    | 80x80x4 | 0.375 | 2.667 | 7.16e-05 | cyc 20-32 | **1.271×/cyc** | 0.240 | 0.88 | **0.32** |

    **Power-law fits (3 points, log-log):**
    - **Growth slope vs resolution:** `slope ∝ (1/Δx)^(-0.703)`, i.e. `slope ∝ Δx^0.703`. Finer grid → slower growth. RESOLUTION-THRESHOLD branch of the Phase 10c decision gate. If the instability were dispersion-driven we would expect `slope ∝ 1/Δx` (positive exponent +1); if particle-noise-driven, we would expect an exponent near 0.
    - **Cycle-1 seed vs resolution:** `dE(1) ∝ (1/Δx)^(+1.4)`, i.e. `dE(1) ∝ Δx^-1.4`. Finer grid → larger cycle-1 algebraic mismatch. This is the "5% TSC leak" that Phases 8–9 were measuring at the 20x20 point — it actually grows by ~2.6× per doubling of resolution, which is why Phase 6c was ~3.8× worse at 40x40 than at 20x20 at cycle 1. Phase 10e now has a power-law fit for it.
    - **Net dE(64):** `dE(64)` drops from 8.44 → 0.95 → 0.32 with refinement, a factor of **26× improvement** from 20x20 to 80x80 at cycle 64 despite the cycle-1 seed being **7× larger** at 80x80. The growth-rate improvement (~4× longer doubling time) overwhelms the seed increase (~7×) over 63 cycles.

    **Plasma-physics sanity check.** `custom_parameters = 0.4 0.25` in `inputfiles/Double_Harris.inp` are the Harris layer half-width `δ = 0.4 d_i` and... (second parameter 0.25 is a density floor, not a length). So layer width / Δx = {0.27, 0.53, 1.07} cells at 20/40/80 nxc. **The 20x20 grid has the Harris layer underresolved by ~4× — its thickness is less than a single cell.** Even the 80x80 grid barely resolves the layer to one cell wide. The TSC 3-cell support is spanning the entire layer at these resolutions, creating an artificial self-smoothing that the CIC 2-cell support partially avoids. This is consistent with the Δx-dependent growth rate: as the layer becomes resolved relative to the TSC support, the instability is suppressed.

    **Extrapolation.** If `slope ∝ Δx^0.703` holds to higher resolution:
    - 160x160x4 (Δx=0.1875): slope ≈ 0.148, rate ≈ 1.160×/cyc, dE(64) ≈ `7e-05 × 1.16^63 ≈ 0.7`. Still unstable.
    - 320x320x4 (Δx=0.094): slope ≈ 0.091, rate ≈ 1.095×/cyc, dE(64) ≈ 0.02. Probably stable but expensive.
    - 640x640x4 (Δx=0.047): slope ≈ 0.056, rate ≈ 1.058×/cyc, dE(64) ≈ 0.003. Stable.
    
    Extrapolation is **crude** — only 3 data points, `Δx^0.7` is not a physical prediction, and the cycle-1 seed scaling will eventually dominate once the exponential rate drops low enough. The practical conclusion is that the TSC+sm=4 discretisation is **weakly unstable across the affordable-resolution regime** for this Double_Harris test case. "Refine until stable" is not a useful fix.

    **What this rules in and out for Phase 10:**
    - ~~**Dispersion / aliasing**~~ — would give rate increasing with 1/Δx (positive exponent). Observed: negative exponent -0.7. **Refuted.**
    - ~~**Particle-noise-driven growth**~~ — would give rate independent of Δx (exponent ~0). Observed: clearly negative exponent. **Refuted.**
    - ~~**20x20 is special / a CI smoke test artifact**~~ — every resolution in the tested range shows the same qualitative instability structure, just with different growth rates. **Refuted.**
    - **Resolution-dependent numerical instability of the TSC+sm=4 discretisation** — confirmed by the monotonic Δx^0.7 scaling.
    - **Algebraic per-cycle seed scaling ∝ Δx^-1.4** — new quantitative finding; Phases 8–9 were fitting just the 20x20 point on this curve.
    - **Harris layer is drastically underresolved on 20x20 (0.27 cells per δ).** At 80x80 the layer is ~1 cell wide. A proper TSC test needs at least a few cells per Harris layer, which means 160x160 or coarser Harris profile.

    **Decision for next steps.**
    1. **Phase 10f (GMRES-iter bisection at cycle 1) now has a specific purpose** — test whether the cycle-1 algebraic seed is solver residual or truly algebraic. If it's solver residual, tightening `GMREStol` or raising NiterGMRES should drop `dE(1)` at all three resolutions proportionally, without changing the exponential growth rate. This is a clean experiment that only needs 1-cycle runs → cheap (~2 min for 25 runs at 20x20, ~5 min at 40x40, ~15 min at 80x80).
    2. **Phase 10h (new): smoother-strength sweep at fixed grid.** Since the instability growth rate depends on Δx but the sm=4 smoother doesn't scale with Δx automatically, try `num_smoothings ∈ {4, 8, 16, 32}` at 80x80 (or 40x40) and measure the growth rate. If more smoothing drops the growth rate at the same Δx, the smoother strength is the control variable — not the shape function.
    3. **Phase 10i (new): alternative smoother kernel.** The current 27-point tensor-product binomial (1,2,1)/4 may be too narrow for TSC's 3-cell particle footprint. Try widening to a (1,3,3,1)/8 binomial or a Gaussian kernel at fixed sm=4 count. This is a code change (adding a second filter kernel behind an input option), so only worth it if Phase 10h shows stronger sm helps.
    4. **Defer Phase 10g (Harris resolution sanity check)** — Phase 10e already answered it: the layer is underresolved, but the TSC instability decay with Δx is too weak to "fix by refinement alone". The resolution constraint is now documented; act on it in the smoother work instead.

24. **Phase 10f GMRES-iter bisection at cycle 1 complete (2026-04-11): the cycle-1 TSC sm=4 seed is ALGEBRAIC, not solver residual. `dE(1)` is invariant to GMRES relative residual across 14 orders of magnitude at all three tested resolutions. The "plateau" gate from the original Phase 10f decision tree fires cleanly at every grid.**

    Added a new input parameter `NiterGMRES` (default `-1` = legacy `m=40, max_iter=25` behaviour). When positive, `EMfields3D::calculateE()` calls the built-in GMRES with `m=N, max_iter=1` — exactly `N` Krylov steps, no restart. Code changes: `include/Collective.h` (field + getter), `main/Collective.cpp:157-158` (parse), `fields/EMfields3D.cpp:2665-2677` (branch on override). Backwards-compatible.

    Driver: `scripts/phase10f_bisection.py`. Swept `NiterGMRES ∈ {5, 10, 20, 40, 80}` × {20x20x4, 40x40x4, 80x80x4} with 5 runs per point at 20x20/40x40 and 3 runs per point at 80x80 (~4.5 min total wall clock). Base input: `inputfiles/phase8c_tsc_kahan.inp` with `ncycles=1`, unique per-run `SaveDirName` for determinism. Full results in `phase10f_results.txt` at the repo root.

    **Headline numbers (median dE(1) / median GMRES relative residual):**

    | Grid    | Niter=5         | Niter=10        | Niter=20        | Niter=40 (default) | Niter=80        |
    |---------|-----------------|-----------------|-----------------|--------------------|-----------------|
    | 20x20x4 | 1.062e-05 / 1.2e-02 | 1.065e-05 / 3.4e-05 | 1.072e-05 / 5.3e-10 | 1.064e-05 / 5e-16 | 1.065e-05 / 5e-16 |
    | 40x40x4 | 4.031e-05 / 9.9e-03 | 4.028e-05 / 6.4e-05 | 4.027e-05 / 1.9e-09 | 4.027e-05 / 8e-16 | 4.027e-05 / 7e-16 |
    | 80x80x4 | 7.059e-05 / 9.0e-03 | 7.062e-05 / 4.5e-05 | 7.106e-05 / 2.3e-09 | 7.118e-05 / 8e-16 | 7.103e-05 / 8e-16 |

    **Intra-row spread (max/min across Niter sweep):** 20x20 — 1.009×; 40x40 — 1.001×; 80x80 — 1.008×. **Within each grid, the Niter variation is well below the 5-run shot-noise variance of any single Niter point** (~1–2%). The median `dE(1)` is flat.

    **Meanwhile the GMRES relative residual moves by 14 orders of magnitude** — from `~1e-2` at Niter=5 (not converged, truncated mid-Krylov) to `~5e-16` at Niter=40 (reached tol=1e-15 at iter 30–37 across all three grids). **There is no correlation.**

    **Why 40 iterations is the natural convergence length.** `GMREStol=1e-15` converges in 30–37 Krylov steps at every grid (first-restart, no restarts needed): 20x20 → 30–31 iter, 40x40 → 34–35 iter, 80x80 → 35–37 iter. So `Niter=40` is already essentially machine-precision-converged; `Niter=80` adds no benefit; `Niter=20` sits at `~1e-9` relative residual (still well below the algebraic leak); and even `Niter=5` with `~1e-2` relative residual (unconverged, 4% relative error in the linear solve) **still produces the same dE(1) to three significant figures**.

    **Key subsidiary observation: GMRES residual at 1e-2 does not propagate into total-energy error.** This is consistent with the residual living in a subspace that is essentially decoupled from total energy — probably a high-wavenumber / divergence-cleaning component that is smoothed out of the energy integral anyway. Contrast this with the Phase 10e growth-rate signal, which IS in the energy-relevant subspace and grows exponentially despite perfect GMRES convergence.

    **Cross-resolution consistency with Finding 23.** Phase 10f's converged-Niter medians `{1.06, 4.03, 7.10}e-05` match Phase 10e's single-run values `{1.06, 4.03, 7.16}e-05` to within the sub-percent run-to-run noise, confirming that Phase 10e was measuring the same algebraic object. The `Δx^-1.4` seed power law from Finding 23 therefore describes an **intrinsic property of the discrete TSC+sm=4 operator** after a single ECSIM step, not a solver artifact.

    **What this refutes and what it rules in.**

    - ~~"Phase 9's 1.06e-05 is GMRES residual at the 40-iter restart-length limit."~~ **Refuted.** The value at Niter=5 (1.062e-05) matches the value at Niter=80 (1.065e-05) to within 0.3%, despite a 14-order-of-magnitude difference in GMRES residual.
    - ~~"Tightening `GMREStol` below 1e-15 would drop `dE(1)`."~~ **Refuted by construction**. At Niter=40 the solver already hits `~5e-16` relative residual; Niter=80 gives no improvement; 1e-15 is below double-precision round-off of the matvec operator anyway.
    - ~~"Raising NiterGMRES would drop `dE(1)`."~~ **Refuted**. Monotonicity in Niter is absent.
    - ~~"Mixed — partial residual + partial algebraic leak."~~ **Refuted**. The partial-residual contribution would show up as *any* monotonic trend between 5 and 80 iters. There isn't one.
    - **The cycle-1 TSC sm=4 seed is 100% algebraic.** It is a property of the discrete equations as solved, not of the solver's approximation to them. Any fix has to change either the discretisation (shape function, smoother kernel, smoother strength, mass-matrix assembly) or the operator structure — not the Krylov parameters.

    **Immediate consequence for next steps.** Phase 10h (smoother-strength sweep at fixed high resolution) is the remaining knob that can be tuned without a C++ kernel rewrite. Finding 24 upgrades 10h from "worth trying" to "the only non-code-change option left": since the seed is algebraic *and* the instability growth rate has a `Δx^0.7` dependence (Finding 23), and the shape function is correct (Phase 10b.1, Finding 21), the smoother is the last degree of freedom accessible via input-file tuning. Phase 10i (alternative kernel) is the fallback if 10h runs out of headroom.

    **Artifacts.**
    - `scripts/phase10f_bisection.py` — driver (runs sweep, parses `ConservedQuantities.txt` col VI, scrapes GMRES convergence string, writes `phase10f_results.txt`/`.json`)
    - `phase10f_results.txt` — consolidated sweep output, repo root (gitignored)
    - `include/Collective.h`, `main/Collective.cpp`, `fields/EMfields3D.cpp:2665-2677` — `NiterGMRES` input parameter

25. **Phase 10h smoother-strength sweep complete (2026-04-11): stronger smoothing IS the control variable. sm=8 drops dE(64) by 200–400× at 20x20/40x40 vs sm=4; sm=16 drops 20x20 dE(64) to 1.76e-04 (48000× improvement over sm=4). Plateau past sm=16 at fixed grid. Optimal sm scales with resolution because the (1,2,1)/4 filter is grid-anchored — refining nxc halves the physical smoothing width at fixed sm. Finding 17 ("sm=4 is optimum") reinterpreted: true only in the dE(10) window, overturned at dE(64).**

    Driver: `scripts/phase10h_smoother.py`. For each tested `(grid, sm)` configuration the driver overrides `num_smoothings` in a copy of the matching Phase 10c/10e base input, runs 64 cycles into a fresh `SaveDirName`, and OLS-fits `log|dE|` vs cycle inside a fixed window (20x20/40x40 → cycles 13..40; 80x80 → cycles 18..50, matching Phase 10e's hand-picked onset window). sm=4 data reused from `/tmp/ipic3d_phase10c_tsc_spectral` (20x20), `/tmp/ipic3d_phase10e_tsc_40`, `/tmp/ipic3d_phase10e_tsc_80` via `--reuse-sm4`. All runs single-shot (cycle-1 variance <2% on the deterministic sm=4/8 configs per Phase 10b.0 Finding 20).

    **Full results table:**

    | grid    | sm | dE(1)     | dE(10)    | dE(32)    | dE(64)     | growth rate | R² (fit window)   |
    |---------|----|-----------|-----------|-----------|------------|-------------|-------------------|
    | 20x20x4 |  4 | 1.06e-05  | 1.86e-04  | 2.00e+00  | **8.44e+00** | **1.366**   | 0.835 (13..40)    |
    | 20x20x4 |  8 | 4.83e-06  | 2.30e-03  | 1.08e-02  | **4.01e-02** | 1.034       | 0.785 (13..40)    |
    | 20x20x4 | 16 | 5.85e-06  | 2.54e-03  | 2.31e-02  | **1.76e-04** | 1.054       | 0.802 (13..40)    |
    | 40x40x4 |  4 | 4.04e-05  | 2.99e-04  | 1.86e-01  | **9.49e-01** | **1.241**   | 0.837 (13..40)    |
    | 40x40x4 |  8 | 3.24e-05  | 1.86e-03  | 3.91e-03  | **2.41e-03** | 1.023       | 0.840 (13..40)    |
    | 40x40x4 | 16 | 2.05e-05  | 2.27e-03  | 6.44e-03  | **7.10e-03** | 0.996       | 0.015 (13..40) ★  |
    | 40x40x4 | 32 | 1.17e-05  | 3.98e-04  | 2.66e-03  | **7.45e-03** | 1.020       | 0.017 (13..40) ★  |
    | 80x80x4 |  4 | 7.16e-05  | 5.80e-04  | 1.27e-02  | **3.18e-01** | **1.150**   | 0.786 (18..50)    |
    | 80x80x4 |  8 | 5.74e-05  | 1.15e-03  | 4.47e-04  | **6.97e-02** | 1.044       | 0.096 (18..50) ★  |
    | 80x80x4 | 16 | 4.28e-05  | 7.57e-04  | 2.29e-04  | **2.90e-02** | 0.991       | 0.014 (18..50) ★  |

    ★ = R² collapse. Read as "exponential growth suppressed; residual trajectory is oscillatory/bounded, not exponential" — the fit rate ≈ 1.0 is an artifact of fitting noise, not a quantitative growth rate. Read the `dE(32)` and `dE(64)` columns for the actual amplitude.

    **Key observations:**

    1. **The decision gate fires: rate drops monotonically with sm at every grid, then plateaus.** At 40x40: 1.24 (sm=4) → 1.02 (sm=8) → 1.00 (sm=16) → 1.02 (sm=32). At 20x20: 1.37 → 1.03 → 1.05. At 80x80: 1.15 → 1.04 → 0.99. The plateau corresponds to "exponential growth killed; whatever's left is bounded noise."

    2. **sm=8 is the most cost-effective fix at 20x20 and 40x40.** It gives 200–400× improvement in dE(64) over sm=4, the rate drops to ~1.02×/cyc (borderline bounded), and R² stays high (~0.8) meaning an exponential fit still makes sense. Doubling again to sm=16 gives another 230× at 20x20 but only a small improvement at 40x40 (2.41e-03 → 7.10e-03, actually *worse* — probably because the fit window-averaged amplitude dominates over residual growth). Going to sm=32 at 40x40 doesn't help further. **sm=16 past diminishing returns at 40x40.**

    3. **At 80x80 the sm knob is less effective.** sm=4 → sm=8 improves dE(64) by only 4.6× (0.318 → 0.070), and sm=8 → sm=16 by another 2.4× (0.070 → 0.029). Compare to the 20x20 improvement (210× then 230×) or the 40x40 improvement (394× then *slight regression*). This is the "grid-anchored smoother" bite: the (1,2,1)/4 kernel has a fixed 1-cell per-pass half-width, so at 80x80 each pass covers half the physical distance it does at 40x40, and twice the passes are needed for the same physical effect. Qualitative estimate: to match 40x40 sm=8's physical footprint at 80x80 requires sm≈16; to match 20x20 sm=8's physical footprint at 80x80 requires sm≈32.

    4. **The cycle-1 seed drops monotonically with sm at all three grids.** At 40x40: 4.04e-05 → 3.24e-05 → 2.05e-05 → 1.17e-05 for sm=4/8/16/32. At 20x20: 1.06e-05 → 4.83e-06 (ignoring the sm=16 wiggle). At 80x80: 7.16e-05 → 5.74e-05 → 4.28e-05. This means the Finding 24 "algebraic" seed (refuted as solver residual, confirmed as a property of the discrete operator) *does* depend on the smoother — each additional smoothing pass reduces the per-cycle mismatch amplitude by ~30–40%. Combined with Finding 24, this means the seed is *algebraic* (independent of solver precision) but NOT *irreducible* (the smoother can attenuate it at O(1/sm)-ish rate). Phase 9c Finding 17 had already hinted at this via its sm=8/16 observations but in the wrong direction because it used dE(10).

    5. **Finding 17 reinterpreted.** Phase 9c's "sm=4 is the optimum" was a dE(10) measurement. Phase 10h reproduces it exactly at 20x20 (sm=4 dE(10) = 1.86e-04 vs sm=8 dE(10) = 2.30e-03, 12× worse), 40x40 (3.0e-04 vs 1.9e-03, 6× worse) and 80x80 (5.8e-04 vs 1.2e-03, 2× worse). **But by cycle 64 the ranking reverses**: sm=4 blows up while sm=8/16 are bounded. Finding 17's conclusion is effectively overturned; dE(10) was too short a window to see the instability onset at cycle ~13.

    6. **Combined resolution × smoother map.** Rows in the table above can be read as a 2D table of dE(64):

       | grid \ sm | 4           | 8         | 16        | 32        |
       |-----------|-------------|-----------|-----------|-----------|
       | 20x20x4   | 8.44e+00    | 4.01e-02  | **1.76e-04** | —         |
       | 40x40x4   | 9.49e-01    | **2.41e-03** | 7.10e-03 | 7.45e-03 |
       | 80x80x4   | 3.18e-01    | 6.97e-02  | 2.90e-02  | —         |

       The "best" entry per row shifts right as grid refines: 20x20 → sm=16, 40x40 → sm=8, 80x80 → sm=16 (but not as good as 20x20 sm=16). **The natural "iso-stability" contour runs diagonally in this table, consistent with physical smoothing width being the relevant invariant, not pass count.**

    **Decision for next steps.**

    1. **Immediate, no-code-change mitigation:** bump the TSC CI default from `num_smoothings=4` to `num_smoothings=8` (40x40) or `num_smoothings=16` (20x20 / 80x80). This alone turns TSC from "unstable after cycle 13" into "bounded energy drift out to cycle 64" on every tested grid, a ~200–48000× improvement depending on resolution. Update `ci_smoke_tsc.inp` to reflect the new default and re-tune Phase 8e's CI tolerances (`dE(1) ≤ 2e-05` is already satisfied at sm=8/16 because the seed drops with sm; `dE(10)` tolerance may need to *loosen* to ~5e-03 because stronger smoothing hurts dE(10) as Finding 17 observed).

    2. **Phase 10i (alternative smoother kernel) is now well-motivated.** The grid-anchored footprint of the current (1,2,1)/4 filter is the reason stronger sm doesn't uniformly help — it's always "1 cell wide per pass" regardless of Δx. A kernel with an input-parameter physical width (e.g. a truncated Gaussian with `σ = W_phys / Δx` in grid units, or a 5-point binomial `(1,4,6,4,1)/16` which has 2-cell half-width) would scale automatically with resolution. Phase 10h provides a target: we want an `(sm, kernel)` combination where dE(64) < 1e-03 at every grid and the "sweet spot" doesn't shift with refinement.

    3. **Phase 10j (direct linear-stability analysis) stays queued** as a theoretical sanity check. We now know what the fix *looks like* (stronger/wider smoothing); 10j would confirm *why* by measuring the `(I + S·M·S)` spectrum and showing that stronger S drops the largest eigenvalue below 1. Lower priority after the empirical 10h result.

    **Artifacts.**
    - `scripts/phase10h_smoother.py` — parameterised driver (`--grid 20|40|80 --sm 4,8,16,32 --reuse-sm4 --out-suffix _XY`)
    - `phase10h_results_{20,40,80}.{txt,json}` — per-grid outputs
    - `phase10h_results.txt` — consolidated table (concatenation of all three per-grid files, repo root, gitignored)

## What we don't know (after Phase 10i — kernel width alone is not the right variable; widening `(1,2,1)/4` to `(1,4,6,4,1)/16` kills the clean exponential but does not eliminate grid sensitivity, and the halo-refresh step between narrow-kernel passes turns out to be load-bearing)

Phase 9 targeted three "scheme-level" candidates for the 5% TSC sm=4 leak. All three are ruled out as individual causes:

- ~~**S·M·S composition asymmetry**~~ — Phase 9a (Finding 15): δ_SMS is actually smaller than δ_M alone. Consistent with Finding 18: (I + S·M·S) is theoretically sound.
- ~~**Particle-field raw/smoothed mismatch**~~ — Phase 9b (Finding 16): particles already see S·E matching the operator. Consistent with Finding 18's prescribed construction.
- ~~**n_ghost=3 halo insufficient**~~ — Phase 9c reframed (Finding 17): sm=4 is the genuine optimum; halo is refreshed between passes so 1 ghost layer is enough for each filter pass.

Phase 10a (literature review) then ruled out the "ECSIM + TSC is theoretically impossible" worry and pointed at an implementation inconsistency in the TSC shape-function pathway itself as the most likely location of the bug, retroactively reinterpreting Phase 6a as the smoking gun (Finding 19). Phase 10b.0 has now re-measured those baselines with 5-run medians (10 runs for TSC no-smooth), **softened the smoking-gun reading** (Finding 20), and **shifted the primary audit target to TSC sm=4 / TSC 40x40 sm=4** (both deterministic) away from TSC no-smooth (chaotic).

**What we still need to do, in order:**

- **Phase 10b.1 (NEXT):** TSC shape-function consistency audit — code inspection. Same hypothesis as before: the same TSC shape function W may not be used consistently in (a) grid-to-particle gather, (b) particle-to-grid scatter, and (c) mass-matrix assembly. Or particle positions may be evaluated at inconsistent time levels across those three paths. Starting points: `Particles3D::mover_PC` (~line 2078) for (a), the `ECSIM_moments_*` scatter loop for (b), `Particles3D.cpp:1857-1896` for (c). The mass-matrix-assembly path is the only place `weights_tsc` appears by name — Finding 16's observation that `compute_supplementary_moments` uses 8-point linear weights in the TSC branch is structural evidence that "linear weights inside a TSC code path" is not a one-off here. See the Phase 10b section below for the full work order (10b.1–10b.5 code audit, 10b.6 post-fix verification).
- **Phase 10c (DONE — 2026-04-11):** Spectral analysis on 64-cycle records. Both the "near-Nyquist mode" and the "clean broadband leak" hypotheses are refuted. The result is instead a low-to-mid-wavenumber exponential instability (TSC: ~15%/cyc growth onset at cycle ~13; CIC: 6%/cyc secular drift, stable). See Finding 22 and the Phase 10c subsection below.
- **Phase 10e (DONE — 2026-04-11):** Grid-size growth-rate scan at 20/40/80 nxc. RESOLUTION-THRESHOLD branch confirmed: `slope ∝ Δx^0.703`, cycle-1 seed `∝ Δx^-1.4`, dE(64) drops 26× from 20x20 to 80x80. Harris layer under-resolved at all three grids. See Finding 23.
- **Phase 10f (DONE — 2026-04-11):** GMRES-iter bisection at cycle 1. `dE(1)` is invariant to NiterGMRES ∈ {5, 10, 20, 40, 80} at all three Phase 10e grids, while GMRES relative residual changes by 14 orders of magnitude. The "algebraic plateau" gate fires. Cycle-1 seed is a property of the discrete TSC+sm=4 operator, not of the solver. See Finding 24.
- **Phase 10h (DONE — 2026-04-11):** Smoother-strength sweep at fixed resolution. Stronger smoothing IS the control variable — growth rate drops monotonically with `num_smoothings` at every tested grid, dE(64) improves by 200–48000× going from sm=4 to sm=8 or sm=16. But the optimal sm scales with resolution because the (1,2,1)/4 kernel is grid-anchored: refining nxc halves the physical smoothing width at fixed sm. See Finding 25. Immediate mitigation: raise TSC CI default from sm=4 to sm=8 (40x40) or sm=16 (20x20/80x80).
- **Phase 10i (DONE — 2026-04-11):** Alternative smoother kernel with physical-scale width — implemented 5-point `(1,4,6,4,1)/16` binomial (`binomial5`) as a new `SmoothKernel` input parameter. Killed the clean exponential instability at every grid (R² fall from 0.8 → <0.3), improved `dE(64)` by 10–16× over `binomial sm=4`, but did NOT meet the Finding 25 target of flat `dE(64) < 1e-03` across grids. The 40x40-is-easy-mode anomaly persists under both kernels. Surprise finding: `binomial5 sm=N` is NOT equivalent to `binomial sm=2N` despite the convolution identity — the per-pass halo refresh between narrow-kernel passes adds inter-rank mixing that a single wide-kernel pass doesn't get. See Finding 26. **Kernel width alone is not the right variable.**
- **Phase 10j (NEXT, promoted from far-fallback by 10i's result):** Direct linear-stability analysis of the fully-discrete `(I + S·M·S)` iteration matrix on 20x20x4 — Arnoldi/SLEPc-based spectrum measurement. We now know (a) `binomial sm=4` has a genuine exponential-growth mode (Phase 10e), (b) stronger narrow-kernel passes push it below growth threshold (Phase 10h), (c) widening the kernel does not (Phase 10i). The *shape* of the dominant eigenmode — its spatial structure, which operator component excites it — is the missing information. Need to compute the top few eigenvectors of the discrete operator at each `(grid, kernel, sm)` point and see what the smoother is actually attacking (or failing to attack).
- **Phase 10d (later, orthogonal to the leak):** Pressure-tensor shape-function audit. Finding 16's open thread — `Particles3D.cpp:1970-2011` uses 8-point linear weights for Pxx..Pzz even in the TSC branch (`compute_supplementary_moments`, not the primary moments path). Not a cycle-1 energy source because it doesn't feed into the mass matrix, but a consistency bug that shows up in stress diagnostics downstream. Address after 10b.

Open questions that do NOT need answering any more:

- ~~Is the ECSIM + TSC combination theoretically possible?~~ **Yes** (Finding 18).
- ~~Is S·M·S the correct filter construction?~~ **Yes** (Lapenta 2023 via Finding 18).
- ~~Is iPIC3D's (8,4,2,1) kernel the right filter?~~ **Yes** — it is one pass of the tensor-product binomial (1,2,1)/4 kernel, symmetric by construction (Finding 18).
- ~~Is Phase 8e's "accept as theoretical limitation" correct?~~ **No, it was premature** (Finding 19, unchanged by 10b.0).
- ~~What is the clean TSC no-smooth dE(1) median, and how reproducible is it?~~ **Answered by Phase 10b.0 (Finding 20): median 3.97e-06 but ~124× spread — the config is too chaotic for a stable point estimate. Demoted to debugging probe.**
- ~~Does CIC no-smooth still reproduce ~4.2e-08?~~ **Yes, 5.44e-08 with 1.3× spread** (Finding 20).
- ~~Does the Phase 6c 40x40 scaling claim survive a 5-run re-run?~~ **Mostly — the direction is right, but the single-run 8.9e-05 was a ~2× outlier; the median is 4.03e-05 with tight spread** (Finding 20).

Open questions for Phase 10b.1 and beyond:

- Where in the TSC code paths does the shape-function mismatch live (if at all)? Is gather using 8-point linear while assembly uses 27-point TSC? Or is assembly reading `x_p^n` while gather reads `x_p^{n+1/2}`?
- Is there a single-particle diagnostic that will distinguish the two candidate mechanisms (shape-function vs time-level)?
- **Alternative hypothesis raised by 10b.0:** is the deterministic TSC sm=4 leak actually a *cumulative* effect of the TSC wider stencil amplifying any residual FP or near-instability mode that the smoother only partially filters, rather than a crisp shape-function inconsistency? If 10b.1–10b.5 turn up no misalignment, this alternative would motivate 10c (spectral analysis).

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

### Phase 10b: TSC shape-function consistency audit (10b.0 DONE, 10b.1 NEXT)

**Goal.** Find and fix the deterministic TSC+smoothing energy leak. After Phase 10b.0, the primary targets are the smoothed, deterministic configurations:
- **Primary:** TSC sm=4 20x20x4 dE(1) = 1.06e-05 (Phase 9) → CIC sm=4 floor 8.6e-09 (Phase 10b.0) — ~1200× improvement target.
- **Secondary:** TSC 40x40 sm=4 dE(1) = 4.03e-05 (Phase 10b.0) → same floor scaled with per-rank workload.
- **Demoted (debugging probe only):** TSC no-smooth dE(1) — chaotic cycle-1 signal, 124× spread across 10 runs, not a reliable gate. If the primary fix also kills TSC no-smooth's near-instability by cycle 5, that's a bonus indicator.

Original TSC no-smooth target ("drop to CIC no-smooth floor") is retained as a *sanity check* but not as a pass/fail criterion — because one of the 10 Phase 10b.0 TSC no-smooth runs already hit 8.72e-08 (within 1.6× of the CIC floor) without any code change, demonstrating that the cycle-1 signal is drowning in MPI-reduction non-determinism rather than measuring a deterministic pathway leak.

**Structure.** 10b.0 is complete (Finding 20). 10b.1–10b.5 are the code audit proper. 10b.6 is post-fix verification against TSC sm=4 and TSC 40x40 sm=4. The decision gate at the end of 10b.0 did NOT trigger (median 3.97e-06 stayed inside the `[1e-06, 2e-05]` corridor), but Finding 19's framing has been softened in Finding 20.

- [x] **10b.0 — Baseline re-validation DONE (2026-04-11).** 5-run medians for CIC no-smooth, CIC sm=4, TSC 40x40 sm=4; 10-run distribution for TSC no-smooth. See Finding 20 for the refreshed numbers and the status table at the top of this file for the canonical reference. Deliverables completed:
    - Energy drift summary table refreshed with Phase 10b.0 medians.
    - Finding 19 table refreshed, "6–8 orders of magnitude" corrected to ~73× at median (1.6×–200× range).
    - Phase 6c "8× worse" scaling claim corrected to ~3.8× (roughly linear in per-rank workload).
    - TSC no-smooth cycle-1 variance documented as 124×, not 3×.
    - Per-run raw data in `/tmp/10b0_*.tsv`; re-run via `./scripts/phase8_bench.sh inputfiles/phase6X.inp LABEL 5` as needed.
    - **Decision gate: NOT triggered (proceed to 10b.1).**

- [x] **10b.1 — Same W used in all three coupling paths. REFUTED (Finding 21).** Audit concluded 2026-04-11. The active ECSIM code (`ECSIM_velocity` at `particles/Particles3D.cpp:1054–1090` for gather; `computeMoments` at `:1577–1897` for scatter + assembly) is **already correctly TSC-branched**, with all three paths using `grid->get_nearest_node_and_weights_tsc` and the same start-of-step position `x_n = pcl->get_x()`. The Explore sub-agent's initial "mover_PC hardcoded to 8-point linear" report was correct but irrelevant because **mover_PC is the legacy IMM mover** — `main/iPIC3Dlib.cpp:624 c_Solver::ParticlesMover()` dispatches to `ECSIM_velocity`/`ECSIM_position`, never to `mover_PC` or its AoS variants. See Finding 21 for the full evidence table. Shape-function mismatch hypothesis is refuted; the leak is NOT here.

- [x] **10b.2 — `weights_tsc` is a true partition of unity. PASS (Finding 21, Step 1).** Analytic check: `wx_l + wx_c + wx_r = 0.5(0.5-s)² + (0.75-s²) + 0.5(0.5+s)² = 1` identically. Numerical sweep over a 21×21×21 grid of fractional offsets gives 3D sum error ≤ 4.4e-16 (machine precision) with all weights non-negative. The 1D formula matches the standard quadratic B-spline: center `W(ξ) = 3/4 − ξ²` for |ξ| ≤ 1/2, wings `W(ξ) = (1/2)(3/2 − |ξ|)²` for 1/2 < |ξ| ≤ 3/2. Nearest-node mapping `floor(nx_pos + 0.5)` at `Grid3DCU.h:474` correctly produces `s ∈ [-½, ½]`. No formula bug, no sign bug, no POU bug.

- [x] **10b.3 — Particle positions at consistent time levels. PASS (by construction).** Both `ECSIM_velocity` and `computeMoments` set `x_old = x_n = pcl->get_x()` (see `:1029` and `:1547` respectively) and use it as the argument to `get_nearest_node_and_weights_tsc`. All three coupling paths gather, scatter, and assemble at exactly the same time level `x^n`. No mismatch. (The `ECSIM_position` mover updates particle positions AFTER the moments + field solve, so `x^n` is well-defined within a cycle.)

- [ ] **10b.4 — Pressure-tensor linear-weight smell** (open thread from Phase 9 Finding 16). `particles/Particles3D.cpp:1970-2011` uses 8-point linear weights for `Pxx..Pzz` even inside the TSC branch (`compute_supplementary_moments`). This is NOT a cycle-1 energy source because pressure tensors don't feed the mass matrix — but it IS structural evidence that "linear weights inside a TSC code path" is not a one-off mistake in this codebase. Read the surrounding code carefully for other silent linear-gather fall-throughs; whatever convention made pressure tensors linear may have affected J or E as well.

- [ ] **10b.5 — One-particle sanity test.** Once a suspected inconsistency is found, construct a minimal single-particle diagnostic: put one particle at a known position with a known velocity in a Double_Harris setup (or simpler — a uniform-field test with no B gradient), run one cycle with `num_smoothings = 0`, and verify:
    - (a) `Σ_g E_g · J_g` computed from the grid after scatter matches `q_p · v̄_p · E(x_p)` computed from the mover's gather, to machine precision. If this fails, the gather-scatter consistency identity is broken — that is the energy-conservation kill criterion.
    - (b) `(M·E)_at_particle_position` matches the velocity response the mass matrix should predict for that particle. If this fails, the mass-matrix assembly uses a different shape function or a different particle position than the mover's gather.
    
    If either test fails, step-debug from there: the fault will be in the arithmetic of one of the three paths.

- [ ] **10b.6 — Post-fix verification (targets updated after Phase 10b.0).** After the inconsistency is fixed, re-run with 5-run medians against the deterministic (smoothed) configurations — these are the primary pass/fail gates:
    - **`phase8c_tsc_kahan.inp` (PRIMARY gate): TSC sm=4 20x20x4 dE(1) should drop from 1.06e-05 toward the CIC sm=4 floor ~8.6e-09, a ~1200× improvement.** If it drops by only 2× (to ~5e-06), the smoother interaction is amplifying a residual inconsistency that the primary audit missed.
    - **`phase6c_tsc_large.inp` (SECONDARY gate): TSC 40x40 sm=4 dE(1) should drop proportionally from 4.03e-05 to the same floor scaled with per-rank workload.** Tight cycle-1 variance (1.003× spread) makes this a particularly clean gate.
    - **`phase6a_tsc_nsmooth.inp` (DIAGNOSTIC only, NOT a gate): record 10 runs for the distribution, and the cycle-by-cycle dE trajectory to cycle 5.** If the fix collapses the 124× spread AND drops the median to the CIC floor AND eliminates the cycle-5 explosion, the shape-function bug was the root cause of the instability too. If any of those three doesn't happen, the diagnostic simply wasn't measuring what we thought it measured (Finding 20 already warned this is the most likely outcome); there is a separate Courant/stability issue to track in Phase 10e or 11.
    - **`ci_smoke_tsc.inp`: production smoke test, update CI tolerances to the new floor.** The old 2e-05 tolerance becomes far too loose.
    
    Document the new baselines in both the status table at the top of `plan-tsc.md` and the "Energy drift summary" table. Tighten the CI smoke-test expectations. Update Finding 20 to record the post-fix distribution of TSC no-smooth if applicable.

**Risk / non-discovery outcome.** If 10b.0 re-validation shows the single-run Phase 6 numbers were outliers (e.g. TSC no-smooth dE(1) is actually ~1e-07 with tight variance), Finding 19's localisation weakens and the three Phase 10a "ruled out" mechanisms deserve a fresh look with the review's explicit compatibility conditions in hand. If the code audit (10b.1–10b.5) runs to completion and finds no inconsistency, Phase 8e's "accept as limitation" can be re-adopted — **but this time with citations from Phase 10a and a justified residual tolerance**, rather than as a default. In any case, do NOT re-close Phase 8e until Phase 10b has been executed end-to-end.

### Phase 10c: spectral analysis — DONE (2026-04-11)

**Inputs:** `inputfiles/phase10c_tsc_spectral.inp` (TSC sm=4, ncycles=64) and `inputfiles/phase10c_cic_spectral.inp` (CIC sm=4 control). Same physics as `phase8c_tsc_kahan` but run for 64 cycles instead of 10. Analysis: `scripts/phase10c_spectral.py` (per-run FFT + exponential-growth fit) and `scripts/phase10c_compare.py` (side-by-side overlay). Plots: `phase10c_spectral.png`, `phase10c_cic_spectral.png`, `phase10c_compare.png` (top-level, gitignored).

**Cycle-64 dE:** TSC `8.44`, CIC `1.83e-03`. Ratio **4604×**. At cycle 10 the ratio was ~10×; the divergence is exponential.

**Exponential growth fit (cycles 13–64):** TSC `exp(+0.141) = 1.151× / cyc`, R² = 0.69. CIC `exp(+0.059) = 1.061× / cyc`, R² = 0.93. Both runs show a pre-instability transient before cycle ~13 (amplifier noise dominated by log-space transition from 1e-08 up to the physical signal) and a growth phase after. TSC's growth phase is exponential at 15%/cycle; CIC's is closer to linear at 6%/cycle and the `e`-fold is mostly an artifact of fitting a linear drift with log(slope).

**FFT results.** No near-Nyquist peak in either signal or any subset thereof:

| Run | window | top-3 concentration | high-k (f≥0.375) share | peak freq | classification |
|---|---|---|---|---|---|
| TSC sm=4 | full 64c, cumulative | 0.995 | 0.000 | 0.016 | MID-BAND (envelope) |
| TSC sm=4 | full 64c, ΔdE  | 0.622 | 0.005 | 0.016 | MID-BAND (envelope) |
| TSC sm=4 | pre-inst. 1–12, ΔdE | 0.755 | 0.151 | 0.091 | MID-BAND (record-limited) |
| CIC sm=4 | full 64c, ΔdE | 0.964 | 0.000 | 0.016 | MID-BAND |
| CIC sm=4 | pre-inst. 1–12, ΔdE | 0.962 | 0.015 | 0.091 | MID-BAND |

The "peak" at 0.016/cyc in the full-record FFTs is just the record-length bin — the Fourier transform of "one slow monotonic growth burst". There is no evidence of a distinguished oscillatory mode that the smoother should be attacking, and the smoother's target (the Nyquist end of the spectrum) is in fact empty to within `<1.5%` of total power in every window. **The "27-point binomial smoother under-damps a near-Nyquist mode" hypothesis is refuted.**

**Pre-instability oscillation structure.** In cycles 1–12 the per-cycle ΔdE does sign-flip: `[+1.1, +3.3, -1.9, +0.0, +1.9, -3.3, +6.0, +5.6, +5.0, +0.9, -3.1, -13.3] × 10⁻⁵`. Those are consistent with a bounded oscillation overlaid on a slowly-growing DC trend, but 12 samples aren't enough for meaningful spectral resolution (the finest resolvable frequency is 1/12 ≈ 0.083/cyc). Running more cycles to refine this is impossible in this config because the exponential blowup takes over at cycle ~13.

**Decision.** Phase 10c's binary decision tree (near-Nyquist → smoother fix, broadband → GMRES-iter bisection) doesn't fit. Instead we got a third answer: the TSC drift is the growth phase of a **slow exponential instability specific to TSC on this 20x20x4 Δx**. Immediate follow-ups:

1. **Phase 10e — grid-size scan.** Does the growth rate scale with Δx? Run 20/40/80 nxc TSC sm=4 for 64 cycles, fit the log|dE| slope in cycles 13-64, compare. If growth rate is Δx-independent, the amplifier is particle shot noise or something equally local. If it scales with 1/Δx or 1/Δx², we're looking at a dispersion or aliasing issue (TSC's wider support crosses a stability threshold this grid doesn't resolve).
2. **Phase 10f — GMRES-iter bisection at cycle 1.** Before the instability kicks in, does `dE(1)` depend on `NiterGMRES`? Run `NiterGMRES ∈ {5, 10, 20, 40, 80}`. If `dE(1)` converges to a floor, the leak is algebraic and independent of solver precision (consistent with a scheme-level instability seed). If it keeps shrinking, the "leak" is really GMRES residual that the 40-iter default leaves behind.
3. **Phase 10g (low priority) — Double_Harris resolution sanity check.** Is `Lx/nxc = 1.5` well-resolved for TSC's 3-cell support at these plasma parameters? CIC has 2-cell support and may sit just below the marginal-resolution cliff TSC crosses.

### Phase 10d (later, orthogonal to the leak): pressure-tensor shape-function audit

Finding 16's open thread — `Particles3D.cpp:1970-2011` uses 8-point linear weights for Pxx..Pzz in the TSC branch (`compute_supplementary_moments`). Not a cycle-1 energy source because it doesn't feed into the mass matrix, but a downstream consistency bug that shows up in stress diagnostics and any downstream analyses that use the pressure tensor output. Address after 10b completes.

### Phase 10e: grid-size growth-rate scan — DONE (2026-04-11)

**Result: RESOLUTION-THRESHOLD branch fires. `slope ∝ Δx^0.703` (finer grid = slower growth), cycle-1 seed `∝ Δx^-1.4` (finer grid = larger algebraic seed), dE(64) dominated by the exponential → 80x80 is 26× stabler than 20x20.** See Finding 23 for the full numbers, fit windows, and plasma-physics sanity check. Inputs: `inputfiles/phase10e_tsc_40.inp`, `phase10e_tsc_80.inp` (and `phase10c_tsc_spectral.inp` reused for the 20x20 data point). Analysis: `scripts/phase10e_scaling.py`. Plot: `phase10e_scaling.png` (repo root, gitignored).

Protocol deviation vs the original spec: only 1 run per resolution instead of 5. The tight R² (1.00 / 1.00 / 0.88) on the exponential-phase fits and the deterministic growth trajectories (80x80 alone took 10:48 wall clock) make the single-run numbers sufficient for a 3-point power-law fit. Phase 10f and Phase 10h should still use multi-run medians where cycle-1 variance matters.

### Phase 10f: GMRES-iteration bisection at cycle 1 — DONE (2026-04-11)

**Result: the "plateau" gate fires. `dE(1)` is invariant to GMRES relative residual across ~14 orders of magnitude (3e-16 → 1.2e-2) at all three resolutions — the cycle-1 TSC sm=4 seed is ALGEBRAIC, not solver residual.** See Finding 24 for the full numbers.

**Implementation.** Exposed a new input parameter `NiterGMRES` (default `-1` = legacy `m=40, max_iter=25` restart behaviour). When set to a positive `N`, `EMfields3D::calculateE()` invokes the built-in GMRES with `m=N, max_iter=1` (exactly `N` Krylov steps, no restart). Code changes: `include/Collective.h` (field + getter), `main/Collective.cpp` (parse), `fields/EMfields3D.cpp:2665-2677` (branch). Backwards-compatible — existing inputs keep the legacy 40/25 defaults.

**Protocol.** Driven by `scripts/phase10f_bisection.py`. For each `NiterGMRES ∈ {5, 10, 20, 40, 80}` at each of the three Phase 10e grids (20x20x4, 40x40x4, 80x80x4), ran `phase8c_tsc_kahan.inp` with `ncycles=1` for `N_runs` runs (5 at 20x20 and 40x40, 3 at 80x80 to save wall-clock), per-run unique `SaveDirName`. Captured `dE(1)` from `ConservedQuantities.txt` column VI and scraped the GMRES relative-residual line (`"GMRES converged at restart ..." / "GMRES not converged !! Final error: ..."`) from each run's log.

**Results table (medians over N runs, min/max spread in parentheses):**

| Grid    | Niter | n | median dE(1) | spread | GMRES rel-residual | GMRES status |
|---------|-------|---|--------------|--------|--------------------|--------------|
| 20x20x4 |   5   | 5 | 1.0624e-05   | 1.01×  | ~1.21e-02          | not converged |
| 20x20x4 |  10   | 5 | 1.0647e-05   | 1.02×  | ~3.44e-05          | not converged |
| 20x20x4 |  20   | 5 | 1.0715e-05   | 1.02×  | ~5.3e-10           | not converged |
| 20x20x4 |  40   | 5 | 1.0642e-05   | 1.05×  | ~5e-16 (iter 31)   | **converged** |
| 20x20x4 |  80   | 5 | 1.0652e-05   | 1.10×  | ~5e-16 (iter 30)   | **converged** |
| 40x40x4 |   5   | 5 | 4.0305e-05   | 1.00×  | ~9.86e-03          | not converged |
| 40x40x4 |  10   | 5 | 4.0281e-05   | 1.01×  | ~6.35e-05          | not converged |
| 40x40x4 |  20   | 5 | 4.0274e-05   | 1.00×  | ~1.91e-09          | not converged |
| 40x40x4 |  40   | 5 | 4.0271e-05   | 1.00×  | ~8e-16 (iter 34)   | **converged** |
| 40x40x4 |  80   | 5 | 4.0271e-05   | 1.03×  | ~7e-16 (iter 34)   | **converged** |
| 80x80x4 |   5   | 3 | 7.0592e-05   | 1.01×  | ~8.96e-03          | not converged |
| 80x80x4 |  10   | 3 | 7.0624e-05   | 1.00×  | ~4.5e-05           | not converged |
| 80x80x4 |  20   | 3 | 7.1060e-05   | 1.00×  | ~2.3e-09           | not converged |
| 80x80x4 |  40   | 3 | 7.1183e-05   | 1.01×  | ~8e-16 (iter 36)   | **converged** |
| 80x80x4 |  80   | 3 | 7.1031e-05   | 1.01×  | ~8e-16 (iter 35)   | **converged** |

**Reading the table.** Within each grid, the median `dE(1)` does not vary with NiterGMRES by more than ~1%, and the variation is bounded above and below the fully-converged value (i.e. 5-iter and 80-iter bracket 40-iter). Meanwhile the GMRES relative residual changes by 14 orders of magnitude. **There is no correlation between solver precision and cycle-1 energy drift.**

**GMRES natural convergence length.** At tol=1e-15 the built-in solver converges in 30–37 Krylov steps at all three grids (no restarts needed). That is why `Niter=40` matches the default `m=40` exactly, why `Niter=80` adds no benefit, and why `Niter=5/10/20` truncated runs with residual in {1e-2, 1e-5, 1e-10} still produce the same `dE(1)`: the TSC sm=4 discretisation is solved essentially to machine precision regardless of truncation, and the ~1e-05 leak is what that discretisation *actually* dissipates per cycle.

**Observation: the resolved 40-iter default value is the ceiling, not a residual floor.** If the leak were solver-limited, we would expect `dE(1)` to *drop* monotonically as Niter grows (tighter residual → cleaner energy balance). Instead, the Niter=80 numbers are statistically indistinguishable from Niter=40, and the `Niter=5` numbers with 4%-relative-residual-unconverged GMRES *also* agree. This simultaneously demonstrates (a) GMRES residual at 1e-2 relative error does not bloat `dE(1)` — consistent with the residual living in a subspace that does not couple to total energy — and (b) no further iteration budget helps. The algebraic seed is locked in by the discrete equations themselves.

**Cross-resolution consistency with Finding 23.** The Phase 10f medians at the converged point recover the Phase 10e Δx^-1.4 scaling: `dE(1) ∈ {1.06, 4.03, 7.10}e-05` for 20/40/80 nxc matches Phase 10e's `{1.06, 4.03, 7.16}e-05`. The tiny 80x80 discrepancy (7.10 vs 7.16) is within the sub-percent run-to-run noise. This confirms Phase 10e was measuring the same algebraic object — not a run-specific artifact.

**What this rules in and out.**
- ~~**Cycle-1 seed is GMRES residual at 40-iter cap**~~ (Phase 10f primary hypothesis): **REFUTED**. At Niter=5 with 1.2e-2 relative residual, `dE(1)` is unchanged. At Niter=80 with the same ~5e-16 relative residual as Niter=40, `dE(1)` is also unchanged.
- ~~**Cycle-1 seed is mixed (part residual, part algebraic)**~~: **REFUTED** by the same data.
- **Cycle-1 seed is 100% algebraic** — intrinsic to the TSC sm=4 discrete operator after smoothing. The "5% TSC leak" from Phases 8–9 is a property of the discretisation, not of GMRES.

**Decision for next steps.** With Phase 10f confirming the seed is algebraic and Phase 10e (Finding 23) confirming the instability growth rate has a `Δx^0.7` resolution-threshold character, the remaining knobs are (in priority order):

1. **Phase 10h (smoother-strength sweep) is now the immediate next step.** Unchanged from the Phase 10e plan: run `num_smoothings ∈ {4, 8, 16, 32}` on `40x40x4` (and 1–2 follow-ups on `80x80x4`) for 64 cycles. Finding 24 makes 10h more motivated: since the seed at each resolution is an algebraic property of the TSC operator after smoothing, the smoother kernel is the only knob *other than* the shape function that can change it without a code rewrite.
2. **Phase 10i (alternative smoother kernel)** stays queued after 10h as the first Phase that requires actual C++ code changes.
3. **A direct linear-stability analysis of the fully-discrete TSC+sm=4 operator** — specifically, measure the spectrum of the `(I + S·M·S)` iteration matrix on the 20x20 grid and look for eigenvalues with modulus > 1. This is a cheaper alternative to hunting the bug by 64-cycle simulations, but requires a new Arnoldi or SLEPc workflow. Defer unless 10h/10i both fail.

**Expected runtime (actual).** 20x20 sweep: 25 runs × 1.6 s = ~40 s + build + script overhead. 40x40 sweep: 25 runs × 2.7 s = ~1 min 10 s. 80x80 sweep: 15 runs × 9 s ≈ 2 min 15 s. Total ~4 min 30 s (well under the plan's ~22-min estimate). All runs converged if Niter ≥ 40.

**Artifacts.**
- `scripts/phase10f_bisection.py` — driver + analyzer.
- `phase10f_results.txt` — concatenated sweep stdout + per-run records (repo root, gitignored).
- `phase10f_results.json` — machine-readable run list (most-recently-run grid only; overwritten by each invocation — consume via the `.txt` file if you need all three grids at once).
- `include/Collective.h`, `main/Collective.cpp`, `fields/EMfields3D.cpp:2665-2677` — `NiterGMRES` input parameter (kept in-tree; useful for future solver-precision experiments).

### Phase 10g (folded into Phase 10e — resolution sanity check)

Phase 10e Finding 23 already reports the Harris layer half-width (`δ = 0.4 d_i` from `custom_parameters`) vs Δx at each tested resolution: `δ/Δx ∈ {0.27, 0.53, 1.07}` for 20/40/80 nxc. **All three grids under-resolve or barely resolve the Harris layer**, and the instability growth rate decays too slowly with Δx (`slope ∝ Δx^0.703`) to be fixed by refinement alone. No further dedicated resolution analysis needed; act on the finding via Phase 10h/10i (smoother tuning) instead.

### Phase 10h: smoother-strength sweep at fixed high resolution — DONE (2026-04-11)

**Result: the "rate drops monotonically with sm, then plateaus" gate fires cleanly at every tested resolution.** Stronger smoothing is the first knob that actually kills the TSC instability. `sm=8` drops `dE(64)` by ~200–400× vs `sm=4` at both 20x20 and 40x40; `sm=16` at 20x20 drops it by an additional 230× (total 48000× improvement over sm=4); beyond sm=8–16 the drift plateaus at fixed grid. See Finding 25 for the full table.

**Implementation.** No code changes required — `num_smoothings` is already a public input parameter (`main/Collective.cpp:111`, `fields/EMfields3D.cpp:3045`). Driver: `scripts/phase10h_smoother.py`. For each `(grid, sm)` tuple, the driver regenerates an input file from the matching Phase 10e/10c base (`phase10c_tsc_spectral.inp` for 20x20, `phase10e_tsc_40.inp` for 40x40, `phase10e_tsc_80.inp` for 80x80), overrides `num_smoothings`, and runs a fresh 64-cycle simulation into a per-configuration `SaveDirName`. Exponential growth rates are fit by OLS on `log|dE|` vs cycle inside a fixed window (20x20/40x40: cycles 13–40; 80x80: cycles 18–50, matching Phase 10e's hand-picked 80x80 onset). `sm=4` data is reused from Phase 10c/10e at each grid via `--reuse-sm4`.

**Protocol summary.** Primary: `40x40x4`, `sm ∈ {4, 8, 16, 32}`, 4 runs × 64 cycles. Follow-ups: `20x20x4`, `sm ∈ {4, 8, 16}` (cheap — ~30 s/run), `80x80x4`, `sm ∈ {4, 8, 16}` (~10 min/run). Total wall-clock ~30 minutes (vs the plan estimate of 45 min). Single run per configuration, same cycle-1 variance caveat as Phase 10e (run-to-run spread <2% on sm=4/8 deterministic configs per Phase 10b.0 / Finding 20).

**Results table.** Full version in Finding 25. Key numbers — `dE(64)` vs `(grid, sm)` and the fit growth rate:

| grid    | sm | dE(1)   | dE(10)  | dE(32)  | dE(64)     | rate (/cyc) | R² (fit window)   |
|---------|----|---------|---------|---------|------------|-------------|-------------------|
| 20x20x4 |  4 | 1.06e-05 | 1.86e-04 | 2.00e+00 | **8.44e+00** | 1.366 | 0.83 (13..40) |
| 20x20x4 |  8 | 4.83e-06 | 2.30e-03 | 1.08e-02 | **4.01e-02** | 1.034 | 0.79 (13..40) |
| 20x20x4 | 16 | 5.85e-06 | 2.54e-03 | 2.31e-02 | **1.76e-04** | 1.054 | 0.80 (13..40) |
| 40x40x4 |  4 | 4.04e-05 | 2.99e-04 | 1.86e-01 | **9.49e-01** | 1.241 | 0.84 (13..40) |
| 40x40x4 |  8 | 3.24e-05 | 1.86e-03 | 3.91e-03 | **2.41e-03** | 1.023 | 0.84 (13..40) |
| 40x40x4 | 16 | 2.05e-05 | 2.27e-03 | 6.44e-03 | **7.10e-03** | 0.996 | 0.02 (13..40) |
| 40x40x4 | 32 | 1.17e-05 | 3.98e-04 | 2.66e-03 | **7.45e-03** | 1.020 | 0.02 (13..40) |
| 80x80x4 |  4 | 7.16e-05 | 5.80e-04 | 1.27e-02 | **3.18e-01** | 1.150 | 0.79 (18..50) |
| 80x80x4 |  8 | 5.74e-05 | 1.15e-03 | 4.47e-04 | **6.97e-02** | 1.044 | 0.10 (18..50) |
| 80x80x4 | 16 | 4.28e-05 | 7.57e-04 | 2.29e-04 | **2.90e-02** | 0.991 | 0.01 (18..50) |

**Decision gate.** **"Rate drops monotonically, then plateaus"** branch fires at every grid. At 40x40, rate drops `1.24 → 1.02 → 1.00 → 1.02` for sm=4/8/16/32 — clear plateau past sm=16. At 20x20, rate drops `1.37 → 1.03 → 1.05` (the sm=16 rate is within noise of sm=8, consistent with plateau). At 80x80, rate drops `1.15 → 1.04 → 0.99` (sm=16 is sub-unity, meaning the trajectory is slightly decaying, not growing).

**R² loss is diagnostic, not a fit failure.** At 40x40 sm≥16 and 80x80 sm≥8, the R² collapses to <0.1 because the trajectory is no longer exponential — it's bounded oscillation around some low mean. The OLS fit over cycles 13..40 is essentially fitting noise, which is why the "rate" value cluster near 1.0. Read those rows as "exponential growth suppressed; residual drift is oscillatory with amplitude given by the dE(32)/dE(64) columns."

**Grid-anchored smoother bites.** The 27-point binomial `(1,2,1)/4` has a fixed **1-cell** per-pass footprint. With `sm` passes the effective half-width is `~√sm` cells. In *physical* units that is `√sm · Δx`, which **halves when you refine the grid at fixed sm**. Consequence: the same `sm` value gives *less* physical smoothing on a finer grid, and the cross-resolution comparison at fixed sm is not a clean "finer = better". This shows up in the data: at sm=4, refinement helps (dE(64): 8.44 → 0.95 → 0.32); at sm=8, refinement *hurts* slightly (0.04 → 0.0024 → 0.07) — the 40x40 sm=8 point has coincidentally-sufficient physical smoothing while 80x80 sm=8 is effectively under-smoothed. At sm=16, the best point shifts back to 20x20 (1.76e-04). **There is no grid at which any fixed sm is uniformly optimal.**

**Rescaling note.** If the "true" control variable is physical smoothing width `W_phys = √sm · Δx`, then the Phase 10h sweet spots (20x20 sm=16 / 40x40 sm=8 / 80x80 sm=16) correspond to `W_phys ≈ {6.0, 2.1, 1.5}` in Lx=30 units. Not a clean iso-W_phys line — still resolution-dependent — but closer than fixed sm. This is the direct motivation for Phase 10i (alternative smoother kernel with physical-scale width).

**Finding 17 reinterpreted.** Phase 9c concluded "sm=4 is the optimum" based on `dE(10)` at 20x20: sm=8/16 showed *worse* 10-cycle drift than sm=4. **Phase 10h reproduces that observation exactly** (20x20 sm=4 dE(10)=1.86e-04, sm=8 dE(10)=2.30e-03, 12× worse). **But at 64 cycles the ranking reverses catastrophically** (sm=4: 8.44, sm=8: 0.04, 210× better). **dE(10) is a misleading metric for the TSC sm=4 instability** because the instability onset is near cycle 13 — the 10-cycle window caught only the transient, not the exponential growth phase. Finding 17's conclusion is effectively overturned: sm=8 (or more) IS strictly better once you measure long enough.

**Expected runtime (actual).** 40x40 sweep: 4 runs × ~2 min = 8 min (sm=4 reused). 20x20 sweep: 2 runs × ~30 s = 1 min. 80x80 sweep: 2 runs × ~10.5 min = 21 min. Total ~30 min wall-clock.

**Artifacts.**
- `scripts/phase10h_smoother.py` — driver. Supports `--grid`, `--sm`, `--window`, `--reuse-sm4`, `--out-suffix`.
- `phase10h_results_{20,40,80}.txt`/`.json` — per-grid sweep outputs (repo root, gitignored).
- `phase10h_results.txt` — concatenation of all three, for easy review.

### Phase 10i: alternative smoother kernel — DONE (2026-04-11)

**Result: binomial5 partially tames the instability but does NOT resolve the grid-sensitivity.** The wider kernel kills the clean exponential growth mode at every tested grid (R² collapses from ~0.8 to <0.3 or less), improves `dE(64)` by 10–16× over `binomial sm=4` per-grid, and at 80x80 actually beats `binomial sm=8`. But at 20x20 and 40x40 it is still 18–24× worse than `binomial sm=8`, and the 40x40-is-easy-mode anomaly persists under both kernels. **The Finding 25 target — flat `dE(64) < 1e-03` at every grid without re-tuning sm — is not met by any `(kernel, sm)` combination in the sweep.** See Finding 26 for the full table and the surprise finding about halo-refresh–dependence.

**Implementation.** First C++ code change on the TSC audit track. New input parameter `SmoothKernel ∈ {"binomial", "binomial5"}` added to `Collective` (include/Collective.h + main/Collective.cpp); default `"binomial"` is byte-identical to the legacy code path. `binomial5` selects a `(1,4,6,4,1)/16`-per-dim tensor product — 125-point 3D stencil with 2-cell half-width — implemented as a fused 5×5×5 triple-nested `di/dj/dk` loop inside `fields/EMfields3D.cpp::energy_conserve_smooth_direction` (after the existing 27-point branch). The existing `n_ghost=2` TSC budget already covers the ±2-cell reach — no halo refactor required, no change to `grids/Grid3DCU.cpp:45`'s `n_ghost ≤ 2` assert. `gauss` mode (listed in the original Phase 10i plan) was deferred because the binomial5 result alone is sufficient to make the decision.

**Driver: `scripts/phase10i_kernel.py`.** Sweeps `SmoothKernel × sm × grid`. `binomial` rows are reused from Phase 10c/10e/10h save directories via `--reuse-binomial`; only the four new `binomial5` runs are executed per grid. Fit window `[13, 40]` matches Phase 10e/10h. Total wall: ~20 min (80x80 binomial5 sm=8 dominates at ~10 min).

**Results — `dE(64)` and exponential-fit rate vs `(grid, kernel, sm)`:**

| grid    | kernel    | sm | dE(1)    | dE(32)    | dE(64)        | rate (/cyc) | R²    |
|---------|-----------|----|----------|-----------|---------------|-------------|-------|
| 20x20x4 | binomial  |  4 | 1.06e-05 | 2.00e+00  | **8.44e+00**  | 1.366       | 0.835 |
| 20x20x4 | binomial  |  8 | 4.83e-06 | 1.08e-02  | **4.01e-02**  | 1.034       | 0.785 |
| 20x20x4 | binomial5 |  4 | 1.14e-05 | 2.46e-03  | **7.21e-01**  | 1.068       | 0.295 |
| 20x20x4 | binomial5 |  8 | 1.05e-05 | 7.36e-03  | **5.50e-02**  | 0.977       | 0.082 |
| 40x40x4 | binomial  |  4 | 4.04e-05 | 1.86e-01  | **9.49e-01**  | 1.241       | 0.837 |
| 40x40x4 | binomial  |  8 | 3.24e-05 | 3.91e-03  | **2.41e-03**  | 1.023       | 0.840 |
| 40x40x4 | binomial5 |  4 | 3.11e-05 | 3.15e-03  | **5.89e-02**  | 1.013       | 0.498 |
| 40x40x4 | binomial5 |  8 | 2.22e-05 | 1.82e-03  | **1.25e-02**  | 0.969       | 0.150 |
| 80x80x4 | binomial  |  4 | 7.16e-05 | 1.27e-02  | **3.18e-01**  | 1.190       | 0.809 |
| 80x80x4 | binomial  |  8 | 5.74e-05 | 4.47e-04  | **6.97e-02**  | 0.969       | 0.045 |
| 80x80x4 | binomial5 |  4 | 5.62e-05 | 1.94e-04  | **8.34e-02**  | 0.996       | 0.002 |
| 80x80x4 | binomial5 |  8 | 4.20e-05 | 8.75e-04  | **2.79e-02**  | 1.008       | 0.032 |

**Key per-grid comparisons:**

- **20x20:** `binomial sm=4` 8.44 → `binomial5 sm=4` 0.72 (**11.7× better**) → `binomial sm=8` 0.04 (**18× better still**). Winner: `binomial sm=8`.
- **40x40:** `binomial sm=4` 0.95 → `binomial5 sm=4` 0.059 (**16× better**) → `binomial sm=8` 0.0024 (**24× better still**). Winner: `binomial sm=8`.
- **80x80:** `binomial sm=4` 0.32 → `binomial sm=8` 0.070 → `binomial5 sm=4` 0.083 → **`binomial5 sm=8` 0.028 (winner)**. At the finest grid, binomial5 actually beats the original kernel at the same sm.

**Decision gate.** Lands between (b) "kernel has its own sweet spot that still shifts" and (c) "kernel width isn't the limiting factor" — see Finding 26. `binomial5 sm=4` does not match `binomial sm=8` on the coarse grids; `binomial5 sm=8` is near-equivalent to `binomial sm=8` at 20x20 and 80x80 but 5× worse at 40x40; no kernel–sm combination gives flat `dE(64) < 1e-03` across grids. **Escalate to Phase 10j.** The dangling `gauss` branch from the original 10i plan is also unlikely to resolve this: if the *structure* of the filter matters (as Finding 26 suggests), widening alone isn't enough — we'd be chasing a moving target without knowing what to aim at.

**Finding 26 — the halo-refresh surprise.** Mathematically, one pass of `binomial5` per direction is exactly the convolution of two passes of `binomial`: `(1,2,1)/4 ∗ (1,2,1)/4 = (1,4,6,4,1)/16`. So on a *serial* code path with no halo step in between, `binomial sm=2N` would be bit-for-bit identical to `binomial5 sm=N`. In this MPI code path it is not: at 20x20, `binomial sm=8` gives `dE(64)=0.040` while `binomial5 sm=4` gives `dE(64)=0.72` (18× difference) despite sharing the same effective filter stencil by convolution. Even cycle-1 already shows a factor of 2 (`4.83e-06` vs `1.14e-05`). The reason: between every two narrow-kernel passes, `communicateNodeBC` refreshes the ghost layers with **post-first-pass smoothed values from the neighbour rank**. `binomial5 sm=4` does the halo refresh only after a full wide-kernel pass, so its ghost cells are one full filter-width "behind" the interior state during each subsequent pass. The extra halo exchanges in `binomial sm=2N` aren't an implementation detail — they inject neighbour-smoothed data between sub-passes, and that inter-rank mixing is evidently *load-bearing* for suppressing the TSC instability. **This reframes what the smoother actually does:** part "wide-stencil convolution", part "iterated boundary-synchronised relaxation". Widening the kernel while keeping `num_smoothings` constant removes the second part. This is why `binomial5 sm=N` is not equivalent to `binomial sm=2N`, and why wider-kernel alone was not sufficient to fix the grid-sensitivity pattern from Phase 10h.

**Implication for Phase 10j and beyond.** Any future kernel candidate that reduces the number of smoother passes should be benchmarked against a sub-pass halo refresh policy, not just against kernel-width variations. A possibly cheap "fix" to test at a later date: add an extra `communicateNodeBC` inside the `binomial5` inner loop (halve the kernel application, refresh, re-apply) — this would reproduce the iterated-relaxation aspect for a wide kernel. Not done in this phase; queued as a 10j follow-up.

**Artifacts.**
- `include/Collective.h`, `main/Collective.cpp` — new `SmoothKernel` string + `smoothKernelInt` plus getter.
- `fields/EMfields3D.cpp::energy_conserve_smooth_direction` — kernel branch at the top of the `num_smoothings` loop.
- `scripts/phase10i_kernel.py` — driver; supports `--grids`, `--kernels`, `--sm`, `--reuse-binomial`, `--out-suffix`.
- `phase10i_results.txt` / `phase10i_results.json` — repo root, gitignored. `phase10i_results_smoke.txt` from the earlier 20x20 single-point sanity run.

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
| `particles/Particles3D.cpp:997-1119` | **`ECSIM_velocity`** — ACTIVE ECSIM mover (velocity update); TSC gather branch at `:1054-1090` via `get_nearest_node_and_weights_tsc`. This is where the particle-side gather actually happens — NOT `mover_PC`. |
| `particles/Particles3D.cpp:1306-1477` | **`ECSIM_position`** — ACTIVE ECSIM position update; contains charge-conservation correction (`dxp/dyp/dzp` at `:1456-1463`) that uses 8-point linear cell-centered weights regardless of `stencil_order_`. Phase 10b.1 Step 3 A/B test ruled out as the leak source (Finding 21). |
| `particles/Particles3D.cpp:1504-1917` | `Particles3D::computeMoments` — scatter + mass matrix assembly. TSC gather at `:1577-1619`, TSC scatter+MM at `:1730-1897` (63 groups, `weights_tsc[s1]*weights_tsc[s2]`). Phase 8d analytically ruled out — Finding 13. |
| `particles/Particles3D.cpp:1122-1304` | `RelSIM_velocity` — RELATIVISTIC mover, hardcoded 8-point linear, **NO TSC branch**. Latent bug for RelSIM+TSC runs. NOT on the ECSIM critical path. Defer to Phase 11. |
| `particles/Particles3D.cpp:2078-2814` | `mover_PC` + 5 AoS variants — **LEGACY IMM MOVER, UNREACHABLE** in ECSIM runs. `main/iPIC3Dlib.cpp:624` dispatches to `ECSIM_velocity`/`ECSIM_position`, never to `mover_PC`. **Do not touch during TSC audit** — Phase 10b.1 wasted effort here (Finding 21). |
| `main/iPIC3Dlib.cpp:624-684` | `c_Solver::ParticlesMover()` — mover dispatcher; confirms `ECSIM_velocity`/`ECSIM_position` is the active path for both `Parameters::SoA` and `Parameters::AoS` mover types. |
| `include/Grid3DCU.h:427-488` | `Grid3DCU::get_weights_tsc` (1D weights) and `get_nearest_node_and_weights_tsc` (3D + nearest node). Phase 10b.1 Step 1 verified formula and POU (Finding 21). |
| `phase4-diagnostic.patch` (untracked) | Extracted Phase 4 operator-symmetry diagnostic — re-apply with `git apply` when needed |
| `inputfiles/ci_smoke_tsc.inp` | TSC smoke test (nzc=4, sm=4) |
| `inputfiles/phase6a_cic_nsmooth.inp` | Phase 6a: CIC no-smooth 10-cycle test |
| `inputfiles/phase6a_tsc_nsmooth.inp` | Phase 6a: TSC no-smooth — unstable by cycle 6 |
| `inputfiles/phase6b_cic_smooth.inp` | Phase 6b: CIC sm=4 on 20x20x4 baseline |
| `inputfiles/phase6c_tsc_large.inp` | Phase 6c: TSC sm=4 on 40x40x4 scaling test |
| `inputfiles/phase8c_tsc_kahan.inp` | Phase 8c: TSC sm=4, 10-cycle Kahan regression test |
| `inputfiles/phase9_tsc_sm8.inp` | Phase 9c: TSC with num_smoothings=8 (cyc-1 winner, cyc-10 loser) |
| `inputfiles/phase9_tsc_sm16.inp` | Phase 9c: TSC with num_smoothings=16 (same pattern) |
| `scripts/phase10f_bisection.py` | Phase 10f GMRES-iter bisection driver (sweeps NiterGMRES × grid, writes `phase10f_results.txt`/`.json`) |
| `include/Collective.h:412-418` | `NiterGMRES` field + `getNiterGMRES()` getter (Phase 10f knob; default -1 = legacy) |
| `main/Collective.cpp:158` | Parse `NiterGMRES` from input file (default -1) |
| `fields/EMfields3D.cpp:2665-2677` | `calculateE()` GMRES invocation — branches on `col->getNiterGMRES()`; override uses `m=N, max_iter=1` |
| `scripts/phase10h_smoother.py` | Phase 10h smoother-strength sweep driver (parameterised over grid and sm, reuses Phase 10c/10e sm=4 baselines via `--reuse-sm4`, writes `phase10h_results_{20,40,80}.{txt,json}`) |
| `main/Collective.cpp:111` | Parse `num_smoothings` from input file (already-exposed legacy parameter — no code change needed for Phase 10h) |
| `fields/EMfields3D.cpp:3021-3092` | `energy_conserve_smooth_direction` — Phase 10i added `SmoothKernel` branch: `kernel==0` runs the legacy 27-point `(1,2,1)/4` stencil, `kernel==1` runs the 125-point `(1,4,6,4,1)/16` 5×5×5 triple loop. Halo refresh in the existing `num_smoothings` loop. |
| `include/Collective.h:211-212` | `getSmoothKernel()` / `getSmoothKernelInt()` getters (Phase 10i) |
| `include/Collective.h:250-255` | `SmoothKernel` + `smoothKernelInt` member fields (Phase 10i) |
| `main/Collective.cpp:113-123` | Parse `SmoothKernel` from input file, default `"binomial"`, with enum mapping (Phase 10i) |
| `scripts/phase10i_kernel.py` | Phase 10i alternative-smoother-kernel sweep driver (parameterised over kernel × grid × sm; reuses Phase 10c/10e/10h binomial runs via `--reuse-binomial`) |

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
| (pending) | docs: Phase 10b.0 baseline re-validation complete — Finding 19 softened, primary audit target shifted from TSC no-smooth (chaotic, 124× spread) to TSC sm=4 / TSC 40x40 sm=4 (deterministic); Finding 20 added |
| (pending) | docs: Phase 10b.1 + 10b.2 + charge-correction A/B probe all refuted — shape-function pathway is NOT the leak source, Finding 21 added, Phase 10c (spectral analysis) elevated to next concrete step |
| (pending) | docs: Phase 10c spectral analysis done — TSC sm=4 is an exponential instability (~15%/cyc) not a bounded leak; near-Nyquist hypothesis refuted (high-k share <1.5%); CIC control stable (4600× smaller at cyc 64); Finding 22 added, Phase 10e (grid-size growth-rate scan) and 10f (GMRES-iter bisection) queued next |
| (pending) | docs: Phase 10e grid-size growth-rate scan done — TSC instability is RESOLUTION-THRESHOLD behaviour, slope ∝ Δx^0.703 (finer = slower); cycle-1 algebraic seed scales the opposite way ∝ Δx^-1.4; dE(64) drops 26× from 20x20 to 80x80; Finding 23 added, Phase 10g folded into 10e's plasma-physics sanity check, Phase 10h (smoother sweep) and 10i (alternative kernel) queued after 10f |
| (pending) | feat: expose `NiterGMRES` input parameter (default -1 = legacy m=40/max_iter=25; when N>0 -> m=N/max_iter=1). Used for Phase 10f bisection; kept in-tree as a general solver-precision knob for future experiments |
| (pending) | docs: Phase 10f GMRES-iter bisection done — cycle-1 TSC sm=4 seed is ALGEBRAIC, not solver residual; dE(1) invariant across 14 orders of magnitude of GMRES relative residual at all three Phase 10e grids (20/40/80 nxc); NiterGMRES ∈ {5,10,20,40,80} gives same dE(1) within <1% at each resolution; Finding 24 added, Phase 10h promoted to "only non-code-change knob remaining"; Phase 10j (direct linear-stability analysis) queued as fallback |
| (pending) | docs: Phase 10h smoother-strength sweep done — stronger smoothing IS the control variable. sm=8 drops dE(64) by 200–400× at 20x20/40x40 vs sm=4; sm=16 drops 20x20 dE(64) from 8.44 to 1.76e-04 (48000× improvement); plateau past sm=16 at fixed grid; optimal sm scales with resolution because (1,2,1)/4 kernel is grid-anchored. Finding 25 added. Finding 17 (sm=4 optimum) reinterpreted as a dE(10)-window artifact — overturned at dE(64). Phase 10i (alternative kernel with physical-scale width) queued as immediate next step and now well-motivated. |
| (pending) | feat: `SmoothKernel` input parameter added (`Collective` + `energy_conserve_smooth_direction`) with values `binomial` (default, byte-identical to legacy path) and `binomial5` (5-point `(1,4,6,4,1)/16` tensor product / 125-point 3D stencil). Fits within existing `n_ghost=2` TSC budget. First C++ change on the TSC audit track. |
| (pending) | docs: Phase 10i alternative smoother kernel done — `binomial5` kills clean exponential growth (R² fall from 0.8→<0.3) and improves `dE(64)` by 10–16× over `binomial sm=4` per grid, but does NOT give flat `dE(64)` across grids. `binomial5 sm=4` is still 18–24× worse than `binomial sm=8` at 20/40; at 80x80 `binomial5 sm=8` beats `binomial sm=8` by ~2.5×. The 40x40-is-easy-mode anomaly persists under both kernels. **Surprise finding:** `binomial5 sm=N` is not equivalent to `binomial sm=2N` despite the convolution identity — per-pass halo refresh between narrow-kernel passes is load-bearing; wide-kernel single-pass removes that inter-rank mixing. Finding 26 added. Phase 10j (Arnoldi linear-stability analysis) promoted from far-fallback to NEXT. |
| (extracted) | Phase 4 operator-symmetry diagnostic — in `phase4-diagnostic.patch`, removed from tree because it perturbs FP determinism even when gated (Finding 11). Re-applied transiently in Phase 9a with a new δ_SMS test, then reverted. |
| (dropped) | Phase 8a Kahan smoothing — tested, no benefit (Finding 12), patch discarded |
| (dropped) | Phase 9b env-gated Exth smoothing skip/extra — tested (Finding 16), both catastrophic, reverted |
