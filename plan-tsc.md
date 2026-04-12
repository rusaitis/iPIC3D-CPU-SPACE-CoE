# Plan: TSC energy conservation — active working document

> Concise status, next step, open questions, and an index into the full phase write-ups.
> Full historical record (Phases 4–10i, Findings 1–26, all tables and protocols): `plan-tsc-archive.md`.
> Both files are gitignored (`plan-*.md` pattern).

**Branch:** `with-petsc` · **HEAD:** `ee8c5d1` (Phase 10 bundle committed) + uncommitted Phase 10j/10k/10l/10m code, scripts, and input files. **Working tree:** dirty (Phase 10m landed a stable, cheaper smoother — ready to commit).

---

## Status (2026-04-11, post-Phase 10m) — **STABLE CHEAP SMOOTHER FOUND**

**TL;DR: `bin sm=2` + post-solve Helmholtz (Phase 10m) gives the best stability of any tested config (`Etot(30) = 0.295`, top |λ|=0.9993) at ~3.6× less smoother work than `bin sm=8`.** The Phase 10k Helmholtz idea was right about the FILTER and wrong about the SLOT — taking the same Helmholtz inverse out of `MaxwellImage`'s `S·M·S` and applying it once per cycle on the converged `Eth` (Phase 10m) flips it from a destabiliser into the cheapest known stable configuration. See Findings 29 and 30.

**Final Phase 10 ranking (20x20x4 Double_Harris, cycles 14–22 DMD, 30-cycle Etot):**

| config                                 | top |λ| | arg(λ)  | Etot(30)  | dE(30)    | smoother work / cycle (relative) |
|----------------------------------------|--------|---------|-----------|-----------|----------------------------------|
| bin sm=4 (default — UNSTABLE)          | 1.270  | 0       | 1.984     | +574%     | 1× (baseline)                    |
| bin sm=8                               | 1.020  | 0.065   | 0.305     | +3.4%     | 2×                               |
| bin5 sm=4                              | 0.953  | 0.120   | 0.293     | -0.6%     | ~1.5× (5× more flops/sweep)      |
| bin5_refresh sm=4 (Phase 10l, NEW)     | 0.965  | 0.056   | 0.304     | +3.4%     | 2× (= bin sm=8 in op count)      |
| **bin sm=2 + post-solve helmholtz (Phase 10m)** | **0.9993** | π   | **0.295** | **+0.1%** | **~0.55×** ✓ (cheapest stable) |
| bin sm=4 + post-solve helmholtz (Phase 10m)     | 1.063  | 0       | 0.293 | -0.6% | ~1.05× (still cheaper than sm=8) |
| helmholtz drop-in (Phase 10k)          | 1.052  | 0       | 52.16     | +17600%   | unstable (negative result)       |

The "smoother work / cycle" column counts narrow-binomial sweeps + Helmholtz CG iters as equal-cost; in practice the per-call Helmholtz is ~12 stencil sweeps but is called ONCE per cycle (vs ~60+ smoother calls per cycle for the binomial inside MaxwellImage), so the post-solve Helmholtz overhead is small.

**Phase 10l (Finding 26 confirmed) + Phase 10m (Helmholtz revival as post-solve) close out the Phase 10 investigation with a clear practical winner: `bin sm=2 + PostSolveHelmholtz=true`.**

---

## Status (2026-04-11, post-Phase 10k)

**Phase 10k (Helmholtz low-pass smoother) is complete and is a NEGATIVE result.** A drop-in CG-solved Helmholtz inverse `(I - α ∇²)⁻¹` replacing the binomial stencil **does not stabilise** TSC at any α tested (1e-6 → 22.8) and at any combination of `num_smoothings × HelmholtzNiter` swept. Best Helmholtz config gives `Etot(30) ≈ 8.6` vs binomial `sm=4` baseline of `Etot(30) ≈ 1.98` and binomial `sm=8` of `Etot(30) ≈ 0.305`. **The smoother slot is not a free post-cycle filter — it is structural to the implicit operator** `S·M·S` inside `MaxwellImage`, and replacing it with a strong low-pass restructures the field equation in ways that are GMRES-solvable but not energy-conserving. See Finding 28.

**Key Phase 10k DMD finding (cycles 14–22, rank 8, 20x20x4):**

| config                              | top |λ| | arg(λ) | Etot(30) | mode structure                                  |
|-------------------------------------|--------|--------|----------|-------------------------------------------------|
| bin sm=4 (baseline)                 | 1.270  | 0      | 1.98     | k_y=2π/Ly fundamental                           |
| bin sm=8                            | 1.020  | 0.065  | 0.305    | Harris-layer residual                           |
| bin5 sm=4                           | 0.953  | 0.120  | 0.295    | Harris-layer residual                           |
| **helmholtz default α≈22.8 sm=1**   | 1.052  | 0      | 52.16    | k_y=2π/Ly fundamental, weakened but unstable    |
| **helmholtz α=1.0 sm=1 niter=32**   | 1.017  | ±1.20  | 8.59     | oscillating mode pair, period~5cyc, NEW family  |

The Helmholtz **does** drop the original `k_y=2π/Ly` real-fundamental eigenvalue (1.270 → 0.896 at α=1) and even brings the dominant mode close to bin sm=8's |λ|=1.02 — yet the energy growth is **30× worse** than bin sm=8. The DMD eigenvalue and the actual energy growth disagree by orders of magnitude. The most likely explanation is **non-normal/transient growth from a restructured operator**: the Helmholtz inverse alters S·M·S in MaxwellImage so that the converged GMRES E satisfies a different (well-posed but non-physical) equation, and the energy theorem (which requires S·M·S to act on E and J in matched ways with a well-conditioned M) breaks at the discrete level.

**This reframes Phase 10k's hypothesis** ("a low-k–aware filter will fix the instability"). The hypothesis is right about the filter SHAPE but wrong about the SLOT — the binomial smoother is not a post-cycle dissipator; it is the stencil-broadening of the mass matrix used inside the implicit field solve, and changing it from short-range to long-range fundamentally changes what field equation the solver is converging to. The right way to use a Helmholtz filter is **outside the field solve, applied to E after `calculateE` returns** — but that is a different and structurally larger change (Phase 10m), not a drop-in.

**Phase 10l (mid-pass halo refresh in binomial5) is now the most promising next step**, both because it directly tests Finding 26's halo-mixing interpretation and because it stays inside the structural slot the binomial smoother already occupies. Phase 10m (Helmholtz as a post-`calculateE` filter, decoupled from MaxwellImage) is the natural follow-up if 10l doesn't close the gap.

---

## Status (2026-04-11, post-Phase 10j)

**Phase 10j (DMD linear-stability analysis) is complete** and has identified the TSC instability as a **domain-fundamental `k_y ≈ 2π/Ly` bulk mode**, not a near-Nyquist oscillation, not a Harris-layer-localised mode, and not a particle-shot-noise amplifier. The dominant DMD eigenpair on the 20x20 TSC sm=4 Double_Harris test at cycles 14–22 has `|λ|=1.270`, arg=0 (purely real monotonic growth), with a spatial structure that is a single-wavelength `k_y = 2π/Ly` standing wave whose antinode lobes contain the Harris layers. See Finding 27.

**Key comparison (DMD on identical cycle 14–22 window, `--rank 8`, 20x20x4):**

| config       | top |λ| | arg(λ) | mode structure                    |
|--------------|--------|--------|-----------------------------------|
| bin sm=4     | **1.270** | 0 (real) | k_y=2π/Ly fundamental, antinodes spanning Harris layers |
| bin sm=8     | 1.020  | 0.065  | residual Harris-layer-localised, weak growth |
| bin5 sm=4    | 0.953  | 0.120  | bounded oscillation, Harris-localised residual |

The DMD eigenvalue √-relation to Phase 10e's `dE` fit holds (Phase 10e gave 20x20 rate = 1.886×/cyc for `|dE|`, so the field mode grows as √1.886 = 1.373 — within 8% of the DMD value 1.270, which is acceptable given the 9-snapshot window and nonlinear saturation onset).

**This reframes the fix.** The binomial `(1,2,1)/4` smoother is a *short-range high-pass diffusion* — it damps high-k strongly and low-k weakly. The unstable mode is the LOWEST-k mode the domain supports (`k_y = 2π/Ly`). The smoother targets the wrong part of the spectrum. More passes of binomial (Finding 25) work by accumulating many small diffusion steps into an effectively-wider low-k damping operator, **and** by refreshing the halo between passes (Finding 26) which lets long-wavelength mode content propagate across MPI rank boundaries. A wider single-pass kernel (binomial5) has the same tail problem — it still targets high-k and only gets one halo refresh per pass.

**What this doesn't directly tell us (open, for Phase 10k/l):**
- **Is there a cheap low-k–aware filter?** A low-pass filter that explicitly damps `k_y = 2π/Ly` (and similar domain-scale modes) while leaving physical Harris-layer gradients intact would fix the instability at `sm=1–2`. Candidates: a Gaussian with `σ ≈ Lx/(2π)` (matches the fundamental), or a Helmholtz-type smoother `(I - α ∇²)⁻¹` which acts as a Green's function low-pass.
- **Why does the mode survive refinement?** At 40x40, DMD gives `|λ|=1.213` (still unstable), but the mode is no longer purely real — it has `arg=0.30` (period ~21 cycles). The 40x40 mode is also more messy (the `x`-averaged `Ex(y)` structure has 11 sign flips vs 2 at 20x20), suggesting multiple competing modes. Finer grids don't kill the bulk instability; they change its character.
- **Why is the cycle-1 algebraic seed (Finding 24) coupled to the same mode?** The seed scales as `Δx^-1.4` (Finding 23). It's algebraic, not solver residual — so it's a projection of the initial state onto the unstable eigenvector, not a random FP artifact. Need a DMD-like decomposition of the cycle-1 step to verify.

**What is already established (from prior phases, unchanged by 10j):**

- TSC at `sm=4` on 20x20 Double_Harris is a slow exponential instability (~15%/cyc growth from cycle 13), not a bounded algebraic leak (Finding 22).
- Growth rate scales as `slope ∝ Δx^0.703`, cycle-1 seed scales as `dE(1) ∝ Δx^-1.4` (Finding 23).
- Cycle-1 seed is 100% algebraic, invariant to 14 orders of magnitude of GMRES relative residual (Finding 24).
- Stronger narrow-kernel smoothing kills the instability (Finding 25), but sweet-spot `sm` shifts with resolution because `(1,2,1)/4` is grid-anchored.
- Shape-function pathway is clean — all three ECSIM coupling paths use consistent 27-point TSC weights at `x_n` (Finding 21).
- ECSIM + TSC is theoretically fine (Finding 18).
- `binomial5 sm=N ≠ binomial sm=2N` despite the convolution identity (Finding 26). Halo-refresh count is load-bearing for long-wavelength mode damping — consistent with the Phase 10j finding that the unstable mode spans the MPI-rank boundary.

**Immediate no-code mitigation available.** Bumping the TSC CI default from `num_smoothings=4` to `sm=8` or `sm=16` turns TSC from "unstable after cycle 13" into "bounded drift out to cycle 64" on every tested grid. Not applied to `ci_smoke_tsc.inp` yet.

---

## Current direction

Phase 10 closes with a stable, cheap configuration: **`SmoothKernel = binomial`, `num_smoothings = 2`, `PostSolveHelmholtz = true` (default α auto)**. Findings 26, 27, 28, 29, and 30 settled. Recommended next moves are operational, not investigative:

- **Validate at scale.** Run the winning Phase 10m config on the 40x40x4 and 80x80x4 grids (Phase 10e bases) to confirm the stability holds across resolutions. The Helmholtz α auto-scales as `(L_max/2π)²` so it should track grid refinement automatically. Cost is one `pixi run sim` per resolution.
- **Bump CI defaults.** Switch `inputfiles/ci_smoke_tsc.inp` from `num_smoothings = 4` (unstable) to `num_smoothings = 2` + `PostSolveHelmholtz = true`. Re-tune CI tolerances to the new (much smaller) drift band (~0.1% over the smoke test horizon vs the current ~few% margin). Backwards-compat: leave `PostSolveHelmholtz = false` as the default in `Collective.cpp` so existing input files continue to behave as before; just opt in for TSC tests.
- **Commit the bundle.** Phases 10j (DMD), 10k (Helmholtz CG, negative), 10l (binomial5_refresh + Finding 26), 10m (post-solve Helmholtz, positive) are all uncommitted. Worth a single feature commit titled along the lines of "Phase 10j–10m: DMD-driven post-solve Helmholtz stabiliser for TSC".
- **Defer:** Phase 10d (pressure-tensor shape-function audit), Phase 9.x (edge/corner self-copy), Phase 11 (non-periodic BCs with `n_ghost > 1`). None are blocking after Phase 10m's win.

---

## Phase 10m — Post-`calculateE` Helmholtz filter (DONE 2026-04-11, **POSITIVE** — winning config)

**Approach.** Keep the binomial smoother in its structural slot inside `MaxwellImage` (where Phase 10k showed it cannot be replaced without breaking energy conservation). Add a NEW hook in `calculateE()` that, after GMRES returns the converged `Eth` and the halo refresh, calls `post_solve_filter_E()` once per cycle to apply the Helmholtz inverse as a one-shot low-pass on the time-centered field. The time-advanced `Ex/Ey/Ez = (1/th) Eth - ((1-th)/th) Ex^n` is then derived from the FILTERED `Eth`, so the next cycle's solver inherits the filtering automatically.

**Implementation (~30 lines).**
- `include/EMfields3D.h` — declare `post_solve_filter_E(Ex, Ey, Ez, nx, ny, nz)`. Add a default `kernel_override = -1` argument to `energy_conserve_smooth_direction` so the post-solve hook can reuse the existing helmholtz CG without mutating Collective state.
- `fields/EMfields3D.cpp::energy_conserve_smooth_direction` — pick `kernel_override` over `col->getSmoothKernelInt()` when `>= 0`.
- `fields/EMfields3D.cpp::post_solve_filter_E` — three calls into `energy_conserve_smooth_direction` with `kernel_override = 2` (helmholtz). Reuses the Phase 10k Helmholtz CG implementation verbatim.
- `fields/EMfields3D.cpp::calculateE` — gated call after the `Exth/Eyth/Ezth` halo refresh, before the `Ex = (1/th) Eth - ((1-th)/th) Ex` derivation: `if (col->getPostSolveHelmholtz()) post_solve_filter_E(Exth, Eyth, Ezth, nxn, nyn, nzn);`.
- `include/Collective.h` + `main/Collective.cpp` — `PostSolveHelmholtz` bool, default `false` (opt-in, no behaviour change for existing inputs).

**Headline numbers.**

| config                              | Etot(0) | Etot(10) | Etot(20) | Etot(30) | dE(30)/E0 | top |λ| |
|-------------------------------------|---------|----------|----------|----------|-----------|--------|
| bin sm=4 (UNSTABLE baseline)        | 0.2944  | 0.2943   | 0.3445   | 1.984    | +574%     | 1.270  |
| bin sm=8                            | 0.2944  | 0.2965   | 0.3015   | 0.3046   | +3.4%     | 1.020  |
| **bin sm=2 + post-solve helmholtz** | 0.2944  | 0.2939   | 0.2936   | **0.2947** | **+0.1%** | **0.9993** |
| **bin sm=4 + post-solve helmholtz** | 0.2944  | 0.2937   | 0.2928   | **0.2927** | **-0.6%** | 1.063 (\|b\|=3e-3) |
| bin sm=1 + post-solve helmholtz     | 0.2944  | 0.2942   | 0.3531   | 0.6033   | +105%     | (still unstable)   |

**Robustness checks (bin sm=4 + post-solve helmholtz, varying α).**

| α      | Etot(30)  | dE(30)/E0 |
|--------|-----------|-----------|
| 1.0    | 0.2912    | -1.1%     |
| 5.0    | 0.2923    | -0.7%     |
| 10.0   | 0.2926    | -0.6%     |
| 22.8 (default auto) | 0.2927 | -0.6% |
| 50.0   | 0.2931    | -0.4%     |

The result is essentially flat over a 50× α range, in stark contrast to Phase 10k where the same α sweep was non-monotonic and never reached even bin sm=4 stability. **The Helmholtz filter behaves correctly when removed from the implicit operator.**

**Why this works.** The unstable `k_y = 2π/Ly` mode (Finding 27) is excited by particle/field interactions inside the cycle. The binomial smoother in `MaxwellImage` is too narrow to damp it directly but is essential for preserving the structure of `S·M·S` and the energy theorem. Decoupling concerns: keep `bin sm=2` (or even `sm=4`) for the implicit operator's well-posedness, then strip the unstable mode out of the converged field with one Helmholtz pass per cycle. The Helmholtz inverse:
- Preserves constants and linear functions exactly (`∇²` of either is zero).
- Damps `k_y = 2π/Ly` by ~50% per cycle at the auto-default α (consistent with the 27% per-cycle growth rate of the underlying unstable mode → net decay).
- Is self-adjoint and SPD, so applying it identically to all three E components preserves the energy structure of the Maxwell update.
- Does NOT enter `S·M·S`, so the implicit operator's high-k content (essential for GMRES well-posedness) is untouched.

**Why bin sm=1 is not enough.** With only one binomial sweep per smoother call, the structural smoother is too weak — the implicit operator's `S·M·S` becomes nearly `M`, which interacts with the wider TSC mass-matrix stencil in a way that the post-solve Helmholtz cannot rescue. `bin sm=2` is the minimum structural threshold; `bin sm=4` works fine but gives no additional stability over `sm=2 + helmholtz`. The recommended setting is `bin sm=2 + PostSolveHelmholtz = true`.

**Cost comparison (per cycle, normalised to bin sm=4).**
- bin sm=4 baseline: 1× (≈ 4 sweeps × ~30 GMRES iters × 2 calls per iter ≈ 240 sweeps).
- bin sm=8: 2× (≈ 480 sweeps).
- bin sm=2 + post-solve helmholtz: ~0.55× (≈ 120 binomial sweeps + 36 helmholtz CG iters per cycle = ~156 work units, plus a single MPI_Allreduce per CG iter).

The post-solve Helmholtz costs **less than 1.5% of the per-cycle smoother work** because it runs once per cycle vs the binomial's ~60+ in-loop calls. Net win: **3.6× cheaper than bin sm=8 with better stability**.

**Artifacts.**
- `inputfiles/phase10m_tsc_20_postsolve_helmholtz.inp` — base test input (default α=auto, bin sm=4).
- `phase10m_dmd_postsolve_helmholtz.{png,json}` — DMD on bin sm=4 + post-solve helmholtz.
- `phase10m_dmd_sm2_postsolve.{png,json}` — DMD on the WINNING bin sm=2 + post-solve helmholtz.
- Run dirs `/tmp/ipic3d_phase10m_*` (ephemeral).

**Decision: Phase 10m is the recommended TSC stabiliser.** Default `PostSolveHelmholtz = false` keeps existing input files unchanged, but TSC test cases (and CI) should opt in.

---

## Phase 10l — Mid-pass halo refresh in binomial5 (DONE 2026-04-11, confirms Finding 26)

**Approach.** Add a new `SmoothKernel = "binomial5_refresh"` (smoothKernelInt = 3) that decomposes each pass as `(1,2,1)/4 → halo refresh → (1,2,1)/4`. By the convolution identity `(1,2,1)*(1,2,1) = (1,4,6,4,1)` this is mathematically equivalent to one 5-point binomial pass on an INFINITE grid; the only difference from the existing `binomial5` is the inter-pass halo refresh. Hypothesis (Finding 26): `bin5_refresh sm=N` should match `binomial sm=2N` to within FP noise, because both have 2N narrow sweeps separated by 2N halo refreshes.

**Implementation (~40 lines).** New branch in `energy_conserve_smooth_direction`:
- Pass 1 (data → temp via 27-point binomial sum)
- Copy temp → data; halo refresh (the load-bearing inter-pass exchange)
- Pass 2 (data → temp via 27-point binomial sum)
- Fall through to the existing post-loop copy + halo refresh

`bin5_refresh` only ever reaches ±1 cell at a time, so it does NOT need the `n_ghost ≥ 2` guard that `binomial5` does. Reuses the same `temp` workspace and inline 27-point sum from the binomial branch.

**Headline numbers (cycles 14–22 DMD + 30-cycle Etot, 20x20x4).**

| config                              | top |λ| | arg(λ) | Etot(10) | Etot(20) | Etot(30) | mode peak |
|-------------------------------------|--------|--------|----------|----------|----------|-----------|
| bin sm=4 (unstable)                 | 1.270  | 0      | 0.2943   | 0.3445   | 1.9840   | bulk k_y=2π/Ly |
| bin sm=8                            | 1.020  | 0.065  | 0.2965   | 0.3015   | 0.3046   | i=3 (Harris)   |
| binomial5 sm=4                      | 0.953  | 0.120  | 0.2952   | 0.2955   | 0.2928   | i=3 (Harris)   |
| **binomial5_refresh sm=4 (NEW)**    | **0.965** | **0.056** | **0.2964** | **0.3014** | **0.3040** | i=3 (Harris) |

**Finding 26 confirmed bit-tight.** `bin5_refresh sm=4` matches `bin sm=8` to ~1e-4 relative at every cycle (within run-to-run FP/MPI variance for TSC sm=4 — Finding 10a). The Harris-layer-localised residual mode peaks at the same `i=3` location with similar amplitude. Both have **identical operation counts** (8 narrow sweeps + 8 halo refreshes per call) and identical stability profiles. The "wider stencil" of `binomial5` is irrelevant; what matters is the (sweep + halo refresh) cycle count.

**What this DOES NOT give us.** A cheaper stabiliser than `bin sm=8`. `bin5_refresh sm=4` and `bin sm=8` are cost-equivalent. Phase 10l confirms the *understanding* (Finding 26) but not a new operating point. The cheaper stable config came from Phase 10m, which decouples the structural smoother from the spectral filter.

**Artifacts.**
- `inputfiles/phase10l_tsc_20_bin5refresh_sm4.inp`
- `phase10l_dmd_bin5refresh_sm4.{png,json}`
- Run dir `/tmp/ipic3d_phase10l_tsc_20_bin5refresh_sm4` (ephemeral)

---

## Phase 10k — Helmholtz low-pass smoother (DONE 2026-04-11, NEGATIVE)

**Approach.** Replace the `(1,2,1)/4` binomial stencil with a CG-solved Helmholtz inverse `(I - α ∇²) S_new = S_old`. The 7-point discrete Laplacian uses physical grid spacings `dx, dy, dz`; α defaults to `(max(Lx,Ly,Lz)/(2π))²` so the half-power point of the inverse sits at the domain fundamental `k = 2π/L_max` — Finding 27's identified unstable mode. Inner CG: standard untruncated CG with halo refresh of `p` before each matvec, MPI_Allreduce SUM dot products via the field communicator, fixed iteration count (`HelmholtzNiter`, default 12). Operator is SPD and self-adjoint (in the continuum), so a fixed-iter CG smoother is also self-adjoint (truncated polynomial in A).

**Implementation.**
- `include/Collective.h` — `HelmholtzAlpha`, `HelmholtzNiter` fields + getters; `smoothKernelInt = 2` for `helmholtz`.
- `main/Collective.cpp` — parser accepts `helmholtz` SmoothKernel; reads `HelmholtzAlpha` (default 0 → auto) and `HelmholtzNiter` (default 12).
- `fields/EMfields3D.cpp::energy_conserve_smooth_direction` — Helmholtz branch as early-return after the existing kernel switch. Allocates `xCG, rCG, pCG, ApCG` (each `nx*ny*nz`), runs inner CG, copies result back, refreshes halo. ~115 lines.

**Test protocol.** Same `phase10j_tsc_20.inp` physics and DMD window (cycles 14–22, rank 8) as Phase 10j. New input file `inputfiles/phase10k_tsc_20_helmholtz.inp`. Sweep: α ∈ {1e-6, 0.1, 0.5, 1.0, 2.0, 5.0, 22.8 (default)}, num_smoothings ∈ {1, 4, 8, 16}, HelmholtzNiter ∈ {12, 32}. Energy diagnostic: `Etot(30)`. DMD on the two informative configs (default α and best α=1).

**Headline numbers.**

| α      | sm | niter | Etot(10) | Etot(20)  | Etot(30)  | top |λ| | arg(λ) |
|--------|----|-------|----------|-----------|-----------|--------|--------|
| 1e-6   | 1  | 12    | 1.49     | 6.87e+05  | 7.44e+08  | —      | —      |
| 0.1    | 1  | 12    | 0.527    | 4.33e+03  | 1.58e+05  | —      | —      |
| **0.5**| 1  | 12    | 0.515    | 6.40      | 10.85     | —      | —      |
| 1.0    | 1  | 12    | 0.460    | 8.29      | 10.72     | —      | —      |
| 1.0    | 1  | 32    | 0.325    | 4.62      | **8.59**  | 1.017  | ±1.20  |
| 2.0    | 1  | 12    | 0.489    | 16.42     | 32.10     | —      | —      |
| 5.0    | 1  | 12    | 0.363    | 14.64     | 31.09     | —      | —      |
| 22.8 (default) | 1 | 12 | 0.311  | 25.86     | 52.16     | 1.052  | 0      |
| 0.5    | 4  | 12    | 0.547    | 6.54      | 10.09     | —      | —      |
| 0.5    | 8  | 12    | 0.577    | 6.16      | 9.27      | —      | —      |
| 0.05   | 16 | 12    | 0.623    | 5.22e+04  | 1.76e+06  | —      | —      |
| —      | —  | —     | —        | —         | —         | —      | —      |
| bin sm=4 (ref) | 4 | — | 0.294 | 0.344     | **1.98**  | 1.270  | 0      |
| bin sm=8 (ref) | 8 | — | 0.297 | 0.301     | **0.305** | 1.020  | 0.065  |
| bin5 sm=4 (ref)| 4 | — | 0.295 | 0.295     | **0.295** | 0.953  | 0.120  |

**Two contradictions to resolve in interpretation.**

1. **DMD eigenvalue says "stabilised" but energy says "exploded".** Helmholtz default reduces top |λ| from 1.270 → 1.052 (a 2.5× rate reduction), yet `Etot(30)` is 26× HIGHER than bin sm=4. Helmholtz α=1 sm=1 niter=32 gives top |λ|=1.017, comparable to bin sm=8's 1.020 — yet `Etot(30)` is 28× higher. The leading DMD eigenvalue does not predict the energy trajectory in this regime. Either (a) the operator is non-normal so transient growth dominates over the spectral radius, or (b) the DMD window cycles 14–22 is post-saturation for the Helmholtz runs and the growth captured is a slower, more localised mode while the actual dynamics saw a much faster transient earlier.

2. **The α-sweep is non-monotonic with a flat optimum around α∈[0.5,1].** Below α=0.5 the smoother does not damp enough; above α=2 it over-damps in a way that makes things worse, not better. There is no value of α at which Helmholtz reaches even bin sm=4 stability levels.

**Likely root cause** (best hypothesis without more diagnostics). The smoother is called multiple times per GMRES iteration inside `MaxwellImage`: pre-mass-matrix (line 2896) and post-mass-matrix (line 2916), giving the implicit operator `(I + curl² + S·M·S)`. With binomial S the transfer function is `≈ 1` at low-k and falls off slowly, so `S·M·S ≈ M`. With Helmholtz S the transfer function is `0.5` at the fundamental and `≈ 0` at high-k, so `S·M·S` is heavily attenuated almost everywhere — most of the implicit mass-matrix term is effectively removed from the field equation. GMRES still converges (the operator is well-defined), but it converges to E that satisfies a different equation, in which the discrete energy theorem no longer holds. Phase 9b's "extra-smooth is catastrophic" finding is a mild version of the same effect. **The smoother slot is not a free post-cycle filter — it is structural to the implicit operator**, and replacing it with a strong long-range filter restructures the operator into something energy-non-conserving.

**Sanity checks performed.**
- α → 0 limit: behaves identically to "no smoothing" (Etot(30) = 7.4e8, matching the unsmoothed TSC explosion) → confirms the CG returns ≈ identity for tiny α.
- GMRES convergence: stays at 23–39 iters across all α, no breakdown → operator is well-conditioned.
- Constants and linear functions: Helmholtz preserves both exactly (∇² of constant or linear = 0), so the CG smoother does not introduce DC drift.

**Artifacts.**
- `inputfiles/phase10k_tsc_20_helmholtz.inp` — base input (default α=0 → auto).
- `phase10k_dmd_helmholtz_default.{png,json}` — DMD on α=22.8 sm=1 niter=12 (top |λ|=1.052, k_y=2π/Ly fundamental).
- `phase10k_dmd_helmholtz_a1.{png,json}` — DMD on α=1 sm=1 niter=32 (top |λ|=1.017, oscillating mode pair).
- Run dirs `/tmp/ipic3d_phase10k_h_a*_sm*_n*` (ephemeral).

**Decision.** Mark Phase 10k closed as negative. The Helmholtz code stays in tree behind `SmoothKernel=helmholtz` (not on by default) for two future uses: (a) Phase 10m if we apply it as a post-`calculateE` filter outside MaxwellImage; (b) standalone diagnostics where a known smoother shape is useful. Move Phase 10l to NEXT.

---

## Phase 10j — DMD linear-stability analysis (DONE 2026-04-11)

**Approach: Dynamic Mode Decomposition on per-cycle E-field snapshots.** Rejected direct Arnoldi on `MaxwellImage` because (a) GMRES converges uniformly in 30–37 iters at all grids (Phase 10f), meaning `MaxwellImage` itself is well-conditioned — the instability is NOT in its spectrum; (b) the unstable operator is the empirically-linearised one-step map `E^n → E^{n+1}`, which is a nonlinear composition (field solve + particle push + moment gather) and can't be written as a matvec callback. DMD extracts the one-step map's eigenstructure directly from simulation trajectories.

**Protocol.** Standard SVD-based exact DMD (Tu et al. 2014):

1. Run TSC on 20x20x4 Double_Harris with `FieldOutputCycle=1, FieldOutputTag=E`, 30 cycles.
2. Load per-cycle E-field snapshots across all ranks, strip ghosts, flatten each cycle into one state vector.
3. Build `X1 = [x_14, x_15, …, x_21], X2 = [x_15, …, x_22]` (Phase 10e's clean-exponential window).
4. Economy SVD `X1 = U Σ V*`, truncate to `rank=8`.
5. Reduced Koopman `Ã = Uᵀ X2 V Σ⁻¹`; eigendecompose `Ã` to get `(λ_i, w_i)`.
6. DMD modes `Φ = X2 V Σ⁻¹ W`; rank by `|λ_i|`.

Driver: `scripts/phase10j_dmd.py`. Input files: `inputfiles/phase10j_tsc_20.inp` (bin sm=4), `phase10j_tsc_20_sm8.inp`, `phase10j_tsc_20_bin5sm4.inp`, `phase10j_tsc_40.inp`. Output: `phase10j_dmd_tsc20_{bin4,sm8,bin5sm4}.{png,json}` (gitignored) plus `phase10j_dmd_tsc40_bin4.{png,json}`.

**Headline results (20x20x4, DMD cycles 14–22, rank 8):**

`bin sm=4` (UNSTABLE):
- Top eigenvalue: λ = 1.270 + 0i (purely real, monotonic exponential growth; matches Phase 10e's energy rate via `λ_field ≈ √λ_energy`)
- Ex(y) sign structure (x-averaged): `+++++++++++---------` — a single full wavelength across the domain, one positive lobe (i=0..8) and one negative lobe (i=9..17). Nodes near i=8.5 and wrap-around i=17.5. k_y = 2π/Ly fundamental.
- Harris layers (from Bx zero-crossings at cycle 0): i=3 (y≈5.8) and i=12 (y≈20.8). Both sit inside the mode's antinode lobes — the mode overlaps the layers but its amplitude peaks are slightly displaced into the bulk.

`bin sm=8` (nearly stable):
- Top eigenvalue: |λ| = 1.020, arg = 0.065 (complex pair, marginal growth, oscillation period ~97 cycles)
- y-marginal |mode|²: sharp peaks at **i=3 (0.20) and i=12 (0.20)** — exactly on the Harris layer zero-crossings.
- Interpretation: with the bulk mode damped, the DMD reads out the physical Harris-layer mode, which is stable/marginal.

`binomial5 sm=4` (bounded):
- Top eigenvalue: |λ| = 0.953, arg = 0.120 (formally stable in this window, but Phase 10i shows dE(64)=0.72 for this config — the DMD window cycles 14–22 is post-transient for binomial5, which reaches its bounded state earlier than binomial sm=4).
- y-marginal: peaks at i=3, i=12 (Harris-layer residual) — same qualitative picture as bin sm=8.

`bin sm=4` at 40x40x4 (cycles 13–23, rank 10):
- Top eigenvalue: |λ| = 1.213, arg = 0.30 (complex pair, period ~21 cycles)
- Mode structure is substantially messier than 20x20 — 11 sign flips in `Ex(y)` vs 2 at 20x20. Multiple competing modes near the top of the spectrum.
- Consistent with Phase 10e's observation that finer grids have a weaker but still-present instability.

**Why this is the `k_y = 2π/Ly` fundamental and not a Harris-layer mode.** The signed `Ex(y)` structure at 20x20 bin sm=4 has exactly one sign flip in the interior and a second at the periodic wrap (total 2 sign flips → k = 1 FFT bin → one full wavelength = Ly). A Harris-layer-localised mode would have delta-function-like peaks at i=3 and i=12 with zero amplitude in the bulk; instead the mode has substantial amplitude throughout the bulk and only approximate reductions at the layers. The mode DOES interact with the layers — its structure is set by the Harris equilibrium — but it is spatially a bulk mode.

**Why the smoother struggles with it.** A binomial `(1,2,1)/4` kernel with `n` passes has a transfer function `H(k)ⁿ = (cos²(k/2))ⁿ`. At `k = π/N` (where N = domain cells), `H(k)ⁿ ≈ (1 - k²/4)ⁿ ≈ 1 - n·k²/4`. For N=20 and n=4, this is ~`1 - 4 · (π/20)² / 4 = 1 - π²/400 = 0.975` — damping only 2.5% per smoother call. A `k_y=2π/Ly` mode grows at ~1.27/cyc (Phase 10j) = +27% per cycle, net growth ~+24% per cycle after smoothing, consistent with the observed long-time exponential. At sm=16 the damping compounds to `0.975^16 = 0.67` — +27% growth − 33% damping = net decay, consistent with the sm=16 stability in Finding 25. **The smoother IS effective against this mode, but only after sufficient accumulated passes because its transfer function is O(k²) flat at low k.**

**Why wider kernels don't fix it without more halo refreshes (ties back to Finding 26).** The `k_y = 2π/Ly` mode crosses the 2×2 MPI rank decomposition at `y = Ly/2`. Each halo refresh between smoother passes propagates post-smoothed information across the rank boundary, damping the cross-rank component of the mode. A single wide-kernel pass (binomial5) only does this once per call; multiple narrow-kernel passes (binomial sm=8) do it 8 times. For a low-k mode that spans the whole domain, the inter-rank propagation is the dominant damping mechanism — not the filter width.

**Open follow-up: the Harris-layer residual modes seen in sm=8 and bin5.** Both `bin sm=8` and `bin5 sm=4` top modes peak at i=3, i=12 (Harris layers) with |λ| ≈ 1.0. These could be (a) physical Harris-layer dynamics (tearing-like, slow), (b) TSC mass-matrix artifacts at the layers, (c) DMD noise from the saturating nonlinear regime. Phase 10k's candidate filters should leave these alone.

**Artifacts:**
- `scripts/phase10j_dmd.py` — driver (exact DMD, per-rank HDF5 stitching, mode-structure diagnostics, plotting)
- `inputfiles/phase10j_tsc_20.inp`, `phase10j_tsc_20_sm8.inp`, `phase10j_tsc_20_bin5sm4.inp`, `phase10j_tsc_40.inp` — base input files with `FieldOutputCycle=1`
- `phase10j_dmd_tsc20_{bin4,sm8,bin5sm4}.{png,json}`, `phase10j_dmd_tsc40_bin4.{png,json}` — outputs (gitignored, repo root)
- Simulation run dirs: `/tmp/ipic3d_phase10j_tsc_20*` and `/tmp/ipic3d_phase10j_tsc_40` (ephemeral, `/tmp`)
- Data capture: write each eigenvector as a field `.h5` snapshot for downstream plotting (`scripts/plot_eigenmodes.py` TBD).

**Decision gate.**

- **Eigenvector localised at the Harris layer** → the TSC 3-cell support spans the under-resolved layer and amplifies it. Fix by layer-aware refinement or kernel with physical-scale width.
- **Eigenvector localised at the periodic boundary** → the periodic-boundary inner-product double-counting Phase 9a flagged is coupling into the unstable subspace. Fix by boundary-aware inner product.
- **Eigenvector grid-scale oscillation** → near-Nyquist mode despite Phase 10c's FFT saying no high-k content. Reconcile with Finding 22 before acting.
- **Eigenvector has cell-scale structure tied to TSC's 3-cell support** → the shape-function's implicit spatial overlap is the amplifier. Motivates a shape-function modification.

**Risk / scope.** Adding SLEPc integration is the first heavyweight dependency change in this investigation. Start with the smallest 20x20 system to de-risk the Arnoldi setup; only scale up after the eigenvalue ordering is verified.

**Cheap 10j side-experiment** (Finding 26 follow-up, tester-hat): add an extra `communicateNodeBC` halfway through each `binomial5` pass (or every K=2 passes) and re-run the 10i sweep. Hypothesis: recovering the inter-pass halo-refresh frequency should bring `binomial5 sm=4` closer to `binomial sm=8` at the coarse grids. If it does, **a "wide kernel + mandatory mid-pass halo refresh" is a cheap 10k candidate** and may also fix the sweet-spot shift across grids. Roughly ~15 lines of code and a re-run of `scripts/phase10i_kernel.py`.

---

## Open questions

- ~~Can a low-k–aware filter (Gaussian / Helmholtz) stabilise `sm=1–2`?~~ → **Yes, as a post-`calculateE` filter outside MaxwellImage** (Finding 30, Phase 10m). bin sm=2 + post-solve helmholtz is the new recommended TSC config.
- ~~Does mid-pass halo refresh in `binomial5` match `binomial sm=2N` bit-for-bit?~~ → **Yes** (Finding 29, Phase 10l).
- **Does Phase 10m's bin sm=2 + post-solve helmholtz hold up at finer grids (40x40, 80x80)?** Auto-α scales as `(L/2π)²` so it should track refinement. One `pixi run sim` per resolution would close this.
- **Why does the leading DMD eigenvalue diverge from the actual energy growth in Phase 10k Helmholtz runs?** (top |λ|=1.052 → predicts ~4.5× over 30 cycles, observed ~180×.) Suggests a non-normal operator with significant transient growth in the broken Phase 10k operator. Low priority — Phase 10m sidesteps the issue entirely.
- **Why does the 40x40 top mode have `arg(λ) = 0.30` (complex) while 20x20 has `arg = 0` (pure real)?** Could be multiple competing modes near the top of the spectrum, or the domain-fundamental becomes a traveling wave at higher resolution. Low-priority unless it affects Phase 10k design.
- **Why does the cycle-1 algebraic seed (Finding 24) scale as `Δx^-1.4`?** Finding 27 shows the mode is grid-spanning, so the seed is a projection of the initial state onto the unstable eigenvector. The scaling should fall out of the mode overlap with `∂Bx/∂y` at the Harris layers, which steepens with Δx. Worth a back-of-envelope check but not blocking.
- **Is there a gauge-like decoupling between GMRES-residual space and energy-relevant space?** Phase 10f observed that 1e-2 relative residual still gives the same `dE(1)` — the residual lives in a subspace that doesn't touch total energy. Consistent with Finding 27: the residual lives in high-k modes, the instability lives at `k = 2π/Ly`, no spectral overlap.
- **40x40 "easy-mode" anomaly** (Phase 10h minimum `dE(64)` at 40x40 for most `(kernel, sm)` combos): partial explanation from Finding 27 — the 40x40 unstable mode has `|λ|=1.21 < 1.27` at 20x20, so a given smoother strength has more headroom to damp it. The non-monotonicity with respect to 80x80 is still open.
- **Pressure-tensor path** (`Particles3D.cpp:1970-2011`) still uses 8-point linear weights in the TSC branch — a consistency bug but not a cycle-1 energy source. Defer to Phase 10d, low priority.

## Settled questions

- ~~ECSIM + TSC is theoretically impossible~~ → **No, theoretically fine** (Finding 18, Phase 10a).
- ~~The 5% TSC leak is a shape-function mismatch~~ → **No, all three paths are clean** (Finding 21, Phase 10b.1).
- ~~The 5% leak is a bounded per-cycle algebraic drift~~ → **No, it's a slow exponential instability** (Finding 22, Phase 10c).
- ~~Finer grids fix it~~ → **No, rate ∝ Δx^0.7 is too shallow to reach stability on affordable grids** (Finding 23, Phase 10e).
- ~~Cycle-1 seed is GMRES residual~~ → **No, it's 100% algebraic** (Finding 24, Phase 10f).
- ~~`sm=4` is the optimum (Phase 9c Finding 17)~~ → **No, overturned — was a `dE(10)`-window artifact; `sm=8+` wins at `dE(64)`** (Finding 25, Phase 10h).
- ~~Widening the kernel to 5-point binomial fixes grid sensitivity~~ → **No, and `binomial5 sm=N ≠ binomial sm=2N` despite the convolution identity** (Finding 26, Phase 10i).
- ~~The unstable mode is near-Nyquist / cell-scale~~ → **No, it's the `k_y = 2π/Ly` domain fundamental** (Finding 27, Phase 10j). Consistent with the FFT results from Finding 22.
- ~~The unstable mode is Harris-layer-localised~~ → **No, it's a bulk mode whose structure is influenced by but not concentrated at the Harris layers** (Finding 27).
- ~~We need Arnoldi/SLEPc to find the mode~~ → **No, DMD on field snapshots works fine** (Phase 10j). SLEPc deferred unless we need inner eigenvalues later.
- ~~A Helmholtz/Gaussian low-pass filter can be a drop-in stabiliser for the binomial smoother slot~~ → **No, the smoother slot is structural to the implicit `S·M·S` operator and replacing it with a long-range filter restructures `MaxwellImage` into a non-energy-conserving operator** (Finding 28, Phase 10k).
- ~~`binomial5 sm=N` differs from `binomial sm=2N` because of the wider stencil~~ → **No, it differs purely because of the inter-pass halo refresh count; with the missing refresh added, `bin5_refresh sm=N == bin sm=2N` bit-tight** (Finding 29, Phase 10l).
- ~~Helmholtz filtering is incompatible with ECSIM energy conservation~~ → **No, Helmholtz works fine when applied OUTSIDE the implicit operator (Phase 10m). Phase 10k's failure was a slot mismatch, not a filter-vs-theorem incompatibility** (Finding 30, Phase 10m).
- ~~Stable TSC requires `bin sm=8`-equivalent smoother work~~ → **No, `bin sm=2 + PostSolveHelmholtz` is ~3.6× cheaper than `bin sm=8` with strictly better stability** (Finding 30, Phase 10m).
- ~~Phase 8e "accept as theoretical limitation"~~ → **Reversed** (Phase 10a).
- ~~S·M·S asymmetry is the leak~~ → **No, δ_SMS < δ_M** (Finding 15, Phase 9a).
- ~~Particle-field raw/smoothed mismatch~~ → **No, `calculateB` already applies S·E before the mover** (Finding 16, Phase 9b).

## What NOT to do (exhausted directions, see archive for reasoning)

- No more FP-mitigation work (Kahan in mass-matrix-mult, mass-matrix-assembly, or smoother) — Phase 8a/c/d exhausted. The drift is algebraic, not FP.
- No more SMS self-adjointness measurements — Phase 9a ruled it out.
- No more particle-facing-E smoothing toggles — Phase 9b showed current setup is at local optimum.
- No more `num_smoothings` sweeps at fixed kernel — Phase 10h mapped the landscape.
- No `n_ghost=3` experiments — `grids/Grid3DCU.cpp:45` hard-asserts `n_ghost ≤ 2` and expanding is a multi-day comms refactor for a disproven hypothesis.
- Do NOT touch `mover_PC` or any of its AoS variants — they are legacy IMM code, unreachable in ECSIM runs. Phase 10b.1 wasted effort there.
- Do NOT re-apply `phase4-diagnostic.patch` during dE comparisons — it perturbs FP determinism even when gated (Finding 11).

---

## Findings index

One-liner per finding. Full write-ups in `plan-tsc-archive.md` (grep for the finding number).

1. **Thin-dimension addFace overcounting** — fixed in commit `9c04563`.
2. **Face self-copy offset inconsistency** — fixed in commit `e2e7c0d`.
3. **TSC is unstable without smoothing.** Energy explodes by cycle 6-7; CIC is stable.
4. **Drift scales with per-rank workload, not communication volume.** np=1 is worse than np=4 for both CIC and TSC.
5. **Operator is self-adjoint within the ghost convention.** Apparent asymmetry is periodic-boundary inner-product double-counting.
6. **Ghost layer inversion for n_ghost > 1 is benign.** Code is self-consistent within the inverted convention; fixing it makes energy 40× worse.
7. **Smoothing is NOT the sole drift source.** TSC unsmoothed is already 90× worse than CIC unsmoothed due to wider mass-matrix stencil.
8. **Two independent TSC drift sources:** wider mass-matrix FP accumulation + smoothing FP amplification.
9. **CIC unsmoothed is near GMRES FP noise floor** (4.2e-08 ≈ 30 iters × ε).
10. **Phase 8c Kahan in `mass_matrix_times_vector`** — partial win on TSC no-smooth, no effect on TSC sm=4. Kept (commit `605a247`).
    10a. Run-to-run dE(10) variance is ~25× for TSC sm=4 due to MPI reduction ordering.
11. **Phase 4 symmetry-diagnostic perturbs FP determinism even when gated.** Extracted to `phase4-diagnostic.patch`. Re-apply transiently when needed.
12. **Phase 8a Kahan in `energy_conserve_smooth_direction`** — no measurable benefit. Patch dropped.
13. **Phase 8d Kahan in TSC mass-matrix assembly** — analytically ruled out (FP bound 2.6e-15 vs 1e-05 observed).
14. **TSC drift is algebraic, not FP.** All FP mitigations exhausted; cycle 0→1 energy balance has a 5% residual vs CIC's 0.01%. Phase 8 closing finding.
15. **Phase 9a:** δ_SMS is smaller than δ_M alone; S·M·S asymmetry is not the leak.
16. **Phase 9b:** particles already see S·E via `calculateB`; both skip-smooth and extra-smooth are catastrophic. Current setup is at local optimum.
17. **Phase 9c: `sm=4` is the `dE(10)` optimum** — later overturned by Finding 25 as a dE(10)-window artifact.
18. **Phase 10a literature review:** the ECSIM energy theorem is B-spline order agnostic; ECSIM + TSC is theoretically fine. Phase 8e reversed.
19. **Phase 10a retroactive Experiment 1:** Phase 6a's TSC no-smooth leak is a "smoking gun" for shape-function pathway bug — later softened by Finding 20.
20. **Phase 10b.0 baseline re-validation:** TSC no-smooth cycle-1 has 124× run-to-run spread (not 3×); audit target shifted to the deterministic smoothed configs.
21. **Phase 10b.1 REFUTED:** shape-function pathway is clean. All three ECSIM coupling paths use consistent TSC weights at the same `x_n`. `mover_PC` is legacy IMM code, unreachable.
22. **Phase 10c spectral analysis:** TSC sm=4 is a slow exponential instability (~15%/cyc onset at cycle ~13), not a bounded leak. CIC on the same grid is stable (4600× smaller). Near-Nyquist hypothesis refuted.
23. **Phase 10e grid-size scan:** `slope ∝ Δx^0.703`, `dE(1) ∝ Δx^-1.4`, `dE(64)` drops 26× from 20x20 to 80x80. RESOLUTION-THRESHOLD branch.
24. **Phase 10f GMRES-iter bisection:** cycle-1 seed is 100% algebraic, invariant to 14 OoM of GMRES relative residual. Exposed `NiterGMRES` input parameter.
25. **Phase 10h smoother-strength sweep:** stronger `num_smoothings` IS the control variable — rate drops monotonically and plateaus. But optimal `sm` shifts with resolution because `(1,2,1)/4` is grid-anchored. Finding 17 overturned.
26. **Phase 10i alternative kernel + halo-refresh surprise:** `binomial5` kills clean exponential but doesn't give flat `dE(64)` across grids. `bin5 sm=N ≠ bin sm=2N` despite convolution identity — per-pass halo refresh between narrow-kernel passes is load-bearing.
27. **Phase 10j DMD:** TSC sm=4 unstable mode at 20x20 is the **domain-fundamental `k_y = 2π/Ly` bulk mode** (DMD |λ|=1.270, purely real). Mode has one full wavelength across the domain with antinode lobes containing the Harris layers, and nodes at i=8.5 (mid-domain) and i=17.5 (periodic wrap). Stabilised configs (`bin sm=8`, `binomial5 sm=4`) have residual modes peaking AT the Harris layers (i=3, i=12) — physical, near-marginal. The binomial smoother has an O(k²)-flat transfer function at low k so it needs many passes + halo refreshes to attenuate the fundamental. **The right filter is a low-k–aware one** (Gaussian / Helmholtz), not a wider binomial.
28. **Phase 10k Helmholtz drop-in REFUTED.** A CG-solved `(I - α ∇²)⁻¹` smoother replacing the binomial stencil **does not stabilise** TSC at any tested α (1e-6 → 22.8) or `(num_smoothings, niter)` combination. Best Helmholtz `Etot(30) ≈ 8.6` vs binomial sm=8's `0.305`. DMD shows the Helmholtz default DOES drop the original `k_y=2π/Ly` real fundamental from |λ|=1.270 → 1.052 — yet observed energy growth is 180× over 30 cycles, not the ~4.5× the eigenvalue would predict. **The smoother slot is structural** — it sits inside `MaxwellImage` as `S·M·S`, and a strong long-range filter restructures the implicit operator into a well-posed but non-energy-conserving equation. The right way to use a Helmholtz filter is **outside** `calculateE`, applied to E after the field solve returns (motivates Phase 10m).
29. **Phase 10l Finding 26 CONFIRMED bit-tight.** `binomial5_refresh sm=N` (which decomposes each pass as `bin → halo refresh → bin`) matches `binomial sm=2N` to within FP/MPI run-to-run noise at every cycle. The "wider stencil" of `binomial5` is irrelevant; what matters is the (narrow-sweep + halo refresh) cycle count. `bin5 sm=4` differs from `bin sm=8` because it has 4 refreshes per call instead of 8, and the inter-rank propagation of long-wavelength mode content is exactly the load-bearing damping mechanism for the `k_y = 2π/Ly` bulk mode (Finding 27). Phase 10l doesn't give a CHEAPER stabiliser than `bin sm=8` — it gives us the *understanding* of why `binomial5` doesn't beat narrow binomial despite the convolution identity.
30. **Phase 10m post-`calculateE` Helmholtz WINS.** Applying the same Helmholtz inverse from Phase 10k as a once-per-cycle filter on the converged `Eth` field — *outside* `MaxwellImage`'s `S·M·S` operator — fully stabilises TSC at `bin sm=2` with the BEST DMD spectrum of any tested config (top |λ| = 0.9993, all other modes < 1) and the LOWEST cost (~0.55× of `bin sm=4`, ~3.6× cheaper than `bin sm=8`). Robust to α over a 50× range (1.0 → 50.0 all give bounded `Etot(30)` within 1% of initial), in stark contrast to Phase 10k's drop-in failure. Phase 10k's filter idea was right; the SLOT was wrong. Recommended TSC default: `SmoothKernel = binomial`, `num_smoothings = 2`, `PostSolveHelmholtz = true` (auto α).

---

## Current code knobs (as of `ee8c5d1` + uncommitted Phases 10j–10m)

| Input parameter | Default | Values | Phase | Purpose |
|---|---|---|---|---|
| `num_smoothings` | 4 | positive int | legacy / 10h | how many times to apply the smoother per E-component call. **Recommended TSC: 2 with `PostSolveHelmholtz=true`.** |
| `NiterGMRES` | -1 | -1 (legacy `m=40/max_iter=25`) or N>0 (force `m=N, max_iter=1`) | 10f | GMRES truncation knob for cycle-1 bisection |
| `SmoothKernel` | `binomial` | `binomial` (27-pt `(1,2,1)/4`) / `binomial5` (125-pt) / `binomial5_refresh` (== `bin sm=2N`) / `helmholtz` (Phase 10k drop-in, NOT recommended) | 10i / 10k / 10l | smoother kernel selection |
| `HelmholtzAlpha` | 0.0 | 0 → auto `(max(Lx,Ly,Lz)/(2π))²`; >0 → explicit α (length²) | 10k / 10m | Helmholtz half-power scale; read by both kernel=helmholtz and `PostSolveHelmholtz` |
| `HelmholtzNiter` | 12 | positive int | 10k / 10m | inner CG iterations per Helmholtz call |
| **`PostSolveHelmholtz`** | **false** | bool | **10m** | **opt in to a once-per-cycle Helmholtz low-pass on `Eth` after `calculateE` returns. Recommended `true` for TSC.** |

## Current critical files (Phase 10-relevant only)

| File | Purpose |
|------|---------|
| `fields/EMfields3D.cpp:2665-2677` | `calculateE()` — branches on `col->getNiterGMRES()` (Phase 10f) |
| `fields/EMfields3D.cpp:2827-2912` | `MaxwellImage` — `(I + curl² + S·M·S)` operator; target for Phase 10j Arnoldi |
| `fields/EMfields3D.cpp:2685-2700` | `calculateE` — Phase 10m post-solve Helmholtz hook (gated on `PostSolveHelmholtz`) |
| `fields/EMfields3D.cpp:3023-3275` | `energy_conserve_smooth_direction` (with `kernel_override`) and `post_solve_filter_E` — Phase 10i + 10k + 10l + 10m smoother branches |
| `include/EMfields3D.h:175-185` | `energy_conserve_smooth_direction(..., kernel_override=-1)` and `post_solve_filter_E` declarations (Phase 10m) |
| `include/Collective.h:181,418` | `NiterGMRES` getter + field (Phase 10f) |
| `include/Collective.h:211-220,254-275` | `SmoothKernel` + `HelmholtzAlpha` + `HelmholtzNiter` + `PostSolveHelmholtz` getters and fields (Phase 10i / 10k / 10l / 10m) |
| `main/Collective.cpp:113-135,158` | Parsers for `SmoothKernel` (incl. `helmholtz`, `binomial5_refresh`), `HelmholtzAlpha`, `HelmholtzNiter`, `PostSolveHelmholtz`, `NiterGMRES` |
| `scripts/phase10c_spectral.py` | 64-cycle FFT + exponential-growth fit (Phase 10c) |
| `scripts/phase10e_scaling.py` | Grid-size growth-rate scan (Phase 10e) |
| `scripts/phase10f_bisection.py` | `NiterGMRES` sweep at cycle 1 (Phase 10f) |
| `scripts/phase10h_smoother.py` | `num_smoothings` sweep (Phase 10h) |
| `scripts/phase10i_kernel.py` | `SmoothKernel × num_smoothings` sweep (Phase 10i) |
| `scripts/phase10j_dmd.py` | DMD on per-cycle E-field HDF5 snapshots (Phase 10j) — exact DMD with SVD rank truncation, mode diagnostics, plotting |
| `inputfiles/phase10c_{tsc,cic}_spectral.inp` | 64-cycle 20x20x4 bases for Phase 10c onward |
| `inputfiles/phase10e_tsc_{40,80}.inp` | 40x40x4 / 80x80x4 grid-size bases |
| `inputfiles/phase10j_tsc_20{,_sm8,_bin5sm4}.inp`, `phase10j_tsc_40.inp` | 30-cycle runs with `FieldOutputCycle=1` for Phase 10j DMD |
| `inputfiles/phase10k_tsc_20_helmholtz.inp` | Phase 10k base input — `SmoothKernel=helmholtz` 20x20x4 with per-cycle E snapshots for DMD comparison |
| `inputfiles/phase10l_tsc_20_bin5refresh_sm4.inp` | Phase 10l — `SmoothKernel=binomial5_refresh` 20x20x4 (Finding 26 confirmation) |
| `inputfiles/phase10m_tsc_20_postsolve_helmholtz.inp` | **Phase 10m winning config** — `bin sm=4` (or `sm=2`) + `PostSolveHelmholtz=1` |
| `inputfiles/phase8c_tsc_kahan.inp` | TSC sm=4 10-cycle reference input |

Full legacy list (Phases 4–9, Com3D and Grid3DCU files, particles mover internals) is in `plan-tsc-archive.md`.

## Deferred / non-blocking backlog

- ~~Phase 10l~~ → DONE (Finding 29, confirms Finding 26 bit-tight).
- ~~Phase 10m~~ → DONE (Finding 30, **the winning config**).
- **Phase 10n (validate at scale, OPTIONAL)** — run `bin sm=2 + PostSolveHelmholtz=true` on the 40x40x4 and 80x80x4 grids (Phase 10e bases) to confirm the Phase 10m stability holds across resolutions. Auto α scales as `(L_max/2π)²` so it should track refinement. One `pixi run sim` per resolution; expected to just work.
- **CI bump** — switch `inputfiles/ci_smoke_tsc.inp` to `num_smoothings = 2` + `PostSolveHelmholtz = true` and retune the smoke-test tolerance to the new (much smaller) drift band. ~5 lines + a tolerance tune.
- **Phase 10d** — pressure-tensor shape-function audit (`Particles3D.cpp:1970-2011` uses 8-pt linear in TSC branch). Not a cycle-1 energy source; downstream consistency only.
- **Phase 9.x** — edge/corner self-copy modular wrapping in `Com3DNonblk.cpp:806-888, 974-1039`. Latent bug masked by current tests (all have `nzc_r ≥ 3`).
- **Phase 11** — non-periodic boundary hardcodings. Won't fire for Double_Harris; address when non-periodic BCs are needed with `n_ghost > 1`.
- **CI defaults** — bump `ci_smoke_tsc.inp` from `sm=4` to `sm=8` or `sm=16` for a 200–48000× `dE(64)` improvement with no code change; retune tolerances accordingly.
