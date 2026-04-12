"""
Phase 10j: Dynamic Mode Decomposition (DMD) on per-cycle E-field snapshots.

Goal: extract the dominant eigenpair(s) of the empirically-linearised ECSIM
one-step map at the TSC sm=4 Double_Harris equilibrium. Phase 10c–10i have
traced the instability under every input-level knob; 10j asks what spatial
pattern the unstable mode actually has.

Method: standard SVD-based DMD (Schmid 2010). Given snapshots
    X  = [x_k0, x_k0+1, ..., x_k1-1]
    X' = [x_k0+1, x_k0+2, ..., x_k1]
where x_k is the flattened E-field at cycle k, we compute the reduced
Koopman operator via
    U, Σ, V* = svd(X)        (rank-r truncated)
    Ã        = U* X' V Σ⁻¹
    λ, w     = eig(Ã)
and the DMD modes are the columns of
    Φ        = X' V Σ⁻¹ w
with |λ_i| giving the per-cycle growth factor of mode i. Phase 10e's
exponential-phase fit at 20x20 reported rate ≈ 1.886×/cyc over cycles 14–22;
the dominant DMD eigenvalue should recover that number and its eigenvector
reveals the spatial structure that the smoother needs to damp.

Usage:
    pixi run python scripts/phase10j_dmd.py \
        --run-dir /tmp/ipic3d_phase10j_tsc_20 \
        --cycles 14 22 \
        --rank 10 \
        --out phase10j_dmd_tsc20.png

Assumes serial HDF5 output: proc{rank}.hdf with fields/{Ex,Ey,Ez}/cycle_{N}.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


# --------------------------------------------------------------------------
# Snapshot loading (multi-rank, per-cycle)
# --------------------------------------------------------------------------

def find_procs(run_dir: Path) -> list[Path]:
    procs = sorted(run_dir.glob("proc*.hdf"))
    if not procs:
        raise SystemExit(f"no proc*.hdf in {run_dir}")
    return procs


def load_topology(run_dir: Path) -> tuple[int, int, int]:
    """Read per-rank cartesian coords from settings.hdf (fallback: from proc0 topology)."""
    s = run_dir / "settings.hdf"
    if s.exists():
        with h5py.File(s, "r") as f:
            if "collective/topology" in f:
                xlen = int(f["collective/topology/XLEN"][()])
                ylen = int(f["collective/topology/YLEN"][()])
                zlen = int(f["collective/topology/ZLEN"][()])
                return xlen, ylen, zlen
    # Fallback: count procs, assume square xy
    procs = find_procs(run_dir)
    n = len(procs)
    # Phase 10j test: 2x2x1
    if n == 4:
        return 2, 2, 1
    raise SystemExit(f"cannot determine topology from {run_dir}")


def per_proc_cart_rank(proc: Path) -> tuple[int, int, int]:
    with h5py.File(proc, "r") as f:
        c = f["topology/cartesian_coord"][...]
    return int(c[0]), int(c[1]), int(c[2])


def load_snapshot(run_dir: Path, cycle: int, component: str,
                  skip_ghosts: bool = True) -> np.ndarray:
    """Load a single field component at a single cycle, stitched across all ranks.

    Returns a numpy array of shape (nxn_global, nyn_global, nzn_global).
    Ghost cells are stripped by default (skip 2 outer layers per dim per rank).
    """
    procs = find_procs(run_dir)
    ng = 2  # TSC default
    # Get per-proc shape from proc0
    with h5py.File(procs[0], "r") as f:
        dset_name = f"fields/{component}/cycle_{cycle}"
        if dset_name not in f:
            raise KeyError(f"{dset_name} not in {procs[0]}")
        shape0 = f[dset_name].shape
    nxn_r, nyn_r, nzn_r = shape0
    if skip_ghosts:
        inner = (nxn_r - 2 * ng, nyn_r - 2 * ng, nzn_r - 2 * ng)
    else:
        inner = shape0

    # Topology
    xlen, ylen, zlen = load_topology(run_dir)
    nxn_g = inner[0] * xlen
    nyn_g = inner[1] * ylen
    nzn_g = inner[2] * zlen
    global_arr = np.zeros((nxn_g, nyn_g, nzn_g), dtype=np.float64)

    for proc in procs:
        cx, cy, cz = per_proc_cart_rank(proc)
        with h5py.File(proc, "r") as f:
            arr = f[f"fields/{component}/cycle_{cycle}"][...]
        if skip_ghosts:
            arr = arr[ng:nxn_r - ng, ng:nyn_r - ng, ng:nzn_r - ng]
        ix0 = cx * inner[0]
        iy0 = cy * inner[1]
        iz0 = cz * inner[2]
        global_arr[ix0:ix0 + inner[0], iy0:iy0 + inner[1], iz0:iz0 + inner[2]] = arr

    return global_arr


def build_snapshot_matrix(run_dir: Path, cycles: list[int],
                          components: list[str],
                          subtract_mean: bool = False) -> tuple[np.ndarray, tuple]:
    """Stack (cycle, component) into a flat column matrix.

    Returns (X, shape_per_component) where X has shape (N_dof, N_snapshots)
    and shape_per_component is (nxn, nyn, nzn) of a single component for
    later un-flattening of DMD modes.
    """
    cols = []
    shape_pc = None
    for cyc in cycles:
        vec_parts = []
        for comp in components:
            a = load_snapshot(run_dir, cyc, comp)
            if shape_pc is None:
                shape_pc = a.shape
            vec_parts.append(a.ravel())
        cols.append(np.concatenate(vec_parts))
    X = np.column_stack(cols)
    if subtract_mean:
        X = X - X.mean(axis=1, keepdims=True)
    return X, shape_pc


# --------------------------------------------------------------------------
# DMD core
# --------------------------------------------------------------------------

@dataclass
class DMDResult:
    eigenvalues: np.ndarray        # (r,) complex — Koopman eigenvalues
    modes: np.ndarray              # (N_dof, r) complex — DMD modes Φ
    amplitudes: np.ndarray         # (r,)  complex — b_i so that x_k = Φ · Λ^k · b
    U: np.ndarray
    sigma: np.ndarray
    Vh: np.ndarray
    rank: int


def dmd(X: np.ndarray, rank: int | None = None) -> DMDResult:
    """Standard exact DMD (Tu et al. 2014).

    X has shape (N_dof, N_snapshots). We split into
        X1 = X[:, :-1]
        X2 = X[:, 1:]
    and find A s.t. X2 ≈ A X1.
    """
    if X.shape[1] < 2:
        raise ValueError("need at least 2 snapshots")
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Economy SVD of X1
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    max_r = min(len(s), X1.shape[1])
    if rank is None:
        rank = max_r
    else:
        rank = min(rank, max_r)
    # Truncate near-zero singular values
    tol = s[0] * max(X1.shape) * np.finfo(float).eps
    r_eff = int(np.sum(s[:rank] > tol))
    rank = max(1, min(rank, r_eff))

    Ur = U[:, :rank]
    sr = s[:rank]
    Vhr = Vh[:rank, :]

    # Reduced Koopman: Ã = Ur^T X2 Vr Σr⁻¹
    A_tilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / sr)
    lam, W = np.linalg.eig(A_tilde)

    # Exact DMD modes (projected): Φ = X2 Vr Σr⁻¹ W
    Phi = X2 @ Vhr.conj().T @ np.diag(1.0 / sr) @ W

    # Initial amplitudes b = pinv(Φ) x0
    b, *_ = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)

    return DMDResult(
        eigenvalues=lam,
        modes=Phi,
        amplitudes=b,
        U=U, sigma=s, Vh=Vh, rank=rank,
    )


# --------------------------------------------------------------------------
# Post-processing / reporting
# --------------------------------------------------------------------------

def sort_by_modulus(res: DMDResult) -> np.ndarray:
    order = np.argsort(-np.abs(res.eigenvalues))
    return order


def mode_energy(res: DMDResult, n_steps: int) -> np.ndarray:
    """Rough ranking by amplitude × mode norm × max eigenvalue weight across window."""
    lam = res.eigenvalues
    b = res.amplitudes
    mode_norms = np.linalg.norm(res.modes, axis=0)
    # integrate |b λ^k|^2 over k∈[0,n_steps-1]; for λ close to 1 this is ~n_steps*|b|^2
    weight = np.abs(b) * mode_norms
    return weight


def unflatten_mode(mode_vec: np.ndarray, shape_pc: tuple, n_components: int) -> np.ndarray:
    """Given a flattened DMD mode with `n_components` concatenated field components,
    return an array of shape (n_components,) + shape_pc."""
    N_pc = int(np.prod(shape_pc))
    out = np.zeros((n_components,) + shape_pc, dtype=mode_vec.dtype)
    for i in range(n_components):
        out[i] = mode_vec[i * N_pc:(i + 1) * N_pc].reshape(shape_pc)
    return out


def localisation_diagnostics(mode_3d: np.ndarray) -> dict:
    """Classify where the mode concentrates its energy.

    mode_3d has shape (n_components, nxn, nyn, nzn). We look at |mode|^2.
    """
    mag2 = (np.abs(mode_3d) ** 2).sum(axis=0)
    total = mag2.sum()
    if total == 0:
        return {"concentration": 0.0}
    mag2 = mag2 / total

    # Marginalise
    py = mag2.sum(axis=(0, 2))   # energy vs y
    px = mag2.sum(axis=(1, 2))   # energy vs x
    pz = mag2.sum(axis=(0, 1))   # energy vs z

    # Harris layer is at y = Ly/2 and y = 0 (double)
    nyn = mag2.shape[1]
    y_mid = nyn // 2
    harris_band = slice(max(0, y_mid - 2), min(nyn, y_mid + 3))
    harris_share = float(py[harris_band].sum())

    # Top cells: fraction of energy in the 10% most-energetic cells
    flat = np.sort(mag2.ravel())[::-1]
    top10 = flat[:max(1, len(flat) // 10)].sum()

    # "Spread": effective entropy of y-marginal
    nz = np.where(py > 0)[0]
    y_entropy = float(-np.sum(py[nz] * np.log(py[nz]))) if nz.size else 0.0
    y_entropy_norm = y_entropy / np.log(nyn) if nyn > 1 else 0.0

    return {
        "harris_band_energy": harris_share,
        "top10pct_energy": float(top10),
        "y_marginal_entropy_norm": y_entropy_norm,
        "peak_x": int(np.argmax(px)),
        "peak_y": int(np.argmax(py)),
        "peak_z": int(np.argmax(pz)),
    }


def plot_modes(res: DMDResult, shape_pc: tuple, components: list[str],
               cycles_used: list[int], top_k: int, out_path: Path,
               title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sort_by_modulus(res)
    weights = mode_energy(res, len(cycles_used))
    combined = np.abs(res.eigenvalues) + 1e-3 * weights / (weights.max() + 1e-12)
    # Actually just rank by modulus; amplitude is secondary
    pick = order[:top_k]

    ncomp = len(components)
    fig, axes = plt.subplots(top_k, ncomp + 1,
                              figsize=(3 * (ncomp + 1), 2.8 * top_k),
                              squeeze=False)

    # First column: eigenvalue spectrum (only on row 0, with the selected modes highlighted)
    ax0 = axes[0, 0]
    ax0.scatter(res.eigenvalues.real, res.eigenvalues.imag, s=18, c="gray", alpha=0.6)
    for rank, idx in enumerate(pick):
        lam = res.eigenvalues[idx]
        ax0.scatter([lam.real], [lam.imag], s=60, c="red", edgecolor="k", zorder=5)
        ax0.annotate(f"{rank+1}", (lam.real, lam.imag), fontsize=8,
                     xytext=(3, 3), textcoords="offset points")
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax0.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4, lw=0.8)
    ax0.axhline(0, c="k", alpha=0.2, lw=0.5)
    ax0.axvline(0, c="k", alpha=0.2, lw=0.5)
    ax0.set_xlabel("Re(λ)")
    ax0.set_ylabel("Im(λ)")
    ax0.set_aspect("equal")
    ax0.set_title("Spectrum")

    # Hide unused first-column axes on later rows
    for r in range(1, top_k):
        axes[r, 0].axis("off")

    # Plot each selected mode's spatial structure (xy slice, z-averaged) for each component
    for row, idx in enumerate(pick):
        mode_vec = res.modes[:, idx]
        mode_3d = unflatten_mode(mode_vec, shape_pc, ncomp)
        lam = res.eigenvalues[idx]
        rate = np.abs(lam)
        freq = np.angle(lam) / (2 * np.pi)
        mag2 = (np.abs(mode_3d) ** 2).sum(axis=0)
        # xy slice averaged over z
        slc = mag2.mean(axis=2)
        ax_tag = axes[row, 1]
        im = ax_tag.imshow(slc.T, origin="lower", aspect="auto", cmap="magma")
        ax_tag.set_title(f"mode {row+1}  |λ|={rate:.3f}  f={freq:.3f}")
        plt.colorbar(im, ax=ax_tag, fraction=0.046, pad=0.04)

        for c in range(ncomp):
            if c + 2 >= ncomp + 1:
                break
            ax = axes[row, c + 2]
            # Real part of the component, z-averaged
            comp_re = mode_3d[c].real.mean(axis=2)
            lim = max(1e-30, np.abs(comp_re).max())
            im = ax.imshow(comp_re.T, origin="lower", aspect="auto",
                           cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"Re({components[c]}) z-avg")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_cycles(raw: list[int]) -> list[int]:
    if len(raw) == 2:
        lo, hi = raw
        return list(range(lo, hi + 1))
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("/tmp/ipic3d_phase10j_tsc_20"),
                    help="directory with proc*.hdf cycle snapshots")
    ap.add_argument("--cycles", type=int, nargs="+",
                    default=[14, 22],
                    help="cycle range [lo, hi] inclusive for the DMD window")
    ap.add_argument("--components", nargs="+", default=["Ex", "Ey", "Ez"],
                    help="field components to include in the state vector")
    ap.add_argument("--rank", type=int, default=10,
                    help="SVD truncation rank")
    ap.add_argument("--top-k", type=int, default=3,
                    help="number of top modes (by |λ|) to report/plot")
    ap.add_argument("--out", type=Path, default=Path("phase10j_dmd.png"),
                    help="output plot path")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional JSON with eigenvalue table + diagnostics")
    ap.add_argument("--subtract-mean", action="store_true",
                    help="subtract time-mean from each snapshot before DMD")
    ap.add_argument("--title", default="",
                    help="title string for the plot")
    args = ap.parse_args()

    cycles = parse_cycles(args.cycles)
    print(f"run-dir: {args.run_dir}")
    print(f"cycles:  {cycles}")
    print(f"components: {args.components}")
    print(f"rank:    {args.rank}")

    X, shape_pc = build_snapshot_matrix(
        args.run_dir, cycles, args.components,
        subtract_mean=args.subtract_mean,
    )
    print(f"snapshot matrix X: {X.shape}  (N_dof × N_snaps), "
          f"per-component shape {shape_pc}")

    res = dmd(X, rank=args.rank)
    print(f"effective DMD rank: {res.rank}  (SVD sing. vals: {res.sigma[:res.rank]})")

    order = sort_by_modulus(res)
    weights = mode_energy(res, X.shape[1])

    print("\nTop eigenvalues (by |λ|):")
    print(f"  {'rank':<6}{'idx':<6}{'|λ|':<12}{'arg(λ)':<12}{'freq/cyc':<12}"
          f"{'|b|':<12}{'|mode|':<12}")
    rows = []
    for r_, i in enumerate(order[:max(args.top_k, 10)]):
        lam = res.eigenvalues[i]
        mode_norm = float(np.linalg.norm(res.modes[:, i]))
        amp = float(np.abs(res.amplitudes[i]))
        row = {
            "rank": r_ + 1,
            "idx": int(i),
            "abs_lambda": float(np.abs(lam)),
            "arg_lambda": float(np.angle(lam)),
            "freq_per_cycle": float(np.angle(lam) / (2 * np.pi)),
            "amplitude": amp,
            "mode_norm": mode_norm,
            "re_lambda": float(lam.real),
            "im_lambda": float(lam.imag),
        }
        rows.append(row)
        print(f"  {row['rank']:<6}{row['idx']:<6}{row['abs_lambda']:<12.4f}"
              f"{row['arg_lambda']:<12.4f}{row['freq_per_cycle']:<12.4f}"
              f"{amp:<12.3e}{mode_norm:<12.3e}")

    # Mode-1 localisation diagnostics
    top_i = order[0]
    top_mode = unflatten_mode(res.modes[:, top_i], shape_pc, len(args.components))
    diag = localisation_diagnostics(top_mode)
    print(f"\nTop mode localisation: {diag}")

    plot_modes(res, shape_pc, args.components, cycles,
               args.top_k, args.out, title=args.title or str(args.run_dir.name))
    print(f"\nPlot: {args.out}")

    if args.json:
        out = {
            "run_dir": str(args.run_dir),
            "cycles": cycles,
            "components": args.components,
            "rank": res.rank,
            "top_eigenvalues": rows,
            "top_mode_localisation": diag,
            "svd_singular_values": [float(x) for x in res.sigma[:res.rank]],
        }
        args.json.write_text(json.dumps(out, indent=2))
        print(f"JSON: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
