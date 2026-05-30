#!/usr/bin/env python3
"""movie_operator.py — "operator movie": physics + the linear system the implicit
field solver actually solves, side by side, evolving in time.

Each frame is a 2x2 figure:

    ┌─────────────────────┬─────────────────────┐
    │  field view 1       │  field view 2       │   ← physics (top)
    ├─────────────────────┼─────────────────────┤
    │  P stiffness map    │  Ritz eigenvalues   │   ← linear algebra (bottom)
    │  ‖P_nn − I‖ on grid │  in the complex plane│
    └─────────────────────┴─────────────────────┘

The point is contrast: a hard problem (Double Harris reconnection) stiffens the
operator where the current sheet forms — the stiffness map develops structure and
the eigenvalue cloud spreads — while an easy problem (Whistler packet) stays
uniform and tightly clustered near 1.

Inputs per cycle:
  * fields        — phdf5 output read with plot_utils.load_field
  * matrix P      — {SimName}_P_cycle{N}.bin (PETSc binary), loaded via petsc4py,
                    produced by running with PrecMatrix=true + PrecDumpCycle=N
  * Ritz eigenvalues — parsed from the run log (PETSc -ksp_view_eigenvalues)

Run the simulation with, e.g.:
    mpirun -np 1 build/iPIC3D inputfiles/Double_Harris_movie.inp \\
        -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_monitor \\
        > dh_movie.log 2>&1        # NB: log OUTSIDE SaveDirName (it gets wiped at startup)

    python scripts/movie_operator.py output/data_reconnection_movie \\
        --inp inputfiles/Double_Harris_movie.inp --log dh_movie.log \\
        --mode reconnection -o output/operator_dh.mp4

Matrix-panel mode:
  * "stiffmap" (default) — per-node ‖P_nn − I‖_F reshaped onto the grid. Needs the
    np=1 node ordering (N == 3·(nxc+1)(nyc+1)(nzc+1)); auto-falls back to "blocknorm".
  * "blocknorm" — matrix-space binned ‖block‖_F heatmap. Works for any decomposition.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import discover_cycles, load_field
from movie_physics_test import (parse_input, guess_inp, is_reconnection,
                                 is_1d_like, compute_Jz, compute_Az)
from plot_ksp_diagnostics import parse_log

try:
    from plot_theme import add_theme_arg, apply_theme
except ImportError:
    add_theme_arg = None
    apply_theme = None

# petsc4py is the required loader for the PETSc binary matrices.
import petsc4py
petsc4py.init(["-no_signal_handler"])
from petsc4py import PETSc
import scipy.sparse as sp


# ── Matrix I/O & reductions ──────────────────────────────────────────────

def load_P(path):
    """Load a PETSc binary matrix into a scipy CSR matrix (serial).

    A global MATMPIAIJ written by MatView from a parallel run loads fine here —
    the PETSc binary format is decomposition-independent.
    """
    viewer = PETSc.Viewer().createBinary(path, "r")
    A = PETSc.Mat().load(viewer)
    indptr, indices, data = A.getValuesCSR()
    n = A.getSize()[0]
    A.destroy()
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _row_index(csr):
    """Per-nonzero row index (CSR has only indptr); cached on the array object."""
    return np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))


def per_node_stiffness(csr):
    """‖P_nn − I‖_F for each 3x3 diagonal node-block.

    Vectorized over nonzeros: a contribution lands in block g = row//3 only when
    the column shares the block (col//3 == row//3); the identity is subtracted on
    the true diagonal (row == col).  Returns an array of length N/3.
    """
    n = csr.shape[0]
    nb = n // 3
    rows = _row_index(csr)
    cols = csr.indices
    vals = csr.data.copy()
    vals[rows == cols] -= 1.0                       # |P - I| on the diagonal
    same_block = (rows // 3) == (cols // 3)
    contrib = np.where(same_block, vals * vals, 0.0)
    block_sq = np.bincount(rows // 3, weights=contrib, minlength=nb)
    return np.sqrt(block_sq)


def binned_block_norm(csr, nbins):
    """Matrix-space ‖block‖_F heatmap binned to nbins x nbins (any decomposition).

    Block = 3x3 node coupling; the identity is subtracted on the diagonal so the
    color shows deviation from I (preconditioner strength), like per_node_stiffness.
    """
    n = csr.shape[0]
    nb = n // 3
    rows = _row_index(csr)
    cols = csr.indices
    vals = csr.data.copy()
    vals[rows == cols] -= 1.0
    rb = (rows // 3) * nbins // nb
    cb = (cols // 3) * nbins // nb
    flat = np.bincount(rb * nbins + cb, weights=vals * vals,
                       minlength=nbins * nbins)
    return np.sqrt(flat.reshape(nbins, nbins))


def grid_node_dims(sim):
    """Interior-node dims (niX, niY, niZ) for the np=1 ordering used by assembleP():
    niX = nxc + 1, etc.  Block index g maps to (i,j,k) in C-order (k fastest)."""
    nxc = int(sim.get("nxc", 0))
    nyc = int(sim.get("nyc", 0))
    nzc = int(sim.get("nzc", 0))
    return nxc + 1, nyc + 1, nzc + 1


# ── Frame assembly (pre-load everything, then animate) ─────────────────────

def matrix_cycles(run_dir, sim_name):
    """Map cycle -> matrix-dump path from {SimName}_P_cycle{N}.bin files."""
    out = {}
    pat = os.path.join(run_dir, f"{sim_name}_P_cycle*.bin")
    for p in glob.glob(pat):
        m = re.search(r"_P_cycle(\d+)\.bin$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def collect(run_dir, sim, log_path, mode, matrix_view, nbins, stride):
    """Gather aligned per-frame data: cycles present in fields AND matrix dumps AND
    the Ritz log.  Returns a dict of lists/arrays ready for rendering."""
    sim_name = str(sim.get("SimulationName", "")).strip().strip('"').strip("'")
    field_cycles = set(discover_cycles(run_dir))
    mat = matrix_cycles(run_dir, sim_name)
    _, _, eigenvalues, singular, _ = parse_log(log_path)
    eig_cycles = {c for c, e in eigenvalues.items() if e}

    cycles = sorted(field_cycles & set(mat) & eig_cycles)[::stride]
    if len(cycles) < 2:
        sys.exit(f"Only {len(cycles)} aligned frames "
                 f"(fields∩matrix∩eig); need ≥2. "
                 f"fields={len(field_cycles)} matrix={len(mat)} eig={len(eig_cycles)}")

    niX, niY, niZ = grid_node_dims(sim)
    Lx, Ly = sim.get("Lx", 1.0), sim.get("Ly", 1.0)
    kz = None  # z slice for field views, set on first load

    data = {"cycles": cycles, "dt": sim.get("dt", 1.0),
            "eig": [eigenvalues[c] for c in cycles],
            "kappa": [singular[c][2] if c in singular else np.nan for c in cycles]}

    f1, f2, mats = [], [], []
    use_stiffmap = (matrix_view == "stiffmap")

    for c in cycles:
        # --- matrix panel ---
        P = load_P(mat[c])
        if use_stiffmap and P.shape[0] == 3 * niX * niY * niZ:
            stiff = per_node_stiffness(P).reshape(niX, niY, niZ)
            mats.append(stiff[:, :, niZ // 2])               # (niX, niY) at z-mid
        else:
            use_stiffmap = False
            mats.append(binned_block_norm(P, nbins))

        # --- field panels ---
        if mode == "reconnection":
            Bx = load_field(run_dir, c, "B", "x")
            By = load_field(run_dir, c, "B", "y")
            if kz is None:
                kz = Bx.shape[2] // 2
            Bx = Bx[:-1, :-1, kz]
            By = By[:-1, :-1, kz]
            nx_i, ny_i = Bx.shape
            dx, dy = Lx / nx_i, Ly / ny_i
            f1.append(By)
            f2.append({"Jz": compute_Jz(Bx, By, dx, dy),
                       "Az": compute_Az(Bx, By, dx, dy)})
        else:  # whistler — 1D along x: transverse B and E profiles
            By = load_field(run_dir, c, "B", "y")
            Bz = load_field(run_dir, c, "B", "z")
            Ey = load_field(run_dir, c, "E", "y")
            Ez = load_field(run_dir, c, "E", "z")
            if kz is None:
                kz = By.shape[2] // 2
            jy = By.shape[1] // 2
            f1.append((By[:-1, jy, kz], Bz[:-1, jy, kz]))
            f2.append((Ey[:-1, jy, kz], Ez[:-1, jy, kz]))

    data["f1"], data["f2"], data["mat"] = f1, f2, mats
    data["stiffmap"] = use_stiffmap
    data["grid"] = (niX, niY, niZ, Lx, Ly)
    return data


# ── Rendering ──────────────────────────────────────────────────────────────

def render(data, sim, mode, args, theme):
    cycles = data["cycles"]
    dt = data["dt"]
    mode_dark = getattr(theme, "mode", "dark") == "dark"
    Lx = sim.get("Lx", 1.0)
    Ly = sim.get("Ly", 1.0)

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0])
    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_f2 = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 0])
    ax_eig = fig.add_subplot(gs[1, 1])

    artists = {}

    # ---------- top: physics ----------
    if mode == "reconnection":
        By0 = data["f1"][0]
        vmaxB = max(float(np.abs(b).max()) for b in data["f1"])
        Jz_all = [d["Jz"] for d in data["f2"]]
        vmaxJ = max(float(np.abs(j).max()) for j in Jz_all) * 0.7
        vmaxA = max(float(np.abs(d["Az"]).max()) for d in data["f2"])
        ext = [0, Lx, 0, Ly]
        cc = "#f38ba8" if mode_dark else "#882255"
        clev = np.linspace(-vmaxA, vmaxA, 21)

        artists["By"] = ax_f1.imshow(By0.T, origin="lower", extent=ext, aspect="equal",
                                     vmin=-vmaxB, vmax=vmaxB, cmap="RdBu_r",
                                     interpolation="bilinear")
        ax_f1.set_title("$B_y$ — reconnecting field (islands)")
        ax_f1.set_xlabel("x"); ax_f1.set_ylabel("y")
        fig.colorbar(artists["By"], ax=ax_f1, shrink=0.82, label="$B_y$")

        artists["Jz"] = ax_f2.imshow(Jz_all[0].T, origin="lower", extent=ext, aspect="equal",
                                     vmin=-vmaxJ, vmax=vmaxJ, cmap="PRGn",
                                     interpolation="bilinear")
        ax_f2.set_title("$J_z$ — out-of-plane current (X-lines / sheets)")
        ax_f2.set_xlabel("x"); ax_f2.set_ylabel("y")
        fig.colorbar(artists["Jz"], ax=ax_f2, shrink=0.82, label="$J_z$")

        nx, ny = By0.shape
        xs = np.linspace(0, Lx, nx); ys = np.linspace(0, Ly, ny)
        artists["_az"] = {"cset": ax_f1.contour(xs, ys, data["f2"][0]["Az"].T,
                                                 levels=clev, colors=cc,
                                                 linewidths=0.6, alpha=0.8),
                          "xs": xs, "ys": ys, "clev": clev, "cc": cc}
    else:  # whistler
        nx = data["f1"][0][0].size
        x = np.linspace(0, Lx, nx, endpoint=False)
        vB = max(max(np.abs(a).max(), np.abs(b).max()) for a, b in data["f1"])
        vE = max(max(np.abs(a).max(), np.abs(b).max()) for a, b in data["f2"])
        cB = ("#89b4fa", "#f38ba8") if mode_dark else ("#4477AA", "#EE6677")
        cE = ("#a6e3a1", "#f9e2af") if mode_dark else ("#228833", "#CCBB44")

        artists["By"], = ax_f1.plot(x, data["f1"][0][0], color=cB[0], lw=1.5, label="$B_y$")
        artists["Bz"], = ax_f1.plot(x, data["f1"][0][1], color=cB[1], lw=1.5, label="$B_z$")
        ax_f1.set_ylim(-1.05 * vB, 1.05 * vB); ax_f1.set_xlim(0, Lx)
        ax_f1.set_title("Transverse magnetic field (R-mode wave)")
        ax_f1.set_xlabel("x"); ax_f1.set_ylabel("$B_\\perp$"); ax_f1.legend(loc="upper right")

        artists["Ey"], = ax_f2.plot(x, data["f2"][0][0], color=cE[0], lw=1.5, label="$E_y$")
        artists["Ez"], = ax_f2.plot(x, data["f2"][0][1], color=cE[1], lw=1.5, label="$E_z$")
        ax_f2.set_ylim(-1.05 * vE, 1.05 * vE); ax_f2.set_xlim(0, Lx)
        ax_f2.set_title("Transverse electric field")
        ax_f2.set_xlabel("x"); ax_f2.set_ylabel("$E_\\perp$"); ax_f2.legend(loc="upper right")

    # ---------- bottom-left: matrix ----------
    mat_all = data["mat"]
    pos = np.concatenate([m[m > 0].ravel() for m in mat_all])
    vmin = max(np.percentile(pos, 1), 1e-12)
    vmax = float(max(m.max() for m in mat_all))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    if data["stiffmap"]:
        artists["mat"] = ax_mat.imshow(mat_all[0].T, origin="lower", norm=norm,
                                       cmap="inferno", aspect="auto",
                                       extent=[0, Lx, 0, Ly],
                                       interpolation="nearest")
        ax_mat.set_xlabel("x"); ax_mat.set_ylabel("y")
        ax_mat.set_title(r"Operator stiffness  $\|P_{nn}-I\|_F$  per grid node")
    else:
        artists["mat"] = ax_mat.imshow(mat_all[0], origin="upper", norm=norm,
                                       cmap="inferno", aspect="equal")
        ax_mat.set_xlabel("block column"); ax_mat.set_ylabel("block row")
        ax_mat.set_title(r"$\|P-I\|_F$ block norms (binned)")
    fig.colorbar(artists["mat"], ax=ax_mat, shrink=0.82, label=r"$\|\cdot\|_F$")

    # ---------- bottom-right: Ritz eigenvalues ----------
    all_re = np.concatenate([[e[0] for e in ev] for ev in data["eig"]])
    all_im = np.concatenate([[e[1] for e in ev] for ev in data["eig"]])
    re_pad = 0.05 * (all_re.max() - all_re.min() + 1e-12)
    im_max = max(abs(all_im).max(), 1e-3)
    ax_eig.set_xlim(all_re.min() - re_pad, all_re.max() + re_pad)
    ax_eig.set_ylim(-1.15 * im_max, 1.15 * im_max)
    ax_eig.axhline(0, color=getattr(theme, "refline_color", "#888"), lw=0.5, alpha=0.5)
    ax_eig.axvline(1, color=getattr(theme, "refline_color", "#888"), lw=0.5, alpha=0.5, ls="--")
    ax_eig.set_xlabel("Re($\\lambda$)"); ax_eig.set_ylabel("Im($\\lambda$)")
    ec = "#f9e2af" if mode_dark else "#EE6677"
    artists["eig"] = ax_eig.scatter(all_re[:1], all_im[:1], s=22, color=ec,
                                    alpha=0.7, edgecolors="none")
    artists["eig_ax"] = ax_eig

    suptitle = fig.suptitle("", fontsize=13)

    def update(i):
        c = cycles[i]
        if mode == "reconnection":
            artists["By"].set_data(data["f1"][i].T)
            artists["Jz"].set_data(data["f2"][i]["Jz"].T)
            az = artists["_az"]
            try:
                az["cset"].remove()
            except (AttributeError, ValueError):
                for coll in getattr(az["cset"], "collections", []):
                    coll.remove()
            az["cset"] = ax_f1.contour(az["xs"], az["ys"], data["f2"][i]["Az"].T,
                                       levels=az["clev"], colors=az["cc"],
                                       linewidths=0.6, alpha=0.8)
        else:
            artists["By"].set_ydata(data["f1"][i][0])
            artists["Bz"].set_ydata(data["f1"][i][1])
            artists["Ey"].set_ydata(data["f2"][i][0])
            artists["Ez"].set_ydata(data["f2"][i][1])

        artists["mat"].set_data(data["mat"][i].T if data["stiffmap"] else data["mat"][i])

        ev = data["eig"][i]
        artists["eig"].set_offsets(np.column_stack([[e[0] for e in ev],
                                                     [e[1] for e in ev]]))
        kappa = data["kappa"][i]
        ktxt = f"  κ≈{kappa:.2f}" if np.isfinite(kappa) else ""
        artists["eig_ax"].set_title(f"Ritz eigenvalues (complex plane){ktxt}")

        suptitle.set_text(f"{args.label}   cycle {c}   t = {c * dt:.2f}   "
                          f"[frame {i + 1}/{len(cycles)}]")
        return []

    return fig, update, len(cycles)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="iPIC3D output dir (SaveDirName) with Fields_* and P_cycle*.bin")
    ap.add_argument("--inp", default=None, help="Path to the .inp (defaults to a heuristic guess)")
    ap.add_argument("--log", default=None, help="Run log with -ksp_view_eigenvalues (default: <run_dir>/run.log)")
    ap.add_argument("--mode", choices=["auto", "reconnection", "whistler"], default="auto")
    ap.add_argument("--matrix-view", choices=["stiffmap", "blocknorm"], default="stiffmap")
    ap.add_argument("--nbins", type=int, default=120, help="Bins for the blocknorm fallback")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth aligned frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("-o", "--output", default="operator_movie.mp4")
    ap.add_argument("--label", default=None, help="Suptitle label (default: run dir basename)")
    if add_theme_arg:
        add_theme_arg(ap)
    args = ap.parse_args()

    if apply_theme:
        theme = apply_theme(args)
    else:
        class _T:
            mode = "dark"; refline_color = "#888888"
        theme = _T()

    inp = args.inp or guess_inp(args.run_dir)
    sim = parse_input(inp) if inp and os.path.isfile(inp) else {}
    if not sim:
        sys.exit("Could not load an .inp — pass --inp (needed for grid dims, Lx/Ly, dt).")

    if args.mode == "auto":
        mode = "reconnection" if is_reconnection(sim) else \
               ("whistler" if is_1d_like(sim) else "reconnection")
    else:
        mode = args.mode

    if args.log is None:
        args.log = os.path.join(args.run_dir, "run.log")
    if not os.path.isfile(args.log):
        sys.exit(f"Log not found: {args.log} (pass --log; it must contain -ksp_view_eigenvalues output)")
    if args.label is None:
        args.label = os.path.basename(args.run_dir.rstrip("/"))

    data = collect(args.run_dir, sim, args.log, mode, args.matrix_view, args.nbins, args.stride)
    if not data["stiffmap"] and args.matrix_view == "stiffmap":
        print("NOTE: matrix size doesn't match the np=1 grid ordering — "
              "falling back to matrix-space binned block norms.")

    fig, update, n = render(data, sim, mode, args, theme)
    anim = FuncAnimation(fig, update, frames=n, interval=1000 // args.fps, blit=False)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(args.output)[1].lower()
    writer = PillowWriter(fps=args.fps) if ext == ".gif" else FFMpegWriter(fps=args.fps, bitrate=3000)

    print(f"Writing {n} frames ({mode}, matrix={'stiffmap' if data['stiffmap'] else 'blocknorm'}) "
          f"to {args.output} at {args.fps} fps...")
    anim.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Done: {args.output}  ({os.path.getsize(args.output) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
