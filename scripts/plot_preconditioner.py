#!/usr/bin/env python3
"""Visualize the PETSc preconditioner matrix P dumped by iPIC3D.

Reads PETSc binary format (MAT_FILE_CLASSID) using numpy, no petsc4py needed.

Usage:
    pixi run python scripts/plot_preconditioner.py output/SimName_P_cycle0.bin
    pixi run python scripts/plot_preconditioner.py output/SimName_P_cycle0.bin --light
    pixi run python scripts/plot_preconditioner.py output/SimName_P_cycle0.bin -o preconditioner.png
    pixi run python scripts/plot_preconditioner.py output/SimName_P_cycle0.bin --prefix DH_baseline
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_theme import add_theme_arg, apply_theme

# PETSc binary format constants
MAT_FILE_CLASSID = 1211216  # from petsc/include/petscmat.h

# Theme-aware colors for Ex/Ey/Ez components
_COMPONENT_COLORS = {
    "dark":  ["#f38ba8", "#a6e3a1", "#89b4fa"],  # Catppuccin red, green, blue
    "light": ["#EE6677", "#228833", "#4477AA"],   # Tol red, green, blue
}
_SPARSITY_COLOR = {"dark": "#89dceb", "light": "#004488"}  # Catppuccin teal / dark blue


def load_petsc_binary_mat(filename):
    """Load a PETSc binary sparse matrix into CSR arrays.

    PETSc binary format for MATMPIAIJ:
      - int32: MAT_FILE_CLASSID (1211216)
      - int32: M (rows)
      - int32: N (cols)
      - int32: nz (total nonzeros)
      - int32[M]: nnz per row
      - int32[nz]: column indices
      - float64[nz]: values
    All integers are big-endian int32, floats are big-endian float64.
    """
    with open(filename, "rb") as f:
        classid = struct.unpack(">i", f.read(4))[0]
        if classid != MAT_FILE_CLASSID:
            raise ValueError(
                f"Not a PETSc matrix file (classid={classid}, expected {MAT_FILE_CLASSID})"
            )
        M = struct.unpack(">i", f.read(4))[0]
        N = struct.unpack(">i", f.read(4))[0]
        nz = struct.unpack(">i", f.read(4))[0]

        row_lengths = np.frombuffer(f.read(M * 4), dtype=">i4").astype(np.int64)
        col_indices = np.frombuffer(f.read(nz * 4), dtype=">i4").astype(np.int64)
        values = np.frombuffer(f.read(nz * 8), dtype=">f8").astype(np.float64)

    row_ptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(row_lengths, out=row_ptr[1:])

    return M, N, row_ptr, col_indices, values


def compute_diagnostics(M, N, row_ptr, col_indices, values):
    """Compute and print matrix diagnostics.  Returns the diagonal array."""
    nz = row_ptr[-1]
    norm_P = np.sqrt(np.sum(values**2))

    diag = np.zeros(min(M, N))
    for i in range(min(M, N)):
        start, end = row_ptr[i], row_ptr[i + 1]
        cols = col_indices[start:end]
        mask = cols == i
        if np.any(mask):
            diag[i] = values[start:end][mask][0]

    off_diag_sum_sq = np.sum(values**2) - np.sum(diag**2)
    diag_shifted_sq = np.sum((diag - 1.0) ** 2)
    norm_PmI = np.sqrt(off_diag_sum_sq + diag_shifted_sq)
    norm_I = np.sqrt(M)

    print(f"\n{'='*50}")
    print("Preconditioner matrix P diagnostics")
    print(f"{'='*50}")
    print(f"  Size:             {M} x {N}")
    print(f"  Nonzeros:         {nz} ({nz / M:.1f} per row)")
    print(f"  ||P||_F           = {norm_P:.6e}")
    print(f"  ||P - I||_F       = {norm_PmI:.6e}")
    print(f"  ||P - I|| / ||I|| = {norm_PmI / norm_I:.6e}")
    print(f"  Diagonal range:     [{diag.min():.4e}, {diag.max():.4e}]")
    print(f"  Off-diagonal |max|: {np.max(np.abs(values[col_indices != np.repeat(np.arange(M), np.diff(row_ptr))])):.4e}")
    print(f"{'='*50}\n")

    return diag


def plot_matrix(M, N, row_ptr, col_indices, values, diag, theme, args):
    """Create 3-panel visualization: sparsity, |P-I| block norms, diagonal."""
    mode = theme.mode
    colors = _COMPONENT_COLORS[mode]
    labels = ["Ex", "Ey", "Ez"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Sparsity pattern ---
    ax = axes[0]
    max_points = 100_000
    if row_ptr[-1] > max_points:
        stride = int(np.ceil(row_ptr[-1] / max_points))
        rows_sparse = np.repeat(np.arange(M), np.diff(row_ptr))[::stride]
        cols_sparse = col_indices[::stride]
    else:
        rows_sparse = np.repeat(np.arange(M), np.diff(row_ptr))
        cols_sparse = col_indices

    ax.scatter(cols_sparse, rows_sparse, s=0.1, marker=".", alpha=0.5,
               c=_SPARSITY_COLOR[mode])
    ax.set_xlim(0, N)
    ax.set_ylim(M, 0)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Sparsity pattern")
    ax.set_aspect("equal")

    # --- Panel 2: |P - I| heatmap (block view) ---
    ax = axes[1]
    block_size = 3
    n_blocks = min(M // block_size, args.max_blocks)
    block_norms = np.zeros((n_blocks, n_blocks))

    for bi in range(n_blocks):
        for r in range(block_size):
            row = bi * block_size + r
            start, end = row_ptr[row], row_ptr[row + 1]
            for idx in range(start, end):
                col = col_indices[idx]
                bj = col // block_size
                if bj >= n_blocks:
                    continue
                val = values[idx]
                if bi == bj and (r == col % block_size):
                    val -= 1.0
                block_norms[bi, bj] += val * val

    block_norms = np.sqrt(block_norms)
    vmin = max(block_norms[block_norms > 0].min(), 1e-16) if np.any(block_norms > 0) else 1e-16
    im = ax.imshow(block_norms, norm=LogNorm(vmin=vmin, vmax=block_norms.max()),
                   cmap="inferno", aspect="equal")
    plt.colorbar(im, ax=ax, label="||block||_F")
    ax.set_xlabel("Block column")
    ax.set_ylabel("Block row")
    shown = f"first {n_blocks}" if n_blocks < M // block_size else "all"
    ax.set_title(f"|P - I| block norms ({shown} blocks)")

    # --- Panel 3: Diagonal values ---
    ax = axes[2]
    x = np.arange(len(diag))
    for c in range(3):
        mask = x % 3 == c
        ax.plot(x[mask], diag[mask], ".", markersize=1.5, color=colors[c],
                alpha=0.6, label=labels[c])
    ax.axhline(y=1.0, color=theme.refline_color, linestyle="--",
               alpha=0.5, label="Identity")
    ax.set_xlabel("DOF index")
    ax.set_ylabel("Diagonal value")
    ax.set_title("P diagonal (deviation from 1 = preconditioner strength)")
    ax.legend(markerscale=5)

    fig.suptitle(f"Preconditioner matrix P  ({M}\u00d7{N}, {row_ptr[-1]} nnz)", fontsize=13)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize PETSc preconditioner matrix")
    parser.add_argument("filename", help="PETSc binary matrix file (e.g. output/SimName_P_cycle0.bin)")
    parser.add_argument("-o", "--output", default=None,
                        help="Save figure to FILE")
    parser.add_argument("--prefix", metavar="NAME",
                        help="Run name prefix — auto-saves next to input file as {PREFIX}_preconditioner.png")
    parser.add_argument("--max-blocks", type=int, default=80,
                        help="Max blocks to show in heatmap (default: 80)")
    add_theme_arg(parser)
    args = parser.parse_args()

    # Default output: same directory as input file
    if args.output is None and args.prefix:
        input_dir = os.path.dirname(args.filename) or "."
        args.output = os.path.join(input_dir, f"{args.prefix}_preconditioner.png")

    theme = apply_theme(args)

    M, N, row_ptr, col_indices, values = load_petsc_binary_mat(args.filename)
    diag = compute_diagnostics(M, N, row_ptr, col_indices, values)
    plot_matrix(M, N, row_ptr, col_indices, values, diag, theme, args)


if __name__ == "__main__":
    main()
