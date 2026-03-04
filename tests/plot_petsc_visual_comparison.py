#!/usr/bin/env python3
"""
plot_petsc_visual_comparison.py — Visual comparison of field solver outputs.

Compares electromagnetic field data produced by different solvers (e.g. GMRES vs PETSc)
to verify they produce equivalent results on Double Harris magnetic reconnection runs.

Usage:
    python3 plot_petsc_visual_comparison.py --gmres <dir> --petsc <dir> [--petsc <dir> ...]
    python3 plot_petsc_visual_comparison.py                  # auto-discovers from test_petsc_output/

HDF5 layout (phdf5 global files):
    <run_dir>/Fields_NNNNN/B_NNNNN.h5  →  /Fields/Bx, /Fields/By, /Fields/Bz  (X, Y, Z)
    <run_dir>/Fields_NNNNN/E_NNNNN.h5  →  /Fields/Ex, /Fields/Ey, /Fields/Ez  (X, Y, Z)
"""

import argparse
import glob
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    print("  WARNING: numpy not installed, skipping visual comparison.")
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("  WARNING: matplotlib not installed, skipping visual comparison.")
    print("  Install with: pip3 install matplotlib")
    sys.exit(0)

try:
    import h5py
except ImportError:
    print("  WARNING: h5py not installed, skipping visual comparison.")
    print("  Install with: pip3 install h5py")
    sys.exit(0)

# ── CLI args ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--gmres', default=None,
                    help='Path to GMRES field output directory (auto-discovered if omitted)')
parser.add_argument('--petsc', action='append', default=None,
                    help='Path to PETSc field output directory (repeatable; auto-discovered if omitted)')
parser.add_argument('--cycle', type=int, default=-1,
                    help='Cycle to compare (-1 = last available)')
parser.add_argument('--vmax-Bx', type=float, default=None,
                    help='Fixed symmetric color limit for Bx field panels')
parser.add_argument('--vmax-Bx-diff', type=float, default=None,
                    help='Fixed symmetric color limit for Bx difference panel')
parser.add_argument('--vmax-Ez', type=float, default=None,
                    help='Fixed symmetric color limit for Ez field panels')
parser.add_argument('--vmax-Ez-diff', type=float, default=None,
                    help='Fixed symmetric color limit for Ez difference panel')
parser.add_argument('--print-bounds', action='store_true',
                    help='Print vmax bounds for all fields and exit (no plot)')
args = parser.parse_args()

# ── Fixed bounds lookup ──────────────────────────────────────────────────
fixed_vmax = {}
if args.vmax_Bx is not None:
    fixed_vmax[("B", "x", "field")] = args.vmax_Bx
if args.vmax_Bx_diff is not None:
    fixed_vmax[("B", "x", "diff")] = args.vmax_Bx_diff
if args.vmax_Ez is not None:
    fixed_vmax[("E", "z", "field")] = args.vmax_Ez
if args.vmax_Ez_diff is not None:
    fixed_vmax[("E", "z", "diff")] = args.vmax_Ez_diff

# ── Auto-discovery ────────────────────────────────────────────────────────
output_dir = os.path.join(script_dir, "test_petsc_output")

if args.gmres is None:
    candidates = sorted(glob.glob(os.path.join(output_dir, "GMRES_*")))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        print("  ERROR: No GMRES directory found in test_petsc_output/. "
              "Use --gmres to specify.")
        sys.exit(1)
    args.gmres = candidates[0]
    print(f"  Auto-discovered GMRES dir: {args.gmres}")

if args.petsc is None:
    candidates = sorted(glob.glob(os.path.join(output_dir, "PETSc_*")))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        print("  ERROR: No PETSc directory found in test_petsc_output/. "
              "Use --petsc to specify.")
        sys.exit(1)
    args.petsc = candidates
    for p in args.petsc:
        print(f"  Auto-discovered PETSc dir: {p}")

# ── Utility functions ─────────────────────────────────────────────────────

def discover_cycles(run_dir):
    """Scan Fields_* directories, extract cycle numbers, return sorted list."""
    pattern = os.path.join(run_dir, "Fields_*")
    cycles = []
    for d in glob.glob(pattern):
        basename = os.path.basename(d)
        m = re.match(r'Fields_(\d+)', basename)
        if m:
            cycles.append(int(m.group(1)))
    return sorted(cycles)


def load_field_2d(run_dir, cycle, field_type, component):
    """Load a 2D slice (z=0 plane) of a field component from phdf5 output.

    Parameters
    ----------
    run_dir : str       Path to simulation output directory
    cycle : int         Cycle number
    field_type : str    "B" or "E"
    component : str     "x", "y", or "z"

    Returns
    -------
    np.ndarray          2D array, shape (Nx, Ny)
    """
    cycle_str = f"{cycle:05d}"
    h5_path = os.path.join(run_dir, f"Fields_{cycle_str}",
                           f"{field_type}_{cycle_str}.h5")
    dataset = f"/Fields/{field_type}{component}"
    with h5py.File(h5_path, "r") as f:
        data = f[dataset][:]  # shape (Nx, Ny, Nz)
    return data[:, :, 0]      # z=0 slice → (Nx, Ny)


def solver_label(run_dir):
    """Extract a human-readable solver label from a run directory name."""
    return os.path.basename(run_dir)

# ── Find common cycles ───────────────────────────────────────────────────
all_dirs = [args.gmres] + args.petsc
all_cycle_sets = [set(discover_cycles(d)) for d in all_dirs]
common_cycles = sorted(set.intersection(*all_cycle_sets))

if not common_cycles:
    print("  ERROR: No common output cycles found across run directories.")
    sys.exit(1)

if args.cycle == -1:
    cycle = common_cycles[-1]
elif args.cycle in common_cycles:
    cycle = args.cycle
else:
    print(f"  ERROR: Requested cycle {args.cycle} not found. "
          f"Available common cycles: {common_cycles}")
    sys.exit(1)

print(f"  Comparing cycle {cycle:05d}")

# ── Fields to compare ────────────────────────────────────────────────────
# Bx = reconnecting field, Ez = out-of-plane reconnection electric field
fields = [("B", "x", "$B_x$"), ("E", "z", "$E_z$")]

# ── Load data ────────────────────────────────────────────────────────────
gmres_label = solver_label(args.gmres)
gmres_data = {}
for ftype, comp, _ in fields:
    gmres_data[(ftype, comp)] = load_field_2d(args.gmres, cycle, ftype, comp)

# ── Print-bounds mode ────────────────────────────────────────────────────
if args.print_bounds:
    # Compute vmax for each field across GMRES and all PETSc dirs
    bound_names = {("B", "x"): "Bx", ("E", "z"): "Ez"}
    for ftype, comp, _ in fields:
        name = bound_names[(ftype, comp)]
        gdata = gmres_data[(ftype, comp)]
        vmax_field = np.max(np.abs(gdata))
        vmax_diff = 0.0
        for petsc_dir in args.petsc:
            pdata = load_field_2d(petsc_dir, cycle, ftype, comp)
            vmax_field = max(vmax_field, np.max(np.abs(pdata)))
            vmax_diff = max(vmax_diff, np.max(np.abs(pdata - gdata)))
        vmax_field = max(vmax_field, 1e-30)
        vmax_diff = max(vmax_diff, 1e-30)
        print(f"vmax_{name}={vmax_field:.6e}")
        print(f"vmax_{name}_diff={vmax_diff:.6e}")
    sys.exit(0)

# ── Generate one figure per PETSc solver ─────────────────────────────────
for petsc_dir in args.petsc:
    petsc_lab = solver_label(petsc_dir)
    petsc_data = {}
    for ftype, comp, _ in fields:
        petsc_data[(ftype, comp)] = load_field_2d(petsc_dir, cycle, ftype, comp)

    # Shape check
    for ftype, comp, label in fields:
        g_shape = gmres_data[(ftype, comp)].shape
        p_shape = petsc_data[(ftype, comp)].shape
        if g_shape != p_shape:
            print(f"  ERROR: Shape mismatch for {label}: "
                  f"GMRES {g_shape} vs {petsc_lab} {p_shape}")
            sys.exit(1)

    # ── Create figure: 2 rows × 3 columns ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    fig.suptitle(f"Field Comparison: {gmres_label} vs {petsc_lab}  (cycle {cycle:05d})",
                 fontsize=14, fontweight='bold', y=0.98)

    for row, (ftype, comp, field_label) in enumerate(fields):
        gdata = gmres_data[(ftype, comp)].T   # transpose: rows=Y, cols=X
        pdata = petsc_data[(ftype, comp)].T
        diff = pdata - gdata

        # Symmetric color scale for field panels (use fixed bounds if provided)
        vmax_field = fixed_vmax.get((ftype, comp, "field")) \
            or max(np.max(np.abs(gdata)), np.max(np.abs(pdata)), 1e-30)
        vmax_diff = fixed_vmax.get((ftype, comp, "diff")) \
            or max(np.max(np.abs(diff)), 1e-30)

        # Error metrics
        max_abs_diff = np.max(np.abs(diff))
        norm_g = np.linalg.norm(gdata)
        l2_rel_err = np.linalg.norm(diff) / norm_g if norm_g > 0 else 0.0

        panels = [
            (axes[row, 0], gdata, f"{field_label} ({gmres_label})",
             'RdBu_r', -vmax_field, vmax_field),
            (axes[row, 1], pdata, f"{field_label} ({petsc_lab})",
             'RdBu_r', -vmax_field, vmax_field),
            (axes[row, 2], diff, f"{field_label} Difference",
             'RdBu_r', -vmax_diff, vmax_diff),
        ]

        for ax, data, title, cmap, vmin, vmax in panels:
            im = ax.imshow(data, origin='lower', aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.tick_params(labelsize=9)

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=8)

        # Annotate difference panel with error metrics
        ax_diff = axes[row, 2]
        ax_diff.text(0.03, 0.97,
                     f"|max diff| = {max_abs_diff:.2e}\n"
                     f"L2 rel err = {l2_rel_err:.2e}",
                     transform=ax_diff.transAxes,
                     fontsize=9, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.85,
                               edgecolor='#cccccc'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save PNG ──────────────────────────────────────────────────────────
    parent_dir = os.path.dirname(args.gmres)
    out_png = os.path.join(parent_dir,
                           f"visual_comparison_{petsc_lab}_cycle{cycle:05d}.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ── Console summary ──────────────────────────────────────────────────
    for ftype, comp, field_label in fields:
        gdata = gmres_data[(ftype, comp)]
        pdata = petsc_data[(ftype, comp)]
        diff = pdata - gdata
        max_abs = np.max(np.abs(diff))
        norm_g = np.linalg.norm(gdata)
        l2_rel = np.linalg.norm(diff) / norm_g if norm_g > 0 else 0.0
        print(f"    {field_label:5s}  |max diff| = {max_abs:.2e}   "
              f"L2 rel err = {l2_rel:.2e}")
