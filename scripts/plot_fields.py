#!/usr/bin/env python3
"""2D field slice visualization from iPIC3D HDF5 dumps.

Reads grid dimensions from the HDF5 file itself — no hardcoded domain
parameters.  Supports arbitrary slice planes (xz, xy, yz), composite
fields (Jy_tot, n_e, etc.), and side-by-side cycle comparison with a
difference panel.

Usage:
    # Single file, xz midplane
    pixi run python scripts/plot_fields.py \
        --files data/Fields_202500.h5 \
        --fields Bz Jy_tot rho_0

    # Two-file comparison with difference column
    pixi run python scripts/plot_fields.py \
        --files data/Fields_100000.h5 data/Fields_202500.h5 \
        --fields Bz Jy_tot rho_0 divB \
        -o scripts/output/field_comparison.png

    # xy plane at z=64, with satellite region overlay
    pixi run python scripts/plot_fields.py \
        --files data/Fields_202500.h5 \
        --plane xy --slice-coord 64.0 \
        --sat-box 0 10 0 15
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from plot_theme import add_theme_arg, apply_theme


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def read_grid_params(h5file, domain=None):
    """Infer grid node counts and cell sizes from the first field dataset.

    Parameters
    ----------
    h5file : h5py.File
    domain : tuple of 3 floats or None
        (Lx, Ly, Lz) domain extents.  If None, tries HDF5 attributes,
        then falls back to index coordinates (dx=dy=dz=1).

    Returns (nxn, nyn, nzn, dx, dy, dz) where n*n = node counts.
    """
    block = h5file["Step#0/Block"]
    first_field = next(iter(block))
    shape = block[first_field]["0"].shape  # (nzn, nyn, nxn)
    nzn, nyn, nxn = shape

    if domain is not None:
        lx, ly, lz = domain
        dx = lx / (nxn - 1)
        dy = ly / (nyn - 1)
        dz = lz / (nzn - 1)
        return nxn, nyn, nzn, dx, dy, dz

    # Try reading domain extents from HDF5 attributes (iPIC3D stores these
    # in some output modes).  Fall back to index coordinates.
    dx = dy = dz = 1.0
    for attr_group in [h5file, h5file.get("Step#0", h5file)]:
        if hasattr(attr_group, "attrs"):
            if "Lx" in attr_group.attrs:
                dx = float(attr_group.attrs["Lx"]) / (nxn - 1)
                dy = float(attr_group.attrs["Ly"]) / (nyn - 1)
                dz = float(attr_group.attrs["Lz"]) / (nzn - 1)
                break

    if dx == 1.0:
        print("  NOTE: No domain extents found in HDF5 or --domain; "
              "using index coordinates (dx=dy=dz=1)")

    return nxn, nyn, nzn, dx, dy, dz


def load_field(h5file, name):
    """Load a field from Step#0/Block/{name}/0 → (nzn, nyn, nxn)."""
    return h5file[f"Step#0/Block/{name}/0"][:]


def available_fields(h5file):
    """List field names available in the HDF5 file."""
    return sorted(h5file["Step#0/Block"].keys())


def extract_cycle(filepath):
    """Extract cycle number from filename like MHDUCLA-Fields_202500.h5."""
    m = re.search(r"_(\d+)\.h5$", str(filepath))
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------

# axis_name → (array index, perpendicular pair)
_PLANE_INFO = {
    "xz": {"normal_idx": 1, "axes": (0, 2), "labels": ("x", "z")},
    "xy": {"normal_idx": 2, "axes": (0, 1), "labels": ("x", "y")},
    "yz": {"normal_idx": 0, "axes": (1, 2), "labels": ("y", "z")},
}


def get_field_slice(h5file, field_key, plane, slice_idx):
    """Extract a 2D slice for a given field key and plane.

    Supports composite fields:
        Jx_tot, Jy_tot, Jz_tot = Jx_0+Jx_1, ...  (total current)
        n_e = -rho_0 * 4π   (electron number density)
        n_i =  rho_1 * 4π   (ion number density)

    Parameters
    ----------
    h5file : h5py.File
    field_key : str
    plane : str  ("xz", "xy", "yz")
    slice_idx : int  (index along the normal axis)

    Returns
    -------
    data_2d : ndarray
    """
    info = _PLANE_INFO[plane]
    ni = info["normal_idx"]

    def _slice_3d(f3d):
        """Slice a (nzn, nyn, nxn) array along the normal axis."""
        # Array is indexed [iz, iy, ix] → axis mapping: x=2, y=1, z=0
        arr_axis = 2 - ni  # x→2, y→1, z→0
        slicing = [slice(None)] * 3
        slicing[arr_axis] = slice_idx
        return f3d[tuple(slicing)]

    # Composite fields
    if re.match(r"J[xyz]_tot$", field_key):
        comp = field_key[1]  # x, y, or z
        f0 = load_field(h5file, f"J{comp}_0")
        f1 = load_field(h5file, f"J{comp}_1")
        return _slice_3d(f0 + f1)
    elif field_key == "n_e":
        return _slice_3d(-load_field(h5file, "rho_0") * 4 * np.pi)
    elif field_key == "n_i":
        return _slice_3d(load_field(h5file, "rho_1") * 4 * np.pi)
    else:
        return _slice_3d(load_field(h5file, field_key))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_FIELD_LABELS = {
    "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$",
    "Ex": r"$E_x$", "Ey": r"$E_y$", "Ez": r"$E_z$",
    "Jx_tot": r"$J_x$ (total)", "Jy_tot": r"$J_y$ (total)",
    "Jz_tot": r"$J_z$ (total)",
    "rho_0": r"$\rho_e$", "rho_1": r"$\rho_i$",
    "n_e": r"$n_e$", "n_i": r"$n_i$",
    "divB": r"$\nabla \cdot \mathbf{B}$",
}


def field_label(key):
    """Human-readable label for a field key."""
    return _FIELD_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fields(files, fields, plane, slice_coord, output,
                sat_box=None, grid_params=None):
    """Plot 2D field slices, optionally comparing two cycles side by side.

    Parameters
    ----------
    files : list of str
        1 or 2 HDF5 field file paths.
    fields : list of str
        Field keys to plot (one row per field).
    plane : str
        "xz", "xy", or "yz".
    slice_coord : float
        Physical coordinate along the normal axis for the slice.
    output : str or None
        Save path; None shows interactively.
    sat_box : tuple of 4 floats or None
        (xmin, xmax, ymin, ymax) in the plane's coordinate system.
        Draws a dashed rectangle overlay.
    grid_params : tuple or None
        (nxn, nyn, nzn, dx, dy, dz).  If None, read from the first file.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle

    info = _PLANE_INFO[plane]
    normal_idx = info["normal_idx"]
    ax0, ax1 = info["axes"]       # indices into (x, y, z)
    xlabel, ylabel = info["labels"]

    # Read grid from first file if not provided
    if grid_params is None:
        with h5py.File(files[0], "r") as h5:
            grid_params = read_grid_params(h5)
    nxn, nyn, nzn = grid_params[:3]
    dx, dy, dz = grid_params[3:]

    sizes = [nxn, nyn, nzn]
    deltas = [dx, dy, dz]

    # Physical coordinate arrays for the in-plane axes
    coord_h = np.arange(sizes[ax0]) * deltas[ax0]
    coord_v = np.arange(sizes[ax1]) * deltas[ax1]

    # Convert slice coordinate to index
    slice_idx = int(round(slice_coord / deltas[normal_idx]))
    slice_idx = min(max(slice_idx, 0), sizes[normal_idx] - 1)
    normal_name = ["x", "y", "z"][normal_idx]

    nfields = len(fields)
    ncycles = len(files)
    has_diff = ncycles == 2

    # --- Layout with nested GridSpecs ---
    if has_diff:
        fig = plt.figure(figsize=(5.5 * 3, 3.2 * nfields))
        outer = gridspec.GridSpec(1, 2, figure=fig,
                                  width_ratios=[2.08, 1.08],
                                  wspace=0.15)
        gs_left = outer[0].subgridspec(nfields, ncycles + 1,
                                       width_ratios=[1] * ncycles + [0.04],
                                       wspace=0.04, hspace=0.08)
        gs_right = outer[1].subgridspec(nfields, 2,
                                        width_ratios=[1, 0.04],
                                        wspace=0.04, hspace=0.08)
    else:
        fig = plt.figure(figsize=(5.5 * ncycles, 3.2 * nfields))
        gs_left = gridspec.GridSpec(nfields, ncycles + 1, figure=fig,
                                    width_ratios=[1] * ncycles + [0.04],
                                    wspace=0.04, hspace=0.08)
        gs_right = None

    cycles_nums = [extract_cycle(f) for f in files]
    stats = []

    for row, fkey in enumerate(fields):
        # Load all slices to compute shared color limits
        slices = []
        for fpath in files:
            h5 = h5py.File(fpath, "r")
            s = get_field_slice(h5, fkey, plane, slice_idx)
            h5.close()
            slices.append(s)

        all_vals = np.concatenate([s.ravel() for s in slices])
        if fkey.startswith(("n_", "N_", "rho")):
            cmap = "viridis"
            vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        else:
            cmap = "RdBu_r"
            vmax = float(np.percentile(np.abs(all_vals), 99))
            vmin = -vmax

        # Plot cycle panels
        row_axes = []
        im = None
        for col, s in enumerate(slices):
            share_y = row_axes[0] if col > 0 else None
            ax = fig.add_subplot(gs_left[row, col], sharey=share_y)
            if col > 0:
                ax.tick_params(labelleft=False)
            row_axes.append(ax)

            im = ax.pcolormesh(coord_h, coord_v, s, shading="nearest",
                               cmap=cmap, vmin=vmin, vmax=vmax)

            if row == 0:
                ax.set_title(f"Cycle {cycles_nums[col]:,}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{field_label(fkey)}\n{ylabel}")
            if row == nfields - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            ax.set_aspect("equal")

            if sat_box is not None:
                rect = Rectangle((sat_box[0], sat_box[2]),
                                 sat_box[1] - sat_box[0],
                                 sat_box[3] - sat_box[2],
                                 linewidth=1.5, edgecolor="lime",
                                 facecolor="none", linestyle="--")
                ax.add_patch(rect)

            stats.append({
                "field": fkey, "cycle": cycles_nums[col],
                "min": float(np.min(s)), "max": float(np.max(s)),
                "mean": float(np.mean(s)), "std": float(np.std(s)),
                "absmax": float(np.max(np.abs(s))),
            })

        # Shared colorbar for cycle panels
        cax = fig.add_subplot(gs_left[row, ncycles])
        fig.colorbar(im, cax=cax)

        # Difference column
        if has_diff:
            diff = slices[1] - slices[0]
            dvmax = float(np.percentile(np.abs(diff), 99))
            if dvmax == 0:
                dvmax = 1e-10

            ax_diff = fig.add_subplot(gs_right[row, 0],
                                      sharey=row_axes[0])
            ax_diff.tick_params(labelleft=False)
            im_diff = ax_diff.pcolormesh(coord_h, coord_v, diff,
                                         shading="nearest", cmap="RdBu_r",
                                         vmin=-dvmax, vmax=dvmax)
            if row == 0:
                ax_diff.set_title(
                    f"Difference\n({cycles_nums[1]:,} − {cycles_nums[0]:,})",
                    fontsize=12)
            if row == nfields - 1:
                ax_diff.set_xlabel(xlabel)
            else:
                ax_diff.tick_params(labelbottom=False)
            ax_diff.set_aspect("equal")

            cax_diff = fig.add_subplot(gs_right[row, 1])
            fig.colorbar(im_diff, cax=cax_diff)

    fig.suptitle(
        f"{plane} plane at {normal_name} = {slice_coord:.1f}  "
        f"(i{normal_name} = {slice_idx})",
        fontsize=14,
    )

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()

    # Print convergence stats
    print(f"\n=== Field Statistics ({plane} plane, "
          f"{normal_name}={slice_coord:.1f}) ===")
    print(f"{'Field':<10} {'Cycle':>10} {'Min':>12} {'Max':>12} "
          f"{'|Max|':>12} {'Std':>12}")
    print("-" * 70)
    for s in stats:
        print(f"{s['field']:<10} {s['cycle']:>10,} {s['min']:>12.4e} "
              f"{s['max']:>12.4e} {s['absmax']:>12.4e} {s['std']:>12.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D field slices from iPIC3D HDF5 dumps"
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="HDF5 field files (1 or 2 for comparison)")
    parser.add_argument("--fields", nargs="+",
                        default=["Bz", "Jy_tot", "rho_0"],
                        help="Fields to plot (default: Bz Jy_tot rho_0)")
    parser.add_argument("--plane", choices=["xz", "xy", "yz"], default="xz",
                        help="Slice plane (default: xz)")
    parser.add_argument("--slice-coord", type=float, default=None,
                        help="Coordinate along normal axis "
                             "(default: domain midpoint)")
    parser.add_argument("--domain", nargs=3, type=float, default=None,
                        metavar=("LX", "LY", "LZ"),
                        help="Domain extents (tries HDF5 attrs if omitted, "
                             "then falls back to index coordinates)")
    parser.add_argument("--sat-box", nargs=4, type=float, default=None,
                        metavar=("H0", "H1", "V0", "V1"),
                        help="Satellite region overlay box in plane coords")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path")
    add_theme_arg(parser)
    args = parser.parse_args()

    domain = tuple(args.domain) if args.domain else None

    # Read grid params (needed for default slice coord and passed to plotter)
    with h5py.File(args.files[0], "r") as h5:
        gp = read_grid_params(h5, domain=domain)

    if args.slice_coord is None:
        normal_idx = _PLANE_INFO[args.plane]["normal_idx"]
        sizes = [gp[0], gp[1], gp[2]]
        deltas = [gp[3], gp[4], gp[5]]
        args.slice_coord = (sizes[normal_idx] - 1) * deltas[normal_idx] / 2.0

    apply_theme(args)
    plot_fields(args.files, args.fields, args.plane, args.slice_coord,
                args.output, sat_box=args.sat_box, grid_params=gp)


if __name__ == "__main__":
    main()
