"""Validate virtual satellite data against HDF5 field dumps.

Compares satellite field values at the first recorded timestep (cycle 202502)
against trilinearly interpolated values from the nearest HDF5 field dump
(cycle 202500). The 2-cycle offset is a known source of small discrepancy.

Usage:
    pixi run python scripts/validate_satellites.py \
        --sat-dir examples/SateliteData \
        --field-file examples/MHDUCLA-Fields_202500.h5

HDF5 layout (H5hut-io format):
    Step#0/Block/{FieldName}/0  — shape (nzn, nyn, nxn) = (321, 131, 461), float32
    Physical coords: x = ix * dx, y = iy * dy, z = iz * dz
    where dx = Lx/nxc, dy = Ly/nyc, dz = Lz/nzc (all 0.4 for MHDUCLA)
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from read_virtual_satellites import (
    FIELD_NAMES,
    FIELD_TO_HDF5,
    check_probe_files,
    read_all_satellites,
)

# MHDUCLA grid parameters
LX, LY, LZ = 184.0, 52.0, 128.0
NXC, NYC, NZC = 460, 130, 320  # cells
DX, DY, DZ = LX / NXC, LY / NYC, LZ / NZC  # all 0.4
NXN, NYN, NZN = NXC + 1, NYC + 1, NZC + 1   # nodes: 461, 131, 321


def load_hdf5_field(h5file, field_name):
    """Load a field array from the H5hut-io file.

    Returns ndarray of shape (nzn, nyn, nxn) = (321, 131, 461).
    """
    path = f"Step#0/Block/{field_name}/0"
    return h5file[path][:]


def interpolate_at_positions(field_3d, positions):
    """Trilinearly interpolate a 3D field at given physical positions.

    Parameters
    ----------
    field_3d : ndarray (nzn, nyn, nxn)
        Field data indexed as [iz, iy, ix].
    positions : ndarray (N, 3)
        Physical (x, y, z) positions.

    Returns
    -------
    values : ndarray (N,)
        Interpolated field values.
    """
    # Convert physical coords to fractional grid indices
    fx = positions[:, 0] / DX
    fy = positions[:, 1] / DY
    fz = positions[:, 2] / DZ

    # Clamp to valid range
    fx = np.clip(fx, 0, NXN - 1)
    fy = np.clip(fy, 0, NYN - 1)
    fz = np.clip(fz, 0, NZN - 1)

    # Integer lower corner
    ix0 = np.clip(fx.astype(int), 0, NXN - 2)
    iy0 = np.clip(fy.astype(int), 0, NYN - 2)
    iz0 = np.clip(fz.astype(int), 0, NZN - 2)

    # Fractional offsets
    dx = fx - ix0
    dy = fy - iy0
    dz = fz - iz0

    # Trilinear interpolation: field indexed as [iz, iy, ix]
    c000 = field_3d[iz0,     iy0,     ix0]
    c001 = field_3d[iz0,     iy0,     ix0 + 1]
    c010 = field_3d[iz0,     iy0 + 1, ix0]
    c011 = field_3d[iz0,     iy0 + 1, ix0 + 1]
    c100 = field_3d[iz0 + 1, iy0,     ix0]
    c101 = field_3d[iz0 + 1, iy0,     ix0 + 1]
    c110 = field_3d[iz0 + 1, iy0 + 1, ix0]
    c111 = field_3d[iz0 + 1, iy0 + 1, ix0 + 1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz


def compute_comparison(sat_values, hdf5_values, name):
    """Compute comparison statistics between satellite and HDF5 values."""
    diff = sat_values - hdf5_values
    abs_diff = np.abs(diff)

    # Relative error (avoid division by zero)
    scale = np.maximum(np.abs(hdf5_values), 1e-20)
    rel_err = abs_diff / scale

    # Correlation
    if np.std(sat_values) > 0 and np.std(hdf5_values) > 0:
        corr = np.corrcoef(sat_values, hdf5_values)[0, 1]
    else:
        corr = float("nan")

    return {
        "name": name,
        "mean_abs_err": np.mean(abs_diff),
        "max_abs_err": np.max(abs_diff),
        "median_rel_err": np.median(rel_err),
        "p95_rel_err": np.percentile(rel_err, 95),
        "correlation": corr,
        "sat_range": (float(np.min(sat_values)), float(np.max(sat_values))),
        "hdf5_range": (float(np.min(hdf5_values)), float(np.max(hdf5_values))),
    }


def validate(sat_dir, field_file, output_fig="satellite_validation.png",
             max_files=None):
    """Run the full validation pipeline."""

    # --- Load satellite data ---
    print("=" * 60)
    print("SATELLITE DATA VALIDATION")
    print("=" * 60)

    print(f"\nLoading satellites from {sat_dir}...")
    pattern = "VirtualSatelliteTraces*.txt"
    sat_dir = Path(sat_dir)

    if max_files:
        # For quick testing, limit number of files
        files = sorted(sat_dir.glob(pattern))[:max_files]
        from read_virtual_satellites import read_satellite_file
        all_coords, all_data = [], []
        ref_cycles = None
        file_map = []
        for f in files:
            c, cy, d = read_satellite_file(f)
            if ref_cycles is None:
                ref_cycles = cy
            all_coords.append(c)
            all_data.append(d)
            file_map.append((f.name, len(c)))
        coords = np.concatenate(all_coords)
        data = np.concatenate(all_data, axis=1)
        cycles = ref_cycles
    else:
        coords, cycles, data, file_map = read_all_satellites(sat_dir)

    n_sats = len(coords)
    print(f"  Satellites: {n_sats}")
    print(f"  Timesteps:  {len(cycles)} "
          f"(cycles {cycles[0]:.0f}–{cycles[-1]:.0f})")
    print(f"  Coord range:")
    print(f"    x: [{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]  "
          f"(domain: [0, {LX}])")
    print(f"    y: [{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]  "
          f"(domain: [0, {LY}])")
    print(f"    z: [{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]  "
          f"(domain: [0, {LZ}])")

    # Use first timestep for comparison
    sat_t0 = data[0, :, :]  # shape (n_sats, 14)
    field_cycle = 202500
    sat_cycle = int(cycles[0])
    print(f"\n  Comparing satellite cycle {sat_cycle} vs HDF5 cycle {field_cycle}")
    print(f"  (offset: {sat_cycle - field_cycle} cycles = "
          f"{(sat_cycle - field_cycle) * 0.1:.1f} time units)")

    # --- Load HDF5 fields ---
    print(f"\nLoading HDF5 fields from {field_file}...")
    h5 = h5py.File(field_file, "r")

    # List available fields
    block = h5["Step#0/Block"]
    print(f"  Available fields: {sorted(block.keys())}")

    # Fields to compare (those present in both satellite data and HDF5)
    compare_fields = []
    for i, fname in enumerate(FIELD_NAMES):
        h5name = FIELD_TO_HDF5.get(fname)
        if h5name and h5name in block:
            compare_fields.append((i, fname, h5name))

    print(f"  Comparing {len(compare_fields)} fields: "
          f"{[f[1] for f in compare_fields]}")

    # --- Interpolate and compare ---
    print("\nInterpolating HDF5 fields at satellite positions...")
    results = []
    sat_vals_dict = {}
    h5_vals_dict = {}

    for col_idx, sat_name, h5_name in compare_fields:
        field_3d = load_hdf5_field(h5, h5_name)
        h5_interp = interpolate_at_positions(field_3d, coords)
        sat_col = sat_t0[:, col_idx]

        stats = compute_comparison(sat_col, h5_interp, sat_name)
        results.append(stats)

        sat_vals_dict[sat_name] = sat_col
        h5_vals_dict[sat_name] = h5_interp

    h5.close()

    # --- Print statistics ---
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"\n{'Field':<8} {'Corr':>8} {'Mean |err|':>12} {'Max |err|':>12} "
          f"{'Med rel%':>10} {'P95 rel%':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<8} {r['correlation']:>8.6f} "
              f"{r['mean_abs_err']:>12.2e} {r['max_abs_err']:>12.2e} "
              f"{r['median_rel_err']*100:>9.4f}% "
              f"{r['p95_rel_err']*100:>9.4f}%")

    print(f"\n{'Field':<8} {'Sat range':>30} {'HDF5 range':>30}")
    print("-" * 70)
    for r in results:
        sr = r["sat_range"]
        hr = r["hdf5_range"]
        print(f"{r['name']:<8} [{sr[0]:>12.4e}, {sr[1]:>12.4e}] "
              f"[{hr[0]:>12.4e}, {hr[1]:>12.4e}]")

    # --- Check Probe files ---
    print(f"\n{'='*60}")
    print("PROBE FILES")
    print("=" * 60)
    probe_report = check_probe_files(sat_dir)
    if probe_report:
        has_data = sum(1 for p in probe_report if p["has_data"])
        print(f"  Found {len(probe_report)} probe files")
        print(f"  With data: {has_data}  |  Header-only: "
              f"{len(probe_report) - has_data}")
        if has_data == 0:
            print("  All probe files contain only headers (no timestep data).")
            print("  This is expected when VirtualProbesOutputCycle = 0 in the input.")
        print(f"\n  Probe locations:")
        for p in probe_report[:6]:
            print(f"    {p['filename']}: ({p['x']:.0f}, {p['y']:.0f}, {p['z']:.0f})")
        if len(probe_report) > 6:
            print(f"    ... and {len(probe_report) - 6} more")
    else:
        print("  No Probe*.txt files found.")

    # --- Generate figure ---
    print(f"\nGenerating validation figure: {output_fig}")
    _plot_validation(coords, sat_vals_dict, h5_vals_dict, results, output_fig)
    print("Done.")

    return results


def _plot_validation(coords, sat_vals, h5_vals, results, output_fig):
    """Generate the multi-panel validation figure."""

    em_fields = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
    em_present = [f for f in em_fields if f in sat_vals]

    n_scatter = len(em_present)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Virtual Satellite Validation: satellite (cycle 202502) vs "
        "HDF5 interpolated (cycle 202500)",
        fontsize=13, y=0.98,
    )

    # --- Panel 1: Scatter plots for EM fields ---
    for i, fname in enumerate(em_present):
        ax = fig.add_subplot(3, n_scatter, i + 1)
        sv = sat_vals[fname]
        hv = h5_vals[fname]

        ax.scatter(hv, sv, s=1, alpha=0.3, rasterized=True)
        lims = [min(hv.min(), sv.min()), max(hv.max(), sv.max())]
        margin = (lims[1] - lims[0]) * 0.05 or 1e-6
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "r-", lw=0.8, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"HDF5 {fname}")
        ax.set_ylabel(f"Satellite {fname}")
        ax.set_aspect("equal")

        # Find correlation for this field
        r = next((r for r in results if r["name"] == fname), None)
        if r:
            ax.set_title(f"{fname}  (r={r['correlation']:.6f})", fontsize=10)

    # --- Panel 2: Spatial map of Bz error ---
    ax_map = fig.add_subplot(3, 2, 3)
    if "Bz" in sat_vals:
        err = sat_vals["Bz"] - h5_vals["Bz"]
        sc = ax_map.scatter(coords[:, 0], coords[:, 2], c=err,
                            s=2, cmap="RdBu_r", alpha=0.5, rasterized=True)
        vmax = np.percentile(np.abs(err), 98)
        if vmax > 0:
            sc.set_clim(-vmax, vmax)
        plt.colorbar(sc, ax=ax_map, label="Bz error (sat − HDF5)")
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("z")
        ax_map.set_title("Bz error: x-z plane (all y)")

    ax_map2 = fig.add_subplot(3, 2, 4)
    if "Bz" in sat_vals:
        sc2 = ax_map2.scatter(coords[:, 0], coords[:, 1], c=err,  # noqa: F821
                              s=2, cmap="RdBu_r", alpha=0.5, rasterized=True)
        if vmax > 0:  # noqa: F821
            sc2.set_clim(-vmax, vmax)
        plt.colorbar(sc2, ax=ax_map2, label="Bz error (sat − HDF5)")
        ax_map2.set_xlabel("x")
        ax_map2.set_ylabel("y")
        ax_map2.set_title("Bz error: x-y plane (all z)")

    # --- Panel 3: Histograms of relative error ---
    ax_hist_b = fig.add_subplot(3, 2, 5)
    ax_hist_e = fig.add_subplot(3, 2, 6)

    for fname in ["Bx", "By", "Bz"]:
        if fname in sat_vals:
            scale = np.maximum(np.abs(h5_vals[fname]), 1e-20)
            rel = np.abs(sat_vals[fname] - h5_vals[fname]) / scale * 100
            rel_clipped = np.clip(rel, 0, 100)
            ax_hist_b.hist(rel_clipped, bins=100, alpha=0.5, label=fname,
                           density=True)
    ax_hist_b.set_xlabel("Relative error (%)")
    ax_hist_b.set_ylabel("Density")
    ax_hist_b.set_title("B-field relative error distribution")
    ax_hist_b.legend()
    ax_hist_b.set_xlim(0, None)

    for fname in ["Ex", "Ey", "Ez"]:
        if fname in sat_vals:
            scale = np.maximum(np.abs(h5_vals[fname]), 1e-20)
            rel = np.abs(sat_vals[fname] - h5_vals[fname]) / scale * 100
            rel_clipped = np.clip(rel, 0, 100)
            ax_hist_e.hist(rel_clipped, bins=100, alpha=0.5, label=fname,
                           density=True)
    ax_hist_e.set_xlabel("Relative error (%)")
    ax_hist_e.set_ylabel("Density")
    ax_hist_e.set_title("E-field relative error distribution")
    ax_hist_e.legend()
    ax_hist_e.set_xlim(0, None)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_fig}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate satellite data against HDF5 field dumps"
    )
    parser.add_argument("--sat-dir", required=True,
                        help="Directory with VirtualSatelliteTraces*.txt")
    parser.add_argument("--field-file", required=True,
                        help="HDF5 field dump file")
    parser.add_argument("--output", default="satellite_validation.png",
                        help="Output figure path (default: satellite_validation.png)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of satellite files (for quick testing)")
    args = parser.parse_args()

    results = validate(args.sat_dir, args.field_file, args.output,
                       max_files=args.max_files)

    # Exit with error if B-field correlations are poor
    b_fields = [r for r in results if r["name"] in ("Bx", "By", "Bz")]
    if b_fields:
        min_corr = min(r["correlation"] for r in b_fields)
        if min_corr < 0.99:
            print(f"\nWARNING: Minimum B-field correlation = {min_corr:.6f} (< 0.99)")
            sys.exit(1)
        else:
            print(f"\nAll B-field correlations >= {min_corr:.6f} — PASS")
