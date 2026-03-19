"""Reader for iPIC3D VirtualSatelliteTraces binary files.

Inspired by the C++ postprocessing tools written by Alexander E. Vapirev,
KU Leuven, Afdeling Plasma-astrofysica (2011-2012).

Binary format (from ECsimlib.cpp WriteVirtual):
  - ASCII header: "nsat\\tnsatx\\tnsaty\\tnsatz\\n" + nsat lines of "x\\ty\\tz\\n"
  - Binary data: nsat * 16 * ntimesteps float32 values
  - Per satellite per timestep: [cycle, sat_idx, Bx, By, Bz, Ex, Ey, Ez,
                                  rho_e, Jxe, Jye, Jze, rho_i, Jxi, Jyi, Jzi]
"""

from pathlib import Path

import numpy as np

# Columns in the 16-float binary record
_RAW_COLS = [
    "cycle", "sat_idx",
    "Bx", "By", "Bz", "Ex", "Ey", "Ez",
    "rho_e", "Jxe", "Jye", "Jze",
    "rho_i", "Jxi", "Jyi", "Jzi",
]

# Field columns (indices 2–15 in the raw record, i.e. 0–13 after stripping cycle/sat_idx)
FIELD_NAMES = _RAW_COLS[2:]

# Derived quantity names (computed from raw fields)
DERIVED_NAMES = [
    "Vxe", "Vye", "Vze", "Vxi", "Vyi", "Vzi",
    "Bmag", "Emag", "Ve_mag", "Vi_mag",
    "axe", "aye", "aze", "axi", "ayi", "azi",
    "ae_mag", "ai_mag",
]

# Map from field name to HDF5 dataset name
FIELD_TO_HDF5 = {
    "Bx": "Bx", "By": "By", "Bz": "Bz",
    "Ex": "Ex", "Ey": "Ey", "Ez": "Ez",
    "rho_e": "rho_0", "Jxe": "Jx_0", "Jye": "Jy_0", "Jze": "Jz_0",
    "rho_i": "rho_1", "Jxi": "Jx_1", "Jyi": "Jy_1", "Jzi": "Jz_1",
}


def read_satellite_file(filepath):
    """Read one VirtualSatelliteTraces binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to the satellite file.

    Returns
    -------
    coords : ndarray (nsat, 3)
        Physical (x, y, z) positions of the satellites.
    cycles : ndarray (ntimesteps,)
        Cycle numbers (from the first satellite's column 0).
    data : ndarray (ntimesteps, nsat, 14)
        Field values: [Bx, By, Bz, Ex, Ey, Ez,
                       rho_e, Jxe, Jye, Jze, rho_i, Jxi, Jyi, Jzi]
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        # --- ASCII header ---
        header = f.readline().decode().strip().split("\t")
        nsat = int(header[0])

        coords = np.empty((nsat, 3), dtype=np.float64)
        for i in range(nsat):
            line = f.readline().decode().strip().split("\t")
            coords[i] = [float(v) for v in line]

        # --- Binary payload ---
        raw = np.fromfile(f, dtype="<f4")

    floats_per_record = 16
    n_records = len(raw) // (nsat * floats_per_record)
    if n_records * nsat * floats_per_record != len(raw):
        raise ValueError(
            f"{filepath.name}: binary size {len(raw)} floats is not divisible by "
            f"nsat={nsat} * 16 = {nsat * floats_per_record}"
        )

    raw = raw.reshape(n_records, nsat, floats_per_record)
    cycles = raw[:, 0, 0]  # cycle number from first satellite
    data = raw[:, :, 2:]   # strip cycle and sat_idx columns

    return coords, cycles, data


def read_all_satellites(directory, file_pattern="VirtualSatelliteTraces*.txt",
                        progress=True):
    """Read all satellite files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing VirtualSatelliteTraces files.
    file_pattern : str
        Glob pattern for satellite files.
    progress : bool
        Print progress updates.

    Returns
    -------
    coords : ndarray (total_sats, 3)
    cycles : ndarray (ntimesteps,)
    data : ndarray (ntimesteps, total_sats, 14)
    file_map : list of (filename, nsat)
        Provenance: which file contributed how many satellites.
    """
    directory = Path(directory)
    files = sorted(directory.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' in {directory}"
        )

    all_coords = []
    all_data = []
    file_map = []
    ref_cycles = None

    for i, fpath in enumerate(files):
        if progress and i % 200 == 0:
            print(f"  Reading file {i+1}/{len(files)}: {fpath.name}")
        coords, cycles, data = read_satellite_file(fpath)

        if ref_cycles is None:
            ref_cycles = cycles
        elif len(cycles) != len(ref_cycles) or cycles[0] != ref_cycles[0]:
            print(f"  WARNING: {fpath.name} has {len(cycles)} timesteps "
                  f"(expected {len(ref_cycles)}), skipping")
            continue

        all_coords.append(coords)
        all_data.append(data)
        file_map.append((fpath.name, len(coords)))

    if progress:
        print(f"  Read {len(files)} files, "
              f"{sum(n for _, n in file_map)} total satellites")

    assert ref_cycles is not None, "No satellite files were read successfully"

    coords = np.concatenate(all_coords, axis=0)
    data = np.concatenate(all_data, axis=1)

    return coords, ref_cycles, data, file_map


def find_nearest_satellite(coords, target_xyz):
    """Find the satellite closest to target (x, y, z).

    Returns
    -------
    idx : int
        Index into the coords/data arrays.
    distance : float
        Euclidean distance to target.
    """
    target = np.asarray(target_xyz, dtype=np.float64)
    dists = np.linalg.norm(coords - target, axis=1)
    idx = int(np.argmin(dists))
    return idx, dists[idx]


def extract_point_timeseries(coords, data, cycles, target_xyz, dt=0.1):
    """Extract a time series at the nearest satellite to target.

    Returns
    -------
    result : dict
        Keys: 'time', 'cycle', field names, 'distance', 'sat_xyz'.
        Values are 1D ndarrays (ntimesteps,) except 'sat_xyz' (3,).
    """
    idx, dist = find_nearest_satellite(coords, target_xyz)

    result = {
        "time": cycles * dt,
        "cycle": cycles,
    }
    for i, name in enumerate(FIELD_NAMES):
        result[name] = data[:, idx, i]
    result["distance"] = dist
    result["sat_xyz"] = coords[idx]

    return result


def compute_derived_quantities(timeseries, dt=0.1):
    """Compute velocities, magnitudes, and accelerations from raw fields.

    Ported from process-virtual-sat-point.cpp (A.E. Vapirev, 2011).

    Parameters
    ----------
    timeseries : dict
        Output of extract_point_timeseries().
    dt : float
        Timestep between cycles.

    Returns
    -------
    timeseries : dict
        Same dict, augmented with DERIVED_NAMES entries.
    """
    eps = 1e-30  # guard against rho ~ 0

    # Bulk velocities V = J / rho
    for comp in ("x", "y", "z"):
        for s, rho_key in [("e", "rho_e"), ("i", "rho_i")]:
            J = timeseries[f"J{comp}{s}"]
            rho = timeseries[rho_key]
            timeseries[f"V{comp}{s}"] = np.where(
                np.abs(rho) > eps, J / rho, 0.0
            )

    # Field magnitudes
    timeseries["Bmag"] = np.sqrt(
        timeseries["Bx"]**2 + timeseries["By"]**2 + timeseries["Bz"]**2
    )
    timeseries["Emag"] = np.sqrt(
        timeseries["Ex"]**2 + timeseries["Ey"]**2 + timeseries["Ez"]**2
    )

    # Velocity magnitudes
    for s in ("e", "i"):
        timeseries[f"V{s}_mag"] = np.sqrt(
            timeseries[f"Vx{s}"]**2
            + timeseries[f"Vy{s}"]**2
            + timeseries[f"Vz{s}"]**2
        )

    # Accelerations: backward difference, 0 at t=0 (matches C++ convention)
    for comp in ("x", "y", "z"):
        for s in ("e", "i"):
            V = timeseries[f"V{comp}{s}"]
            a = np.empty_like(V)
            a[0] = 0.0
            a[1:] = (V[1:] - V[:-1]) / dt
            timeseries[f"a{comp}{s}"] = a

    for s in ("e", "i"):
        timeseries[f"a{s}_mag"] = np.sqrt(
            timeseries[f"ax{s}"]**2
            + timeseries[f"ay{s}"]**2
            + timeseries[f"az{s}"]**2
        )

    return timeseries


def extract_plane_timeseries(coords, data, cycles, plane, plane_coord,
                             tolerance=None, dt=0.1):
    """Extract all satellites in a 2D plane from the full dataset.

    Parameters
    ----------
    coords : ndarray (nsat, 3)
    data : ndarray (ntimesteps, nsat, 14)
    cycles : ndarray (ntimesteps,)
    plane : str
        "xy", "xz", or "yz" — the plane to slice.
    plane_coord : float
        Coordinate value along the normal axis (e.g. z=5.0 for an XY plane).
    tolerance : float or None
        Match tolerance. If None, auto-detected from minimum satellite spacing
        along the normal axis.
    dt : float
        Timestep for time axis.

    Returns
    -------
    result : dict
        Keys: 'coords_2d' (N, 2), 'data' (ntimesteps, N, 14),
        'cycles', 'time', 'indices', 'plane', 'plane_coord'.
    """
    axis_map = {"xy": 2, "xz": 1, "yz": 0}
    if plane not in axis_map:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")
    normal_axis = axis_map[plane]
    in_plane_axes = [i for i in range(3) if i != normal_axis]

    normal_vals = coords[:, normal_axis]

    if tolerance is None:
        unique_vals = np.unique(normal_vals)
        if len(unique_vals) > 1:
            tolerance = np.min(np.diff(unique_vals)) * 0.5
        else:
            tolerance = 0.5

    mask = np.abs(normal_vals - plane_coord) <= tolerance
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No satellites found in {plane} plane at "
            f"{['x','y','z'][normal_axis]}={plane_coord} "
            f"(tolerance={tolerance:.3f})"
        )

    return {
        "coords_2d": coords[indices][:, in_plane_axes],
        "data": data[:, indices, :],
        "cycles": cycles,
        "time": cycles * dt,
        "indices": indices,
        "plane": plane,
        "plane_coord": plane_coord,
    }


def check_probe_files(directory):
    """Check Probe*.txt files — report whether they contain data or are header-only.

    Returns
    -------
    report : list of dict
        One entry per probe file with keys: filename, x, y, z, has_data, n_header_lines.
    """
    directory = Path(directory)
    probes = sorted(directory.glob("Probe*.txt"))
    report = []
    for p in probes:
        lines = p.read_text().strip().splitlines()
        header_lines = sum(1 for l in lines if l.startswith("#"))
        data_lines = len(lines) - header_lines

        # Parse location from header
        x = y = z = None
        for l in lines:
            if "x =" in l:
                x = float(l.split("=")[1])
            elif "y =" in l:
                y = float(l.split("=")[1])
            elif "z =" in l:
                z = float(l.split("=")[1])

        report.append({
            "filename": p.name,
            "x": x, "y": y, "z": z,
            "has_data": data_lines > 0,
            "n_header_lines": header_lines,
            "n_data_lines": data_lines,
        })
    return report


# --- CLI for quick inspection ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read and inspect iPIC3D virtual satellite data"
    )
    parser.add_argument("--sat-dir", required=True,
                        help="Directory with VirtualSatelliteTraces*.txt")
    parser.add_argument("--single-file", default=None,
                        help="Read only a single file (for quick test)")
    parser.add_argument("--target", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Extract time series at nearest satellite to (X,Y,Z)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Timestep for time axis (default: 0.1)")
    args = parser.parse_args()

    if args.single_file:
        coords, cycles, data = read_satellite_file(args.single_file)
        print(f"File: {args.single_file}")
        print(f"  Satellites: {len(coords)}")
        print(f"  Timesteps:  {len(cycles)} (cycles {cycles[0]:.0f}–{cycles[-1]:.0f})")
        print(f"  Coord range: x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]"
              f"  y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]"
              f"  z=[{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")
        print(f"  Data shape: {data.shape}")
    else:
        print("Reading all satellite files...")
        coords, cycles, data, fmap = read_all_satellites(args.sat_dir)
        print(f"\nSummary:")
        print(f"  Total satellites: {len(coords)}")
        print(f"  Timesteps:        {len(cycles)} (cycles {cycles[0]:.0f}–{cycles[-1]:.0f})")
        print(f"  Coord range: x=[{coords[:,0].min():.1f}, {coords[:,0].max():.1f}]"
              f"  y=[{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]"
              f"  z=[{coords[:,2].min():.1f}, {coords[:,2].max():.1f}]")
        print(f"  Data shape: {data.shape}")

    if args.target:
        ts = extract_point_timeseries(coords, data, cycles, args.target,
                                      dt=args.dt)
        xyz = ts["sat_xyz"]
        print(f"\nNearest satellite to ({args.target}):")
        print(f"  Position: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})")
        print(f"  Distance: {ts['distance']:.3f}")
        print(f"\nFirst 5 timesteps:")
        print(f"  {'cycle':>8s}  {'Bx':>12s}  {'By':>12s}  {'Bz':>12s}  "
              f"{'Ex':>12s}  {'Ey':>12s}  {'Ez':>12s}")
        for i in range(min(5, len(ts["cycle"]))):
            print(f"  {ts['cycle'][i]:>8.0f}  {ts['Bx'][i]:>12.4e}  "
                  f"{ts['By'][i]:>12.4e}  {ts['Bz'][i]:>12.4e}  "
                  f"{ts['Ex'][i]:>12.4e}  {ts['Ey'][i]:>12.4e}  "
                  f"{ts['Ez'][i]:>12.4e}")

    # Check probes
    probe_report = check_probe_files(Path(args.sat_dir))
    if probe_report:
        print(f"\nProbe files: {len(probe_report)} found")
        has_data = sum(1 for p in probe_report if p["has_data"])
        print(f"  With data: {has_data}, Header-only: {len(probe_report) - has_data}")
        if has_data == 0:
            print("  (All probe files are header-only — no timestep data recorded)")
