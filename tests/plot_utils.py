#!/usr/bin/env python3
"""
plot_utils.py — Shared utilities for iPIC3D test plotting scripts.

Consolidates HDF5 I/O, cycle discovery, and dataclasses used by
plot_l2_timeseries.py and plot_field_comparison.py.
"""

import csv
import dataclasses
import glob
import os
import re
import statistics
import sys
from typing import Optional


# ── Import guard ─────────────────────────────────────────────────────────

def require_imports(*module_names: str) -> None:
    """Check that all required modules are importable; exit(2) if any missing.

    Exit code 2 (not 0) lets callers like test.sh distinguish
    "missing dependency" from "success".
    """
    missing = []
    for name in module_names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"  WARNING: missing Python packages: {', '.join(missing)}")
        print(f"  Install with: pip3 install {' '.join(missing)}")
        sys.exit(2)


# ── Cycle discovery ──────────────────────────────────────────────────────

def discover_cycles(run_dir: str) -> list:
    """Scan Fields_* directories, extract cycle numbers, return sorted list."""
    pattern = os.path.join(run_dir, "Fields_*")
    cycles = []
    for d in glob.glob(pattern):
        m = re.match(r"Fields_(\d+)", os.path.basename(d))
        if m:
            cycles.append(int(m.group(1)))
    return sorted(cycles)


def find_common_cycles(*run_dirs: str) -> list:
    """Return sorted cycle numbers present in all run directories."""
    return sorted(set.intersection(*(set(discover_cycles(d)) for d in run_dirs)))


def solver_label(run_dir: str) -> str:
    """Extract a human-readable solver label from a run directory path."""
    return os.path.basename(run_dir)


# ── HDF5 field I/O ───────────────────────────────────────────────────────

def load_field(run_dir: str, cycle: int, field_type: str, component: str,
               z_slice: Optional[int] = None):
    """Load a field component from phdf5 output.

    Parameters
    ----------
    run_dir : str       Path to simulation output directory.
    cycle : int         Cycle number.
    field_type : str    "B" or "E".
    component : str     "x", "y", or "z".
    z_slice : int or None
        If None, return the full 3D array (Nx, Ny, Nz).
        If an integer, return a 2D slice ``data[:, :, z_slice]``.

    Returns
    -------
    np.ndarray
        Shape (Nx, Ny, Nz) when z_slice is None, or (Nx, Ny) otherwise.

    Raises
    ------
    FileNotFoundError   If the HDF5 file does not exist.
    KeyError            If the dataset is not found inside the file.
    """
    import h5py

    cycle_str = f"{cycle:05d}"
    h5_path = os.path.join(run_dir, f"Fields_{cycle_str}",
                           f"{field_type}_{cycle_str}.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    dataset = f"/Fields/{field_type}{component}"
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset '{dataset}' not found in {h5_path}. "
                           f"Available: {list(f['/Fields'].keys())}")
        data = f[dataset][:]

    if z_slice is not None:
        return data[:, :, z_slice]
    return data


# ── ConservedQuantities parser ───────────────────────────────────────────

def parse_conserved_quantities(run_dir: str) -> dict:
    """Parse ConservedQuantities.txt from a simulation run directory.

    Returns dict with arrays: cycle, E_energy, B_energy, KE, total_energy,
    delta_energy. Returns empty dict if the file is missing or has no data.
    """
    path = os.path.join(run_dir, "ConservedQuantities.txt")
    if not os.path.isfile(path):
        return {}

    result = {k: [] for k in
              ('cycle', 'E_energy', 'B_energy', 'KE', 'total_energy', 'delta_energy')}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('I', '=', 'V')):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) < 12:
                continue
            result['cycle'].append(int(vals[0]))
            result['E_energy'].append(vals[1])
            result['B_energy'].append(vals[5])
            result['KE'].append(vals[9])
            result['total_energy'].append(vals[10])
            result['delta_energy'].append(vals[11])

    return result if result['cycle'] else {}


# ── Profile file discovery ────────────────────────────────────────────────

def grid_tag_from_dir(dir_path: str) -> str:
    """Extract grid tag (e.g. '100x100x1') from a run directory name."""
    m = re.search(r'(\d+x\d+x\d+)$', os.path.basename(dir_path))
    return m.group(1) if m else ''


def solver_from_dir(dir_path: str, grid_tag: str) -> str:
    """Extract solver label by stripping the grid suffix from dir name."""
    name = os.path.basename(dir_path)
    if name.endswith('_' + grid_tag):
        return name[:-(len(grid_tag) + 1)]
    return name


def discover_run_groups(output_dir: str, ref_solver: str = '',
                        require_fields: bool = False):
    """Group run directories by grid tag, identify ref vs test within each.

    Parameters
    ----------
    output_dir : str
        Directory containing solver run dirs (e.g. GMRES_100x100x1/).
    ref_solver : str
        Reference solver name.  Auto-detected from ref_solver.txt if empty.
    require_fields : bool
        If True, only include directories that contain Fields_* output.

    Returns
    -------
    list of (grid_tag, ref_dir, test_dirs) tuples, sorted by grid size.
    """
    if not ref_solver:
        ref_file = os.path.join(output_dir, 'ref_solver.txt')
        if os.path.isfile(ref_file):
            with open(ref_file) as f:
                ref_solver = f.read().strip()

    # Collect all run directories
    all_dirs = sorted(glob.glob(os.path.join(output_dir, '*')))
    run_dirs = [d for d in all_dirs
                if os.path.isdir(d) and grid_tag_from_dir(d)]

    if require_fields:
        run_dirs = [d for d in run_dirs if discover_cycles(d)]

    # Group by grid tag
    groups = {}  # grid_tag -> list of dirs
    for d in run_dirs:
        tag = grid_tag_from_dir(d)
        groups.setdefault(tag, []).append(d)

    # Sort grids by first dimension
    sorted_tags = sorted(groups, key=lambda t: int(t.split('x')[0]))

    result = []
    for tag in sorted_tags:
        dirs = groups[tag]
        # Find ref: match ref_solver prefix, prefer GMRES, fall back to first
        ref_dir = None
        if ref_solver:
            for d in dirs:
                if solver_from_dir(d, tag) == ref_solver:
                    ref_dir = d
                    break
        if ref_dir is None:
            for d in dirs:
                if solver_from_dir(d, tag) == 'GMRES':
                    ref_dir = d
                    break
        if ref_dir is None:
            ref_dir = dirs[0]

        test_dirs = [d for d in dirs if d != ref_dir]
        if test_dirs:
            result.append((tag, ref_dir, test_dirs))

    return result


def parse_grid_from_filename(path: str) -> int:
    """Extract grid N from filename like profile_GMRES_100x100x1.csv -> 100."""
    base = os.path.splitext(os.path.basename(path))[0]
    grid_part = base.rsplit('_', 1)[-1]
    try:
        return int(grid_part.split('x')[0])
    except (ValueError, IndexError):
        return 0


def discover_profile_files(profile_dir: str, largest_only: bool = True) -> tuple:
    """Find profile_*.csv files, optionally keeping only the largest grid.

    Returns (profile_files, max_grid) where max_grid is the N value of
    the largest grid found, or 0 if no files matched.
    """
    all_files = sorted(glob.glob(os.path.join(profile_dir, "profile_*.csv")))
    if not all_files:
        return [], 0

    max_grid = max(parse_grid_from_filename(f) for f in all_files)
    if largest_only:
        files = [f for f in all_files if parse_grid_from_filename(f) == max_grid]
    else:
        files = all_files
    return files, max_grid


# ── Field spec parsing ───────────────────────────────────────────────────

def parse_field_specs(spec_str: str) -> list:
    """Parse 'Bx,Ez,By' into [('B','x','$B_x$'), ('E','z','$E_z$'), ...].

    Each spec must be exactly two characters: field type (B/E) and
    component (x/y/z).  Invalid specs are skipped with a warning.
    """
    fields = []
    for spec in spec_str.split(','):
        spec = spec.strip()
        if len(spec) == 2:
            field_type = spec[0].upper()
            component = spec[1].lower()
            field_latex = f"${field_type}_{component}$"
            fields.append((field_type, component, field_latex))
        else:
            print(f"  WARNING: Unrecognized field spec '{spec}', skipping.")
    return fields


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class PowerLawFit:
    """Power-law fit L2(t) = A * t^alpha in log-log space.

    Typical growth modes in reconnection simulations:
      alpha ~ 0.5  diffusive (random walk of truncation errors)
      alpha ~ 1.0  linear accumulation (systematic bias per step)
      alpha ~ 2.0  quadratic (instability amplification)
    """
    alpha: float
    r_squared: float
    amplitude: float
    log_amplitude: float
    knee_cycle: Optional[float] = None
    alpha_early: Optional[float] = None
    log_amplitude_early: Optional[float] = None
    alpha_late: Optional[float] = None
    log_amplitude_late: Optional[float] = None
    r_squared_piecewise: Optional[float] = None


@dataclasses.dataclass
class L2ErrorSeries:
    """Per-field L2 relative error over output cycles.

    L2 relative error = ||test - ref||_2 / ||ref||_2
    """
    cycles: list = dataclasses.field(default_factory=list)
    l2_relative: list = dataclasses.field(default_factory=list)
    max_absolute: list = dataclasses.field(default_factory=list)
    ref_norm: list = dataclasses.field(default_factory=list)


# ── Profile aggregation ──────────────────────────────────────────────────

def parse_profile_filename(name: str):
    """Extract (solver_label, grid_tag) from profile_{solver}_{NxNxN}.csv.

    Returns (None, None) if the filename doesn't match.
    """
    m = re.match(r'^profile_(.+?)_(\d+x\d+x\d+)\.csv$', name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def sum_profile(csv_path: str) -> dict:
    """Sum per-cycle timings from a profile CSV.

    Returns dict with field_s, moments_s, mover_s, cycles, mean_iters,
    or None if the file is empty.
    """
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    field_s = sum(float(r['field_solver_s']) for r in rows)
    moments_s = sum(float(r['moment_gatherer_s']) for r in rows)
    mover_s = sum(float(r['particle_mover_s']) for r in rows)
    cycles = len(rows)

    iters = [int(r['iterations']) for r in rows
             if r.get('iterations', '')]
    mean_iters = statistics.mean(iters) if iters else 0

    return {
        'field_s': field_s,
        'moments_s': moments_s,
        'mover_s': mover_s,
        'cycles': cycles,
        'mean_iters': mean_iters,
    }


def aggregate_profiles_to_csv(output_dir: str, ref_solver: str = '',
                               np_str: str = '8') -> str:
    """Build results.csv from profile_*.csv files in output_dir.

    Parameters
    ----------
    output_dir : str   Directory containing profile_*.csv files.
    ref_solver : str   Reference solver name (auto-detected if empty).
    np_str : str       Number of MPI processes (CSV metadata).

    Returns
    -------
    str  Path to the written results.csv.
    """
    # Discover profile CSVs
    profiles = {}  # (solver, grid_tag) -> csv_path
    for name in sorted(os.listdir(output_dir)):
        solver, grid_tag = parse_profile_filename(name)
        if solver:
            profiles[(solver, grid_tag)] = os.path.join(output_dir, name)

    if not profiles:
        raise FileNotFoundError(
            f"No profile_*.csv files found in {output_dir}")

    solvers = list(dict.fromkeys(k[0] for k in profiles))
    grids = list(dict.fromkeys(k[1] for k in profiles))
    grids.sort(key=lambda g: int(g.split('x')[0]))

    # Determine reference solver
    if not ref_solver:
        ref_file = os.path.join(output_dir, 'ref_solver.txt')
        if os.path.isfile(ref_file):
            with open(ref_file) as f:
                ref_solver = f.read().strip()
    if not ref_solver:
        # Prefer GMRES as default reference
        ref_solver = 'GMRES' if 'GMRES' in solvers else solvers[0]

    # Sum timings
    data = {}
    n_cycles = None
    for (solver, grid_tag), path in profiles.items():
        result = sum_profile(path)
        if result:
            data[(solver, grid_tag)] = result
            if n_cycles is None:
                n_cycles = result['cycles']

    # Write CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    nzc = int(grids[0].split('x')[2]) if grids else 1

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)

        header = ['grid_size']
        for solver in solvers:
            header.extend([solver, f'{solver}_std', f'{solver}_iters',
                           f'{solver}_converged', f'{solver}_residual'])
        header.extend(['np', 'cycles', 'nzc', 'ref_solver',
                        'ref_moments_s', 'ref_mover_s',
                        'ref_field_pct', 'ref_moments_pct', 'ref_mover_pct'])
        w.writerow(header)

        for grid_tag in grids:
            nxc = int(grid_tag.split('x')[0])
            row = [nxc if len(grids) > 1 else grid_tag]

            ref_field = ref_moments = ref_mover = None

            for solver in solvers:
                d = data.get((solver, grid_tag))
                if d:
                    row.extend([
                        f"{d['field_s']:.4f}", '', f"{d['mean_iters']:.0f}",
                        '', '',
                    ])
                    if solver == ref_solver:
                        ref_field = d['field_s']
                        ref_moments = d['moments_s']
                        ref_mover = d['mover_s']
                else:
                    row.extend(['NA', '', '', '', ''])

            row.extend([np_str, n_cycles or '?', nzc, ref_solver])

            if ref_field is not None and ref_moments is not None:
                total = ref_field + ref_moments + ref_mover
                row.extend([
                    f"{ref_moments:.4f}", f"{ref_mover:.4f}",
                    f"{ref_field/total*100:.1f}",
                    f"{ref_moments/total*100:.1f}",
                    f"{ref_mover/total*100:.1f}",
                ])
            else:
                row.extend(['NA', 'NA', 'NA', 'NA', 'NA'])

            w.writerow(row)

    return csv_path
