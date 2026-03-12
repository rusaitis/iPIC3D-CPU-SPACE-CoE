#!/usr/bin/env python3
"""
plot_utils.py — Shared utilities for iPIC3D test plotting scripts.

Consolidates HDF5 I/O, cycle discovery, and dataclasses used by
plot_l2_timeseries.py and plot_field_comparison.py.
"""

import dataclasses
import glob
import os
import re
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
