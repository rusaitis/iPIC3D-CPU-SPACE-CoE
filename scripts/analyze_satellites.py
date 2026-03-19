#!/usr/bin/env python3
"""Virtual satellite analysis: time series, FFT, and plane heatmaps.

Inspired by the C++ postprocessing tools written by Alexander E. Vapirev,
KU Leuven, Afdeling Plasma-astrofysica (2011-2012).

Uses the reader from read_virtual_satellites.py and theming from
tests/plot_theme.py.

Examples
--------
  # Time series at a point
  pixi run python scripts/analyze_satellites.py \\
      --sat-dir data/SateliteData --target 92 26 64 --mode timeseries

  # Power spectrum with reference frequencies
  pixi run python scripts/analyze_satellites.py \\
      --sat-dir data/SateliteData --target 92 26 64 \\
      --mode fft --B0x 0.0097 --mass-ratio 256

  # 2D plane snapshot
  pixi run python scripts/analyze_satellites.py \\
      --sat-dir data/SateliteData --mode plane \\
      --plane xz --plane-coord 26.0 --fields Bz --cycle-idx 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np

__all__ = [
    "compute_reference_frequencies",
    "compute_power_spectrum", "compute_all_spectra",
    "plot_timeseries", "plot_spectra", "plot_plane_snapshot",
]

# Ensure project root paths are importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_project_root / "tests"))
from read_virtual_satellites import (
    DERIVED_NAMES,
    FIELD_NAMES,
    compute_derived_quantities,
    extract_plane_timeseries,
    extract_point_timeseries,
    read_all_satellites,
)

# ---------------------------------------------------------------------------
# Reference frequencies (normalized plasma units)
# ---------------------------------------------------------------------------

def compute_reference_frequencies(timeseries, mass_ratio, B0x):
    """Compute characteristic plasma frequencies from initial densities.

    In iPIC3D's normalized Gaussian units:
      - wci = B0x                               (ion cyclotron)
      - wpi = sqrt(n0)                           (ion plasma)
      - wce = wci * mass_ratio                   (electron cyclotron)
      - wlh = 1/sqrt(1/(wce*wci) + 1/wpi^2)     (lower hybrid)
      - wpe = wpi * sqrt(mass_ratio)             (electron plasma)
      - wuh = sqrt(wce^2 + wpe^2)                (upper hybrid)

    Parameters
    ----------
    timeseries : dict
        Must contain 'rho_e' and 'rho_i' arrays.
    mass_ratio : float
        mi/me (e.g. 256).
    B0x : float
        Background magnetic field magnitude.

    Returns
    -------
    freqs : dict
        Keys: 'wci', 'wpi', 'wce', 'wpe', 'wlh', 'wuh' (angular frequencies).
    """
    # iPIC3D stores charge density as rho = sign(q)*n/(4π) — see units.txt.
    # Multiply by 4π to recover n, subtract species to get n_i + n_e, /2 for average.
    n0 = (timeseries["rho_i"][0] - timeseries["rho_e"][0]) * 4 * np.pi / 2

    wci = B0x                        # ion cyclotron frequency (= B0 in normalized units)
    wpi = np.sqrt(max(n0, 1e-30))    # ion plasma frequency
    wce = wci * mass_ratio           # electron cyclotron frequency
    wpe = wpi * np.sqrt(mass_ratio)  # electron plasma frequency
    # Lower hybrid: standard approximation 1/ωlh² = 1/(ωce·ωci) + 1/ωpi²
    # (drops ωci² correction; valid for mi >> me)
    wlh = 1.0 / np.sqrt(1.0 / max(wce * wci, 1e-30) + 1.0 / max(wpi**2, 1e-30))
    wuh = np.sqrt(wce**2 + wpe**2)   # upper hybrid frequency

    return {"wci": wci, "wpi": wpi, "wce": wce, "wpe": wpe, "wlh": wlh, "wuh": wuh}


# ---------------------------------------------------------------------------
# FFT spectral analysis
# ---------------------------------------------------------------------------

def compute_power_spectrum(signal, dt, istart=0, istop=None):
    """Compute the one-sided amplitude spectrum of a real signal.

    No windowing (rectangular window) — fine for identifying dominant modes,
    but expect spectral leakage for broadband signals. Peak-normalized to
    match the C++ postprocessing convention.

    Parameters
    ----------
    signal : 1D array
    dt : float
        Sampling interval.
    istart, istop : int
        Time window indices (applied before FFT).

    Returns
    -------
    freq_rad : 1D array
        Angular frequencies (rad / time unit).
    amplitude : 1D array
        Absolute value of the FFT, normalized by peak (matches C++ convention).
    """
    sig = signal[istart:istop]
    N = len(sig)
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    spectrum = np.fft.rfft(sig)
    amplitude = np.abs(spectrum)

    peak = amplitude.max()
    if peak > 0:
        amplitude /= peak

    freq_rad = np.fft.rfftfreq(N, d=dt) * 2 * np.pi

    return freq_rad, amplitude


def compute_all_spectra(timeseries, dt, istart=0, istop=None):
    """Batch-FFT all field and derived quantities in a timeseries dict.

    Returns
    -------
    spectra : dict
        Keys matching FIELD_NAMES + DERIVED_NAMES (where present),
        values are (freq_rad, amplitude) tuples.
    """
    spectra = {}
    for name in FIELD_NAMES + DERIVED_NAMES:
        if name in timeseries:
            freq, amp = compute_power_spectrum(
                timeseries[name], dt, istart, istop
            )
            spectra[name] = (freq, amp)
    return spectra


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _import_plotting():
    """Import matplotlib and the iPIC3D plot theme."""
    import matplotlib.pyplot as plt
    from plot_theme import add_theme_arg, apply_theme
    return plt, add_theme_arg, apply_theme


def _save_or_show(fig, output):
    """Save figure to file or show interactively."""
    import matplotlib.pyplot as plt
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_timeseries(timeseries, fields=None, output=None, B0x=None):
    """Multi-panel stacked time series plot.

    Parameters
    ----------
    timeseries : dict
        From extract_point_timeseries + compute_derived_quantities.
    fields : list of str or None
        Field names per panel. If None, uses default grouping.
    output : str or None
        Save path. If None, shows interactively.
    B0x : float or None
        If given, x-axis is wci*t instead of time.
    """
    plt, _, _ = _import_plotting()

    if fields is None:
        fields = [
            ["Bx", "By", "Bz"],
            ["Ex", "Ey", "Ez"],
            ["Vx_e", "Vy_e", "Vz_e"],
            ["Vx_i", "Vy_i", "Vz_i"],
            ["rho_e", "rho_i"],
            ["Ep_e", "Ep_i"],  # non-ideal E field (frozen-in violation)
        ]
        # Only include panels whose fields are present
        fields = [
            group for group in fields
            if any(f in timeseries for f in group)
        ]
    elif isinstance(fields[0], str):
        fields = [fields]

    time = timeseries["time"]
    if B0x is not None and B0x > 0:
        time = time * B0x
        xlabel = r"$\omega_{ci} \cdot t$"
    else:
        xlabel = "Time"

    npanels = len(fields)
    fig, axes = plt.subplots(npanels, 1, figsize=(10, 2.5 * npanels),
                             sharex=True)
    if npanels == 1:
        axes = [axes]

    for ax, group in zip(axes, fields):
        for name in group:
            if name in timeseries:
                ax.plot(time, timeseries[name], label=name, linewidth=1.2)
        ax.set_ylabel(", ".join(group))
        ax.legend(loc="upper right", fontsize=8, ncol=min(len(group), 3))

    axes[-1].set_xlabel(xlabel)

    xyz = timeseries.get("sat_xyz")
    if xyz is not None:
        fig.suptitle(
            f"Virtual satellite at ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})",
            fontsize=12,
        )

    fig.tight_layout()
    _save_or_show(fig, output)


def plot_spectra(spectra, ref_freqs=None, fields=None, output=None,
                 log_y=True, sat_xyz=None):
    """Power spectrum plot with optional reference frequency lines.

    Parameters
    ----------
    spectra : dict
        From compute_all_spectra(). Keys -> (freq_rad, amplitude).
    ref_freqs : dict or None
        From compute_reference_frequencies(). Draws vertical lines.
    sat_xyz : array-like or None
        (x, y, z) of the satellite, used in the title.
    fields : list of str or None
        Which spectra to plot. If None, plots Bx, By, Bz, Ex, Ey, Ez.
    output : str or None
    log_y : bool
    """
    plt, _, _ = _import_plotting()

    if fields is None:
        fields = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
    fields = [f for f in fields if f in spectra]

    if not fields:
        print("No matching spectra to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for name in fields:
        freq, amp = spectra[name]
        # Skip DC component (index 0)
        if ref_freqs and ref_freqs.get("wci", 0) > 0:
            freq_norm = freq[1:] / ref_freqs["wci"]
            ax.set_xlabel(r"$\omega / \omega_{ci}$")
        else:
            freq_norm = freq[1:]
            ax.set_xlabel(r"$\omega$ (rad / time unit)")
        ax.plot(freq_norm, amp[1:], label=name, linewidth=1.0)

    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=9)

    # Reference frequency lines — distinct colors and text labels.
    # Only draw lines that fall within the data's Nyquist range;
    # clip x-axis to just past the highest visible reference line.
    if ref_freqs:
        wci = ref_freqs.get("wci", 0)
        ref_styles = [
            (r"$\omega_{ci}$", "wci",  "#e06c75", "-.",  1.2),
            (r"$\omega_{pi}$", "wpi",  "#e5c07b", "--",  1.2),
            (r"$\omega_{lh}$", "wlh",  "#98c379", ":",   1.2),
            (r"$\omega_{pe}$", "wpe",  "#c678dd", "-.",  1.2),
            (r"$\omega_{ce}$", "wce",  "#56b6c2", "--",  1.2),
            (r"$\omega_{uh}$", "wuh",  "#d19a66", ":",   1.2),
        ]
        if wci > 0:
            # Determine max frequency in the data
            max_freq_norm = max(
                (spectra[f][0][-1] / wci) for f in fields if f in spectra
            )
            drawn_xmax = 0
            yhi = 0.95
            for label, key, color, ls, lw in ref_styles:
                if key in ref_freqs and ref_freqs[key] > 0:
                    x = ref_freqs[key] / wci
                    if x > max_freq_norm:
                        continue  # beyond Nyquist — skip
                    ax.axvline(x, color=color, ls=ls, linewidth=lw, alpha=0.7)
                    ax.text(x, yhi, f" {label}", color=color, fontsize=11,
                            transform=ax.get_xaxis_transform(),
                            va="top", ha="left", rotation=90)
                    yhi -= 0.12
                    drawn_xmax = max(drawn_xmax, x)
            # Clip x-axis to ~20% past the highest drawn reference line,
            # but only if at least one line was drawn.  When no lines are
            # visible (all above Nyquist, or no --B0x), show the full range.
            if drawn_xmax > 0:
                ax.set_xlim(0, drawn_xmax * 1.2)
            ax.legend(fontsize=9)

    if sat_xyz is not None:
        ax.set_title(f"Power spectrum at ({sat_xyz[0]:.1f}, {sat_xyz[1]:.1f}, {sat_xyz[2]:.1f})")
    else:
        ax.set_title("Power spectrum")
    fig.tight_layout()
    _save_or_show(fig, output)


def plot_plane_snapshot(plane_data, field_name, cycle_idx=0, output=None):
    """Heatmap for one field from a plane slice at one timestep.

    Parameters
    ----------
    plane_data : dict
        From extract_plane_timeseries().
    field_name : str
    cycle_idx : int
        Index into the time axis.
    output : str or None
    """
    plt, _, _ = _import_plotting()

    field_idx = FIELD_NAMES.index(field_name)
    values = plane_data["data"][cycle_idx, :, field_idx]
    coords = plane_data["coords_2d"]
    plane = plane_data["plane"]

    # Build 2D grid from scattered satellite positions
    axis_labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    xlabel, ylabel = axis_labels[plane]

    ux = np.unique(coords[:, 0])
    uy = np.unique(coords[:, 1])

    if len(ux) * len(uy) == len(coords):
        # Regular grid
        grid = np.full((len(uy), len(ux)), np.nan)
        ix = np.searchsorted(ux, coords[:, 0])
        iy = np.searchsorted(uy, coords[:, 1])
        grid[iy, ix] = values

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(ux, uy, grid, shading="nearest", cmap="RdBu_r")
    else:
        # Irregular: scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.scatter(
            coords[:, 0], coords[:, 1], c=values,
            cmap="RdBu_r", s=10,
        )

    plt.colorbar(im, ax=ax, label=field_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cycle = plane_data["cycles"][cycle_idx]
    normal = {"xy": "z", "xz": "y", "yz": "x"}[plane]
    ax.set_title(
        f"{field_name}  |  {normal}={plane_data['plane_coord']:.1f}"
        f"  |  cycle {cycle:.0f}"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_or_show(fig, output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="iPIC3D virtual satellite analysis "
                    "(time series, FFT, plane snapshots)"
    )
    parser.add_argument("--sat-dir", required=True,
                        help="Directory with VirtualSatelliteTraces*.txt")
    parser.add_argument("--target", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Satellite position (nearest match)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Timestep between cycles (default: 0.1)")
    parser.add_argument("--B0x", type=float, default=None,
                        help="Background Bx for reference frequencies")
    parser.add_argument("--mass-ratio", type=float, default=256.0,
                        help="mi/me mass ratio (default: 256)")
    parser.add_argument("--mode", choices=["timeseries", "fft", "plane"],
                        default="timeseries",
                        help="Analysis mode (default: timeseries)")
    parser.add_argument("--fields", type=str, default=None,
                        help="Comma-separated field names (e.g. Bx,By,Bz)")
    parser.add_argument("--fft-window", nargs=2, type=int, default=None,
                        metavar=("START", "STOP"),
                        help="Cycle index range for FFT window")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz",
                        help="Plane for --mode plane (default: xz)")
    parser.add_argument("--plane-coord", type=float, default=None,
                        help="Normal-axis coordinate for plane slice")
    parser.add_argument("--cycle-idx", type=int, default=-1,
                        help="Time index for plane snapshot (default: last)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path (shows plot if omitted)")
    parser.add_argument("--light", action="store_true",
                        help="Use light theme")

    args = parser.parse_args()

    # Parse fields
    fields = args.fields.split(",") if args.fields else None

    # Load data
    print("Reading satellite data...")
    coords, cycles, data, _ = read_all_satellites(args.sat_dir)
    print(f"  {len(coords)} satellites, {len(cycles)} timesteps")

    # Apply theme (lazy import to avoid matplotlib dependency for non-plot use)
    from plot_theme import apply_theme
    apply_theme(args)

    if args.mode == "plane":
        if args.plane_coord is None:
            parser.error("--plane-coord required for --mode plane")
        if fields is None:
            fields = ["Bz"]

        plane_data = extract_plane_timeseries(
            coords, data, cycles, args.plane, args.plane_coord, dt=args.dt
        )
        print(f"  Plane {args.plane} at {args.plane_coord}: "
              f"{len(plane_data['indices'])} satellites")

        for fname in fields:
            out = args.output
            if out and len(fields) > 1:
                stem = Path(out).stem
                out = str(Path(out).with_stem(f"{stem}_{fname}"))
            plot_plane_snapshot(plane_data, fname, args.cycle_idx, output=out)
        return

    # Point-based modes need --target
    if args.target is None:
        parser.error("--target X Y Z required for timeseries/fft modes")

    ts = extract_point_timeseries(coords, data, cycles, args.target, dt=args.dt)
    ts = compute_derived_quantities(ts, dt=args.dt)
    xyz = ts["sat_xyz"]
    print(f"  Nearest satellite: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"
          f"  distance={ts['distance']:.3f}")

    if args.mode == "timeseries":
        plot_timeseries(ts, fields=fields, output=args.output, B0x=args.B0x)

    elif args.mode == "fft":
        istart = args.fft_window[0] if args.fft_window else 0
        istop = args.fft_window[1] if args.fft_window else None

        spectra = compute_all_spectra(ts, args.dt, istart, istop)

        ref_freqs = None
        if args.B0x is not None:
            ref_freqs = compute_reference_frequencies(
                ts, args.mass_ratio, args.B0x
            )
            print(f"  Reference frequencies:")
            for k, v in ref_freqs.items():
                print(f"    {k} = {v:.6f}")

        plot_spectra(spectra, ref_freqs=ref_freqs, fields=fields,
                     output=args.output, sat_xyz=xyz)


if __name__ == "__main__":
    main()
