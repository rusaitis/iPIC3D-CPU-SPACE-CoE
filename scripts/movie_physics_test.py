#!/usr/bin/env python3
"""movie_physics_test.py — animated visualization for a physics validation run.

Auto-detects geometry from the iPIC3D output and chooses the appropriate
visualization:

  * 1D-like (`nxc / max(nyc, nzc) >= 4`):
      Top panel  — animated line plot δF(x) at current cycle.
      Bottom    — Hovmöller diagram δF(x, t) growing as the movie plays.
      Used by PlaneEMWave / AlfvenWave / WhistlerPacket.

  * 2D (everything else):
      Single animated pcolor of the chosen field component.
      Used by ObliqueAlfvenWave.

The probe direction is always along x (the wave-propagation axis for the
existing physics tests). The y and z are sliced through the middle.

Usage:
    pixi run python scripts/movie_physics_test.py data_AlfvenWave \\
        --inp inputfiles/AlfvenWave.inp -o alfven.mp4

If `--inp` is omitted the script tries `inputfiles/<Case>.inp` from the
output dir name (heuristic: strips `data_` prefix and `_np{N}` suffix).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import discover_cycles, load_field

try:
    from plot_theme import add_theme_arg, apply_theme
except ImportError:
    add_theme_arg = None
    apply_theme = None


def parse_input(path):
    """Pull (Lx, Ly, dt, nxc, nyc, nzc, Case, …) from an .inp file."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if '=' not in line:
                continue
            lhs, rhs = (s.strip() for s in line.split('=', 1))
            try:
                out[lhs] = float(rhs.split()[0])
            except ValueError:
                out[lhs] = rhs
    return out


def guess_inp(out_dir):
    base = os.path.basename(out_dir.rstrip('/'))
    case = base
    if case.startswith('data_'):
        case = case[len('data_'):]
    for suffix in ('_np1', '_np2', '_np4', '_np8', '_np16'):
        if case.endswith(suffix):
            case = case[: -len(suffix)]
            break
    candidate = os.path.join(os.path.dirname(out_dir.rstrip('/')) or '.',
                             'inputfiles', f'{case}.inp')
    return candidate if os.path.isfile(candidate) else None


def is_1d_like(sim):
    """Heuristic: is this a 1D-along-x configuration?"""
    nxc = sim.get('nxc', 1.0)
    nyc = sim.get('nyc', 1.0)
    nzc = sim.get('nzc', 1.0)
    return nxc / max(nyc, nzc) >= 4.0


def is_reconnection(sim):
    """Detect a Double_Harris (or similar) reconnection setup."""
    case = str(sim.get('Case', '')).strip().strip('"').strip("'")
    return case.startswith('Double_Harris') or case == 'GEM' or case == 'GEMnoPert'


def compute_Jz(Bx, By, dx, dy):
    """Out-of-plane current density (∝ ∂B_y/∂x - ∂B_x/∂y).

    Drops the global prefactor (c/4π) — only direction/sign matter for
    visualization. Inputs are 2D node-centred slices.
    """
    dBy_dx = np.gradient(By, dx, axis=0)
    dBx_dy = np.gradient(Bx, dy, axis=1)
    return dBy_dx - dBx_dy


def compute_Az(Bx, By, dx, dy):
    """In-plane vector potential A_z(x, y).

    For 2D B with B_z = 0, A_z satisfies B_x = ∂A_z/∂y, B_y = -∂A_z/∂x.
    Reconstruct by integrating B_y along x at j=0 to get A_z(x, 0), then
    integrating B_x along y at each x. The field lines (= contours of A_z)
    show magnetic islands as closed loops and X-points as saddles.

    Determined up to an additive global constant; we subtract the mean for
    a clean colorbar centred on zero.
    """
    Az_bottom = -np.cumsum(By[:, 0]) * dx                    # shape (Nx,)
    # Cumulative integration of B_x along y, with row 0 = 0 baseline.
    Az_y_inc = np.zeros_like(Bx)
    Az_y_inc[:, 1:] = np.cumsum(Bx[:, :-1], axis=1) * dy
    Az = Az_bottom[:, None] + Az_y_inc
    Az -= Az.mean()
    return Az


def render_1d(panels, sim, args, theme):
    """Animated 1D line plot (top) + Hovmöller x-t diagram (bottom)."""
    panel = panels[0]
    cycles = panel['cycles']
    frames = panel['frames']                 # shape (n_t, nx)
    Lx = sim.get('Lx', frames.shape[1])
    dt = sim.get('dt', 1.0)
    nx = frames.shape[1]
    x = np.linspace(0, Lx, nx, endpoint=False)
    t = np.array(cycles, dtype=float) * dt

    vmax = float(np.abs(frames).max())
    vmin = -vmax

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6.5), constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 1.4]})

    line_color = '#f38ba8' if theme.mode == 'dark' else '#EE6677'
    line, = ax_top.plot(x, frames[0], color=line_color, lw=1.6)
    ax_top.set_xlim(0, Lx)
    ax_top.set_ylim(1.05 * vmin, 1.05 * vmax)
    ax_top.set_xlabel('x')
    ax_top.set_ylabel(f'δ{args.field}{args.component}')
    ax_top.set_title('Field along propagation axis')
    ax_top.axhline(0, color=theme.zero_line_color, lw=0.6, alpha=0.5)

    # Hovmöller: pre-fill with nan, reveal frame-by-frame.
    hov = np.full_like(frames, np.nan)
    extent = [0, Lx, t[0], t[-1]]
    im = ax_bot.imshow(hov, origin='lower', aspect='auto',
                       extent=extent, vmin=vmin, vmax=vmax,
                       cmap='RdBu_r', interpolation='nearest')
    ax_bot.set_xlabel('x')
    ax_bot.set_ylabel('t')
    ax_bot.set_title('Hovmöller diagram (x, t) — periodic-BC stress')
    fig.colorbar(im, ax=ax_bot, shrink=0.85,
                 label=f'δ{args.field}{args.component}')

    suptitle = fig.suptitle('', fontsize=11)

    def update(i):
        line.set_ydata(frames[i])
        hov[: i + 1] = frames[: i + 1]
        im.set_data(hov)
        suptitle.set_text(
            f"{args.label or os.path.basename(args.output_dirs[0].rstrip('/'))}   "
            f"cycle {cycles[i]}   t = {t[i]:.2f}")
        return [line, im, suptitle]

    return fig, update, len(cycles)


def render_reconnection(panel, sim, args, theme):
    """3-panel stacked view tailored for reconnection — B_y on top, J_z in
    the middle (sharpens at X-lines), A_z + field-line contours on the bottom
    (closed loops = magnetic islands, X-shape = X-points).
    """
    cycles = panel['cycles']
    By_t   = panel['By']        # (n_t, nx, ny)
    Jz_t   = panel['Jz']
    Az_t   = panel['Az']
    Lx = sim.get('Lx', By_t.shape[1])
    Ly = sim.get('Ly', By_t.shape[2])
    dt = sim.get('dt', 1.0)

    vmax_B = float(np.abs(By_t).max())
    vmax_J = float(np.abs(Jz_t).max()) * 0.7   # damp range so X-lines stand out
    vmax_A = float(np.abs(Az_t).max())

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.5),
                              constrained_layout=True, sharex=True)
    ax_By, ax_Jz, ax_Az = axes

    extent = [0, Lx, 0, Ly]
    im_By = ax_By.imshow(By_t[0].T, origin='lower', extent=extent, aspect='equal',
                          vmin=-vmax_B, vmax=vmax_B, cmap='RdBu_r',
                          interpolation='bilinear')
    ax_By.set_ylabel('y')
    ax_By.set_title('B_y  —  reconnecting in-plane field (dipole pairs = islands)')
    fig.colorbar(im_By, ax=ax_By, shrink=0.85, label='B_y')

    im_Jz = ax_Jz.imshow(Jz_t[0].T, origin='lower', extent=extent, aspect='equal',
                          vmin=-vmax_J, vmax=vmax_J, cmap='PRGn',
                          interpolation='bilinear')
    ax_Jz.set_ylabel('y')
    ax_Jz.set_title('J_z  —  out-of-plane current  (sharp ridges = current sheets / X-lines)')
    fig.colorbar(im_Jz, ax=ax_Jz, shrink=0.85, label='J_z')

    # A_z + contour lines (the actual magnetic field lines projected to the plane).
    im_Az = ax_Az.imshow(Az_t[0].T, origin='lower', extent=extent, aspect='equal',
                          vmin=-vmax_A, vmax=vmax_A, cmap='gray',
                          interpolation='bilinear', alpha=0.55)
    nx, ny = By_t.shape[1], By_t.shape[2]
    xs = np.linspace(0, Lx, nx)
    ys = np.linspace(0, Ly, ny)
    contour_color = '#f38ba8' if theme.mode == 'dark' else '#882255'
    contour_levels = np.linspace(-vmax_A, vmax_A, 21)
    cset = ax_Az.contour(xs, ys, Az_t[0].T, levels=contour_levels,
                          colors=contour_color, linewidths=0.7, alpha=0.85)
    ax_Az.set_xlabel('x')
    ax_Az.set_ylabel('y')
    ax_Az.set_title('A_z  —  vector potential  (contours = magnetic field lines)')
    fig.colorbar(im_Az, ax=ax_Az, shrink=0.85, label='A_z')

    suptitle = fig.suptitle('', fontsize=11)
    cset_holder = {'cset': cset}

    def update(i):
        cyc = cycles[i]
        im_By.set_data(By_t[i].T)
        im_Jz.set_data(Jz_t[i].T)
        im_Az.set_data(Az_t[i].T)
        # Contours can't be set_data'd — remove the previous QuadContourSet
        # and re-draw. (matplotlib ≥3.8 removed .collections; use .remove().)
        try:
            cset_holder['cset'].remove()
        except (AttributeError, ValueError):
            for coll in getattr(cset_holder['cset'], 'collections', []):
                coll.remove()
        cset_holder['cset'] = ax_Az.contour(
            xs, ys, Az_t[i].T, levels=contour_levels,
            colors=contour_color, linewidths=0.7, alpha=0.85)
        suptitle.set_text(
            f"{args.label or os.path.basename(args.output_dirs[0].rstrip('/'))}   "
            f"cycle {cyc}   t = {cyc * dt:.2f}")
        return [im_By, im_Jz, im_Az, suptitle]

    return fig, update, len(cycles)


def render_2d(panels, sim, args, _theme):
    """Side-by-side (or single) animated 2D pcolor — like movie_oblique_alfven."""
    n_panels = len(panels)
    Lx = sim.get('Lx', 1.0)
    Ly = sim.get('Ly', 1.0)
    dt = sim.get('dt', 1.0)

    vmax = max(float(np.abs(p['frames']).max()) for p in panels)
    vmin = -vmax

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0),
                              squeeze=False, constrained_layout=True)
    axes = axes[0]

    titles = args.titles or [
        os.path.basename(d.rstrip('/')) for d in args.output_dirs
    ]
    images = []
    for ax, panel, title in zip(axes, panels, titles):
        im = ax.imshow(panel['frames'][0].T, origin='lower',
                       extent=[0, Lx, 0, Ly], aspect='equal',
                       vmin=vmin, vmax=vmax, cmap='RdBu_r',
                       interpolation='bilinear')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        images.append(im)
    fig.colorbar(images[-1], ax=axes.tolist() if n_panels > 1 else axes,
                 shrink=0.85, label=f'{args.field}{args.component}')

    suptitle = fig.suptitle('', fontsize=11)

    def update(i):
        cyc = panels[0]['cycles'][i]
        for im, panel in zip(images, panels):
            im.set_data(panel['frames'][i].T)
        suptitle.set_text(f'cycle {cyc}   t = {cyc * dt:.2f}')
        return list(images) + [suptitle]

    n_frames = min(len(p['cycles']) for p in panels)
    return fig, update, n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dirs', nargs='+',
                    help='One or two iPIC3D output dirs (1 = single, 2 = side-by-side).')
    ap.add_argument('--inp', default=None,
                    help='Path to .inp file (defaults to inputfiles/<Case>.inp heuristic).')
    ap.add_argument('--field', default='B', choices=['B', 'E'])
    ap.add_argument('--component', default='y', choices=['x', 'y', 'z'])
    ap.add_argument('--y-slice', type=int, default=None,
                    help='y-cell index for 1D probe (default: middle).')
    ap.add_argument('--z-slice', type=int, default=None,
                    help='z-cell index for the 2D slice / probe (default: middle).')
    ap.add_argument('--stride', type=int, default=2,
                    help='Use every Nth output cycle (default 2).')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('-o', '--output', default='physics_movie.mp4',
                    help='Output movie file (.mp4 or .gif).')
    ap.add_argument('--titles', nargs='+', default=None,
                    help='Custom panel titles for 2D side-by-side.')
    ap.add_argument('--label', default=None,
                    help='Label used in 1D suptitle (default: dir basename).')
    ap.add_argument('--mode', choices=['auto', '1d', '2d', 'reconnection'], default='auto',
                    help='Force 1D / 2D / reconnection rendering (default: auto-detect).')
    if add_theme_arg:
        add_theme_arg(ap)
    args = ap.parse_args()

    if apply_theme:
        theme = apply_theme(args)
    else:
        class _T:
            mode = 'dark'
            zero_line_color = '#cccccc'
        theme = _T()

    n_panels = len(args.output_dirs)
    if n_panels > 2:
        sys.exit("Pass at most 2 output dirs (single or side-by-side).")

    inp_path = args.inp or guess_inp(args.output_dirs[0])
    sim = parse_input(inp_path) if inp_path and os.path.isfile(inp_path) else {}

    if args.mode == 'auto':
        if is_reconnection(sim):
            mode = 'reconnection'
        elif is_1d_like(sim):
            mode = '1d'
        else:
            mode = '2d'
    else:
        mode = args.mode

    if mode == 'reconnection' and n_panels > 1:
        print("WARNING: reconnection mode supports a single dir — using first only.")
        args.output_dirs = args.output_dirs[:1]
        n_panels = 1

    panels = []
    for d in args.output_dirs:
        cycles = discover_cycles(d)[::args.stride]
        if len(cycles) < 4:
            sys.exit(f"{d}: only {len(cycles)} snapshots after stride {args.stride}")
        sample = load_field(d, cycles[0], args.field, args.component)
        ny = sample.shape[1]
        nz = sample.shape[2]
        j = args.y_slice if args.y_slice is not None else ny // 2
        k = args.z_slice if args.z_slice is not None else nz // 2

        if mode == 'reconnection':
            # Need both B_x and B_y to reconstruct J_z and A_z.
            Lx = sim.get('Lx', 1.0)
            Ly = sim.get('Ly', 1.0)
            By_frames = []
            Jz_frames = []
            Az_frames = []
            for c in cycles:
                Bx_full = load_field(d, c, 'B', 'x')[:-1, :-1, k]
                By_full = load_field(d, c, 'B', 'y')[:-1, :-1, k]
                nx_i, ny_i = Bx_full.shape
                dx = Lx / nx_i
                dy = Ly / ny_i
                By_frames.append(By_full)
                Jz_frames.append(compute_Jz(Bx_full, By_full, dx, dy))
                Az_frames.append(compute_Az(Bx_full, By_full, dx, dy))
            panels.append({
                'dir': d, 'cycles': cycles,
                'By': np.array(By_frames),
                'Jz': np.array(Jz_frames),
                'Az': np.array(Az_frames),
            })
        elif mode == '1d':
            frames = np.array([
                load_field(d, c, args.field, args.component)[:-1, j, k]
                for c in cycles])
            panels.append({'dir': d, 'cycles': cycles, 'frames': frames})
        else:   # 2d
            frames = np.array([
                load_field(d, c, args.field, args.component)[:-1, :-1, k]
                for c in cycles])
            panels.append({'dir': d, 'cycles': cycles, 'frames': frames})

    if mode == '1d' and n_panels > 1:
        # Two-panel 1D not implemented — fall back to first panel only.
        print("WARNING: 1D mode with 2 dirs — only the first dir is rendered.")
        panels = panels[:1]

    if mode == 'reconnection':
        fig, update, n_frames = render_reconnection(panels[0], sim, args, theme)
    elif mode == '1d':
        fig, update, n_frames = render_1d(panels, sim, args, theme)
    else:
        fig, update, n_frames = render_2d(panels, sim, args, theme)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // args.fps, blit=False)

    out_dir_for_movie = os.path.dirname(args.output)
    if out_dir_for_movie:
        os.makedirs(out_dir_for_movie, exist_ok=True)

    ext = os.path.splitext(args.output)[1].lower()
    if ext == '.gif':
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    mode_label = {'1d': '1D + Hovmöller', '2d': '2D pcolor',
                  'reconnection': '3-panel reconnection'}[mode]
    print(f"Writing {n_frames} frames ({mode_label}) "
          f"to {args.output} at {args.fps} fps...")
    anim.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Done: {args.output}  ({os.path.getsize(args.output) / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
