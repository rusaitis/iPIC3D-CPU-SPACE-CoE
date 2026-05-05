#!/usr/bin/env python3
"""movie_oblique_alfven.py — animate Bz across the periodic box for the
oblique shear Alfvén wave test.

Produces an MP4 (or GIF) showing the diagonal wavefronts propagating across
the periodic boundary cleanly — the visual signature of correctly-handled
2-axis cross-rank halos.

Usage:
    pixi run python scripts/movie_oblique_alfven.py data_ObliqueAlfvenWave \\
        -o oblique_alfven.mp4

Optionally compares np=1 and np=4 side-by-side to make the equivalence visual:
    pixi run python scripts/movie_oblique_alfven.py \\
        data_ObliqueAlfvenWave data_ObliqueAlfvenWave_np4 \\
        -o oblique_alfven_compare.mp4
"""

import argparse
import os
import sys

import numpy as np
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_dirs', nargs='+',
                    help='One or two iPIC3D output dirs (1 = single panel, 2 = side-by-side)')
    ap.add_argument('--inp', default=None,
                    help='Optional .inp file (defaults to inputfiles/<Case>.inp from first dir)')
    ap.add_argument('--field', default='B', choices=['B', 'E'])
    ap.add_argument('--component', default='z', choices=['x', 'y', 'z'])
    ap.add_argument('--z-slice', type=int, default=None,
                    help='z-cell index for the 2D slice (default: middle)')
    ap.add_argument('--stride', type=int, default=2,
                    help='Use every Nth output cycle (default 2, so ~200 frames)')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('-o', '--output', default='oblique_alfven.mp4',
                    help='Output movie file (.mp4 or .gif)')
    ap.add_argument('--titles', nargs='+', default=None,
                    help='Custom panel titles (one per output_dir)')
    if add_theme_arg:
        add_theme_arg(ap)
    args = ap.parse_args()

    if apply_theme:
        apply_theme(args)

    n_panels = len(args.output_dirs)
    if n_panels > 2:
        sys.exit("Pass at most 2 output dirs (single or side-by-side).")

    # Load all snapshots up front (small grid → small memory)
    panels = []
    for d in args.output_dirs:
        cycles = discover_cycles(d)[::args.stride]
        if len(cycles) < 4:
            sys.exit(f"{d}: only {len(cycles)} snapshots after stride")
        nzn = load_field(d, 0, args.field, args.component).shape[2]
        z = args.z_slice if args.z_slice is not None else nzn // 2
        frames = np.array([load_field(d, c, args.field, args.component)[:, :, z]
                           for c in cycles])
        panels.append({'dir': d, 'cycles': cycles, 'frames': frames, 'z': z})

    inp_path = args.inp
    if inp_path is None:
        # Heuristic: if dir is data_X or data_X_np4 then the .inp is inputfiles/X.inp
        first_dir = os.path.basename(args.output_dirs[0].rstrip('/'))
        case = first_dir.replace('data_', '').rstrip('_np4').rstrip('_np2').rstrip('_np8')
        if case.endswith('_'):
            case = case[:-1]
        candidate = os.path.join(os.path.dirname(args.output_dirs[0]), 'inputfiles', f'{case}.inp')
        if os.path.exists(candidate):
            inp_path = candidate
    sim = parse_input(inp_path) if inp_path and os.path.exists(inp_path) else {}
    Lx = sim.get('Lx', 10.0)
    Ly = sim.get('Ly', 10.0)
    dt = sim.get('dt', 0.25)

    # Common color limits
    vmax = max(abs(p['frames']).max() for p in panels)
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
        t = cyc * dt
        for im, panel in zip(images, panels):
            im.set_data(panel['frames'][i].T)
        suptitle.set_text(f'cycle {cyc}   t = {t:.2f}   '
                          f'(oblique Alfvén, k = (2π/Lx, 2π/Ly), 45° from B0)')
        return images + [suptitle]

    n_frames = min(len(p['cycles']) for p in panels)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // args.fps, blit=False)

    ext = os.path.splitext(args.output)[1].lower()
    if ext == '.gif':
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    print(f"Writing {n_frames} frames to {args.output} at {args.fps} fps...")
    anim.save(args.output, writer=writer)
    print(f"Done: {args.output}  ({os.path.getsize(args.output) / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
