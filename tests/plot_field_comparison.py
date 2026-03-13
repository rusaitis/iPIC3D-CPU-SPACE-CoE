#!/usr/bin/env python3
"""
plot_field_comparison.py — Visual comparison of field solver outputs.

Compares electromagnetic field data produced by different solvers to verify
they produce equivalent results on Double Harris magnetic reconnection runs.

Usage:
    python3 plot_field_comparison.py --ref <dir> --test <dir> [--test <dir> ...]
    python3 plot_field_comparison.py                  # auto-discovers from test_output/

HDF5 layout (phdf5 global files):
    <run_dir>/Fields_NNNNN/B_NNNNN.h5  →  /Fields/Bx, /Fields/By, /Fields/Bz  (X, Y, Z)
    <run_dir>/Fields_NNNNN/E_NNNNN.h5  →  /Fields/Ex, /Fields/Ey, /Fields/Ez  (X, Y, Z)
"""

import argparse
import glob
import os
import re
import sys

from plot_utils import (require_imports, discover_cycles, solver_label,
                        load_field, find_common_cycles, parse_field_specs,
                        discover_run_groups)

require_imports("numpy", "matplotlib", "h5py")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py


# ── Utility functions ─────────────────────────────────────────────────────

def compute_pct_diff(test_data, ref_data, peak_ref=None):
    """Difference normalized by peak of reference field, in percent."""
    if peak_ref is None:
        peak_ref = np.max(np.abs(ref_data))
    if peak_ref == 0:
        return np.zeros_like(ref_data), 0.0
    return (test_data - ref_data) / peak_ref * 100.0, peak_ref


# ── Field row renderer ──────────────────────────────────────────────────

def _render_field_row(fig, axes_row, ref_2d, test_2d, field_label,
                      ref_label, test_label, fixed_vmax, ftype, comp,
                      diff_mode, diff_label_suffix, theme):
    """Render one field row (ref, test, difference) in the comparison figure.

    Parameters
    ----------
    fig : Figure           Parent figure (needed for colorbars).
    axes_row : array       Three Axes objects [ref, test, diff].
    ref_2d, test_2d        2D field arrays (already z-sliced and transposed).
    field_label : str      LaTeX label like "$B_x$".
    ref_label, test_label  Human-readable solver names.
    fixed_vmax : dict      Pre-computed bounds.
    ftype, comp : str      Field type and component for bound lookup.
    diff_mode : str        "percent" or "absolute".
    diff_label_suffix : str  Suffix for difference panel title.
    theme : Theme          Active plot theme.
    """
    diff = test_2d - ref_2d

    # Symmetric color scale for field panels
    vmax_field = fixed_vmax.get((ftype, comp, "field")) \
        or max(np.max(np.abs(ref_2d)), np.max(np.abs(test_2d)), 1e-30)

    # Error metrics (always computed in absolute terms)
    max_abs_diff = np.max(np.abs(diff))
    norm_ref = np.linalg.norm(ref_2d)
    l2_rel_err = np.linalg.norm(diff) / norm_ref if norm_ref > 0 else 0.0

    # Difference panel data — percent or absolute
    if diff_mode == "percent":
        peak_ref = fixed_vmax.get((ftype, comp, "peak_ref")) \
            or np.max(np.abs(ref_2d))
        pct_diff, peak_ref = compute_pct_diff(test_2d, ref_2d, peak_ref=peak_ref)
        diff_panel_data = pct_diff
        max_pct_diff = np.max(np.abs(pct_diff))
        vmax_diff = fixed_vmax.get((ftype, comp, "pct_diff")) \
            or max(max_pct_diff, 1e-30)
        diff_title = f"{field_label} {diff_label_suffix}"
        pct_fmt = f"{max_pct_diff:.2f}" if max_pct_diff >= 1 else f"{max_pct_diff:.4f}"
        annotation = (f"|max diff| = {max_abs_diff:.2e}\n"
                      f"max |%diff| = {pct_fmt}%\n"
                      f"L2 rel err = {l2_rel_err:.2e}")
    else:
        diff_panel_data = diff
        vmax_diff = fixed_vmax.get((ftype, comp, "diff")) \
            or max(max_abs_diff, 1e-30)
        diff_title = f"{field_label} Difference"
        annotation = (f"|max diff| = {max_abs_diff:.2e}\n"
                      f"L2 rel err = {l2_rel_err:.2e}")

    panels = [
        (axes_row[0], ref_2d, f"{field_label} ({ref_label})",
         'berlin', -vmax_field, vmax_field),
        (axes_row[1], test_2d, f"{field_label} ({test_label})",
         'berlin', -vmax_field, vmax_field),
        (axes_row[2], diff_panel_data, diff_title,
         'berlin', -vmax_diff, vmax_diff),
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
    ax_diff = axes_row[2]
    ax_diff.text(0.03, 0.97, annotation,
                 transform=ax_diff.transAxes,
                 fontsize=9, fontfamily='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=theme.annot_face,
                           alpha=theme.annot_alpha,
                           edgecolor=theme.annot_edge))


# ── Comparison figure generator ───────────────────────────────────────────

def generate_comparison(cycle, test_dir, ref_data_cache, fields,
                        fixed_vmax, diff_mode, ref_dir, ref_label,
                        theme=None, output_dir=None):
    """Generate a comparison figure for one cycle and one test directory.

    Parameters
    ----------
    cycle : int
    test_dir : str           Test run directory.
    ref_data_cache : dict    {(ftype, comp): np.ndarray} for the reference solver.
    fields : list            [(ftype, comp, latex_label), ...]
    fixed_vmax : dict        Pre-computed bounds; keys like (ftype, comp, "field"|"diff"|"pct_diff")
    diff_mode : str          "percent" or "absolute"
    ref_dir : str            Reference run directory (for output path).
    ref_label : str          Reference run label.
    theme : Theme or None    Active theme (falls back to plot_theme.active)
    output_dir : str or None Directory for output PNGs (default: parent of ref_dir)
    """
    if theme is None:
        from plot_theme import active as _active
        theme = _active
    test_lab = solver_label(test_dir)
    test_data = {}
    for ftype, comp, _ in fields:
        try:
            test_data[(ftype, comp)] = load_field(test_dir, cycle, ftype, comp,
                                                  z_slice=0)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ERROR: Could not load {ftype}{comp} cycle {cycle} "
                  f"from {test_lab}: {e}")
            return

    # Shape check
    for ftype, comp, label in fields:
        r_shape = ref_data_cache[(ftype, comp)].shape
        t_shape = test_data[(ftype, comp)].shape
        if r_shape != t_shape:
            print(f"  ERROR: Shape mismatch for {label}: "
                  f"{ref_label} {r_shape} vs {test_lab} {t_shape}")
            sys.exit(1)

    # ── Create figure: N_fields rows × 3 columns ──────────────────────────
    n_rows = len(fields)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing
    diff_label_suffix = "Diff (% of peak)" if diff_mode == "percent" else "Difference"
    fig.suptitle(f"Field Comparison: {ref_label} vs {test_lab}  (cycle {cycle:05d})",
                 fontsize=14, fontweight='bold', y=0.95)

    for row, (ftype, comp, field_label) in enumerate(fields):
        rdata = ref_data_cache[(ftype, comp)].T   # transpose: rows=Y, cols=X
        tdata = test_data[(ftype, comp)].T
        _render_field_row(fig, axes[row], rdata, tdata, field_label,
                          ref_label, test_lab, fixed_vmax, ftype, comp,
                          diff_mode, diff_label_suffix, theme)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save PNG ──────────────────────────────────────────────────────────
    parent_dir = output_dir if output_dir else os.path.dirname(ref_dir)
    out_png = os.path.join(parent_dir,
                           f"visual_comparison_{test_lab}_cycle{cycle:05d}.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ── Console summary ──────────────────────────────────────────────────
    for ftype, comp, field_label in fields:
        rdata = ref_data_cache[(ftype, comp)]
        tdata = test_data[(ftype, comp)]
        diff = tdata - rdata
        max_abs = np.max(np.abs(diff))
        norm_r = np.linalg.norm(rdata)
        l2_rel = np.linalg.norm(diff) / norm_r if norm_r > 0 else 0.0
        extra = ""
        if diff_mode == "percent":
            peak_ref = fixed_vmax.get((ftype, comp, "peak_ref")) \
                or np.max(np.abs(rdata))
            if peak_ref > 0:
                extra = f"   max |%diff| = {max_abs / peak_ref * 100:.4f}%"
        print(f"    {field_label:5s}  |max diff| = {max_abs:.2e}   "
              f"L2 rel err = {l2_rel:.2e}{extra}")


# ── Main entry point ──────────────────────────────────────────────────────

def main(argv=None):
    from plot_theme import add_theme_arg, apply_theme

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ref', default=None,
                        help='Reference run directory (auto-discovered if omitted)')
    parser.add_argument('--test', action='append', default=None,
                        help='Test run directory (repeatable; auto-discovered if omitted)')
    # Legacy aliases (hidden, kept for backwards compatibility)
    parser.add_argument('--gmres', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--petsc', action='append', default=None,
                        help=argparse.SUPPRESS)
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
    parser.add_argument('--vmax-Bx-pct', type=float, default=None,
                        help='Fixed %% scale for Bx difference panel (percent mode)')
    parser.add_argument('--vmax-Ez-pct', type=float, default=None,
                        help='Fixed %% scale for Ez difference panel (percent mode)')
    parser.add_argument('--fields', default='Bx,Ez',
                        help='Field components to compare (default: Bx,Ez)')
    parser.add_argument('--diff-mode', choices=['percent', 'absolute'], default='percent',
                        help='Difference panel normalization (default: percent)')
    parser.add_argument('--all-cycles', action='store_true',
                        help='Generate one PNG per common cycle with consistent colorbar scale')
    parser.add_argument('--print-bounds', action='store_true',
                        help='Print vmax bounds for all fields and exit (no plot)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for output PNGs (default: next to reference directory)')
    parser.add_argument('--dir', default=None,
                        help='Output directory to auto-discover run dirs from '
                             '(default: tests/test_output)')
    add_theme_arg(parser)

    args = parser.parse_args(argv)
    theme = apply_theme(args)

    # ── Merge legacy aliases (--gmres → --ref, --petsc → --test) ────────
    if args.ref is None and args.gmres is not None:
        import warnings
        warnings.warn("--gmres is deprecated, use --ref instead", FutureWarning)
        args.ref = args.gmres
    elif args.ref is not None and args.gmres is not None:
        parser.error("Cannot specify both --ref and --gmres")

    if args.test is None and args.petsc is not None:
        import warnings
        warnings.warn("--petsc is deprecated, use --test instead", FutureWarning)
        args.test = args.petsc
    elif args.test is not None and args.petsc is not None:
        args.test.extend(args.petsc)

    # ── Fixed bounds lookup ──────────────────────────────────────────────
    fixed_vmax = {}
    if args.vmax_Bx is not None:
        fixed_vmax[("B", "x", "field")] = args.vmax_Bx
    if args.vmax_Bx_diff is not None:
        fixed_vmax[("B", "x", "diff")] = args.vmax_Bx_diff
    if args.vmax_Ez is not None:
        fixed_vmax[("E", "z", "field")] = args.vmax_Ez
    if args.vmax_Ez_diff is not None:
        fixed_vmax[("E", "z", "diff")] = args.vmax_Ez_diff
    if args.vmax_Bx_pct is not None:
        fixed_vmax[("B", "x", "pct_diff")] = args.vmax_Bx_pct
    if args.vmax_Ez_pct is not None:
        fixed_vmax[("E", "z", "pct_diff")] = args.vmax_Ez_pct

    # ── Auto-discovery ────────────────────────────────────────────────────
    output_dir = args.dir or os.path.join(script_dir, "test_output")

    if args.ref is not None or args.test is not None:
        # Explicit --ref/--test: single group, original behavior
        if args.ref is None or args.test is None:
            parser.error("--ref and --test must both be specified, "
                         "or both omitted for auto-discovery.")
        run_groups = [('explicit', args.ref, args.test)]
    else:
        run_groups = discover_run_groups(output_dir, require_fields=True)
        if not run_groups:
            print("  ERROR: No run directories with field output found in "
                  f"{output_dir}. Use --ref/--test to specify manually.")
            sys.exit(1)
        for tag, ref_dir, test_dirs in run_groups:
            names = [solver_label(d) for d in test_dirs]
            print(f"  Grid {tag}: ref={solver_label(ref_dir)}, "
                  f"test={', '.join(names)}")

    # ── Fields to compare ────────────────────────────────────────────────
    fields = parse_field_specs(args.fields)
    if not fields:
        print("  ERROR: No valid fields specified.")
        sys.exit(1)

    # ── Process each grid group ──────────────────────────────────────────
    for grid_tag, ref_dir, test_dirs in run_groups:
        print(f"\n  ── {grid_tag} ──")

        all_dirs = [ref_dir] + test_dirs
        all_cycle_sets = [set(discover_cycles(d)) for d in all_dirs]
        common_cycles = sorted(set.intersection(*all_cycle_sets))

        if not common_cycles:
            print(f"  WARNING: No common output cycles for {grid_tag}, skipping.")
            continue

        if args.cycle == -1:
            cycle = common_cycles[-1]
        elif args.cycle in common_cycles:
            cycle = args.cycle
        else:
            print(f"  WARNING: Requested cycle {args.cycle} not found for "
                  f"{grid_tag}. Available: {common_cycles}. Skipping.")
            continue

        ref_label = solver_label(ref_dir)

        # ── Print-bounds mode ────────────────────────────────────────────
        if args.print_bounds:
            print(f"  Comparing cycle {cycle:05d}")
            ref_data = {}
            for ftype, comp, _ in fields:
                ref_data[(ftype, comp)] = load_field(ref_dir, cycle, ftype,
                                                     comp, z_slice=0)
            bound_names = {("B", "x"): "Bx", ("E", "z"): "Ez"}
            for ftype, comp, _ in fields:
                name = bound_names.get((ftype, comp), f"{ftype}{comp}")
                rdata = ref_data[(ftype, comp)]
                peak_ref = np.max(np.abs(rdata))
                vmax_field = peak_ref
                vmax_diff = 0.0
                vmax_pct = 0.0
                for test_dir in test_dirs:
                    try:
                        tdata = load_field(test_dir, cycle, ftype, comp,
                                           z_slice=0)
                    except (FileNotFoundError, KeyError) as e:
                        print(f"  WARNING: Could not load {ftype}{comp} from "
                              f"{solver_label(test_dir)}: {e}")
                        continue
                    vmax_field = max(vmax_field, np.max(np.abs(tdata)))
                    abs_diff = np.max(np.abs(tdata - rdata))
                    vmax_diff = max(vmax_diff, abs_diff)
                    if peak_ref > 0:
                        vmax_pct = max(vmax_pct, abs_diff / peak_ref * 100.0)
                vmax_field = max(vmax_field, 1e-30)
                vmax_diff = max(vmax_diff, 1e-30)
                print(f"vmax_{name}={vmax_field:.6e}")
                print(f"vmax_{name}_diff={vmax_diff:.6e}")
                print(f"vmax_{name}_pct={vmax_pct:.6e}")
            continue

        # ── All-cycles mode: two-pass with consistent colorbars ──────────
        grid_fixed = dict(fixed_vmax)  # copy so per-grid bounds don't leak

        if args.all_cycles:
            print(f"  All-cycles mode: {len(common_cycles)} cycles to process")

            global_peak_ref = {}
            global_vmax_pct = {}
            global_vmax_field = {}

            for c in common_cycles:
                ref_cache = {}
                for ftype, comp, _ in fields:
                    ref_cache[(ftype, comp)] = load_field(
                        ref_dir, c, ftype, comp, z_slice=0)
                    peak = np.max(np.abs(ref_cache[(ftype, comp)]))
                    global_peak_ref[(ftype, comp)] = max(
                        global_peak_ref.get((ftype, comp), 0.0), peak)
                    global_vmax_field[(ftype, comp)] = max(
                        global_vmax_field.get((ftype, comp), 0.0), peak)

                for test_dir in test_dirs:
                    for ftype, comp, _ in fields:
                        try:
                            tdata = load_field(test_dir, c, ftype, comp,
                                               z_slice=0)
                        except (FileNotFoundError, KeyError) as e:
                            print(f"  WARNING: Could not load {ftype}{comp} "
                                  f"cycle {c} from "
                                  f"{solver_label(test_dir)}: {e}")
                            continue
                        global_vmax_field[(ftype, comp)] = max(
                            global_vmax_field[(ftype, comp)],
                            np.max(np.abs(tdata)))
                        diff = tdata - ref_cache[(ftype, comp)]
                        peak_ref = global_peak_ref[(ftype, comp)]
                        if peak_ref > 0:
                            max_pct = (np.max(np.abs(diff))
                                       / peak_ref * 100.0)
                            global_vmax_pct[(ftype, comp)] = max(
                                global_vmax_pct.get((ftype, comp), 0.0),
                                max_pct)

            for ftype, comp, _ in fields:
                if (ftype, comp, "field") not in grid_fixed:
                    grid_fixed[(ftype, comp, "field")] = max(
                        global_vmax_field.get((ftype, comp), 1e-30), 1e-30)
                grid_fixed[(ftype, comp, "peak_ref")] = max(
                    global_peak_ref.get((ftype, comp), 1e-30), 1e-30)
                if (ftype, comp, "pct_diff") not in grid_fixed:
                    grid_fixed[(ftype, comp, "pct_diff")] = max(
                        global_vmax_pct.get((ftype, comp), 1e-30), 1e-30)
                if (ftype, comp, "diff") not in grid_fixed:
                    grid_fixed[(ftype, comp, "diff")] = max(
                        global_vmax_pct.get((ftype, comp), 0.0)
                        * global_peak_ref.get((ftype, comp), 1e-30)
                        / 100.0, 1e-30)

            print("  Global bounds (pass 1):")
            for ftype, comp, _ in fields:
                name = f"{ftype}{comp}"
                print(f"    {name}: peak_ref="
                      f"{grid_fixed[(ftype, comp, 'peak_ref')]:.6e}  "
                      f"vmax_pct="
                      f"{grid_fixed[(ftype, comp, 'pct_diff')]:.6e}%")

            for c in common_cycles:
                print(f"  Comparing cycle {c:05d}")
                ref_cache = {}
                for ftype, comp, _ in fields:
                    ref_cache[(ftype, comp)] = load_field(
                        ref_dir, c, ftype, comp, z_slice=0)
                for test_dir in test_dirs:
                    generate_comparison(c, test_dir, ref_cache, fields,
                                        grid_fixed, args.diff_mode,
                                        ref_dir, ref_label, theme=theme,
                                        output_dir=args.output_dir)

        else:
            print(f"  Comparing cycle {cycle:05d}")
            ref_data = {}
            for ftype, comp, _ in fields:
                ref_data[(ftype, comp)] = load_field(
                    ref_dir, cycle, ftype, comp, z_slice=0)
            for test_dir in test_dirs:
                generate_comparison(cycle, test_dir, ref_data, fields,
                                    grid_fixed, args.diff_mode,
                                    ref_dir, ref_label, theme=theme,
                                    output_dir=args.output_dir)


if __name__ == "__main__":
    main()
