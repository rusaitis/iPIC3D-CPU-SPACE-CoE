#!/usr/bin/env python3
"""
plot_field_comparison.py — Visual comparison of field solver outputs.

Compares electromagnetic field data produced by different solvers (e.g. GMRES vs PETSc)
to verify they produce equivalent results on Double Harris magnetic reconnection runs.

Usage:
    python3 plot_field_comparison.py --gmres <dir> --petsc <dir> [--petsc <dir> ...]
    python3 plot_field_comparison.py                  # auto-discovers from test_petsc_output/

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
                        load_field, find_common_cycles, parse_field_specs)

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

def generate_comparison(cycle, petsc_dir, gmres_data_cache, fields,
                        fixed_vmax, diff_mode, gmres_dir, gmres_label,
                        theme=None, output_dir=None):
    """Generate a comparison figure for one cycle and one test directory.

    Parameters
    ----------
    cycle : int
    petsc_dir : str
    gmres_data_cache : dict  {(ftype, comp): np.ndarray}
    fields : list            [(ftype, comp, latex_label), ...]
    fixed_vmax : dict        Pre-computed bounds; keys like (ftype, comp, "field"|"diff"|"pct_diff")
    diff_mode : str          "percent" or "absolute"
    gmres_dir : str          Reference run directory (for output path)
    gmres_label : str        Reference run label
    theme : Theme or None    Active theme (falls back to plot_theme.active)
    output_dir : str or None Directory for output PNGs (default: parent of gmres_dir)
    """
    if theme is None:
        from plot_theme import active as _active
        theme = _active
    petsc_lab = solver_label(petsc_dir)
    petsc_data = {}
    for ftype, comp, _ in fields:
        try:
            petsc_data[(ftype, comp)] = load_field(petsc_dir, cycle, ftype, comp,
                                                   z_slice=0)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ERROR: Could not load {ftype}{comp} cycle {cycle} "
                  f"from {petsc_lab}: {e}")
            return

    # Shape check
    for ftype, comp, label in fields:
        g_shape = gmres_data_cache[(ftype, comp)].shape
        p_shape = petsc_data[(ftype, comp)].shape
        if g_shape != p_shape:
            print(f"  ERROR: Shape mismatch for {label}: "
                  f"GMRES {g_shape} vs {petsc_lab} {p_shape}")
            sys.exit(1)

    # ── Create figure: N_fields rows × 3 columns ──────────────────────────
    n_rows = len(fields)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing
    diff_label_suffix = "Diff (% of peak)" if diff_mode == "percent" else "Difference"
    fig.suptitle(f"Field Comparison: {gmres_label} vs {petsc_lab}  (cycle {cycle:05d})",
                 fontsize=14, fontweight='bold', y=0.95)

    for row, (ftype, comp, field_label) in enumerate(fields):
        gdata = gmres_data_cache[(ftype, comp)].T   # transpose: rows=Y, cols=X
        pdata = petsc_data[(ftype, comp)].T
        _render_field_row(fig, axes[row], gdata, pdata, field_label,
                          gmres_label, petsc_lab, fixed_vmax, ftype, comp,
                          diff_mode, diff_label_suffix, theme)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save PNG ──────────────────────────────────────────────────────────
    parent_dir = output_dir if output_dir else os.path.dirname(gmres_dir)
    out_png = os.path.join(parent_dir,
                           f"visual_comparison_{petsc_lab}_cycle{cycle:05d}.png")
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ── Console summary ──────────────────────────────────────────────────
    for ftype, comp, field_label in fields:
        gdata = gmres_data_cache[(ftype, comp)]
        pdata = petsc_data[(ftype, comp)]
        diff = pdata - gdata
        max_abs = np.max(np.abs(diff))
        norm_g = np.linalg.norm(gdata)
        l2_rel = np.linalg.norm(diff) / norm_g if norm_g > 0 else 0.0
        extra = ""
        if diff_mode == "percent":
            peak_ref = fixed_vmax.get((ftype, comp, "peak_ref")) \
                or np.max(np.abs(gdata))
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
    parser.add_argument('--gmres', default=None,
                        help='Path to GMRES field output directory (auto-discovered if omitted)')
    parser.add_argument('--petsc', action='append', default=None,
                        help='Path to PETSc field output directory (repeatable; auto-discovered if omitted)')
    parser.add_argument('--ref', default=None,
                        help='Reference run directory (solver-agnostic alias for --gmres)')
    parser.add_argument('--test', action='append', default=None,
                        help='Test run directory (solver-agnostic alias for --petsc; repeatable)')
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
                        help='Directory for output PNGs (default: next to GMRES directory)')
    add_theme_arg(parser)

    args = parser.parse_args(argv)
    theme = apply_theme(args)

    # ── Merge solver-agnostic aliases ────────────────────────────────────
    if args.gmres is None and args.ref is not None:
        args.gmres = args.ref
    elif args.gmres is not None and args.ref is not None:
        parser.error("Cannot specify both --gmres and --ref")

    if args.petsc is None and args.test is not None:
        args.petsc = args.test
    elif args.petsc is not None and args.test is not None:
        args.petsc.extend(args.test)

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

    # ── Find common cycles ───────────────────────────────────────────────
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

    # ── Fields to compare ────────────────────────────────────────────────
    fields = parse_field_specs(args.fields)
    if not fields:
        print("  ERROR: No valid fields specified.")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────
    gmres_label = solver_label(args.gmres)
    gmres_data = {}
    for ftype, comp, _ in fields:
        gmres_data[(ftype, comp)] = load_field(args.gmres, cycle, ftype, comp,
                                               z_slice=0)

    # ── Print-bounds mode ────────────────────────────────────────────────
    if args.print_bounds:
        bound_names = {("B", "x"): "Bx", ("E", "z"): "Ez"}
        for ftype, comp, _ in fields:
            name = bound_names[(ftype, comp)]
            gdata = gmres_data[(ftype, comp)]
            peak_ref = np.max(np.abs(gdata))
            vmax_field = peak_ref
            vmax_diff = 0.0
            vmax_pct = 0.0
            for petsc_dir in args.petsc:
                try:
                    pdata = load_field(petsc_dir, cycle, ftype, comp, z_slice=0)
                except (FileNotFoundError, KeyError) as e:
                    print(f"  WARNING: Could not load {ftype}{comp} from "
                          f"{solver_label(petsc_dir)}: {e}")
                    continue
                vmax_field = max(vmax_field, np.max(np.abs(pdata)))
                abs_diff = np.max(np.abs(pdata - gdata))
                vmax_diff = max(vmax_diff, abs_diff)
                if peak_ref > 0:
                    vmax_pct = max(vmax_pct, abs_diff / peak_ref * 100.0)
            vmax_field = max(vmax_field, 1e-30)
            vmax_diff = max(vmax_diff, 1e-30)
            print(f"vmax_{name}={vmax_field:.6e}")
            print(f"vmax_{name}_diff={vmax_diff:.6e}")
            print(f"vmax_{name}_pct={vmax_pct:.6e}")
        sys.exit(0)

    # ── All-cycles mode: two-pass with consistent colorbars ──────────────
    if args.all_cycles:
        print(f"  All-cycles mode: {len(common_cycles)} cycles to process")

        # Pass 1: scan all cycles to find global peak_ref and max pct_diff per field
        global_peak_ref = {}   # (ftype, comp) → float
        global_vmax_pct = {}   # (ftype, comp) → float
        global_vmax_field = {} # (ftype, comp) → float

        for c in common_cycles:
            gdata_cache = {}
            for ftype, comp, _ in fields:
                gdata_cache[(ftype, comp)] = load_field(args.gmres, c, ftype, comp,
                                                        z_slice=0)
                peak = np.max(np.abs(gdata_cache[(ftype, comp)]))
                global_peak_ref[(ftype, comp)] = max(
                    global_peak_ref.get((ftype, comp), 0.0), peak)
                global_vmax_field[(ftype, comp)] = max(
                    global_vmax_field.get((ftype, comp), 0.0), peak)

            for petsc_dir in args.petsc:
                for ftype, comp, _ in fields:
                    try:
                        pdata = load_field(petsc_dir, c, ftype, comp, z_slice=0)
                    except (FileNotFoundError, KeyError) as e:
                        print(f"  WARNING: Could not load {ftype}{comp} cycle {c} "
                              f"from {solver_label(petsc_dir)}: {e}")
                        continue
                    global_vmax_field[(ftype, comp)] = max(
                        global_vmax_field[(ftype, comp)], np.max(np.abs(pdata)))
                    diff = pdata - gdata_cache[(ftype, comp)]
                    peak_ref = global_peak_ref[(ftype, comp)]
                    if peak_ref > 0:
                        max_pct = np.max(np.abs(diff)) / peak_ref * 100.0
                        global_vmax_pct[(ftype, comp)] = max(
                            global_vmax_pct.get((ftype, comp), 0.0), max_pct)

        # Inject global bounds into fixed_vmax (CLI overrides take precedence)
        for ftype, comp, _ in fields:
            if (ftype, comp, "field") not in fixed_vmax:
                fixed_vmax[(ftype, comp, "field")] = max(
                    global_vmax_field.get((ftype, comp), 1e-30), 1e-30)
            fixed_vmax[(ftype, comp, "peak_ref")] = max(
                global_peak_ref.get((ftype, comp), 1e-30), 1e-30)
            if (ftype, comp, "pct_diff") not in fixed_vmax:
                fixed_vmax[(ftype, comp, "pct_diff")] = max(
                    global_vmax_pct.get((ftype, comp), 1e-30), 1e-30)
            if (ftype, comp, "diff") not in fixed_vmax:
                # For absolute mode within all-cycles, also provide stable bounds
                fixed_vmax[(ftype, comp, "diff")] = max(
                    global_vmax_pct.get((ftype, comp), 0.0)
                    * global_peak_ref.get((ftype, comp), 1e-30) / 100.0, 1e-30)

        print("  Global bounds (pass 1):")
        bound_names = {("B", "x"): "Bx", ("E", "z"): "Ez"}
        for ftype, comp, _ in fields:
            name = bound_names[(ftype, comp)]
            print(f"    {name}: peak_ref={fixed_vmax[(ftype, comp, 'peak_ref')]:.6e}  "
                  f"vmax_pct={fixed_vmax[(ftype, comp, 'pct_diff')]:.6e}%")

        # Pass 2: render each cycle
        for c in common_cycles:
            print(f"  Comparing cycle {c:05d}")
            gdata_cache = {}
            for ftype, comp, _ in fields:
                gdata_cache[(ftype, comp)] = load_field(args.gmres, c, ftype, comp,
                                                        z_slice=0)
            for petsc_dir in args.petsc:
                generate_comparison(c, petsc_dir, gdata_cache, fields,
                                    fixed_vmax, args.diff_mode,
                                    args.gmres, gmres_label, theme=theme,
                                    output_dir=args.output_dir)

    else:
        # ── Single-cycle mode (original behavior) ────────────────────────
        for petsc_dir in args.petsc:
            generate_comparison(cycle, petsc_dir, gmres_data, fields,
                                fixed_vmax, args.diff_mode,
                                args.gmres, gmres_label, theme=theme,
                                output_dir=args.output_dir)


if __name__ == "__main__":
    main()
