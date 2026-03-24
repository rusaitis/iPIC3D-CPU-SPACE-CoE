#!/usr/bin/env python3
"""
plot_l2_timeseries.py — L2 error accumulation analysis over simulation cycles.

Computes L2 relative error between a reference run and one or more test runs
at each output cycle, fits power-law growth models, and generates a diagnostic
plot with local exponent analysis.

Usage:
    python3 scripts/plot_l2_timeseries.py --ref DIR --test DIR [--test DIR ...]
    python3 scripts/plot_l2_timeseries.py                          # auto-discover
    python3 scripts/plot_l2_timeseries.py --csv --gmrestol 1e-15   # with CSV output

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
                        load_field, parse_field_specs, PowerLawFit, L2ErrorSeries,
                        discover_run_groups)

require_imports("numpy", "matplotlib", "h5py")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py


# ── Fitting and analysis ─────────────────────────────────────────────────

def _r_squared(y_obs, y_pred):
    """Coefficient of determination in log space."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _filter_positive(cycles, l2_vals):
    """Keep only entries where both cycle > 0 and L2 > 0; return float arrays."""
    mask = np.array([v > 0 and c > 0 for c, v in zip(cycles, l2_vals)])
    return (np.array(cycles, dtype=float)[mask],
            np.array(l2_vals, dtype=float)[mask])


def fit_powerlaw(cycles, l2_vals):
    """Fit L2 = A * t^alpha in log-log space, with optional piecewise split.

    Performs linear regression on log(L2) vs log(t) to determine the
    power-law exponent alpha.  If a piecewise fit (two segments with a
    knee) improves R^2 by >= 0.05, the knee location and per-segment
    exponents are included.

    Returns PowerLawFit or None if insufficient data.
    """
    if len(cycles) < 3:
        return None
    cycles_pos, l2_pos = _filter_positive(cycles, l2_vals)
    if len(cycles_pos) < 3:
        return None

    log_c = np.log(cycles_pos)
    log_v = np.log(l2_pos)

    # Global power-law: log(L2) = log(A) + alpha * log(t)
    alpha, log_A = np.polyfit(log_c, log_v, 1)
    A = np.exp(log_A)
    r2 = _r_squared(log_v, log_A + alpha * log_c)

    result = PowerLawFit(alpha=alpha, r_squared=r2,
                         amplitude=A, log_amplitude=log_A)

    # Try piecewise power-law: split at each candidate index
    n_pts = len(cycles_pos)
    if n_pts < 6:
        return result

    best_ssr = np.inf
    best_k = None
    best_left = best_right = None

    for k in range(2, n_pts - 2):
        # Left segment
        alpha_l, logA_l = np.polyfit(log_c[:k], log_v[:k], 1)
        pred_l = logA_l + alpha_l * log_c[:k]
        # Right segment
        alpha_r, logA_r = np.polyfit(log_c[k:], log_v[k:], 1)
        pred_r = logA_r + alpha_r * log_c[k:]

        ssr = np.sum((log_v[:k] - pred_l) ** 2) + np.sum((log_v[k:] - pred_r) ** 2)
        if ssr < best_ssr:
            best_ssr = ssr
            best_k = k
            best_left = (alpha_l, logA_l)
            best_right = (alpha_r, logA_r)

    if best_k is not None and best_left is not None:
        pred_pw = np.empty_like(log_v)
        pred_pw[:best_k] = best_left[1] + best_left[0] * log_c[:best_k]
        pred_pw[best_k:] = best_right[1] + best_right[0] * log_c[best_k:]
        r2_pw = _r_squared(log_v, pred_pw)

        if r2_pw - r2 >= 0.05:
            result = PowerLawFit(
                alpha=alpha, r_squared=r2,
                amplitude=A, log_amplitude=log_A,
                knee_cycle=float(cycles_pos[best_k]),
                alpha_early=best_left[0],
                log_amplitude_early=best_left[1],
                alpha_late=best_right[0],
                log_amplitude_late=best_right[1],
                r_squared_piecewise=r2_pw,
            )

    return result


def compute_local_exponent(cycles, l2_vals):
    """Finite-difference d(log L2) / d(log t) between consecutive cycles.

    This gives the instantaneous power-law exponent at each midpoint,
    useful for detecting regime transitions (e.g. diffusive -> linear)
    that a single global fit would miss.

    Returns (midpoint_cycles, local_alpha) arrays, or (None, None) if
    insufficient data.
    """
    cycles_pos, l2_pos = _filter_positive(cycles, l2_vals)
    if len(cycles_pos) < 2:
        return None, None

    log_c = np.log(cycles_pos)
    log_v = np.log(l2_pos)

    d_log_c = np.diff(log_c)
    d_log_v = np.diff(log_v)

    # Avoid division by zero at identical cycle values
    good = d_log_c != 0
    mids = 0.5 * (cycles_pos[:-1] + cycles_pos[1:])
    local_alpha = np.where(good, d_log_v / d_log_c, 0.0)

    return mids[good], local_alpha[good]


def _format_l2_with_pct(val):
    """Format L2 value with parenthetical percentage when > 1e-6."""
    s = f"{val:.2e}"
    if val > 1e-6:
        pct = val * 100
        if pct >= 0.01:
            s += f" ({pct:.3f}%)"
        else:
            s += f" ({pct:.1e}%)"
    return s


def _annotate_fit(ax, text, xy, color):
    """Place a fit-parameter annotation with consistent style."""
    from plot_theme import active as _theme
    face = _theme.annot_face if _theme else 'white'
    ax.annotate(text, xy=xy, fontsize=8, color=color,
                xytext=(8, 12), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=face,
                          alpha=0.7, edgecolor=color, linewidth=0.5))


def _amplification(l2_relative):
    """Total amplification factor = L2_last / L2_first.

    Measures how much the L2 error grew from the first nonzero value to
    the last cycle.  A value of 1.0 means no growth; >>1 indicates
    accumulation (expected in implicit PIC over many cycles).
    """
    nz = [v for v in l2_relative if v > 0]
    if not nz or nz[0] <= 0:
        return 0.0
    return l2_relative[-1] / nz[0]


# ── Plot helpers ─────────────────────────────────────────────────────────

def _plot_fit_overlay(ax, cycles_arr, l2_arr, fit, style, run_index=0):
    """Draw power-law fit curves and annotations on the L2 error panel."""
    if fit is None:
        return

    from plot_theme import active as _theme
    annot_face = _theme.annot_face if _theme else 'white'

    c_fit = np.array(cycles_arr, dtype=float)
    knee = fit.knee_cycle

    if knee is not None:
        # Piecewise power-law: two segments
        left = c_fit[c_fit <= knee]
        right = c_fit[c_fit >= knee]
        if len(left) > 0:
            y_left = np.exp(fit.log_amplitude_early +
                            fit.alpha_early * np.log(left))
            ax.semilogy(left, y_left,
                        color=style['color'], linestyle='--',
                        linewidth=1, alpha=0.5)
        if len(right) > 0:
            y_right = np.exp(fit.log_amplitude_late +
                             fit.alpha_late * np.log(right))
            ax.semilogy(right, y_right,
                        color=style['color'], linestyle=':',
                        linewidth=1, alpha=0.5)
        # Vertical knee marker
        ax.axvline(x=knee, color=style['color'],
                   linestyle='-', linewidth=0.5, alpha=0.3)
        # Transient fill (label appears in legend)
        transient_cycles = int(knee - c_fit[0])
        ax.axvspan(c_fit[0], knee,
                   color=style['color'], alpha=0.05, zorder=0,
                   label=f'transient ({transient_cycles} cycles)')
        # Annotation with early/late exponents (axes coords, stacked per run)
        ann_text = (f"\u03b1={fit.alpha_early:.1f}/{fit.alpha_late:.1f}"
                    f" (R\u00b2={fit.r_squared_piecewise:.2f})")
        ann_y = 0.92 - 0.10 * run_index
        ax.annotate(ann_text,
                    xy=(0.03, ann_y), xycoords='axes fraction',
                    fontsize=8, color=style['color'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=annot_face,
                              alpha=0.7, edgecolor=style['color'],
                              linewidth=0.5))
    else:
        # Single power-law fit curve
        fit_y = fit.amplitude * c_fit ** fit.alpha
        ax.semilogy(c_fit, fit_y,
                    color=style['color'], linestyle='--',
                    linewidth=1, alpha=0.5)
        # Annotation (axes coords, stacked per run)
        ann_text = f"\u03b1={fit.alpha:.2f} (R\u00b2={fit.r_squared:.2f})"
        ann_y = 0.92 - 0.10 * run_index
        ax.annotate(ann_text,
                    xy=(0.03, ann_y), xycoords='axes fraction',
                    fontsize=8, color=style['color'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=annot_face,
                              alpha=0.7, edgecolor=style['color'],
                              linewidth=0.5))

    # Final-value annotation with percentage (staggered per run)
    last_val = l2_arr[-1]
    if last_val > 0:
        lbl = _format_l2_with_pct(last_val)
        ax.annotate(
            lbl, xy=(cycles_arr[-1], last_val),
            fontsize=7, color=style['color'],
            xytext=(5, -10 - 14 * run_index), textcoords='offset points',
            ha='left', va='top')


def _plot_local_exponent(ax, cycles_arr, l2_arr, style, theme):
    """Draw local power-law exponent in the bottom panel."""
    mids, loc_alpha = compute_local_exponent(cycles_arr, l2_arr)
    if mids is not None and len(mids) > 0:
        ax.plot(mids, loc_alpha,
                color=style['color'], marker=style['marker'],
                linestyle=style['ls'], linewidth=1.2, markersize=3,
                markeredgecolor=theme.marker_edge, markeredgewidth=0.3,
                alpha=0.85)


# ── Main entry point ─────────────────────────────────────────────────────

def main(argv=None):
    from plot_theme import add_theme_arg, apply_theme, get_solver_style

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ref', '--gmres', default=None,
                        help='Reference run directory (auto-discovered if omitted)')
    parser.add_argument('--test', '--petsc', action='append', default=None,
                        help='Test run directory (repeatable; auto-discovered if omitted)')
    parser.add_argument('--fields', default='Bx,Ez',
                        help='Field components to compare (default: Bx,Ez)')
    parser.add_argument('--gmrestol', type=float, default=1e-15,
                        help='Solver tolerance for reference line (default: 1e-15)')
    parser.add_argument('--csv', action='store_true',
                        help='Write results_l2_error.csv alongside PNG')
    parser.add_argument('--output', default=None,
                        help='Override output PNG path')
    parser.add_argument('--dir', default=None,
                        help='Output directory to auto-discover run dirs from '
                             '(default: tests/test_output)')
    add_theme_arg(parser)

    args = parser.parse_args(argv)
    theme = apply_theme(args)

    # ── Parse field specifications ────────────────────────────────────────
    fields = parse_field_specs(args.fields)
    if not fields:
        print("  ERROR: No valid fields specified.")
        sys.exit(1)

    # ── Auto-discovery ────────────────────────────────────────────────────
    output_dir = args.dir or os.path.join(script_dir, "test_output")

    if args.ref is not None or args.test is not None:
        if args.ref is None or args.test is None:
            parser.error("--ref and --test must both be specified, "
                         "or both omitted for auto-discovery.")
        run_groups = [('explicit', args.ref, args.test)]
    else:
        run_groups = discover_run_groups(output_dir, require_fields=True)
        if not run_groups:
            print(f"  ERROR: No run directories with field output found in "
                  f"{output_dir}. Use --ref/--test to specify manually.")
            sys.exit(1)
        for tag, ref_dir, test_dirs in run_groups:
            names = [solver_label(d) for d in test_dirs]
            print(f"  Grid {tag}: ref={solver_label(ref_dir)}, "
                  f"test={', '.join(names)}")

    def get_style(dir_path):
        """Get plot style for a directory via the shared theme module."""
        return get_solver_style(dir_path)

    # ── Process each grid group ──────────────────────────────────────────
    for grid_tag, ref_dir, test_dirs in run_groups:
        print(f"\n  ── {grid_tag} ──")

        ref_cycles = set(discover_cycles(ref_dir))
        all_test_cycles = [set(discover_cycles(t)) for t in test_dirs]
        common_cycles = sorted(ref_cycles.intersection(*all_test_cycles))

        if not common_cycles:
            print(f"  WARNING: No common output cycles for {grid_tag}, "
                  "skipping.")
            continue

        print(f"  Found {len(common_cycles)} common cycles: "
              f"{common_cycles[0]} .. {common_cycles[-1]}")

        ref_label = solver_label(ref_dir)

        # ── Compute N_dof and round-off floor ────────────────────────────
        sample_ftype, sample_comp = fields[0][0], fields[0][1]
        try:
            cycle_str = f"{common_cycles[0]:05d}"
            h5_path = os.path.join(ref_dir, f"Fields_{cycle_str}",
                                   f"{sample_ftype}_{cycle_str}.h5")
            with h5py.File(h5_path, "r") as f:
                n_dof = int(np.prod(
                    f[f"/Fields/{sample_ftype}{sample_comp}"].shape))
        except Exception:
            n_dof = 10000  # fallback

        eps_machine = np.finfo(np.float64).eps
        roundoff_floor = np.sqrt(n_dof) * eps_machine

        # ── Compute L2 errors (one cycle at a time to limit memory) ──────
        results = {t: {} for t in test_dirs}

        for ftype, comp, field_latex in fields:
            for test_dir in test_dirs:
                results[test_dir][(ftype, comp)] = L2ErrorSeries()

            for cycle in common_cycles:
                try:
                    ref_data = load_field(ref_dir, cycle, ftype, comp)
                except (OSError, KeyError) as e:
                    print(f"  WARNING: Could not load ref {ftype}{comp} "
                          f"cycle {cycle}: {e}")
                    continue

                ref_norm = np.linalg.norm(ref_data)

                for test_dir in test_dirs:
                    try:
                        test_data = load_field(test_dir, cycle, ftype, comp)
                    except (OSError, KeyError) as e:
                        print(f"  WARNING: Could not load {ftype}{comp} "
                              f"cycle {cycle} from "
                              f"{solver_label(test_dir)}: {e}")
                        continue

                    if ref_data.shape != test_data.shape:
                        print(f"  WARNING: Shape mismatch at cycle {cycle} "
                              f"for {field_latex}: {ref_data.shape} vs "
                              f"{test_data.shape}, skipping.")
                        continue

                    diff = test_data - ref_data
                    l2_rel = (np.linalg.norm(diff) / ref_norm
                              if ref_norm > 0 else 0.0)
                    max_abs = float(np.max(np.abs(diff)))

                    series = results[test_dir][(ftype, comp)]
                    series.cycles.append(cycle)
                    series.l2_relative.append(l2_rel)
                    series.max_absolute.append(max_abs)
                    series.ref_norm.append(ref_norm)

        for test_dir in test_dirs:
            print(f"  Processed: {solver_label(test_dir)}")

        # ── Plot ─────────────────────────────────────────────────────────
        n_fields = len(fields)
        fig = plt.figure(figsize=(7 * n_fields, 7), dpi=150,
                         layout='constrained')
        gs = fig.add_gridspec(2, n_fields, height_ratios=[3, 1], hspace=0.08)

        fig.suptitle(f"L2 Error Accumulation  (ref: {ref_label})",
                     fontsize=13, fontweight='bold', y=1.02)

        fit_cache = {}

        for fi, (ftype, comp, field_latex) in enumerate(fields):
            ax_top = fig.add_subplot(gs[0, fi])
            ax_bot = fig.add_subplot(gs[1, fi], sharex=ax_top)
            any_nonzero = False

            for ri, test_dir in enumerate(test_dirs):
                test_lab = solver_label(test_dir)
                data = results[test_dir][(ftype, comp)]
                cycles_arr = data.cycles
                l2_arr = data.l2_relative

                if not cycles_arr:
                    continue

                style = get_style(test_dir)

                all_zero = all(v == 0 for v in l2_arr)
                if all_zero:
                    ax_top.plot([], [], color=style['color'],
                                linestyle=style['ls'],
                                marker=style['marker'],
                                label=f"{test_lab} (bit-identical)")
                    continue

                any_nonzero = True

                ax_top.semilogy(
                    cycles_arr, l2_arr,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['ls'], linewidth=1.8, markersize=5,
                    markeredgecolor=theme.marker_edge, markeredgewidth=0.5,
                    label=test_lab, alpha=0.9, zorder=3)

                fit = fit_powerlaw(cycles_arr, l2_arr)
                fit_cache[(test_dir, ftype, comp)] = fit
                _plot_fit_overlay(ax_top, cycles_arr, l2_arr, fit, style,
                                  run_index=ri)

                _plot_local_exponent(ax_bot, cycles_arr, l2_arr, style, theme)

            if any_nonzero:
                ax_top.axhline(y=args.gmrestol, color=theme.refline_color,
                               linestyle='--', linewidth=0.8, alpha=0.6)
                ax_top.text(0.5, args.gmrestol,
                            f'solver tol = {args.gmrestol:.0e}',
                            transform=ax_top.get_yaxis_transform(),
                            fontsize=8, color=theme.refline_color,
                            va='bottom', ha='center')
                ax_top.axhline(y=roundoff_floor, color=theme.refline_color,
                               linestyle=':', linewidth=0.8, alpha=0.6)
                ax_top.text(0.5, roundoff_floor,
                            f'round-off floor = {roundoff_floor:.1e}',
                            transform=ax_top.get_yaxis_transform(),
                            fontsize=8, color=theme.refline_color,
                            va='bottom', ha='center')
            else:
                ax_top.text(
                    0.5, 0.5,
                    "All test runs are bit-identical\nwith the reference",
                    transform=ax_top.transAxes, ha='center', va='center',
                    fontsize=12, color=theme.text_secondary)

            if any_nonzero:
                _ref_color = ('#9399b2' if theme.mode == 'dark'
                              else '#8890a8')
                for ref_alpha, ref_lbl in [(0.5, '0.5'), (1.0, '1'),
                                           (2.0, '2')]:
                    ax_bot.axhline(y=ref_alpha, color=_ref_color,
                                   linestyle='--', linewidth=0.5, alpha=0.5)
                    ax_bot.text(
                        ax_bot.get_xlim()[1] if ax_bot.get_xlim()[1] > 0
                        else cycles_arr[-1],
                        ref_alpha, ref_lbl,
                        fontsize=7, color=_ref_color, alpha=0.7,
                        ha='right', va='bottom')
                ax_bot.set_ylim(-0.5, 6)
                ax_bot.set_ylabel(r'Error growth exponent $\alpha$',
                                  fontsize=9)
                ax_bot.set_title(
                    r'$L_2 \propto t^\alpha$  (local $\alpha$ between '
                    r'consecutive cycles)',
                    fontsize=9, loc='left', color=theme.text_secondary,
                    style='italic')

            ax_bot.set_xlabel('Cycle', fontsize=11)
            plt.setp(ax_top.get_xticklabels(), visible=False)
            ax_top.set_ylabel('$L_2$ relative error', fontsize=11)
            ax_top.set_title(field_latex, fontsize=13, fontweight='bold')
            leg = ax_top.legend(fontsize=9, loc='lower right')
            for text in leg.get_texts():
                if 'transient' in text.get_text():
                    text.set_alpha(0.4)
            ax_top.grid(True, which='both', alpha=0.3)
            ax_bot.grid(True, which='both', alpha=0.3)
            for ax in (ax_top, ax_bot):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        # ── Save PNG ─────────────────────────────────────────────────────
        if args.output:
            out_png = args.output
        else:
            parent_dir = os.path.dirname(ref_dir)
            suffix = f"_{grid_tag}" if len(run_groups) > 1 else ""
            out_png = os.path.join(parent_dir,
                                   f"results_l2_error{suffix}.png")

        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_png}")

        # ── CSV output ───────────────────────────────────────────────────
        if args.csv:
            csv_path = os.path.splitext(out_png)[0] + ".csv"
            with open(csv_path, 'w') as f:
                f.write("cycle,ref_label,test_label,field,"
                        "l2_rel_err,max_abs_diff,ref_l2_norm\n")
                for test_dir in test_dirs:
                    test_lab = solver_label(test_dir)
                    for ftype, comp, field_latex in fields:
                        data = results[test_dir][(ftype, comp)]
                        for i, cycle in enumerate(data.cycles):
                            f.write(
                                f"{cycle},{ref_label},{test_lab},"
                                f"{ftype}{comp},"
                                f"{data.l2_relative[i]:.6e},"
                                f"{data.max_absolute[i]:.6e},"
                                f"{data.ref_norm[i]:.6e}\n")
            print(f"  Saved: {csv_path}")

        # ── Console summary ──────────────────────────────────────────────
        print()
        print("L2 Error Accumulation Summary")
        print("\u2500" * 96)
        print(f"Reference: {ref_label}")
        print(f"Per-step solver tol: {args.gmrestol:.0e}    "
              f"Round-off floor (\u221aN\u00b7\u03b5): {roundoff_floor:.1e}")
        print("Note: L2 errors accumulate over cycles; "
              "per-step floors are lower bounds, not targets.")
        print()

        header = f"  {'Test':<26s} {'Field':>5s}   {'L2 (last cycle)':>22s}"
        print(header)
        print("  " + "\u2500" * 55)

        for test_dir in test_dirs:
            test_lab = solver_label(test_dir)
            for ftype, comp, field_latex in fields:
                data = results[test_dir][(ftype, comp)]
                field_name = f"{ftype}{comp}"

                if not data.cycles:
                    print(f"  {test_lab:<26s} {field_name:>5s}   "
                          f"{'N/A':>22s}")
                    continue

                l2_last = data.l2_relative[-1]

                if all(v == 0 for v in data.l2_relative):
                    print(f"  {test_lab:<26s} {field_name:>5s}   "
                          f"{'0 (bit-identical)':>22s}")
                    continue

                l2_str = _format_l2_with_pct(l2_last)
                print(f"  {test_lab:<26s} {field_name:>5s}   "
                      f"{l2_str:>22s}")


if __name__ == "__main__":
    main()
