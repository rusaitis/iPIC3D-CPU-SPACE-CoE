#!/usr/bin/env python3
"""
plot_l2_timeseries.py — L2 error accumulation analysis over simulation cycles.

Computes L2 relative error between a reference run and one or more test runs
at each output cycle, fits power-law growth models, and generates a diagnostic
plot with local exponent analysis.

Usage:
    python3 tests/plot_l2_timeseries.py --ref DIR --test DIR [--test DIR ...]
    python3 tests/plot_l2_timeseries.py                          # auto-discover
    python3 tests/plot_l2_timeseries.py --csv --gmrestol 1e-15   # with CSV output

HDF5 layout (phdf5 global files):
    <run_dir>/Fields_NNNNN/B_NNNNN.h5  →  /Fields/Bx, /Fields/By, /Fields/Bz  (X, Y, Z)
    <run_dir>/Fields_NNNNN/E_NNNNN.h5  →  /Fields/Ex, /Fields/Ey, /Fields/Ez  (X, Y, Z)
"""

import argparse
import glob
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    print("  WARNING: numpy not installed, skipping L2 timeseries analysis.")
    print("  Install with: pip3 install numpy")
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("  WARNING: matplotlib not installed, skipping L2 timeseries plot.")
    print("  Install with: pip3 install matplotlib")
    sys.exit(0)

try:
    import h5py
except ImportError:
    print("  WARNING: h5py not installed, skipping L2 timeseries analysis.")
    print("  Install with: pip3 install h5py")
    sys.exit(0)

# ── CLI args ──────────────────────────────────────────────────────────────
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
                    help='Write l2_timeseries.csv alongside PNG')
parser.add_argument('--output', default=None,
                    help='Override output PNG path')
args = parser.parse_args()

# ── Parse field specifications ────────────────────────────────────────────
fields = []
for spec in args.fields.split(','):
    spec = spec.strip()
    if len(spec) == 2:
        field_type = spec[0].upper()
        component = spec[1].lower()
        label = f"${field_type}_{component}$"
        fields.append((field_type, component, label))
    else:
        print(f"  WARNING: Unrecognized field spec '{spec}', skipping.")

if not fields:
    print("  ERROR: No valid fields specified.")
    sys.exit(1)

# ── Auto-discovery ────────────────────────────────────────────────────────
output_dir = os.path.join(script_dir, "test_petsc_output")

if args.ref is None:
    candidates = sorted(glob.glob(os.path.join(output_dir, "GMRES_*")))
    candidates = [c for c in candidates if os.path.isdir(c)
                  and not os.path.basename(c).startswith("GMRES_2")]
    if not candidates:
        print("  ERROR: No GMRES reference directory found in test_petsc_output/.")
        print("         Use --ref to specify.")
        sys.exit(1)
    args.ref = candidates[0]
    print(f"  Auto-discovered ref dir: {args.ref}")

if args.test is None:
    all_dirs = sorted(glob.glob(os.path.join(output_dir, "*")))
    args.test = [d for d in all_dirs
                 if os.path.isdir(d) and d != args.ref
                 and re.search(r'_\d+x\d+x\d+$', os.path.basename(d))]
    if not args.test:
        print("  ERROR: No test directories found in test_petsc_output/.")
        print("         Use --test to specify.")
        sys.exit(1)
    for t in args.test:
        print(f"  Auto-discovered test dir: {t}")

# ── Styles ────────────────────────────────────────────────────────────────
_styles = {
    "PETSc_gmres":  {"color": "#EE6677", "marker": "s", "ls": "-"},
    "PETSc_bcgs":   {"color": "#228833", "marker": "^", "ls": "--"},
    "PETSc_fgmres": {"color": "#AA3377", "marker": "D", "ls": "-."},
    "PETSc_tfqmr":  {"color": "#CCBB44", "marker": "v", "ls": ":"},
    "GMRES_2":      {"color": "#66CCEE", "marker": "d", "ls": "--"},
}
_extra_colors = ["#882255", "#332288", "#117733", "#999933", "#CC6677"]


def get_style(dir_path):
    """Get plot style for a directory, matching solver label prefix."""
    basename = os.path.basename(dir_path)
    key = re.sub(r'_\d+x\d+x\d+$', '', basename)
    if key in _styles:
        return _styles[key]
    for k, v in _styles.items():
        if basename.startswith(k):
            return v
    idx = hash(basename) % len(_extra_colors)
    return {"color": _extra_colors[idx], "marker": "o", "ls": "-"}


# ── Utility functions ─────────────────────────────────────────────────────

def discover_cycles(run_dir):
    """Scan Fields_* directories, extract cycle numbers, return sorted list."""
    pattern = os.path.join(run_dir, "Fields_*")
    cycles = []
    for d in glob.glob(pattern):
        m = re.match(r'Fields_(\d+)', os.path.basename(d))
        if m:
            cycles.append(int(m.group(1)))
    return sorted(cycles)


def load_field_3d(run_dir, cycle, field_type, component):
    """Load full 3D field array from phdf5 output.

    Returns
    -------
    np.ndarray  shape (Nx, Ny, Nz)
    """
    cycle_str = f"{cycle:05d}"
    h5_path = os.path.join(run_dir, f"Fields_{cycle_str}",
                           f"{field_type}_{cycle_str}.h5")
    dataset = f"/Fields/{field_type}{component}"
    with h5py.File(h5_path, "r") as f:
        return f[dataset][:]


def solver_label(run_dir):
    return os.path.basename(run_dir)


def _r_squared(y_obs, y_pred):
    """Coefficient of determination in log space."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_powerlaw(cycles, l2_vals):
    """Fit L2 = A·t^α in log-log space, with optional piecewise split.

    Returns dict with keys:
        alpha, r2, A           — global power-law fit
        knee_cycle             — split cycle (None if piecewise not triggered)
        alpha_early, alpha_late, r2_piecewise — piecewise fit (None if not triggered)
    Returns None if insufficient data.
    """
    if len(cycles) < 3:
        return None
    mask = np.array([v > 0 and c > 0 for c, v in zip(cycles, l2_vals)])
    c = np.array(cycles, dtype=float)[mask]
    v = np.array(l2_vals, dtype=float)[mask]
    if len(c) < 3:
        return None

    log_c = np.log(c)
    log_v = np.log(v)

    # Global power-law: log(L2) = log(A) + α·log(t)
    alpha, log_A = np.polyfit(log_c, log_v, 1)
    A = np.exp(log_A)
    r2 = _r_squared(log_v, log_A + alpha * log_c)

    result = dict(alpha=alpha, r2=r2, A=A,
                  knee_cycle=None, alpha_early=None,
                  alpha_late=None, r2_piecewise=None)

    # Try piecewise power-law: split at each candidate index
    n = len(c)
    if n < 6:
        return result

    best_ssr = np.inf
    best_k = None
    best_left = best_right = None

    for k in range(2, n - 2):
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
            result['knee_cycle'] = float(c[best_k])
            result['alpha_early'] = best_left[0]
            result['alpha_late'] = best_right[0]
            result['r2_piecewise'] = r2_pw

    return result


def compute_local_exponent(cycles, l2_vals):
    """Δlog(L2)/Δlog(t) at midpoints — instantaneous power-law exponent.

    Returns (midpoint_cycles, local_alpha) arrays, or (None, None) if
    insufficient data.
    """
    mask = np.array([v > 0 and c > 0 for c, v in zip(cycles, l2_vals)])
    c = np.array(cycles, dtype=float)[mask]
    v = np.array(l2_vals, dtype=float)[mask]
    if len(c) < 2:
        return None, None

    log_c = np.log(c)
    log_v = np.log(v)

    d_log_c = np.diff(log_c)
    d_log_v = np.diff(log_v)

    # Avoid division by zero at identical cycle values
    good = d_log_c != 0
    mids = 0.5 * (c[:-1] + c[1:])
    local_alpha = np.where(good, d_log_v / d_log_c, 0.0)

    return mids[good], local_alpha[good]


def _format_l2_with_pct(val):
    """Format L2 value with parenthetical percentage when > 1e-6."""
    s = f"{val:.2e}"
    if val > 1e-6:
        pct = val * 100
        if pct >= 0.01:
            s += f" ({pct:.2f}%)"
        else:
            s += f" ({pct:.1e}%)"
    return s


# ── Find common cycles ───────────────────────────────────────────────────
ref_cycles = set(discover_cycles(args.ref))
all_test_cycles = [set(discover_cycles(t)) for t in args.test]
common_cycles = sorted(ref_cycles.intersection(*all_test_cycles))

if not common_cycles:
    print("  ERROR: No common output cycles found across run directories.")
    sys.exit(1)

print(f"  Found {len(common_cycles)} common cycles: "
      f"{common_cycles[0]} .. {common_cycles[-1]}")

ref_label = solver_label(args.ref)

# ── Compute N_dof and round-off floor ────────────────────────────────────
sample_ftype, sample_comp = fields[0][0], fields[0][1]
try:
    cycle_str = f"{common_cycles[0]:05d}"
    h5_path = os.path.join(args.ref, f"Fields_{cycle_str}",
                           f"{sample_ftype}_{cycle_str}.h5")
    with h5py.File(h5_path, "r") as f:
        n_dof = int(np.prod(f[f"/Fields/{sample_ftype}{sample_comp}"].shape))
except Exception:
    n_dof = 10000  # fallback

eps_machine = np.finfo(np.float64).eps
roundoff_floor = np.sqrt(n_dof) * eps_machine

# ── Compute L2 errors (one cycle at a time to limit memory) ──────────────
# Load ref data once per (field, cycle), then compare against each test dir.
# results[test_dir][(field_type, comp)] = {
#     'cycles': [...], 'l2_rel': [...], 'max_abs': [...], 'ref_norm': [...]
# }
results = {t: {} for t in args.test}

for ftype, comp, flabel in fields:
    for test_dir in args.test:
        results[test_dir][(ftype, comp)] = {
            'cycles': [], 'l2_rel': [], 'max_abs': [], 'ref_norm': [],
        }

    for cycle in common_cycles:
        try:
            ref_data = load_field_3d(args.ref, cycle, ftype, comp)
        except (OSError, KeyError) as e:
            print(f"  WARNING: Could not load ref {ftype}{comp} cycle {cycle}: {e}")
            continue

        ref_norm = np.linalg.norm(ref_data)

        for test_dir in args.test:
            try:
                test_data = load_field_3d(test_dir, cycle, ftype, comp)
            except (OSError, KeyError) as e:
                print(f"  WARNING: Could not load {ftype}{comp} cycle {cycle} "
                      f"from {solver_label(test_dir)}: {e}")
                continue

            if ref_data.shape != test_data.shape:
                print(f"  WARNING: Shape mismatch at cycle {cycle} for {flabel}: "
                      f"{ref_data.shape} vs {test_data.shape}, skipping.")
                continue

            diff = test_data - ref_data
            l2_rel = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else 0.0
            max_abs = float(np.max(np.abs(diff)))

            r = results[test_dir][(ftype, comp)]
            r['cycles'].append(cycle)
            r['l2_rel'].append(l2_rel)
            r['max_abs'].append(max_abs)
            r['ref_norm'].append(ref_norm)

for test_dir in args.test:
    print(f"  Processed: {solver_label(test_dir)}")

# ── Plot ──────────────────────────────────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'axes.facecolor': '#fafafa',
})

n_fields = len(fields)
fig = plt.figure(figsize=(7 * n_fields, 7), dpi=150,
                 layout='constrained')
gs = fig.add_gridspec(2, n_fields, height_ratios=[3, 1], hspace=0.08)

fig.suptitle(f"L2 Error Accumulation  (ref: {ref_label})",
             fontsize=13, fontweight='bold', y=0.98)

# Cache fit results: fit_cache[(test_dir, ftype, comp)] = fit_powerlaw result
fit_cache = {}

for fi, (ftype, comp, flabel) in enumerate(fields):
    ax_top = fig.add_subplot(gs[0, fi])
    ax_bot = fig.add_subplot(gs[1, fi], sharex=ax_top)
    any_nonzero = False

    for test_dir in args.test:
        test_lab = solver_label(test_dir)
        data = results[test_dir][(ftype, comp)]
        cycles_arr = data['cycles']
        l2_arr = data['l2_rel']

        if not cycles_arr:
            continue

        style = get_style(test_dir)

        # Check for bit-identical runs (all L2 = 0)
        all_zero = all(v == 0 for v in l2_arr)
        if all_zero:
            ax_top.plot([], [], color=style['color'], linestyle=style['ls'],
                        marker=style['marker'],
                        label=f"{test_lab} (bit-identical)")
            continue

        any_nonzero = True

        ax_top.semilogy(cycles_arr, l2_arr,
                        color=style['color'], marker=style['marker'],
                        linestyle=style['ls'], linewidth=1.8, markersize=5,
                        markeredgecolor='white', markeredgewidth=0.5,
                        label=test_lab, alpha=0.9, zorder=3)

        # Power-law fit overlay
        fit = fit_powerlaw(cycles_arr, l2_arr)
        fit_cache[(test_dir, ftype, comp)] = fit
        if fit is not None:
            c_fit = np.array([c for c in cycles_arr if c > 0], dtype=float)
            knee = fit['knee_cycle']

            if knee is not None:
                # Piecewise power-law: two segments with different dashes
                left = c_fit[c_fit <= knee]
                right = c_fit[c_fit >= knee]
                if len(left) > 0:
                    y_left = np.exp(np.log(fit['A']) +
                                    fit['alpha'] * np.log(left))
                    # Re-fit left segment for its own A
                    mask_l = c_fit <= knee
                    vals_l = np.array([l2_arr[i] for i, c in enumerate(
                        cycles_arr) if c > 0 and c <= knee])
                    if len(vals_l) >= 2:
                        alpha_l = fit['alpha_early']
                        logA_l = np.mean(np.log(vals_l) -
                                         alpha_l * np.log(
                                             c_fit[mask_l][:len(vals_l)]))
                        y_left = np.exp(logA_l + alpha_l * np.log(left))
                    ax_top.semilogy(left, y_left,
                                    color=style['color'], linestyle='--',
                                    linewidth=1, alpha=0.5)
                if len(right) > 0:
                    vals_r = np.array([l2_arr[i] for i, c in enumerate(
                        cycles_arr) if c > 0 and c >= knee])
                    if len(vals_r) >= 2:
                        alpha_r = fit['alpha_late']
                        logA_r = np.mean(np.log(vals_r) -
                                         alpha_r * np.log(right[:len(vals_r)]))
                        y_right = np.exp(logA_r + alpha_r * np.log(right))
                    else:
                        y_right = fit['A'] * right ** fit['alpha']
                    ax_top.semilogy(right, y_right,
                                    color=style['color'], linestyle=':',
                                    linewidth=1, alpha=0.5)
                # Vertical knee marker
                ax_top.axvline(x=knee, color=style['color'],
                               linestyle='-', linewidth=0.5, alpha=0.3)
                # Transient fill
                ax_top.axvspan(c_fit[0], knee,
                               color=style['color'], alpha=0.08, zorder=0)
                # Annotation with early/late exponents
                ann_x = cycles_arr[-1] * 0.65
                ann_idx = max(0, len(cycles_arr) * 2 // 3)
                ann_y = l2_arr[min(ann_idx, len(l2_arr) - 1)]
                ax_top.annotate(
                    f"\u03b1={fit['alpha_early']:.1f}/{fit['alpha_late']:.1f}"
                    f" (R\u00b2={fit['r2_piecewise']:.2f})",
                    xy=(ann_x, ann_y), fontsize=8, color=style['color'],
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7,
                              edgecolor=style['color'], linewidth=0.5))
            else:
                # Single power-law fit curve
                fit_y = fit['A'] * c_fit ** fit['alpha']
                ax_top.semilogy(c_fit, fit_y,
                                color=style['color'], linestyle='--',
                                linewidth=1, alpha=0.5)
                # Annotation
                mid_idx = len(cycles_arr) // 2
                ax_top.annotate(
                    f"\u03b1={fit['alpha']:.2f} (R\u00b2={fit['r2']:.2f})",
                    xy=(cycles_arr[mid_idx], l2_arr[mid_idx]),
                    fontsize=8, color=style['color'],
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7,
                              edgecolor=style['color'], linewidth=0.5))

            # Final-value annotation with percentage
            last_val = l2_arr[-1]
            if last_val > 0:
                lbl = _format_l2_with_pct(last_val)
                ax_top.annotate(
                    lbl, xy=(cycles_arr[-1], last_val),
                    fontsize=7, color=style['color'],
                    xytext=(5, -10), textcoords='offset points',
                    ha='left', va='top')

        # Local exponent in bottom panel
        mids, loc_alpha = compute_local_exponent(cycles_arr, l2_arr)
        if mids is not None and len(mids) > 0:
            ax_bot.plot(mids, loc_alpha,
                        color=style['color'], marker=style['marker'],
                        linestyle=style['ls'], linewidth=1.2, markersize=3,
                        markeredgecolor='white', markeredgewidth=0.3,
                        alpha=0.85)

    # Reference lines on top panel (only meaningful on log scale)
    if any_nonzero:
        ax_top.axhline(y=args.gmrestol, color='gray', linestyle='--',
                       linewidth=0.8, alpha=0.6,
                       label=f'solver tol (per step) = {args.gmrestol:.0e}')
        ax_top.axhline(y=roundoff_floor, color='gray', linestyle=':',
                       linewidth=0.8, alpha=0.6,
                       label=(f'round-off floor '
                              f'($\\sqrt{{N}}\\cdot\\varepsilon$)'
                              f' = {roundoff_floor:.1e}'))
        # Annotate the gap between per-step floors and cumulative error
        ax_top.text(0.98, 0.02,
                    'ref. lines = per-step precision;\n'
                    'data = cumulative error over N cycles',
                    transform=ax_top.transAxes, fontsize=6.5,
                    color='#888888', ha='right', va='bottom',
                    style='italic')
    else:
        ax_top.text(0.5, 0.5,
                    "All test runs are bit-identical\nwith the reference",
                    transform=ax_top.transAxes, ha='center', va='center',
                    fontsize=12, color='#666666')

    # Bottom panel: reference exponent lines
    if any_nonzero:
        for ref_alpha, ref_lbl in [(0.5, 'diffusive'), (1.0, 'linear'),
                                   (2.0, 'quadratic')]:
            ax_bot.axhline(y=ref_alpha, color='#999999', linestyle=':',
                           linewidth=0.6, alpha=0.5)
            ax_bot.text(ax_bot.get_xlim()[1] if ax_bot.get_xlim()[1] > 0
                        else cycles_arr[-1],
                        ref_alpha + 0.1, ref_lbl,
                        fontsize=6, color='#999999', ha='right', va='bottom')
        ax_bot.set_ylim(-0.5, 6)
        ax_bot.set_ylabel(r'Error growth exponent $\alpha$', fontsize=9)
        ax_bot.set_title(
            r'$L_2 \propto t^\alpha$  (local $\alpha$ between consecutive cycles)',
            fontsize=8, loc='left', color='#666666', style='italic')

    ax_bot.set_xlabel('Cycle', fontsize=11)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel('$L_2$ relative error', fontsize=11)
    ax_top.set_title(flabel, fontsize=13, fontweight='bold')
    ax_top.legend(fontsize=8, loc='upper left')
    ax_top.grid(True, which='both', alpha=0.3)
    ax_bot.grid(True, which='both', alpha=0.3)
    for ax in (ax_top, ax_bot):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Layout handled by constrained_layout=True

# ── Save PNG ──────────────────────────────────────────────────────────────
if args.output:
    out_png = args.output
else:
    parent_dir = os.path.dirname(args.ref)
    out_png = os.path.join(parent_dir, "l2_timeseries.png")

fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out_png}")

# ── CSV output ────────────────────────────────────────────────────────────
if args.csv:
    csv_path = os.path.splitext(out_png)[0] + ".csv"
    with open(csv_path, 'w') as f:
        f.write("cycle,ref_label,test_label,field,"
                "l2_rel_err,max_abs_diff,ref_l2_norm\n")
        for test_dir in args.test:
            test_lab = solver_label(test_dir)
            for ftype, comp, flabel in fields:
                data = results[test_dir][(ftype, comp)]
                for i, cycle in enumerate(data['cycles']):
                    f.write(f"{cycle},{ref_label},{test_lab},"
                            f"{ftype}{comp},{data['l2_rel'][i]:.6e},"
                            f"{data['max_abs'][i]:.6e},"
                            f"{data['ref_norm'][i]:.6e}\n")
    print(f"  Saved: {csv_path}")

    # Summary CSV with power-law fit parameters
    summary_path = os.path.splitext(out_png)[0] + "_summary.csv"
    with open(summary_path, 'w') as f:
        f.write("test_label,field,alpha,r2,l2_last,amplification,"
                "knee_cycle,alpha_early,alpha_late\n")
        for test_dir in args.test:
            test_lab = solver_label(test_dir)
            for ftype, comp, flabel in fields:
                data = results[test_dir][(ftype, comp)]
                if not data['cycles']:
                    continue
                fit = fit_cache.get((test_dir, ftype, comp))
                l2_last = data['l2_rel'][-1]
                nz = [v for v in data['l2_rel'] if v > 0]
                amplif = l2_last / nz[0] if nz and nz[0] > 0 else 0.0
                if fit is not None:
                    knee_s = (f"{fit['knee_cycle']:.1f}"
                              if fit['knee_cycle'] else "")
                    ae_s = (f"{fit['alpha_early']:.4f}"
                            if fit['alpha_early'] is not None else "")
                    al_s = (f"{fit['alpha_late']:.4f}"
                            if fit['alpha_late'] is not None else "")
                    f.write(f"{test_lab},{ftype}{comp},"
                            f"{fit['alpha']:.4f},{fit['r2']:.4f},"
                            f"{l2_last:.6e},{amplif:.2f},"
                            f"{knee_s},{ae_s},{al_s}\n")
                else:
                    f.write(f"{test_lab},{ftype}{comp},"
                            f",,{l2_last:.6e},{amplif:.2f},,,\n")
    print(f"  Saved: {summary_path}")

# ── Console summary ──────────────────────────────────────────────────────
print()
print("L2 Error Accumulation Summary")
print("\u2500" * 96)
print(f"Reference: {ref_label}")
print(f"Per-step solver tol: {args.gmrestol:.0e}    "
      f"Round-off floor (\u221aN\u00b7\u03b5): {roundoff_floor:.1e}  "
      f"(N={n_dof} DOFs)")
print("Note: L2 errors accumulate over cycles; "
      "per-step floors are lower bounds, not targets.")
print()

header = (f"  {'Test':<26s} {'Field':>5s}  {'L2@last':>18s}  "
          f"{'\u03b1':>6s}  {'R\u00b2':>5s}  {'Amplif.':>8s}  {'Knee':>5s}")
print(header)
print("  " + "\u2500" * 90)

for test_dir in args.test:
    test_lab = solver_label(test_dir)
    for ftype, comp, flabel in fields:
        data = results[test_dir][(ftype, comp)]
        field_name = f"{ftype}{comp}"

        if not data['cycles']:
            print(f"  {test_lab:<26s} {field_name:>5s}  "
                  f"{'N/A':>18s}  {'N/A':>6s}  {'N/A':>5s}  "
                  f"{'N/A':>8s}  {'N/A':>5s}")
            continue

        l2_last = data['l2_rel'][-1]

        if all(v == 0 for v in data['l2_rel']):
            print(f"  {test_lab:<26s} {field_name:>5s}  "
                  f"{'0 (bit-identical)':>18s}  {'N/A':>6s}  {'N/A':>5s}  "
                  f"{'N/A':>8s}  {'\u2014':>5s}")
            continue

        l2_str = _format_l2_with_pct(l2_last)
        nz = [v for v in data['l2_rel'] if v > 0]
        amplif = l2_last / nz[0] if nz and nz[0] > 0 else 0.0
        amplif_str = f"{amplif:.1f}x"

        fit = fit_cache.get((test_dir, ftype, comp))
        if fit is not None:
            alpha_str = f"{fit['alpha']:.2f}"
            r_sq_str = f"{fit['r2']:.2f}"
            if fit['knee_cycle'] is not None:
                knee_str = f"{fit['knee_cycle']:.0f}"
                alpha_str = (f"{fit['alpha_early']:.1f}/"
                             f"{fit['alpha_late']:.1f}")
                r_sq_str = f"{fit['r2_piecewise']:.2f}"
            else:
                knee_str = "\u2014"
        else:
            alpha_str = "N/A"
            r_sq_str = "N/A"
            knee_str = "\u2014"

        print(f"  {test_lab:<26s} {field_name:>5s}  "
              f"{l2_str:>18s}  {alpha_str:>6s}  {r_sq_str:>5s}  "
              f"{amplif_str:>8s}  {knee_str:>5s}")
