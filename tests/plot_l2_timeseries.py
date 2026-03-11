#!/usr/bin/env python3
"""
plot_l2_timeseries.py — L2 error accumulation analysis over simulation cycles.

Computes L2 relative error between a reference run and one or more test runs
at each output cycle, fits exponential growth rates, and generates a diagnostic
plot with significance analysis.

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


def fit_growth(cycles, l2_vals):
    """Fit log(L2) = intercept + gamma * cycle via numpy.polyfit.

    Returns (gamma, r_squared, intercept) or None if insufficient data.
    gamma is the Lyapunov-like growth rate per cycle.
    """
    if len(cycles) < 3:
        return None
    mask = [v > 0 for v in l2_vals]
    c = [cycles[i] for i in range(len(cycles)) if mask[i]]
    v = [l2_vals[i] for i in range(len(l2_vals)) if mask[i]]
    if len(c) < 3:
        return None
    c_arr = np.array(c, dtype=float)
    log_v = np.log(v)
    gamma, intercept = np.polyfit(c_arr, log_v, 1)
    ss_res = np.sum((log_v - (gamma * c_arr + intercept)) ** 2)
    ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return gamma, r_sq, intercept


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
    sample = load_field_3d(args.ref, common_cycles[0], sample_ftype, sample_comp)
    n_dof = sample.size
except Exception:
    n_dof = 10000  # fallback

eps_machine = np.finfo(np.float64).eps
roundoff_floor = np.sqrt(n_dof) * eps_machine

# ── Compute L2 errors (one cycle at a time to limit memory) ──────────────
# results[test_dir][(field_type, comp)] = {
#     'cycles': [...], 'l2_rel': [...], 'max_abs': [...], 'ref_norm': [...]
# }
results = {}

for test_dir in args.test:
    test_lab = solver_label(test_dir)
    results[test_dir] = {}
    for ftype, comp, flabel in fields:
        l2_vals = []
        max_abs_vals = []
        ref_norm_vals = []
        valid_cycles = []

        for cycle in common_cycles:
            try:
                ref_data = load_field_3d(args.ref, cycle, ftype, comp)
                test_data = load_field_3d(test_dir, cycle, ftype, comp)
            except (OSError, KeyError) as e:
                print(f"  WARNING: Could not load {ftype}{comp} cycle {cycle}: {e}")
                continue

            if ref_data.shape != test_data.shape:
                print(f"  WARNING: Shape mismatch at cycle {cycle} for {flabel}: "
                      f"{ref_data.shape} vs {test_data.shape}, skipping.")
                continue

            diff = test_data - ref_data
            ref_norm = np.linalg.norm(ref_data)
            l2_rel = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else 0.0
            max_abs = float(np.max(np.abs(diff)))

            valid_cycles.append(cycle)
            l2_vals.append(l2_rel)
            max_abs_vals.append(max_abs)
            ref_norm_vals.append(ref_norm)

        results[test_dir][(ftype, comp)] = {
            'cycles': valid_cycles,
            'l2_rel': l2_vals,
            'max_abs': max_abs_vals,
            'ref_norm': ref_norm_vals,
        }

    print(f"  Processed: {test_lab}")

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
fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5.5), dpi=150,
                         squeeze=False)
axes = axes[0]

fig.suptitle(f"L2 Error Accumulation  (ref: {ref_label})",
             fontsize=13, fontweight='bold', y=0.98)

for fi, (ftype, comp, flabel) in enumerate(fields):
    ax = axes[fi]
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
            # Can't plot 0 on log scale — add invisible line for legend
            ax.plot([], [], color=style['color'], linestyle=style['ls'],
                    marker=style['marker'], label=f"{test_lab} (bit-identical)")
            continue

        any_nonzero = True

        ax.semilogy(cycles_arr, l2_arr,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['ls'], linewidth=1.8, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.5,
                    label=test_lab, alpha=0.9)

        # Exponential fit overlay
        fit = fit_growth(cycles_arr, l2_arr)
        if fit is not None:
            gamma, r_sq, intercept = fit
            fit_y = np.exp(intercept + gamma * np.array(cycles_arr))
            ax.semilogy(cycles_arr, fit_y,
                        color=style['color'], linestyle='--', linewidth=1,
                        alpha=0.5)
            # Annotate growth rate near the midpoint of the curve
            mid_idx = len(cycles_arr) // 2
            ax.annotate(
                f"\u03b3={gamma:.3f}/cyc (R\u00b2={r_sq:.2f})",
                xy=(cycles_arr[mid_idx], l2_arr[mid_idx]),
                fontsize=8, color=style['color'],
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7,
                          edgecolor=style['color'], linewidth=0.5))

    # Reference lines (only meaningful on log scale)
    if any_nonzero:
        ax.axhline(y=args.gmrestol, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.6,
                   label=f'GMREStol = {args.gmrestol:.0e}')
        ax.axhline(y=roundoff_floor, color='gray', linestyle=':',
                   linewidth=0.8, alpha=0.6,
                   label=f'\u221aN\u00b7\u03b5 = {roundoff_floor:.1e}')
    else:
        ax.text(0.5, 0.5,
                "All test runs are bit-identical\nwith the reference",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='#666666')

    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('L2 relative error', fontsize=11)
    ax.set_title(flabel, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])

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

# ── Console summary ──────────────────────────────────────────────────────
print()
print("L2 Error Accumulation Summary")
print("\u2500" * 80)
print(f"Reference: {ref_label}")
print(f"Solver tol: {args.gmrestol:.0e}    "
      f"Round-off floor: {roundoff_floor:.1e} (N={n_dof})")
print()

header = (f"  {'Test':<30s} {'Field':>5s}  {'L2@first':>10s}  "
          f"{'L2@last':>10s}  {'\u03b3 (/cyc)':>10s}  {'R\u00b2':>5s}")
print(header)
print("  " + "\u2500" * 78)

for test_dir in args.test:
    test_lab = solver_label(test_dir)
    for ftype, comp, flabel in fields:
        data = results[test_dir][(ftype, comp)]
        field_name = f"{ftype}{comp}"

        if not data['cycles']:
            print(f"  {test_lab:<30s} {field_name:>5s}  "
                  f"{'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>5s}")
            continue

        l2_first = data['l2_rel'][0]
        l2_last = data['l2_rel'][-1]

        if all(v == 0 for v in data['l2_rel']):
            print(f"  {test_lab:<30s} {field_name:>5s}  "
                  f"{'0 (ident.)':>10s}  {'0':>10s}  "
                  f"{'N/A':>10s}  {'N/A':>5s}")
            continue

        fit = fit_growth(data['cycles'], data['l2_rel'])
        if fit is not None:
            gamma, r_sq, _ = fit
            gamma_str = f"{gamma:.4f}"
            r_sq_str = f"{r_sq:.2f}"
        else:
            gamma_str = "N/A"
            r_sq_str = "N/A"

        print(f"  {test_lab:<30s} {field_name:>5s}  "
              f"{l2_first:>10.2e}  {l2_last:>10.2e}  "
              f"{gamma_str:>10s}  {r_sq_str:>5s}")
