#!/usr/bin/env python3
"""
plot_petsc_test_results.py — Generate scaling plots from a scaling_results.csv file.

Usage:
    python3 plot_petsc_test_results.py [path/to/scaling_results.csv] [--loglog] [--ref-slope N] [--sort]

The CSV path defaults to scaling_results.csv next to this script.
The output PNG is written alongside the CSV with the same base name.

All configuration (solver labels, np, cycles, nzc, breakdown data) is read
directly from the CSV. Environment variables are still accepted as overrides
for backward compatibility when called from test_petsc_scaling.sh.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter
except ImportError:
    print("  WARNING: matplotlib not installed, skipping plot generation.")
    print("  Install with: pip3 install matplotlib")
    raise SystemExit(0)

# ── CLI args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('csv', nargs='?', default=None, metavar='CSV_FILE',
                    help='Path to CSV (default: scaling_results.csv next to this script)')
parser.add_argument('--loglog', action='store_true',
                    help='Use log-log axes for the time plot (default: linear)')
parser.add_argument('--ref-slope', type=float, default=None, metavar='N',
                    help='Show reference slope N^exp (e.g. --ref-slope 2)')
parser.add_argument('--sort', action='store_true',
                    help='Group improvement bars by solver (sorted by avg improvement)')
parser.add_argument('--profile-smooth', type=int, default=None, metavar='N',
                    help='Smooth profile plot by averaging over N cycles')
args = parser.parse_args()

# ── Modern styling ────────────────────────────────────────────────────────
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
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.25,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'axes.facecolor': '#fafafa',
})

# ── I/O paths ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

if args.csv is not None:
    csv_path = args.csv
elif "CSV_FILE" in os.environ:
    csv_path = os.environ["CSV_FILE"]
else:
    csv_path = os.path.join(script_dir, "scaling_results.csv")

if "PLOT_FILE" in os.environ:
    plot_path = os.environ["PLOT_FILE"]
else:
    plot_path = os.path.splitext(csv_path)[0] + ".png"

# ── Infer metadata from CSV header and first data row ─────────────────────
METADATA_COLS = {
    'grid_size', 'np', 'cycles', 'nzc',
    'gmres_moments_s', 'gmres_mover_s',
    'gmres_field_pct', 'gmres_moments_pct', 'gmres_mover_pct',
}

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    header_fields = reader.fieldnames or []
    first_row = next(reader, {})

# Solver labels: columns that are not metadata and not suffixed
inferred_labels = [c for c in header_fields
                   if c not in METADATA_COLS
                   and not c.endswith(('_std', '_iters', '_converged', '_residual'))]

solver_labels = os.environ["SOLVER_LABEL_LIST"].split() \
    if "SOLVER_LABEL_LIST" in os.environ else inferred_labels

nzc    = int(os.environ.get("NZC",    first_row.get('nzc',    '1') or '1'))
np_    =     os.environ.get("NP",     first_row.get('np',     '?') or '?')
cycles =     os.environ.get("CYCLES", first_row.get('cycles', '?') or '?')

# ── Breakdown data: from CSV columns, or env-var override ─────────────────
def parse_floats(env_key):
    return [float(x) if x != "NA" else None
            for x in os.environ.get(env_key, "").split() if x]

if "BD_GRIDS_LIST" in os.environ:
    bd_grids   = [int(x) for x in os.environ.get("BD_GRIDS_LIST", "").split() if x]
    bd_field   = parse_floats("BD_FIELD_LIST")
    bd_moments = parse_floats("BD_MOMENTS_LIST")
    bd_mover   = parse_floats("BD_MOVER_LIST")
else:
    bd_grids, bd_field, bd_moments, bd_mover = [], [], [], []
    gmres_label = solver_labels[0] if solver_labels else "GMRES"
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            bd_grids.append(int(row['grid_size']))
            f_val = row.get(gmres_label,      'NA')
            m_val = row.get('gmres_moments_s', 'NA')
            p_val = row.get('gmres_mover_s',   'NA')
            bd_field.append(  float(f_val) if f_val not in ('NA', '') else None)
            bd_moments.append(float(m_val) if m_val not in ('NA', '') else None)
            bd_mover.append(  float(p_val) if p_val not in ('NA', '') else None)

# ── Main timing data ───────────────────────────────────────────────────────
grids        = []    # numeric grid sizes (for plotting x-axis)
grid_strings = []    # original grid_size strings (for labels)
solver_times = {label: [] for label in solver_labels}
solver_stds  = {label: [] for label in solver_labels}

with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
        gs = row["grid_size"]
        grid_strings.append(gs)
        # grid_size may be numeric (scaling mode) or "NxNxN" string (single-grid)
        try:
            grids.append(int(gs))
        except ValueError:
            # String like "100x100x1": use index as x position
            grids.append(len(grids))
        for label in solver_labels:
            val     = row.get(label,           "NA")
            std_val = row.get(f"{label}_std",  "NA")
            solver_times[label].append(float(val)     if val     not in ("NA", "") else None)
            solver_stds[label].append( float(std_val) if std_val not in ("NA", "") else None)

# Detect single-row mode (single-grid comparison)
single_row = len(grids) == 1

# ── Styles ─────────────────────────────────────────────────────────────────
styles = {
    "GMRES":        {"color": "#4477AA", "marker": "o", "ls": "-"},
    "PETSc":        {"color": "#EE6677", "marker": "s", "ls": "-"},
    "PETSc_gmres":  {"color": "#EE6677", "marker": "s", "ls": "-"},
    "PETSc_bcgs":   {"color": "#228833", "marker": "^", "ls": "--"},
    "PETSc_fgmres": {"color": "#AA3377", "marker": "D", "ls": "-."},
    "PETSc_tfqmr":  {"color": "#CCBB44", "marker": "v", "ls": ":"},
}
default_style = {"color": "#607D8B", "marker": "x", "ls": ":"}

# ── Time unit ──────────────────────────────────────────────────────────────
all_times = [t for tlist in solver_times.values() for t in tlist if t is not None]
if all_times and max(all_times) < 1.0:
    unit, scale = 'ms', 1000.0
elif all_times and max(all_times) >= 120.0:
    unit, scale = 'min', 1 / 60.0
else:
    unit, scale = 's', 1.0

# ── Profile file discovery ─────────────────────────────────────────────────
profile_dir = os.environ.get("PROFILE_DIR", os.path.dirname(csv_path))
profile_files_all = sorted(glob.glob(os.path.join(profile_dir, "profile_*.csv")))

def parse_grid(path):
    """Extract grid N from filename like profile_GMRES_100x100x1.csv -> 100."""
    base = os.path.splitext(os.path.basename(path))[0]
    grid_part = base.rsplit('_', 1)[-1]
    try:
        return int(grid_part.split('x')[0])
    except (ValueError, IndexError):
        return 0

# Keep only the largest grid when multiple grids exist
if profile_files_all:
    max_grid = max(parse_grid(f) for f in profile_files_all)
    profile_files = [f for f in profile_files_all if parse_grid(f) == max_grid]
else:
    profile_files = []
    max_grid = 0

# ── Layout ─────────────────────────────────────────────────────────────────
has_breakdown = bool(bd_grids) and any(m is not None for m in bd_moments)
has_profile = single_row and bool(profile_files)
has_bottom = has_breakdown or has_profile

fig = plt.figure(figsize=(15, 10 if has_bottom else 6))
gs  = GridSpec(2 if has_bottom else 1, 2, figure=fig,
               hspace=0.45, wspace=0.35,
               top=0.90, bottom=0.10, left=0.07, right=0.97)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

date_str = datetime.now().strftime("%Y-%m-%d")
if single_row:
    fig.suptitle(
        f'iPIC3D Solver Comparison  —  {grid_strings[0]}  ·  {np_} procs · {cycles} cycles · {date_str}',
        fontsize=12, color='#444444', y=0.97
    )
else:
    fig.suptitle(
        f'iPIC3D Field Solver Scaling  —  {np_} procs · {cycles} cycles · {date_str}',
        fontsize=12, color='#444444', y=0.97
    )

grid_label   = f'Grid size (N × N × {nzc})' if nzc > 1 else 'Grid size (N × N)'
tick_labels  = grid_strings if single_row else \
               [f'{g}×{nzc}' if nzc > 1 else str(g) for g in grids]
tick_rotation = 45 if len(grids) > 6 else 0
tick_ha      = 'right' if tick_rotation else 'center'

x_label_map = dict(zip(grids, tick_labels))
x_fmt        = FuncFormatter(lambda x, _: x_label_map.get(round(x), f'{x:.10g}'))

# ── Left panel ────────────────────────────────────────────────────────────
plain_fmt = FuncFormatter(lambda x, _: f'{x:.10g}')

if single_row:
    # Single-grid mode: bar chart of solver times
    bar_x = list(range(len(solver_labels)))
    bar_vals = []
    bar_errs = []
    bar_colors = []
    for label in solver_labels:
        t = solver_times[label][0]
        e = solver_stds[label][0]
        bar_vals.append(t * scale if t is not None else 0)
        bar_errs.append(e * scale if e is not None else 0)
        s = styles.get(label, default_style)
        bar_colors.append(s["color"])

    bars = ax1.bar(bar_x, bar_vals, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5,
                   yerr=bar_errs if any(e > 0 for e in bar_errs) else None,
                   capsize=5)
    for bar, val in zip(bars, bar_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + max(bar_vals) * 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(bar_x)
    ax1.set_xticklabels(solver_labels, fontsize=10)
    ax1.set_ylabel(f'Field solver time ({unit})', fontsize=12)
    ax1.set_title(f'Field Solver Time  ({grid_strings[0]})', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

else:
    # Multi-grid mode: line plot of solver times vs grid size
    for label in solver_labels:
        times = solver_times[label]
        stds  = solver_stds[label]
        valid = [(g, t, e) for g, t, e in zip(grids, times, stds) if t is not None]
        if not valid:
            continue
        vg, vt_raw, vstd_raw = zip(*valid)
        vt = [t * scale for t in vt_raw]
        s  = styles.get(label, default_style)
        ax1.plot(vg, vt, marker=s["marker"], linestyle=s["ls"], color=s["color"],
                 linewidth=2, markersize=7, label=label, markeredgecolor='white',
                 markeredgewidth=0.8)
        has_errs = any(e is not None for e in vstd_raw)
        if has_errs:
            ve = [e * scale if e is not None else 0 for e in vstd_raw]
            lo = [t - e for t, e in zip(vt, ve)]
            hi = [t + e for t, e in zip(vt, ve)]
            ax1.fill_between(vg, lo, hi, color=s["color"], alpha=0.12)

    # Optional reference slope (--ref-slope N)
    if args.ref_slope is not None:
        exp = args.ref_slope
        all_positive = [t * scale for t in all_times if t > 0]
        if len(grids) >= 2 and all_positive:
            x0, x1 = grids[0], grids[-1]
            x_mid  = math.exp((math.log(x0) + math.log(x1)) / 2)
            y_mid  = math.exp(sum(math.log(t) for t in all_positive) / len(all_positive)) * 0.3
            C      = y_mid / x_mid ** exp
            ax1.plot([x0, x1], [C * x0 ** exp, C * x1 ** exp],
                     color='#999999', linestyle=':', linewidth=1.5, alpha=0.5,
                     label=f'$\\propto N^{{{exp:g}}}$')

    ax1.set_xlabel(grid_label, fontsize=12)
    ax1.set_ylabel(f'Field solver time ({unit})', fontsize=12)
    ax1.set_title('Field Solver Time vs Grid Size', fontsize=13, fontweight='bold')
    if args.loglog:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(plain_fmt)
    ax1.set_xticks(grids)
    ax1.xaxis.set_major_formatter(x_fmt)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

# ── Right panel: speedup vs GMRES ─────────────────────────────────────────
gmres_times  = solver_times.get("GMRES", [])
petsc_labels = [l for l in solver_labels if l != "GMRES"]

# Collect per-solver average improvements for the summary annotation
avg_improvements = {}

if gmres_times and petsc_labels:
    # Pre-compute improvement percentages and averages for all PETSc solvers
    solver_pcts = {}
    for label in petsc_labels:
        times = solver_times[label]
        pcts  = [(gt - pt) / gt * 100
                 if gt is not None and pt is not None and gt > 0 else 0
                 for gt, pt in zip(gmres_times, times)]
        solver_pcts[label] = pcts
        valid_pcts = [p for p in pcts if p != 0]
        if valid_pcts:
            avg_improvements[label] = sum(valid_pcts) / len(valid_pcts)

    if args.sort:
        # ── Sorted mode: group bars by solver, sorted by avg improvement ──
        sorted_labels = sorted(petsc_labels,
                               key=lambda l: avg_improvements.get(l, 0),
                               reverse=True)
        n_runs    = len(grids)
        group_gap = 1.5
        bar_width = 0.95

        for gi, label in enumerate(sorted_labels):
            pcts        = solver_pcts[label]
            s           = styles.get(label, default_style)
            group_start = gi * (n_runs + group_gap)
            x_bars      = [group_start + i for i in range(n_runs)]

            bars = ax2.bar(x_bars, pcts, width=bar_width, color=s["color"],
                           alpha=0.8, edgecolor='white', linewidth=0.5)
            # Percentage labels on bars
            for bar, pct in zip(bars, pcts):
                if abs(pct) > 0.5:
                    y  = pct + (0.4 if pct >= 0 else -0.4)
                    va = 'bottom' if pct >= 0 else 'top'
                    ax2.text(bar.get_x() + bar.get_width()/2, y,
                             f'{pct:.0f}%', ha='center', va=va,
                             fontsize=9, fontweight='bold')

        # Major ticks: solver names centered under each group
        group_centers = [gi * (n_runs + group_gap) + (n_runs - 1) / 2
                         for gi in range(len(sorted_labels))]
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(sorted_labels, fontsize=9)
        ax2.tick_params(axis='x', which='major', length=0, pad=15)

        # Minor ticks: run numbers under each bar
        all_bar_x = []
        all_bar_labels = []
        for gi in range(len(sorted_labels)):
            group_start = gi * (n_runs + group_gap)
            for i in range(n_runs):
                all_bar_x.append(group_start + i)
                all_bar_labels.append(str(i + 1))
        ax2.set_xticks(all_bar_x, minor=True)
        ax2.set_xticklabels(all_bar_labels, minor=True, fontsize=7,
                            color='#666666')
        ax2.tick_params(axis='x', which='minor', length=0, pad=2)

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Improvement vs GMRES (%)', fontsize=12)
        ax2.set_title('PETSc Improvement vs Built-in GMRES  (by solver)',
                      fontsize=13, fontweight='bold')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    else:
        # ── Default mode: group bars by grid size ─────────────────────────
        x_positions = list(range(len(grids)))
        bar_width   = 0.7 / len(petsc_labels)

        for j, label in enumerate(petsc_labels):
            pcts    = solver_pcts[label]
            s       = styles.get(label, default_style)
            offsets = [x + (j - len(petsc_labels)/2 + 0.5) * bar_width
                       for x in x_positions]
            bars    = ax2.bar(offsets, pcts, width=bar_width, color=s["color"],
                              alpha=0.8, edgecolor='white', linewidth=0.5,
                              label=label)
            for bar, pct in zip(bars, pcts):
                if abs(pct) > 0.5:
                    y  = pct + (0.3 if pct >= 0 else -0.3)
                    va = 'bottom' if pct >= 0 else 'top'
                    ax2.text(bar.get_x() + bar.get_width()/2, y,
                             f'{pct:.0f}%', ha='center', va=va,
                             fontsize=7, fontweight='bold')

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel(grid_label, fontsize=12)
        ax2.set_ylabel('Improvement vs GMRES (%)', fontsize=12)
        ax2.set_title('PETSc Improvement vs Built-in GMRES',
                      fontsize=13, fontweight='bold')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(tick_labels, fontsize=9)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax2.legend(fontsize=9)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

# ── Bottom panel ──────────────────────────────────────────────────────────
ax3 = None

if has_profile:
    # ── Per-cycle timing profile from profile CSVs ─────────────────────
    ax3 = fig.add_subplot(gs[1, :])

    component_styles = [
        ('field_solver_s',     'Field solver',     '#4477AA'),
        ('particle_mover_s',   'Particle mover',   '#EE6677'),
        ('moment_gatherer_s',  'Moment gatherer',  '#228833'),
    ]
    solver_linestyles = ['-', '--', '-.', ':']

    for si, pf in enumerate(profile_files):
        base = os.path.splitext(os.path.basename(pf))[0]
        # "profile_PETSc_gmres_100x100x1" → remove prefix and grid suffix
        parts = base[len("profile_"):]
        label = re.sub(r'_\d+x\d+x\d+$', '', parts)

        ls = solver_linestyles[si % len(solver_linestyles)]

        with open(pf, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        cycles_col = [int(r['cycle']) for r in rows]

        for col, comp_name, color in component_styles:
            if col not in rows[0]:
                continue
            vals = [float(r[col]) for r in rows]
            plot_cycles, plot_vals = cycles_col, vals
            if args.profile_smooth and args.profile_smooth > 1:
                n = args.profile_smooth
                plot_cycles, plot_vals = [], []
                for i in range(0, len(vals), n):
                    plot_vals.append(sum(vals[i:i+n]) / len(vals[i:i+n]))
                    plot_cycles.append(sum(cycles_col[i:i+n]) / len(cycles_col[i:i+n]))
            ax3.plot(plot_cycles, plot_vals, color=color, linestyle=ls,
                     linewidth=1.5, alpha=0.85,
                     label=f'{label} \u2014 {comp_name}')

    ax3.set_xlabel('Cycle', fontsize=12)
    ax3.set_ylabel('Time per cycle (s)', fontsize=12)
    grid_tag = f'{max_grid}\u00d7{max_grid}' if profile_files else ''
    ax3.set_title(f'Per-Cycle Timing Profile  ({grid_tag} grid)',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

elif has_breakdown:
    # ── Component time breakdown (PROFILING build only) ────────────────
    ax3   = fig.add_subplot(gs[1, :])
    x_pos = list(range(len(bd_grids)))
    bar_w = 0.55

    fracs_field, fracs_moments, fracs_mover = [], [], []
    for ft, mt, pt in zip(bd_field, bd_moments, bd_mover):
        total = sum(v for v in [ft, mt, pt] if v is not None)
        if total > 0:
            fracs_field.append((ft or 0) / total)
            fracs_moments.append((mt or 0) / total)
            fracs_mover.append((pt or 0) / total)
        else:
            fracs_field.append(0); fracs_moments.append(0); fracs_mover.append(0)

    bottom_f = fracs_moments
    bottom_p = [m + f for m, f in zip(fracs_moments, fracs_field)]

    ax3.bar(x_pos, fracs_moments, bar_w, label='Moments (CalculateMoments)',
            color='#228833', alpha=0.85, edgecolor='white')
    ax3.bar(x_pos, fracs_field,   bar_w, bottom=bottom_f,
            label='Field solver (FieldSolver)',
            color='#4477AA', alpha=0.85, edgecolor='white')
    ax3.bar(x_pos, fracs_mover,   bar_w, bottom=bottom_p,
            label='Particle mover (ParticlesMover)',
            color='#EE6677', alpha=0.85, edgecolor='white')
    ax3.plot(x_pos, fracs_field, 'o--', color='#004488',
             linewidth=1.5, markersize=5, zorder=5, label='Field solver fraction')

    bd_tick_labels = [f'{g}\u00d7{nzc}' if nzc > 1 else str(g) for g in bd_grids]
    ax3.set_xlabel(grid_label, fontsize=12)
    ax3.set_ylabel('Fraction of component time', fontsize=12)
    ax3.set_title('Time Breakdown by Component  (GMRES baseline)',
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bd_tick_labels, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

# ── Summary annotation ────────────────────────────────────────────────────
if args.sort and avg_improvements:
    grid_dim = f'N × N × {nzc}' if nzc > 1 else 'N × N'
    run_legend = f'Runs ({grid_dim} cells):  ' + '   '.join(
        f'{i+1} = {tick_labels[i]}' for i in range(len(grids)))
    ax2.text(0.5, -0.12, run_legend, ha='center', va='top',
             transform=ax2.transAxes, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                       edgecolor='#cccccc', alpha=0.9))

if avg_improvements:
    # Sort by descending average improvement
    sorted_avg = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
    # Build colored summary using fig.text for the label, then ax.annotate per solver
    import matplotlib.transforms as mtransforms
    summary_y = -0.05
    fig.text(0.5, summary_y, '', ha='center', va='top')  # anchor point
    # Use a single-line approach with colored text segments via fig.transFigure
    from matplotlib.offsetbox import AnchoredText, HPacker, TextArea
    text_areas = [TextArea('Avg improvement:  ', textprops=dict(
        fontsize=11, fontfamily='monospace', fontweight='bold', color='#333333'))]
    for i, (lbl, avg) in enumerate(sorted_avg):
        s = styles.get(lbl, default_style)
        sep = '    ' if i < len(sorted_avg) - 1 else ''
        text_areas.append(TextArea(f'{lbl}: {avg:+.1f}%{sep}', textprops=dict(
            fontsize=11, fontfamily='monospace', fontweight='bold', color=s["color"])))
    packed = HPacker(children=text_areas, align='baseline', pad=0, sep=0)
    ab = matplotlib.offsetbox.AnchoredOffsetbox(
        loc='upper center', child=packed, frameon=True,
        bbox_to_anchor=(0.5, summary_y), bbox_transform=fig.transFigure,
        pad=0.4, borderpad=0.4,
        prop=dict(size=11))
    ab.patch.set_boxstyle('round,pad=0.4')
    ab.patch.set_facecolor('#f0f0f0')
    ab.patch.set_edgecolor('#cccccc')
    ab.patch.set_alpha(0.9)
    fig.add_artist(ab)

# ── Save ───────────────────────────────────────────────────────────────────

# Re-apply after tight_layout (log-scale axes can lose custom formatters)
if not single_row:
    ax1.xaxis.set_major_formatter(x_fmt)
    if args.loglog:
        ax1.yaxis.set_major_formatter(plain_fmt)
axes_to_rotate = [ax1, ax2] + ([ax3] if ax3 is not None else [])
if tick_rotation:
    for ax in axes_to_rotate:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(tick_rotation)
            lbl.set_ha(tick_ha)

plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Plot saved to: {plot_path}")
