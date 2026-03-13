#!/usr/bin/env python3
"""
plot_timing_common.py — Shared CSV loading, data structures, and rendering
helpers for iPIC3D solver comparison plots (plot_single_grid / plot_scaling).
"""

import csv
import dataclasses
import os
import re

from plot_utils import parse_conserved_quantities, discover_profile_files


# ── Constants ────────────────────────────────────────────────────────────

METADATA_COLS = {
    'grid_size', 'np', 'cycles', 'nzc', 'ref_solver',
    # New column names
    'ref_moments_s', 'ref_mover_s',
    'ref_field_pct', 'ref_moments_pct', 'ref_mover_pct',
    # Legacy column names (backwards compat)
    'gmres_moments_s', 'gmres_mover_s',
    'gmres_field_pct', 'gmres_moments_pct', 'gmres_mover_pct',
}


# ── Data container ───────────────────────────────────────────────────────

@dataclasses.dataclass
class ResultsData:
    """All data loaded from a results CSV and associated profile/energy files."""
    csv_path: str
    solver_labels: list
    grids: list
    grid_strings: list
    solver_times: dict       # label -> [float|None, ...]
    solver_stds: dict        # label -> [float|None, ...]
    nzc: int
    np_: str
    cycles: str
    single_row: bool
    # Breakdown data (PROFILING builds)
    bd_grids: list
    bd_field: list
    bd_moments: list
    bd_mover: list
    # Profile data
    profile_dir: str
    profile_files: list
    max_grid: int
    # Energy data (per-solver ConservedQuantities)
    energy_data: dict        # label -> dict
    profile_labels: list     # solver labels matching profile_files
    ref_solver: str = ""     # reference solver label (from CSV or first solver)

    @property
    def ref_label(self):
        """Label of the reference solver (first solver if not explicit)."""
        return self.ref_solver or (self.solver_labels[0] if self.solver_labels else "GMRES")

    @property
    def ref_times(self):
        """Timing data for the reference solver."""
        return self.solver_times.get(self.ref_label, [])

    @property
    def comparison_labels(self):
        """Solver labels excluding the reference."""
        ref = self.ref_label
        return [l for l in self.solver_labels if l != ref]

    # Legacy aliases for backwards compatibility
    @property
    def gmres_times(self):
        return self.ref_times

    @property
    def petsc_labels(self):
        return self.comparison_labels

    @property
    def all_times(self):
        return [t for tlist in self.solver_times.values()
                for t in tlist if t is not None]

    @property
    def has_breakdown(self):
        return bool(self.bd_grids) and any(m is not None for m in self.bd_moments)

    @property
    def has_profile(self):
        return self.single_row and bool(self.profile_files)

    @property
    def has_energy(self):
        return self.single_row and bool(self.energy_data)

    @property
    def tick_labels(self):
        if self.single_row:
            return self.grid_strings
        nzc = self.nzc
        return [f'{g}\u00d7{nzc}' if nzc > 1 else str(g) for g in self.grids]

    @property
    def grid_label(self):
        return (f'Grid size (N \u00d7 N \u00d7 {self.nzc})'
                if self.nzc > 1 else 'Grid size (N \u00d7 N)')


# ── CSV loading ──────────────────────────────────────────────────────────

def load_results_csv(csv_path):
    """Load all data from a results.csv and discover associated files.

    Returns a ResultsData instance with all timing, breakdown, profile,
    and energy data populated.
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        header_fields = reader.fieldnames or []
        first_row = next(reader, {})

    # Solver labels: columns that are not metadata and not suffixed
    inferred_labels = [c for c in header_fields
                       if c not in METADATA_COLS
                       and not c.endswith(('_std', '_iters',
                                           '_converged', '_residual'))]

    solver_labels = (os.environ["SOLVER_LABEL_LIST"].split()
                     if "SOLVER_LABEL_LIST" in os.environ else inferred_labels)

    nzc    = int(os.environ.get("NZC",    first_row.get('nzc',    '1') or '1'))
    np_    =     os.environ.get("NP",     first_row.get('np',     '?') or '?')
    cycles =     os.environ.get("CYCLES", first_row.get('cycles', '?') or '?')

    # Reference solver: from env, CSV column, or default to first solver
    ref_solver = (os.environ.get("REF_SOLVER")
                  or first_row.get('ref_solver', '')
                  or (solver_labels[0] if solver_labels else ''))

    # ── Breakdown data ────────────────────────────────────────────────
    def _parse_floats(env_key):
        return [float(x) if x != "NA" else None
                for x in os.environ.get(env_key, "").split() if x]

    if "BD_GRIDS_LIST" in os.environ:
        bd_grids   = [int(x) for x in os.environ.get("BD_GRIDS_LIST", "").split() if x]
        bd_field   = _parse_floats("BD_FIELD_LIST")
        bd_moments = _parse_floats("BD_MOMENTS_LIST")
        bd_mover   = _parse_floats("BD_MOVER_LIST")
    else:
        bd_grids, bd_field, bd_moments, bd_mover = [], [], [], []
        ref_label_bd = ref_solver or (solver_labels[0] if solver_labels else "GMRES")
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                gs = row['grid_size']
                try:
                    bd_grids.append(int(gs))
                except ValueError:
                    bd_grids.append(len(bd_grids))
                f_val = row.get(ref_label_bd,       'NA')
                # Support both new (ref_*) and legacy (gmres_*) column names
                m_val = row.get('ref_moments_s', row.get('gmres_moments_s', 'NA'))
                p_val = row.get('ref_mover_s',   row.get('gmres_mover_s',   'NA'))
                bd_field.append(  float(f_val) if f_val not in ('NA', '') else None)
                bd_moments.append(float(m_val) if m_val not in ('NA', '') else None)
                bd_mover.append(  float(p_val) if p_val not in ('NA', '') else None)

    # ── Main timing data ──────────────────────────────────────────────
    grids        = []
    grid_strings = []
    solver_times = {label: [] for label in solver_labels}
    solver_stds  = {label: [] for label in solver_labels}

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            gs = row["grid_size"]
            grid_strings.append(gs)
            try:
                grids.append(int(gs))
            except ValueError:
                grids.append(len(grids))
            for label in solver_labels:
                val     = row.get(label,           "NA")
                std_val = row.get(f"{label}_std",  "NA")
                solver_times[label].append(
                    float(val) if val not in ("NA", "") else None)
                solver_stds[label].append(
                    float(std_val) if std_val not in ("NA", "") else None)

    single_row = len(grids) == 1

    # ── Profile & energy discovery ────────────────────────────────────
    profile_dir = os.environ.get("PROFILE_DIR", os.path.dirname(csv_path))
    profile_files, max_grid = discover_profile_files(profile_dir)

    energy_data = {}
    profile_labels = []

    for pf in profile_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        parts = base[len("profile_"):]
        label = re.sub(r'_\d+x\d+x\d+$', '', parts)
        grid_suffix = re.search(r'_(\d+x\d+x\d+)$', parts)
        run_dir_name = (f"{label}_{grid_suffix.group(1)}"
                        if grid_suffix else parts)
        run_dir = os.path.join(os.path.dirname(pf), run_dir_name)
        profile_labels.append(label)
        cq = parse_conserved_quantities(run_dir)
        if cq:
            energy_data[label] = cq

    return ResultsData(
        csv_path=csv_path,
        solver_labels=solver_labels,
        grids=grids,
        grid_strings=grid_strings,
        solver_times=solver_times,
        solver_stds=solver_stds,
        nzc=nzc,
        np_=np_,
        cycles=cycles,
        single_row=single_row,
        bd_grids=bd_grids,
        bd_field=bd_field,
        bd_moments=bd_moments,
        bd_mover=bd_mover,
        profile_dir=profile_dir,
        profile_files=profile_files,
        max_grid=max_grid,
        energy_data=energy_data,
        profile_labels=profile_labels,
        ref_solver=ref_solver,
    )


# ── Rendering utilities ──────────────────────────────────────────────────

def bar_label_fontsize(n_bars: int) -> int:
    """Scale bar-label font size down when bars are crowded."""
    return max(8, min(20, 60 // max(n_bars, 1)))


def darken_color(color, factor=0.45):
    """Return a darker version of a matplotlib color."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def auto_time_unit(all_times):
    """Pick time unit and scale factor based on magnitudes.

    Returns (unit_label, scale_factor).
    """
    if all_times and max(all_times) < 1.0:
        return 'ms', 1000.0
    elif all_times and max(all_times) >= 120.0:
        return 'min', 1 / 60.0
    return 's', 1.0


def compute_improvements(data):
    """Compute per-solver improvement percentages vs the reference solver.

    Returns (solver_pcts, avg_improvements) where:
      solver_pcts:      dict label -> [pct, ...] per grid
      avg_improvements: dict label -> float average improvement
    """
    ref_times = data.ref_times
    comparison_labels = data.comparison_labels

    if not ref_times or not comparison_labels:
        return {}, {}

    solver_pcts = {}
    avg_improvements = {}
    for label in comparison_labels:
        times = data.solver_times[label]
        pcts = [(rt - ct) / rt * 100
                if rt is not None and ct is not None and rt > 0 else 0
                for rt, ct in zip(ref_times, times)]
        solver_pcts[label] = pcts
        valid_pcts = [p for p in pcts if p != 0]
        if valid_pcts:
            avg_improvements[label] = sum(valid_pcts) / len(valid_pcts)

    return solver_pcts, avg_improvements


def render_improvement_bars(ax, data, solver_pcts, avg_improvements,
                            sort, theme):
    """Render improvement bar chart vs the reference solver on the given axis.

    Handles both sorted-by-solver and grouped-by-grid modes.
    """
    from matplotlib.ticker import FuncFormatter
    from plot_theme import get_solver_style

    comparison_labels = data.comparison_labels
    ref_label = data.ref_label
    grids = data.grids
    tick_labels = data.tick_labels

    if sort:
        # ── Sorted mode: group bars by solver ────────────────────────
        sorted_labels = sorted(comparison_labels,
                               key=lambda l: avg_improvements.get(l, 0),
                               reverse=True)
        n_runs    = len(grids)
        group_gap = 1.5
        bar_width = 0.95
        n_bars    = len(sorted_labels) * len(grids)
        label_fs  = bar_label_fontsize(n_bars)

        for gi, label in enumerate(sorted_labels):
            pcts        = solver_pcts[label]
            s           = get_solver_style(label)
            group_start = gi * (n_runs + group_gap)
            x_bars      = [group_start + i for i in range(n_runs)]

            bars = ax.bar(x_bars, pcts, width=bar_width, color=s["color"],
                          alpha=0.8, edgecolor='white', linewidth=0.5)
            for bar, pct in zip(bars, pcts):
                if abs(pct) > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2, pct / 2,
                            f'{pct:.0f}%', ha='center', va='center',
                            fontsize=label_fs, fontweight='heavy',
                            color=darken_color(s["color"]))

        # Major ticks: solver names centered under each group
        group_centers = [gi * (n_runs + group_gap) + (n_runs - 1) / 2
                         for gi in range(len(sorted_labels))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(sorted_labels, fontsize=9)
        ax.tick_params(axis='x', which='major', length=0, pad=15)

        # Minor ticks: run numbers under each bar
        all_bar_x = []
        all_bar_labels = []
        for gi in range(len(sorted_labels)):
            group_start = gi * (n_runs + group_gap)
            for i in range(n_runs):
                all_bar_x.append(group_start + i)
                all_bar_labels.append(str(i + 1))
        ax.set_xticks(all_bar_x, minor=True)
        ax.set_xticklabels(all_bar_labels, minor=True, fontsize=7,
                           color=theme.text_secondary)
        ax.tick_params(axis='x', which='minor', length=0, pad=2)

        ax.axhline(y=0, color=theme.zero_line_color, linestyle='--',
                    linewidth=1, alpha=0.5)
        ax.set_ylabel(f'Improvement vs {ref_label} (%)', fontsize=14)
        ax.set_title(f'Improvement vs {ref_label}  (by solver)',
                     fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    else:
        # ── Default mode: group bars by grid size ────────────────────
        x_positions = list(range(len(grids)))
        bar_width   = 0.7 / len(comparison_labels)
        n_bars      = len(comparison_labels) * len(grids)
        label_fs    = bar_label_fontsize(n_bars)

        for j, label in enumerate(comparison_labels):
            pcts    = solver_pcts[label]
            s       = get_solver_style(label)
            offsets = [x + (j - len(comparison_labels)/2 + 0.5) * bar_width
                       for x in x_positions]
            bars    = ax.bar(offsets, pcts, width=bar_width, color=s["color"],
                             alpha=0.8, edgecolor=theme.bar_edge, linewidth=0.5,
                             label=label)
            for bar, pct in zip(bars, pcts):
                if abs(pct) > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2, pct / 2,
                            f'{pct:.0f}%', ha='center', va='center',
                            fontsize=label_fs, fontweight='heavy',
                            color=darken_color(s["color"]))

        ax.axhline(y=0, color=theme.zero_line_color, linestyle='--',
                    linewidth=1, alpha=0.5)
        ax.set_xlabel(data.grid_label, fontsize=12)
        ax.set_ylabel(f'Improvement vs {ref_label} (%)', fontsize=14)
        ax.set_title(f'Improvement vs {ref_label}',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def render_summary_annotation(fig, gs, ax2, n_rows, data, avg_improvements,
                              sort, theme):
    """Render the 'Average improvement' annotation between row 0 and row 1.

    Also renders the run-legend text on ax2 when in --sort mode.
    """
    import matplotlib
    from matplotlib.offsetbox import HPacker, TextArea
    from plot_theme import get_solver_style

    nzc = data.nzc
    grids = data.grids
    tick_labels = data.tick_labels

    if sort and avg_improvements:
        grid_dim = f'N \u00d7 N \u00d7 {nzc}' if nzc > 1 else 'N \u00d7 N'
        run_legend = f'Runs ({grid_dim} cells):  ' + '   '.join(
            f'{i+1} = {tick_labels[i]}' for i in range(len(grids)))
        ax2.text(0.5, -0.12, run_legend, ha='center', va='top',
                 transform=ax2.transAxes, fontsize=9, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=theme.box_face,
                           edgecolor=theme.box_edge, alpha=theme.box_alpha))

    if avg_improvements:
        sorted_avg = sorted(avg_improvements.items(),
                            key=lambda x: x[1], reverse=True)
        row_bottoms, row_tops, _, _ = gs.get_grid_positions(fig)
        row0_bottom = row_bottoms[0]
        row1_top = row_tops[1] if n_rows > 1 else row0_bottom - 0.05
        summary_y = 0.5 * (row0_bottom + row1_top) + 0.015

        text_areas = [TextArea('Average improvement:  ', textprops=dict(
            fontsize=11, fontfamily='monospace', fontweight='bold',
            color=theme.text_color))]
        for i, (lbl, avg) in enumerate(sorted_avg):
            s = get_solver_style(lbl)
            sep = '    ' if i < len(sorted_avg) - 1 else ''
            text_areas.append(TextArea(
                f'{lbl}: {avg:+.1f}%{sep}', textprops=dict(
                    fontsize=11, fontfamily='monospace', fontweight='bold',
                    color=s["color"])))
        packed = HPacker(children=text_areas, align='baseline', pad=0, sep=0)
        ab = matplotlib.offsetbox.AnchoredOffsetbox(
            loc='upper center', child=packed, frameon=True,
            bbox_to_anchor=(0.5, summary_y), bbox_transform=fig.transFigure,
            pad=0.4, borderpad=0.4, prop=dict(size=11))
        ab.patch.set_boxstyle('round,pad=0.4')
        ab.patch.set_facecolor(theme.box_face)
        ab.patch.set_edgecolor(theme.box_edge)
        ab.patch.set_alpha(theme.box_alpha)
        fig.add_artist(ab)


def apply_tick_rotation(axes, grids):
    """Apply 45-degree tick rotation when there are many grid sizes."""
    tick_rotation = 45 if len(grids) > 6 else 0
    tick_ha = 'right' if tick_rotation else 'center'
    if tick_rotation:
        for ax in axes:
            if ax is not None:
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(tick_rotation)
                    lbl.set_ha(tick_ha)


# ── I/O path resolution ─────────────────────────────────────────────────

def resolve_csv_path(args):
    """Resolve the CSV path from CLI args or env vars.

    Fallback order: CLI arg → CSV_FILE env → tests/results.csv → tests/test_output/results.csv.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if getattr(args, 'csv', None) is not None:
        return args.csv
    elif "CSV_FILE" in os.environ:
        return os.environ["CSV_FILE"]
    # Prefer results.csv next to script; fall back to test_output/
    default = os.path.join(script_dir, "results.csv")
    if os.path.isfile(default):
        return default
    alt = os.path.join(script_dir, "test_output", "results.csv")
    if os.path.isfile(alt):
        return alt
    return default  # return original path so error message makes sense


def resolve_plot_path(args, csv_path):
    """Resolve the output PNG path from CLI args or env vars."""
    if getattr(args, 'output', None) is not None:
        return args.output
    elif "PLOT_FILE" in os.environ:
        return os.environ["PLOT_FILE"]
    return os.path.splitext(csv_path)[0] + ".png"
