"""
log_patterns.py — Shared regex patterns for parsing iPIC3D simulation logs.

Single source of truth for convergence and timing patterns used by:
  extract_profile.py, validate.py, summarize.py, plot_ksp_diagnostics.py,
  and tests/test.sh (via import).

Patterns must match the actual output from GMRES.cpp and PETSC.cpp.
"""

import re

# ── Convergence iteration patterns ───────────────────────────────────────
# Each entry: (regex, iter_extractor_fn)
# iter_extractor takes a re.Match and returns the iteration count.

ITER_PATTERNS = [
    # GMRES: "GMRES converged at restart R; iteration K with error: E"
    (re.compile(r'GMRES converged at restart (\d+); iteration (\d+) with error:'),
     lambda m: int(m.group(1)) * 20 + int(m.group(2))),
    # GMRES zero-iter: "GMRES converged without iterations"
    (re.compile(r'GMRES converged without iterations'),
     lambda m: 0),
    # PETSc success: "PETSc KSP converged: cycle=C iterations=I residual=R"
    (re.compile(r'PETSc KSP converged: cycle=\d+ iterations=(\d+) residual=([\d.eE+\-]+)'),
     lambda m: int(m.group(1))),
    # PETSc failure: "WARNING: PETSc KSP did NOT converge: cycle=C reason=R iterations=I residual=R"
    (re.compile(r'PETSc KSP did NOT converge: cycle=\d+ reason=\S+ iterations=(\d+) residual=([\d.eE+\-]+)'),
     lambda m: int(m.group(1))),
]

# ── Convergence success/failure patterns (for pass/fail checks) ──────────
# Simple patterns that just detect convergence vs failure (no extraction).

CONV_SUCCESS = [
    re.compile(r'GMRES converged at restart \d+; iteration \d+'),
    re.compile(r'GMRES converged without iterations'),
    re.compile(r'PETSc KSP converged:'),
]

CONV_FAIL = [
    re.compile(r'GMRES not converged'),
    re.compile(r'PETSc KSP did NOT converge'),
]

# ── Timing patterns ──────────────────────────────────────────────────────
# Cumulative timing counters printed each cycle by iPIC3D.

TIMING_PATTERNS = [
    ('field',  re.compile(r'Field solver\s*:\s*([\d.eE+\-]+)')),
    ('mover',  re.compile(r'Particle mover\s*:\s*([\d.eE+\-]+)')),
    ('moment', re.compile(r'Moment gatherer\s*:\s*([\d.eE+\-]+)')),
    ('write',  re.compile(r'Write data\s*:\s*([\d.eE+\-]+)')),
]

# ── Full convergence patterns with residual (for detailed parsing) ───────
# Each entry: (name, success_regex, fail_regex,
#              success_iter_extractor, success_residual_group,
#              fail_iter_extractor, fail_residual_group)

CONVERGENCE_PATTERNS = [
    ("GMRES",
     re.compile(r'GMRES converged at restart (\d+); iteration (\d+) with error:\s*([\d.eE+\-]+)'),
     re.compile(r'GMRES not converged.*Final error:\s*([\d.eE+\-]+)'),
     lambda m: int(m.group(1)) * 20 + int(m.group(2)),
     3,     # residual group index in success regex
     None,  # no iter count for GMRES failure
     1),    # residual group index in fail regex
    ("PETSc",
     re.compile(r'PETSc KSP converged: cycle=\d+ iterations=(\d+) residual=([\d.eE+\-]+)'),
     re.compile(r'PETSc KSP did NOT converge: cycle=\d+ reason=\S+ iterations=(\d+) residual=([\d.eE+\-]+)'),
     lambda m: int(m.group(1)),
     2,
     lambda m: int(m.group(1)),
     2),
]

# Special case: GMRES zero-iteration convergence
GMRES_ZERO_ITER = re.compile(r'GMRES converged without iterations')
