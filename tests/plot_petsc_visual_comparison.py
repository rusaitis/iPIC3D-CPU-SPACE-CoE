#!/usr/bin/env python3
"""
plot_petsc_visual_comparison.py — Visual comparison of field solver outputs.

Compares electromagnetic field data produced by different solvers (e.g. GMRES vs PETSc)
to verify they produce equivalent results.

Usage:
    python3 plot_petsc_visual_comparison.py --gmres <hdf5_dir> --petsc <hdf5_dir> [--petsc <hdf5_dir> ...]

The directories should contain the HDF5 field output from iPIC3D runs.
"""
import argparse
import sys

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--gmres', required=True,
                    help='Path to GMRES field output directory')
parser.add_argument('--petsc', action='append', required=True,
                    help='Path to PETSc field output directory (repeatable)')
parser.add_argument('--cycle', type=int, default=-1,
                    help='Cycle to compare (-1 = last available)')
args = parser.parse_args()

# TODO: Load HDF5 field data, compute differences, plot comparisons
print(f"GMRES output: {args.gmres}")
for p in args.petsc:
    print(f"PETSc output: {p}")
print("Visual comparison not yet implemented.")
