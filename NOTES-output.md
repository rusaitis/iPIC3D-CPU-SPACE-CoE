# iPIC3D Output Reference

## Output Methods

iPIC3D supports three output backends, selected via `WriteMethod` in the input file:

| Method | Value | Description |
|--------|-------|-------------|
| **Parallel HDF5** | `phdf5` | Each output cycle writes separate `.h5` files using MPI-IO collective writes. All ranks contribute to a single globally-assembled dataset per file. Recommended for production runs. |
| **Serial HDF5** | `shdf5` | Each MPI rank writes its own `.hdf` file (`proc0.hdf`, `proc1.hdf`, ...) containing its local subdomain. Datasets are indexed by cycle (`/fields/Bx/cycle_10`). Simpler but requires post-processing to reassemble the global domain. |
| **VTK** | `pvtk` / `nbcvtk` | MPI collective binary VTK files (`.vtk`). Limited variable selection (B, E, Je, Ji, rho for fields; rho, PXX–PZZ for moments via `MomentsOutputTag`). Legacy; not recommended for new work. |

## Output Precision

Controlled by `output_data_precision` (default: `DOUBLE`):

| Value | HDF5 type | Size |
|-------|-----------|------|
| `DOUBLE` | `H5T_IEEE_F64LE` | 8 bytes/value |
| `SINGLE` | `H5T_IEEE_F32LE` | 4 bytes/value |

Applies to both field/moment and particle outputs. VTK output is always single precision.

---

## Output Cycle Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FieldOutputCycle` | 100 | Write fields and moments every N cycles (<=0 disables) |
| `ParticlesOutputCycle` | 0 | Write particle data every N cycles (<=0 disables) |
| `ParticlesDownsampleOutputCycle` | 1 | Write downsampled particles every N cycles (<=0 disables) |
| `ParticlesDownsampleFactor` | 1 | Keep every Nth particle (1 = no downsampling) |
| `DiagnosticsOutputCycle` | `FieldOutputCycle` | Write text diagnostics every N cycles |
| `RestartOutputCycle` | 5000 | Write restart files every N cycles (final cycle always written) |

---

## Field and Moment Output Variables

Controlled by `FieldOutputTag` — a space-separated string of tags. If a tag is not listed, that data is **not written**. The `+` signs are optional separators (ignored by the parser; only whitespace-delimited tokens matter).

### Electromagnetic Fields

| Tag | Variables | Grid location | Description |
|-----|-----------|---------------|-------------|
| `B` | Bx, By, Bz | Nodes | Magnetic field (internal, excludes external contribution) |
| `E` | Ex, Ey, Ez | Nodes | Electric field |
| `B_c` | Bxc, Byc, Bzc | Cell centres | Magnetic field interpolated to cell centres |
| `B_ext` | Bx_ext, By_ext, Bz_ext | Nodes | External magnetic field |
| `phi` | phi | Cell centres | Electrostatic potential |
| `divergence` | rhoc_avg, div_E_avg | Cell centres | Charge density and div(E) averages (Gauss's law diagnostic) |

### Moments (per species)

| Tag | Variables | Grid location | Description | Prerequisite |
|-----|-----------|---------------|-------------|-------------|
| `rho` | rho | Nodes | Total charge density (all species summed) | — |
| `rho_s` | rho | Nodes | Charge density per species | — |
| `J` | Jx, Jy, Jz | Nodes | Total current density (all species summed) | — |
| `J_s` | Jx, Jy, Jz | Nodes | Current density per species | — |
| `pressure` | pXX, pXY, pXZ, pYY, pYZ, pZZ | Nodes | Pressure tensor per species (6 independent components of the symmetric 2nd-order moment) | Always computed |
| `E_flux` | EFx, EFy, EFz | Nodes | Energy flux per species (3rd velocity moment contracted to vector) | Always computed |
| `H_flux` | Qxxx, Qxxy, Qxyy, Qxzz, Qyyy, Qyzz, Qzzz, Qxyz, Qxxz, Qyyz | Nodes | Heat flux tensor per species (10 independent components of the fully symmetric 3rd-order moment) | Requires `SaveHeatFluxTensor = true` in input file |

### Energies (serial HDF5 only)

These are written by the `shdf5` backend only, **not** by `phdf5`:

| Tag | Path | Description |
|-----|------|-------------|
| `K_energy` | `/energy/kinetic/species_N/cycle_C` | Kinetic energy per species (scalar) |
| `B_energy` | `/energy/magnetic/cycle_C` | Magnetic field energy (scalar) |
| `E_energy` | `/energy/electric/cycle_C` | Electric field energy (scalar) |

### Example `FieldOutputTag` values

```ini
# Minimal (E and B only)
FieldOutputTag = B + E

# Standard (fields + per-species moments)
FieldOutputTag = B + E + rho + J + J_s + rho_s

# Full moments including pressure tensor and energy flux
FieldOutputTag = B + E + rho + J + J_s + rho_s + pressure + E_flux

# Everything (requires SaveHeatFluxTensor = true for H_flux)
FieldOutputTag = B + E + rho + J + J_s + rho_s + pressure + E_flux + H_flux
```

---

## Particle Output Variables

Controlled by `ParticlesOutputTag`:

| Tag | Variables | Description |
|-----|-----------|-------------|
| `position` | x, y, z | Particle positions |
| `velocity` | u, v, w | Particle velocities |
| `q` | q | Macro-particle charge (constant within a species; one scalar value) |
| `ID` | ID | Particle tracking ID (integer) |

### Downsampled Particles

Controlled by `ParticlesDownsampleOutputTag`:

| Tag | Variables | Description |
|-----|-----------|-------------|
| `position_DS` | x, y, z | Downsampled positions (every `ParticlesDownsampleFactor`-th particle) |
| `velocity_DS` | u, v, w | Downsampled velocities |
| `q_DS` | q | Downsampled charges |

---

## File Layout

### Parallel HDF5 (`phdf5`)

One directory per output cycle, with separate files per variable category and species:

```
data_dir/
├── Fields_00010/
│   ├── E_00010.h5          # Ex, Ey, Ez
│   └── B_00010.h5          # Bx, By, Bz
├── Moments_00010/
│   ├── rho_species_0_00010.h5
│   ├── J_species_0_00010.h5
│   ├── Pressure_species_0_00010.h5    # pXX, pXY, pXZ, pYY, pYZ, pZZ
│   ├── E_flux_species_0_00010.h5      # EFx, EFy, EFz
│   ├── H_flux_species_0_00010.h5      # Qxxx, ..., Qyyz (10 components)
│   ├── rho_species_1_00010.h5
│   └── ...                            # one file per species per variable
├── Particles_00010/
│   ├── species_0_00010.h5   # position (N×3), velocity (N×3), q (1×1)
│   ├── species_1_00010.h5
│   └── ...
├── ConservedQuantities.txt
├── SpeciesQuantities.txt
├── SimulationData.txt
├── restart0.hdf ... restartN.hdf
└── <input_file>.inp          # if copied manually
```

**HDF5 internal structure (phdf5):**

Fields file (e.g. `E_00010.h5`):
```
/Fields/Ex    dataset (NX+1, NY+1, NZ+1)   float64 or float32
/Fields/Ey    dataset (NX+1, NY+1, NZ+1)
/Fields/Ez    dataset (NX+1, NY+1, NZ+1)
```

Moments file (e.g. `Pressure_species_0_00010.h5`):
```
/Moments/species_0/pXX   dataset (NX+1, NY+1, NZ+1)
/Moments/species_0/pXY   ...
/Moments/species_0/pXZ   ...
/Moments/species_0/pYY   ...
/Moments/species_0/pYZ   ...
/Moments/species_0/pZZ   ...
```

Particles file (e.g. `species_0_00010.h5`):
```
/Particles/species_0/position   dataset (Nglobal, 3)    # [x, y, z] per particle
/Particles/species_0/velocity   dataset (Nglobal, 3)    # [u, v, w] per particle
/Particles/species_0/q          dataset (1, 1)           # scalar macro-particle charge
```

### Serial HDF5 (`shdf5`)

One file per MPI rank, all cycles accumulated inside:

```
data_dir/
├── proc0.hdf
├── proc1.hdf
├── ...
└── procN.hdf
```

Internal structure per file:
```
/collective/...                              # simulation parameters (Lx, Ly, Nxc, etc.)
/topology/...                                # MPI topology (XLEN, YLEN, coordinates, neighbours)
/fields/Bx/cycle_0       (nx, ny, nz)       # local subdomain, no ghost cells
/fields/Bx/cycle_10      ...
/fields/By/cycle_0       ...
/moments/Jx/cycle_0      ...
/moments/species_0/Jx/cycle_0    ...
/moments/species_0/rho/cycle_0   ...
/moments/species_0/pXX/cycle_0   ...
/particles/species_0/x/cycle_0   (nop,)      # 1D array of local particles
/particles/species_0/u/cycle_0   ...
/particles/species_0/q/cycle_0   ...
/energy/kinetic/species_0/cycle_0  (scalar)  # only in shdf5
/energy/magnetic/cycle_0           (scalar)
```

---

## Array Ordering and Grid Conventions

### Grid dimensions

- **Nodes** (NXN, NYN, NZN) = (nxc + 3, nyc + 3, nzc + 3) including 1 ghost cell on each side
- **Cells** (NXC, NYC, NZC) = (nxc + 2, nyc + 2, nzc + 2) including ghost cells
- **Output** excludes ghost cells: node-based data has shape `(nxc+1, nyc+1, nzc+1)`, cell-centred data has shape `(nxc, nyc, nzc)`

### Memory layout

Data is stored in **row-major (C) order** with the loop nesting `for i (x) → for j (y) → for k (z)`, so **z varies fastest** in the linear buffer:

```
flat_index = i * (ny * nz) + j * nz + k
```

HDF5 datasets inherit this ordering. When reading with Python (h5py/numpy), the array shape is `(nx, ny, nz)` with `array[ix, iy, iz]`.

For 2D simulations (nzc = 1), the output shape is `(nx, ny, 2)` at nodes — the two z-planes represent the single interior node plus the periodic image.

### Coordinate system

- Origin at (0, 0, 0), domain extends to (Lx, Ly, Lz)
- Node positions: `x_i = i * dx`, where `dx = Lx / nxc`
- The grid is uniform Cartesian with spacing `(dx, dy, dz) = (Lx/nxc, Ly/nyc, Lz/nzc)`

---

## Units

All quantities are in **Gaussian-CGS units normalized to ion plasma parameters** (see `Documentation/units.txt`):

| Quantity | Normalized to | Symbol |
|----------|---------------|--------|
| Length | Ion skin depth | d_i = c/ω_pi |
| Time | Inverse ion plasma frequency | ω_pi⁻¹ |
| Velocity | Speed of light | c |
| Magnetic field | Reference field B₀ | (such that v_A = B₀/√(4πn₀m_i)) |
| Electric field | B₀ | |
| Charge density | n₀ q_i / (4π) | Note: the stored ρ = n·q/(4π), multiply by 4π for physical charge density |
| Pressure tensor | n₀ m_i c² | Components pXX etc. are the full 2nd velocity moment (not thermal only) |
| Current density | n₀ q_i c / (4π) | |
| Particle charge | Macro-particle charge q = (ρ_init × cell_volume) / Np_cell | |

**Important:** The charge density `rho_s` stored in the output is `n·q/(4π)`, consistent with Gaussian units where Gauss's law reads `∇·E = 4πρ`. The VTK output code multiplies by 4π before writing; the HDF5 output does **not**.

---

## Text Diagnostics

Written every `DiagnosticsOutputCycle` cycles to the output directory:

### `ConservedQuantities.txt`
Columns:
1. Cycle
2. Total electric field energy
3. Electric field energy (Ex)
4. Electric field energy (Ey)
5. Electric field energy (Ez)
6. Total magnetic field energy
7. Magnetic field energy (Bx)
8. Magnetic field energy (By)
9. Magnetic field energy (Bz)

### `SpeciesQuantities.txt`
One row per species per cycle. Columns:
1. Cycle
2. Species index
3. Momentum
4. Total kinetic energy
5. Bulk kinetic energy
6. Thermal kinetic energy

### `SimulationData.txt`
A human-readable summary of simulation parameters (grid, species, solver settings) written once at startup.

---

## Special Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SaveHeatFluxTensor` | `false` | Must be `true` to allocate and compute 3rd-order moment tensors (required for `H_flux` output tag) |
| `MomentsOutputTag` | `" "` | Separate tag string used only by VTK output (`pvtk`/`nbcvtk`) for moments. Accepts: `rho`, `PXX`, `PXY`, `PXZ`, `PYY`, `PYZ`, `PZZ` |
| `output_data_precision` | `DOUBLE` | `SINGLE` or `DOUBLE` for HDF5 outputs |

---

## Reading Output in Python

### Parallel HDF5 (phdf5) — global arrays, ready to use

```python
import h5py
import numpy as np

# Read Bz at cycle 10
with h5py.File("data_dir/Fields_00010/B_00010.h5", "r") as f:
    Bz = f["Fields/Bz"][:]        # shape (nxc+1, nyc+1, nzc+1)

# Read electron (species 0) pressure tensor at cycle 10
with h5py.File("data_dir/Moments_00010/Pressure_species_0_00010.h5", "r") as f:
    pXX = f["Moments/species_0/pXX"][:]   # shape (nxc+1, nyc+1, nzc+1)
    pYY = f["Moments/species_0/pYY"][:]
    pZZ = f["Moments/species_0/pZZ"][:]
    # Scalar pressure (isotropic part): P = (pXX + pYY + pZZ) / 3

# Read particle positions at cycle 10
with h5py.File("data_dir/Particles_00010/species_0_00010.h5", "r") as f:
    pos = f["Particles/species_0/position"][:]  # shape (Nparticles, 3)
    vel = f["Particles/species_0/velocity"][:]  # shape (Nparticles, 3)
```

### Serial HDF5 (shdf5) — requires subdomain reassembly

```python
# Each procN.hdf contains only the local subdomain for rank N.
# Reassemble by reading topology info from each file and stitching
# based on /topology/cartesian_coord.
```

---

## Virtual Satellite Traces (dead code — not currently wired up)

The codebase contains infrastructure for "virtual satellites" — a grid of uniformly-spaced probe points that sample field and moment data every cycle, producing high-cadence time traces at fixed locations. This is useful for spectral analysis (FFT) and wave diagnostics where `FieldOutputCycle` is too coarse.

**Current status:** The method `c_Solver::WriteVirtualSatelliteTraces()` exists in `main/iPIC3Dlib.cpp:1139` and is declared in `include/iPic3D.h:81`, but it is **never called** from the simulation loop (`WriteOutput()` or `iPIC3D.cpp`). The member variables `nsat` (number of satellites per dimension) and `cqsat` (output filename) are declared in `include/iPic3D.h` but **never initialized**. Calling this method in its current state would use uninitialized values.

A rich postprocessing toolchain for the expected output files exists in `postprocessing_tools/c++/read-virt-sat-files-iPIC/`.

### How it was designed to work

The method places `nsat³` probe points on a regular sub-grid within each MPI subdomain. For each satellite at indices `(isat, jsat, ksat)` where `0 ≤ isat < nsat`:

```
index_x = 1 + isat * (nxc_local / nsat) + nxc_local / nsat / 2
index_y = 1 + jsat * (nyc_local / nsat) + nyc_local / nsat / 2
index_z = 1 + ksat * (nzc_local / nsat) + nzc_local / nsat / 2
```

This places each satellite at the **centre** of evenly-divided blocks within the local subdomain.

### Variables recorded per satellite point

At each probe point, one tab-separated line is appended per cycle to a text file (one file per MPI rank). The 14 columns per satellite are:

| Columns | Variable | Description |
|---------|----------|-------------|
| 1–3 | Bx, By, Bz | Magnetic field |
| 4–6 | Ex, Ey, Ez | Electric field |
| 7–9 | Jxe, Jye, Jze | Electron current (species 0 + 2 combined) |
| 10–12 | Jxi, Jyi, Jzi | Ion current (species 1 + 3 combined) |
| 13 | ρe | Electron density (species 0 + 2) |
| 14 | ρi | Ion density (species 1 + 3) |

**Constraint:** The method hard-codes `assert_eq(ns, 4)` — it requires exactly 4 species (the standard Double Harris configuration with two electron + two ion populations).

### Expected output files

Each MPI rank would produce: `VirtualSatelliteTraces<rank>.txt`

Each line contains `nsat³ × 14` tab-separated values (all satellite points for that rank concatenated), with one line per cycle.

### What would be needed to reactivate

1. Initialize `nsat` (e.g. from an input file parameter or hard-coded) and `cqsat` (output path) in `c_Solver::Init()`
2. Call `WriteVirtualSatelliteTraces()` from within the cycle loop in `WriteOutput()` or `iPIC3D.cpp`
3. Optionally generalize beyond the 4-species assumption and add a configurable output cadence

### Postprocessing tools

The `postprocessing_tools/c++/read-virt-sat-files-iPIC/` directory contains C++ codes and shell scripts for:

- **Point extraction** (`process-virtual-satellite-files-point.sh`) — find the satellite closest to user-specified (x,y,z) coordinates across all rank files, extract its time series
- **Line/trace extraction** (`process-virtual-satellite-files-block-single-var.sh`) — extract a 1D trace along X, Y, or Z for a single variable, output as a gnuplot-friendly block matrix
- **FFT analysis** (`fft-cpp-code-point/`) — compute frequency spectra from extracted point data
- **Gnuplot generators** (`gnuplot-script-generators/`) — auto-generate plotting scripts for all of the above
- **Run stitching** — combine data from restart-continued runs into continuous time series
