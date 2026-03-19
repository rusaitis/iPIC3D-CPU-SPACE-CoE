# iPIC3D Grid, Coordinates, and Output Reference

Quick reference for understanding iPIC3D's spatial discretization and
reconstructing coordinates from HDF5 output files.

## 1. Units and normalization

All quantities are in **Gaussian units**, normalized to ion plasma parameters
(see `Documentation/units.txt`):

| Quantity | Normalization |
|----------|---------------|
| Length   | `d_i` (ion skin depth) |
| Time     | `1 / f_{p,i}` (inverse ion plasma frequency) |
| Speed    | `c = 1` |
| Mass     | `m_i = 1` |
| Charge   | `e = 1` |
| Density  | `rho = 1/(4*pi)` |
| Energy   | `m_i * c^2` |
| Permeability | `mu_0 = 1` |

The Alfven speed equals `B_0` from the input file.

The code constant `FourPI = 16*atan(1.0)` = 4π appears throughout the field
equations and energy diagnostics. Maxwell's equations carry explicit 4π factors
because this is Gaussian CGS — there is no separate μ₀ or ε₀ parameter. The
`μ₀ = 1` entry in the table (matching `Documentation/units.txt`) reflects that
no permeability parameter exists in the code; the physics is entirely in the 4π
prefactors below.

### Physics equations as implemented

Each equation below shows the **exact prefactors from the code**, with source
file references. All quantities are in the normalized Gaussian units above.

#### Faraday's law — `calculateB()` (`fields/EMfields3D.cpp:2937`)

```
B_c^{n+1} = B_c^n − c·Δt · ∇×E^{n+θ}
```

- Code: `addscale(-c * dt, 1, Bxc, curlN2C(Exth, ...), ...)`
- E^{n+θ} = θ·E^{n+1} + (1−θ)·E^n is the θ-weighted electric field
  (Crank–Nicolson when θ = 0.5, i.e. `th` in the input file)
- Curl is taken on nodes → result on cell centers

#### Maxwell source (RHS) — `MaxwellSource()` (`fields/EMfields3D.cpp:2731–2739`)

```
b = E^n + θΔt · [c·(∇×B^n) − 4π · J_h · invVOL]
```

- `temp2 = curlC2N(Bxc, ...)` = ∇×B on nodes (line 2693)
- `tempX = th*dt*(c*temp2X - FourPI*Jx_tot*invVOL)` (line 2731)
- `tempX += Ex` (line 2737), giving the full RHS
- `J_h` is the half-step current from moment gathering (ECSIM); `invVOL = 1/(dx·dy·dz)`
- This is the right-hand side of the implicit system A·E^{n+1} = b

#### Implicit field operator (LHS) — `MaxwellImage()` (`fields/EMfields3D.cpp:2825–2870`)

```
A·E = E + (cθΔt)² · ∇×(∇×E) + θΔt · 4π · M·E / ΔV
```

- `factor = c*th*dt*c*th*dt` = (cθΔt)² (line 2825)
- Double curl: `curlN2C` then `curlC2N` (lines 2816, 2822)
- `imageX = E + factor * ∇×(∇×E)` (line 2830)
- Mass matrix: `temp2X = dt*th*FourPI*MEx` (line 2847), then `*= invVOL` (line 2859)
- Final: `imageX += temp2X` (line 2868)
- M is the ECSIM mass matrix (symmetric, from moment gathering)
- This is the matrix-free operator applied by GMRES or PETSc's `MatShell`

#### Lorentz force / particle mover — `mover_PC()` (`particles/Particles3D.cpp:1929–2085`)

```
Ω = (q/m)·(Δt/2c)·B                                     [gyration parameter]
v_t = v^n + (q/m)·(Δt/2c)·E                              [half E-kick]
v_avg = (v_t + v_t×Ω + (v_t·Ω)Ω) / (1 + Ω²)            [Boris rotation]
x^{n+1} = x^n + v_avg · Δt
v^{n+1} = 2·v_avg − v^n
```

- `qdto2mc = qom * dto2 / c` (line 1929) — the `1/c` is the Gaussian Lorentz
  force: **F** = q(**E** + **v**×**B**/c)
- Boris rotation: `Omx = qdto2mc*Bxl` (line 2052), `ut = uorig + qdto2mc*Exl`
  (line 2061), velocity update (lines 2068–2070)
- Position uses full Δt (line 2080); velocity uses the extrapolation
  v^{n+1} = 2v_avg − v^n (line 2083)
- Predictor–corrector: the inner loop (NiterMover iterations) refines the
  average position for field interpolation

#### Energy diagnostics (`fields/EMfields3D.cpp:6103–6181`, `particles/Particles3Dcomm.cpp:1430–1438`)

```
U_E = ∫ E²/(8π) dV     →  code: 0.5 * dV * E² / FourPI     (nodes, line 6111)
U_B = ∫ B²/(8π) dV     →  code: 0.5 * dV * B² / FourPI     (cell centers, line 6176)
K   = Σ ½ m v²         →  code: 0.5 * (q/qom) * (u²+v²+w²) (line 1437)
K_rel = Σ (γ−1)mc²     →  code: (q/qom) * (γ−1) * c²        (line 1433)
```

- `q/qom = q/(q/m) = m`, so both kinetic energy formulas give the standard result
- `0.5/FourPI = 1/(8π)` — Gaussian electromagnetic energy density

#### Density normalization (`fields/EMfields3D.cpp:3826`)

```
ρ_stored = ρ_init / (4π)
```

The 1/(4π) factor is absorbed at initialization. It cancels with the explicit
`4π` in the Maxwell source term (line 2731: `FourPI * J * invVOL`), keeping
the equations self-consistent. The convention matches Gauss's law ∇·E = 4πρ.

#### Ampère's law — initial current (`fields/EMfields3D.cpp:4090`)

```
J = c · ∇×B / (4π)
```

- Code: `Jxs[0] = c * curlC2N(Bxc, ...) / FourPI`
- This is the Gaussian Ampère's law (static limit): ∇×B = (4π/c)J,
  rearranged for J. Used to initialize electron current from the magnetic
  field configuration.

#### Dipole field and magnetosphere scaling

**Coordinate system constraint:** iPIC3D uses a uniform Cartesian grid
exclusively — `Grid3DCU` = "Grid 3D Cartesian Uniform." All differential
operators (curl, grad, div in `Grid3DCU.h`) assume Cartesian geometry; there is
no support for cylindrical, spherical, or curvilinear coordinates. The dipole
and magnetosphere cases handle inherently spherical geometry entirely on this
Cartesian mesh: `B_ext` is computed in Cartesian components using the radial
distance formula, fields are zeroed inside the sphere radius `L_square`, and
particles entering the sphere are deleted each cycle. The 3-argument coordinate
accessors `getXN(i,j,k)` (which currently just return `node_xcoord[i]`) are
retained for a possible future extension to deformed logically-Cartesian meshes
(`Grid3DCU.h:266–268`, issue #40).

For magnetosphere simulations, iPIC3D uses an external dipole field initialized
by `initDipole()` / `initDipole2D()` (`fields/EMfields3D.cpp:4873–4968`).

**Key parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `L_square` | 5.0 | Planet radius (d_i) — inner boundary sphere |
| `x_center`, `y_center`, `z_center` | 5.0 | Dipole center position |
| `B1x`, `B1y`, `B1z` | — | Dipole field strength components |

**Dipole formula** (outside the sphere, r > L_square):
```
a = L_square
fac = −B1z · a³ / r⁵
Bx_ext = 3·(x−xc)·(z−zc) · fac
By_ext = 3·(y−yc)·(z−zc) · fac
Bz_ext = [2(z−zc)² − (x−xc)² − (y−yc)²] · fac
```
Inside the sphere (r ≤ L_square): B_ext = 0, particles are deleted.

The effective magnetic moment is **M = B1z · L_square³**.

**Magnetosphere scaling trick:** Real Earth has R_E ≈ 1000 d_i, far too large
for kinetic resolution. Set L_square = O(1) d_i to compress the magnetosphere
so that reconnection sites and boundary layers fall within a tractable domain.
Adjust B1z so that the field at the region of interest has the desired strength.
There is no separate scaling ratio parameter — the compression is implicit in
the choice of L_square, B1z, and domain size.

**Particle removal:** `deleteParticlesInsideSphere()` (`main/iPIC3Dlib.cpp:725`)
removes particles that enter the planet interior each cycle. For 2D dipoles,
`deleteParticlesInsideSphere2DPlaneXZ()` is used instead (line 730).

## 2. Grid extent and spacing

The domain is `[0, Lx] x [0, Ly] x [0, Lz]` with uniform spacing:

```
dx = Lx / Nxc
dy = Ly / Nyc
dz = Lz / Nzc
```

where `Nxc`, `Nyc`, `Nzc` are the **global number of cells** per dimension.
Computed in `Collective::init_derived_parameters()` (`main/Collective.cpp:1327`).

## 3. Node vs center grid (staggered)

iPIC3D uses a staggered grid with two sets of points:

- **Nodes** (vertices): `Nxc + 1` points per dimension. All primary field
  quantities (E, B, moments) are stored and output here.
- **Centers** (cell midpoints): `Nxc` points per dimension. B is also stored
  on centers internally; other center quantities are interpolated.

### Coordinate formulas

From `grids/Grid3DCU.cpp:146-160`:

```
node_xcoord[i]   = xStart + (i - 1) * dx       // i = 0..nxn-1
center_xcoord[i] = 0.5 * (node_xcoord[i] + node_xcoord[i+1])
```

For the **global** grid (single process, `xStart = 0`):

```
x_node[i]   = (i - 1) * dx      for i = 0..Nxc+2   (including ghosts)
x_center[i] = x_node[i] + dx/2
```

**Key consequence:** `0` and `Lx` are **node positions** (cell edges), not
cell centers. The first center sits at `dx/2`, the last at `Lx - dx/2`.

### Index conventions

Active (non-ghost) indices on a local subdomain:

- Nodes: `i = 1 .. nxn-2`
- Centers: `i = 1 .. nxc-2`

Index 0 and the last index are ghost layers (see next section).

## 4. Ghost cells

Each local subdomain has **1 ghost cell/node on each side** per dimension:

```
nxc = nxc_r + 2          // local cells including ghosts
nxn = nxc + 1            // local nodes including ghosts
```

The ghost region starts at:

```
xStart_g = xStart - dx   // Grid3DCU.cpp:120
```

So `node_xcoord[0] = xStart - dx` (ghost) and `node_xcoord[1] = xStart`
(first real node).

## 5. MPI decomposition

Cartesian topology: `XLEN x YLEN x ZLEN` processes (total `np = XLEN*YLEN*ZLEN`).

Each rank's local subdomain origin (`Grid3DCU.cpp:94-101`):

```
nxc_rr = ceil(Nxc / XLEN)           // regular subdomain cell count
xWidth = dx * nxc_rr
xStart = rank_x * xWidth
```

The uppermost rank in each dimension may be truncated (fewer cells) if `Nxc`
is not evenly divisible by `XLEN`.

## 6. Field storage summary

| Quantity | Stored on | Internal arrays | Output grid |
|----------|-----------|-----------------|-------------|
| E (`Ex`, `Ey`, `Ez`) | nodes | `arr3_double` | nodes |
| B (`Bxn`, `Byn`, `Bzn`) | nodes | `arr3_double` | nodes |
| B (`Bxc`, `Byc`, `Bzc`) | centers | `arr3_double` | not output |
| rho, J, pressure | nodes | `arr3_double` / `arr4_double` | nodes |

Interpolation between grids: `interpN2C()` and `interpC2N()` in
`Grid3DCU` (`include/Grid3DCU.h:112-118`).

## 7. HDF5 output coordinate system

### What is written

- **No coordinate arrays** are stored in the HDF5 files.
- Grid parameters (`Lx`, `Ly`, `Lz`, `Nxc`, etc.) are stored as HDF5
  attributes by `CreatePHDF5file()`, which receives `L[3]` and `dglob[3]`.
- All field datasets are on the **node grid**: global shape
  `(Nxc+1, Nyc+1, Nzc+1)`.

### Ghost cell stripping

The `pack_sca_nodes()` function (`ParallelIO.cpp:64-72`) packs only real
nodes, skipping ghost index 0:

```cpp
for (int i = 1; i <= lx; ++i)
    for (int j = 1; j <= ly; ++j)
        for (int k = 1; k <= lz; ++k)
            out[p++] = getf(i, j, k);
```

Each rank writes its local chunk via collective MPI-IO hyperslabs. The last
rank in each dimension writes one extra node (the shared boundary node) to
fill the `+1` in the global shape.

### Particle output

Particle positions `(x, y, z)` are written as raw physical coordinates in
the same normalized units. No rescaling needed.

## 8. Reconstructing coordinates from output

Given an HDF5 field dataset of shape `(Nx+1, Ny+1, Nz+1)`:

```python
# Read grid params from HDF5 attributes or input file
Nxc, Nyc, Nzc = Nx, Ny, Nz       # number of cells = shape - 1
dx, dy, dz = Lx/Nxc, Ly/Nyc, Lz/Nzc

# Node coordinates (what the output represents)
x_node = np.arange(Nxc + 1) * dx   # [0, dx, 2*dx, ..., Lx]
y_node = np.arange(Nyc + 1) * dy
z_node = np.arange(Nzc + 1) * dz

# Cell-center coordinates (if needed)
x_center = (np.arange(Nxc) + 0.5) * dx   # [dx/2, 3*dx/2, ..., Lx - dx/2]
y_center = (np.arange(Nyc) + 0.5) * dy
z_center = (np.arange(Nzc) + 0.5) * dz
```

For a 2D simulation (e.g., GEM reconnection in x-y), the z dimension
typically has `Nzc = 1`, so `z_node = [0, Lz]`.

### Particle coordinates

Already in physical units — use directly. Particle positions span `[0, Lx]`
(continuous, not snapped to grid).

## Key source files

| File | What to look at |
|------|-----------------|
| `grids/Grid3DCU.cpp:38-163` | Grid init, coordinate arrays, ghost setup |
| `include/Grid3DCU.h:253-274` | `getXN`/`getXC` accessors, `calcXN` |
| `main/Collective.cpp:1322-1331` | `dx`/`dy`/`dz` computation |
| `inputoutput/ParallelIO.cpp:64-72` | `pack_sca_nodes` — ghost stripping |
| `inputoutput/ParallelIO.cpp:133-195` | Field output writing (hyperslabs) |
| `Documentation/units.txt` | Full normalization reference |
