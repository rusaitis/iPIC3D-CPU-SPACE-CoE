/* 
 * iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!************************************************************************* EMfields3D.h - ElectroMagnetic fields definition ------------------- begin : May 2008 copyright : KU Leuven developers : Stefano Markidis, Giovanni Lapenta ************************************************************************* */

#ifndef EMfields3D_H
#define EMfields3D_H

#include <iostream>
#include "asserts.h"
#include "ipicfwd.h"
#include "Alloc.h"
#include "Basic.h"
#include "Neighbouring_Nodes.h"
#include <memory>
#include <iomanip>

using namespace std;
/*! Electromagnetic fields and sources defined for each local grid, and for an implicit maxwell's solver @date May 2008 @par Copyright: (C) 2008 KUL @author Stefano Markidis, Giovanni Lapenta. @version 3.0 */

// dimension of vectors used in fieldForPcls
const int DFIELD_3or4 = 4; // 4 pads with garbage but is needed for alignment

class Particles3Dcomm;
class Moments10;
class ECSIM_Moments13;
#ifdef USE_PETSC
class PetscSolver;
#endif
class EMfields3D                // :public Field
{
public:
    /*! constructor */
    EMfields3D(Collective * col, Grid * grid, VirtualTopology3D *vct);
    /*! destructor */
    ~EMfields3D();

    void setAllzero();

    //! ======================================================================================================= !//

    //? ---------- Initial field distributions (Non Relativistic) ---------- ?//

    //* Initialise electromagnetic fields with constant values
    void init();

    //* Initialise beam
    void initBEAM(double x_center, double y_center, double z_center, double radius);
    
    //* Initialise GEM challenge 
    void initGEM();

    void initOriginalGEM();
    
    //* Initialise double Harris sheets for magnetic reconnection
    void init_double_Harris();

    void init_double_Harris_hump();
    
    //* Initialise GEM challenge with dipole-like tail without perturbation
    void initGEMDipoleLikeTailNoPert();
    
    //* Initialise GEM challenge with no Perturbation
    void initGEMnoPert();

    //* Initialise from BATSRUS
    #ifdef BATSRUS
        void initBATSRUS();
    #endif

    //* Random initial fields
    void initRandomField();
    
    //* Initialise force free field (JxB=0)
    void initForceFree();
    
    //* Initialise rotated magnetic field
    void initEM_rotate(double B, double theta);
    
    //* Add a perturbation to charge density
    void AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid * grid);
    
    //* Add perturbation to the EM field
    void AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid * grid);
    
    //* Initialise a combination of magnetic dipoles
    void initDipole();
    void initDipole2D();
    
    //* Initialise magnetic nulls
    void initNullPoints();
    
    //* Initialise Taylor-Green flow
    void initTaylorGreen();

    //* Initialise fields for shear velocity in fluid finite Larmor radius (FLR) equilibrium (Cerri et al. 2013)
    void init_KHI_FLR(); 

    //? ---------- Initial particle distributions (Relativistic) ---------- ?//

    //? Quasi-1D ion-electron shock (Relativistic and Non relativistic)
    void initShock1D();

    //? Relativistic double Harris for pair plasma: Maxwellian background, drifting particles in the sheets
    void init_Relativistic_Double_Harris_pairs();

    //? Relativistic double Harris for ion-electron plasma: Maxwellian background, drifting particles in the sheets
    void init_Relativistic_Double_Harris_ion_electron();

    //! ======================================================================================================= !//

    /*! Calculate Electric field using the implicit Maxwell solver */
    void calculateE();
#ifdef USE_PETSC
    void setPetscSolver(PetscSolver *solver) { petscSolver_ = solver; }
#endif
    /*! Image of Poisson Solver (for SOLVER) */
    void PoissonImage(double *image, double *vector);
    /*! Image of Maxwell Solver (for Solver) */
    void MaxwellImage(double *im, double *vector);
    /*! Maxwell source term (for SOLVER) */
    void MaxwellSource(double *bkrylov);
    /*! Impose a constant charge inside a spherical zone of the domain */
    void ConstantChargePlanet(double R, double x_center, double y_center, double z_center);
    void ConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center);
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBC();
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBCv2();
    /*! Calculate Magnetic field with the implicit solver: calculate B defined on nodes With E(n+ theta) computed, the magnetic field is evaluated from Faraday's law */
    void calculateB();

    //? ---------- Boundary Conditions ---------- ?//
    
    //* B boundary for GEM (cell centres) - this assumes non-periodic boundaries along Y *//
    void fixBcGEM();
    
    //* B boundary for GEM (nodes) - this assumes non-periodic boundaries along Y *//
    void fixBnGEM();
    
    //* B boundary for forcefree *//
    void fixBforcefree();

    //* Boundary conditions for magnetic field *//
    // void fixBC_B();

    //? ----------------------------------------- ?//

    /*! Calculate rho hat, Jx hat, Jy hat, Jz hat */
    void calculateHatFunctions();

    void C2NB();

    //* Compute the product of mass matrix with vector "V = (Vx, Vy, Vz)"
    void mass_matrix_times_vector(double* MEx, double* MEy, double* MEz, const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int i, int j, int k);

    //* Energy-conserving smoothing
    void energy_conserve_smooth(arr3_double data_X, arr3_double data_Y, arr3_double data_Z, int nx, int ny, int nz);
    //* kernel_override: -1 (default) -> use col->getSmoothKernelInt();
    //*                  >=0          -> force this kernel (Phase 10m: post-solve Helmholtz reuse)
    void energy_conserve_smooth_direction(double*** data, int nx, int ny, int nz, int dir, int kernel_override = -1);

    //* Phase 10m: post-`calculateE` Helmholtz low-pass applied once per cycle to (Ex, Ey, Ez)
    //* OUTSIDE the implicit operator. Decoupled from `S·M·S` so it does not restructure
    //* `MaxwellImage` the way Phase 10k's drop-in attempt did.
    void post_solve_filter_E(arr3_double Ex_field, arr3_double Ey_field, arr3_double Ez_field, int nx, int ny, int nz);

    //* Step 22: enforce equality of the periodic-duplicate interior nodes (indices n_ghost_ and
    //*          nxn-n_ghost_-1 along each periodic axis) on the solved E-field arrays.
    //* The Maxwell solver treats the two images of the same physical periodic node as independent
    //* DOFs. communicateNodeBC refreshes ghost faces only, so interior duplicates can disagree by
    //* O(relative solver tol). See plan-energy-conservation.md Step 21 Sub-test (c).
    void unify_periodic_duplicates(arr3_double Exf, arr3_double Eyf, arr3_double Ezf, int nx, int ny, int nz);

    //* Step 25 diagnostic: at end of calculateE, print the two grid-side work integrals
    //*   I_J = dt · <Eth, Jxh>_unique     I_M = dt · <Eth, M·Eth>_unique
    //* Summation range is the unique-node interior [n_ghost_, n{x,y,z}n - n_ghost_ - 1)
    //* matching get_E_field_energy, so the prints align with ConservedQuantities.txt
    //* columns and external scripts can derive R_part / R_field per cycle.
    void dump_cycle_identity(int cycle);

    //* Step 27 diagnostic: at the end of the first moment gather, print Frobenius-norm,
    //* max-abs, and sum statistics of the 9-component stored mass matrix. Cheap probe
    //* designed as the iPIC3D endpoint for a cross-code byte compare against an ECSIM
    //* dump. Shape-preserving (unique-node interior only) so differences are localised
    //* and not polluted by halo wrap.
    void dump_mass_matrix_stats(int cycle);

    //* Step 32: raw-binary IEEE-754 double dump of all node fields at a given
    //* cycle for cross-code (iPIC3D ↔ ECSIM) byte diff. Writes one file per
    //* array to `{dir}/fields_cycle{N}_{name}.bin` plus a
    //* `fields_cycle{N}.meta.txt` index. Row-major C order, k (z) fastest.
    //* Cycle 0 = post-init (pre-solve) snapshot; cycle 1 = post-solve-1.
    void dump_cycle_fields(int cycle, const std::string& dir);

    //* Step 34b: programmatic self-adjointness probe for MaxwellImage.
    //* Generates two deterministic pseudo-random Krylov-space vectors u, v,
    //* applies the matrix-free operator A via MaxwellImage to each, and
    //* reports <A·u, v> − <u, A·v>. A self-adjoint A makes the gap purely
    //* FP round-off (~1e-13 relative at DoubleGEM scale). Any systematic
    //* asymmetry above ~1e-12 rel indicates a structural break in one of the
    //* six ECSIM-exact energy-identity conditions (#2: operator self-adjoint;
    //* #6: consistent periodic-DOF handling). Gated by input flag
    //* VerifyAdjoint; runs once at the chosen cycle (default cycle 1).
    void probe_adjointness(int cycle);

    //* Lapenta-2023 (Physics, 5, 72) adds one condition to the 2017 proof:
    //* when smoothing fires (Smooth>0), the smoothing matrix S must be
    //* symmetric (S_{gg'} = S_{g'g}) for exact energy conservation. The
    //* probe applies S component-wise via `energy_conserve_smooth_direction`
    //* to two deterministic pseudo-random Krylov vectors u, v and reports
    //* <S·u, v> − <u, S·v> in raw, unify, and unique-DOF variants. The raw
    //* gap measures input-inconsistency + ghost-handling; the unique gap
    //* measures the matrix symmetry the 2023 paper calls out.
    void probe_smooth_symmetry(int cycle);

    //* Step 38: per-stage dump inside MaxwellImage. `set_mi_dump_target(cycle)` is
    //* called from the main loop so MaxwellImage knows when to dump. Dumps happen
    //* only on the *first* matvec call of the target cycle (so cost is one
    //* set of binary files per run, not per Krylov iteration).
    void set_mi_dump_target(int cycle) { mi_dump_target_cycle_ = cycle; mi_matvec_count_ = 0; }
    void dump_maxwell_stage(const char* stage_name, arr3_double aX, arr3_double aY, arr3_double aZ);

    /*! communicate ghost for densities and interp rho from node to center */
    void interpDensitiesN2C();

    void setZeroDensities();
    void setZeroPrimaryMoments();
    void setZeroDerivedMoments();
    void setZeroTertiaryMoments();

    //! Set all elements of mass matrix to 0.0 !//
    void setZeroMassMatrix();

    //! Step 68b: Kahan-compensated gather (KahanGather=true) helpers.
    //* `setZeroKahanGatherCompensation` zeroes the companion compensation
    //* buffers at the start of each gather cycle; `foldKahanGatherCompensation`
    //* folds the accumulated compensation back into the primary field and
    //* zeroes it. Called right before `communicateInterp` / `communicateNode_P`
    //* so the halo exchange ships the ε²-accurate per-rank sum.
    void setZeroKahanGatherCompensation();
    void foldKahanGatherCompensation();
    /*! Sum rhon and J over species */
    void sumOverSpecies();
        /*! Sum rhon over species */
    // void sumOverSpeciesRho();
    /*! Sum current over different species */
    // void sumOverSpeciesJ();
    void sumOverSpecies_supplementary();
    void interpolateCenterSpecies(int is); 
    /*! Smoothing after the interpolation* */
    void smooth(arr3_double vector, int type);
    /*! SPECIES: Smoothing after the interpolation for species fields* */
    void smooth(double value, arr4_double vector, int is, int type);
    /*! smooth the electric field */
    void smoothE();

    /*! copy the field data to the array used to move the particles */
    void set_fieldForPcls();
    
    //* Communicate ghost cells for grid -> particles interpolation - IMM
    void communicateGhostP2G(int is);

    //* Communicate ghost cells for grid -> particles interpolation - ECSIM
    void communicateGhostP2G_ecsim(int is);
    void communicateGhostP2G_mass_matrix();

    //* Communicate ghost cells for grid -> particles interpolation - ECSIM output only
    void communicateGhostP2G_supplementary_moments(int is);

    /*! sum moments (interp_P2G) versions */
    void sumMoments(const Particles3Dcomm* part);
    void sumMoments_AoS(const Particles3Dcomm* part);
    void sumMoments_AoS_intr(const Particles3Dcomm* part);
    void sumMoments_vectorized(const Particles3Dcomm* part);
    void sumMoments_vectorized_AoS(const Particles3Dcomm* part);
    void sumMomentsOld(const Particles3Dcomm& pcls);
    /*! add accumulated moments to the moments for a given species */
    //void addToSpeciesMoments(const TenMoments & in, int is);

    //* ECSIM/RelSIM moments
    void add_Rho(double weight[8], int X, int Y, int Z, int is);
    //* Single-node deposit helpers used by the TSC (quadratic) stencil path.
    //* The 8-corner add_Rho/add_Jxh/... helpers above hardcode a 2x2x2 pattern,
    //* which is too rigid for the 27-node TSC support. These accept a single
    //* (value, node) pair so the per-particle TSC code can call them in a 27-loop.
    // Race guard: ECSIM gather runs inside `#pragma omp parallel for` over particles
    // (Particles3D.cpp:1518,1530); multiple threads scatter into the same grid nodes.
    // atomic update fixes the read-modify-write race (no lost contributions); FP order
    // remains thread-schedule-dependent so run-to-run results still vary at ULP.
    inline void add_Rho_node(double value, int X, int Y, int Z, int is)
    {
        #pragma omp atomic update
        rhons[is][X][Y][Z] += value * invVOL;
    }
    inline void add_Jxh_node(double value, int X, int Y, int Z, int is)
    {
        #pragma omp atomic update
        Jxhs[is][X][Y][Z] += value;
    }
    inline void add_Jyh_node(double value, int X, int Y, int Z, int is)
    {
        #pragma omp atomic update
        Jyhs[is][X][Y][Z] += value;
    }
    inline void add_Jzh_node(double value, int X, int Y, int Z, int is)
    {
        #pragma omp atomic update
        Jzhs[is][X][Y][Z] += value;
    }

    //* Step 68b: Neumaier (Kahan-Babuska) compensated add helper.
    //* Used by the `*_kahan` deposit variants below when `KahanGather=true`.
    //* The companion `comp` scalar accumulates the floating-point residual
    //* that a plain `sum += term` would discard; folding it back at the end
    //* of the gather yields an ε²-accurate total independent of add order.
    static inline void kahan_add(double& sum, double& comp, double term)
    {
        const double t = sum + term;
        if (std::fabs(sum) >= std::fabs(term))
            comp += (sum - t) + term;
        else
            comp += (term - t) + sum;
        sum = t;
    }

    //* Step 68b: Kahan-compensated single-node deposit helpers (TSC + CIC
    //* alike). No atomic pragma: `KahanGather=true` also pins the gather to
    //* num_threads=1, so these are safe and cheaper than the atomic variants.
    inline void add_Rho_node_kahan(double value, int X, int Y, int Z, int is)
    {
        kahan_add(rhons[is][X][Y][Z], rhons_c[is][X][Y][Z], value * invVOL);
    }
    inline void add_Jxh_node_kahan(double value, int X, int Y, int Z, int is)
    {
        kahan_add(Jxhs[is][X][Y][Z], Jxhs_c[is][X][Y][Z], value);
    }
    inline void add_Jyh_node_kahan(double value, int X, int Y, int Z, int is)
    {
        kahan_add(Jyhs[is][X][Y][Z], Jyhs_c[is][X][Y][Z], value);
    }
    inline void add_Jzh_node_kahan(double value, int X, int Y, int Z, int is)
    {
        kahan_add(Jzhs[is][X][Y][Z], Jzhs_c[is][X][Y][Z], value);
    }
    
    void add_Jxh(double weight[8], int X, int Y, int Z, int is);
    void add_Jyh(double weight[8], int X, int Y, int Z, int is);
    void add_Jzh(double weight[8], int X, int Y, int Z, int is);

    void add_Mass(double value[3][3], int X, int Y, int Z, int ind);

    //* Step 68b: Kahan-compensated 8-corner CIC deposits + mass matrix. See
    //* the companion `*_c` arrays and `kahan_add` helper below. Used when
    //* `KahanGather=true`.
    void add_Rho_kahan(double weight[8], int X, int Y, int Z, int is);
    void add_Jxh_kahan(double weight[8], int X, int Y, int Z, int is);
    void add_Jyh_kahan(double weight[8], int X, int Y, int Z, int is);
    void add_Jzh_kahan(double weight[8], int X, int Y, int Z, int is);
    void add_Mass_kahan(double value[3][3], int X, int Y, int Z, int ind);

    //* ECSIM/RelSIM supplementary moments
    void add_Jx(double weight[8], int X, int Y, int Z, int is);
    void add_Jy(double weight[8], int X, int Y, int Z, int is);
    void add_Jz(double weight[8], int X, int Y, int Z, int is);

    void add_N(double weight[8], int X, int Y, int Z, int is);

    void add_Pxx(double weight[8], int X, int Y, int Z, int is);
    void add_Pxy(double weight[8], int X, int Y, int Z, int is);
    void add_Pxz(double weight[8], int X, int Y, int Z, int is);
    void add_Pyy(double weight[8], int X, int Y, int Z, int is);
    void add_Pyz(double weight[8], int X, int Y, int Z, int is);
    void add_Pzz(double weight[8], int X, int Y, int Z, int is);

    void add_E_flux_x(double weight[8], int X, int Y, int Z, int is);
    void add_E_flux_y(double weight[8], int X, int Y, int Z, int is);
    void add_E_flux_z(double weight[8], int X, int Y, int Z, int is);

    void add_Qxxx(double weight[8], int X, int Y, int Z, int is);
    void add_Qyyy(double weight[8], int X, int Y, int Z, int is);
    void add_Qzzz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxyz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxxy(double weight[8], int X, int Y, int Z, int is);
    void add_Qxxz(double weight[8], int X, int Y, int Z, int is);
    void add_Qyyz(double weight[8], int X, int Y, int Z, int is);
    void add_Qxyy(double weight[8], int X, int Y, int Z, int is);
    void add_Qxzz(double weight[8], int X, int Y, int Z, int is);
    void add_Qyzz(double weight[8], int X, int Y, int Z, int is);
    
    /*! adjust densities on boundaries that are not periodic */
    void adjustNonPeriodicDensities(int is);

    /*! Perfect conductor boundary conditions LEFT wall */
    void perfectConductorLeft(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                              const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                              int dir);
    /*! Perfect conductor boundary conditions RIGHT wall */
    void perfectConductorRight(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                               const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                               int dir);
    /*! Perfect conductor boundary conditions for source LEFT wall */
    void perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);
    /*! Perfect conductor boundary conditions for source RIGHT wall */
    void perfectConductorRightS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);

    /*! Calculate the sysceptibility tensor on the boundary */
    void sustensorRightX(double **susxx, double **susyx, double **suszx);
    void sustensorLeftX (double **susxx, double **susyx, double **suszx);
    void sustensorRightY(double **susxy, double **susyy, double **suszy);
    void sustensorLeftY (double **susxy, double **susyy, double **suszy);
    void sustensorRightZ(double **susxz, double **susyz, double **suszz);
    void sustensorLeftZ (double **susxz, double **susyz, double **suszz);

    //? Potential array
    arr3_double getPHI() {return PHI;}

    //* Field components
    const_arr4_pfloat get_fieldForPcls() { return fieldForPcls; }

    //? Electric Field (nodes)
    double getEx(int X, int Y, int Z) const { return Ex.get(X,Y,Z);  }
    double getEy(int X, int Y, int Z) const { return Ey.get(X,Y,Z);  }
    double getEz(int X, int Y, int Z) const { return Ez.get(X,Y,Z);  }
    arr3_double getEx() { return Ex;  }
    arr3_double getEy() { return Ey;  }
    arr3_double getEz() { return Ez;  }

    double getEx_ext(int X, int Y, int Z) const { return Ex_ext.get(X,Y,Z); }
    double getEy_ext(int X, int Y, int Z) const { return Ey_ext.get(X,Y,Z); }
    double getEz_ext(int X, int Y, int Z) const { return Ez_ext.get(X,Y,Z); }
    arr3_double getEx_ext() { return Ex_ext; }
    arr3_double getEy_ext() { return Ey_ext; }
    arr3_double getEz_ext() { return Ez_ext; }
    
    //? Magnetic field (nodes)
    double getBx(int X, int Y, int Z) const { return Bxn.get(X,Y,Z); }
    double getBy(int X, int Y, int Z) const { return Byn.get(X,Y,Z); }
    double getBz(int X, int Y, int Z) const { return Bzn.get(X,Y,Z); }
    arr3_double getBx() { return Bxn; }
    arr3_double getBy() { return Byn; }
    arr3_double getBz() { return Bzn; }

    double getBx_ext(int X, int Y, int Z) const { return Bx_ext.get(X,Y,Z); }
    double getBy_ext(int X, int Y, int Z) const { return By_ext.get(X,Y,Z); }
    double getBz_ext(int X, int Y, int Z) const { return Bz_ext.get(X,Y,Z); }
    arr3_double getBx_ext() { return Bx_ext; }
    arr3_double getBy_ext() { return By_ext; }
    arr3_double getBz_ext() { return Bz_ext; }

    double getBxc_ext(int X, int Y, int Z) const { return Bxc_ext.get(X,Y,Z); }
    double getByc_ext(int X, int Y, int Z) const { return Byc_ext.get(X,Y,Z); }
    double getBzc_ext(int X, int Y, int Z) const { return Bzc_ext.get(X,Y,Z); }
    arr3_double getBxc_ext() { return Bxc_ext; }
    arr3_double getByc_ext() { return Byc_ext; }
    arr3_double getBzc_ext() { return Bzc_ext; }

    double getBxc(int X, int Y, int Z) const { return Bxc.get(X,Y,Z); }
    double getByc(int X, int Y, int Z) const { return Byc.get(X,Y,Z); }
    double getBzc(int X, int Y, int Z) const { return Bzc.get(X,Y,Z); }
    arr3_double getBxc() { return Bxc; };
    arr3_double getByc() { return Byc; };
    arr3_double getBzc() { return Bzc; };

    double getBxTot(int X, int Y, int Z) const { return Bxn.get(X,Y,Z) + Bx_ext.get(X,Y,Z); }
    double getByTot(int X, int Y, int Z) const { return Byn.get(X,Y,Z) + By_ext.get(X,Y,Z); }
    double getBzTot(int X, int Y, int Z) const { return Bzn.get(X,Y,Z) + Bz_ext.get(X,Y,Z); }
    arr3_double getBxTot() { addscale(1.0,Bxn,Bx_ext,Bx_tot,nxn,nyn,nzn); return Bx_tot; }
    arr3_double getByTot() { addscale(1.0,Byn,By_ext,By_tot,nxn,nyn,nzn); return By_tot; }
    arr3_double getBzTot() { addscale(1.0,Bzn,Bz_ext,Bz_tot,nxn,nyn,nzn); return Bz_tot; }

    //* Densities (s --> of each species)
    arr3_double getRHOn() { return rhon; }
    double getRHOn(int X, int Y, int Z) const { return rhon.get(X, Y, Z); }
    
    arr4_double getRHOns() { return rhons; }
    double getRHOns(int X, int Y, int Z, int is) const { return rhons.get(is, X, Y, Z); }

    arr4_double getRHOcs() { return rhocs; }
    
    double getRHOcs(int X, int Y, int Z, int is) const { return rhocs.get(is, X, Y, Z); }
    arr3_double getRHOc_avg() { return rhoc_avg; }
    double getRHOc_avg(int X, int Y, int Z) const { return rhoc_avg.get(X, Y, Z); }

    //* Current (s --> of each species)
    double getJxs(int X,int Y,int Z,int is) const { return Jxs.get(is,X,Y,Z); }
    double getJys(int X,int Y,int Z,int is) const { return Jys.get(is,X,Y,Z); }
    double getJzs(int X,int Y,int Z,int is) const { return Jzs.get(is,X,Y,Z); }
    arr4_double getJxs() { return Jxs; }
    arr4_double getJys() { return Jys; }
    arr4_double getJzs() { return Jzs; }

    double getJxhs(int X, int Y,int Z, int is) const { return Jxhs.get(is,X,Y,Z); }
    double getJyhs(int X, int Y,int Z, int is) const { return Jyhs.get(is,X,Y,Z); }
    double getJzhs(int X, int Y,int Z, int is) const { return Jzhs.get(is,X,Y,Z); }
    arr4_double getJxhs() { return Jxhs; }
    arr4_double getJyhs() { return Jyhs; }
    arr4_double getJzhs() { return Jzhs; }

    //* Current (overall)
    arr3_double getJx() { return Jx; }
    arr3_double getJy() { return Jy; }
    arr3_double getJz() { return Jz; }
    double getJx(int X,int Y,int Z) const { return Jx.get(X,Y,Z); }
    double getJy(int X,int Y,int Z) const { return Jy.get(X,Y,Z); }
    double getJz(int X,int Y,int Z) const { return Jz.get(X,Y,Z); }

    arr3_double getJxh() { return Jxh; }
    arr3_double getJyh() { return Jyh; }
    arr3_double getJzh() { return Jzh; }

    arr3_double getJx_ext() { return Jx_ext; }
    arr3_double getJy_ext() { return Jy_ext; }
    arr3_double getJz_ext() { return Jz_ext; }

    //* Pressure Tensor
    arr4_double getpXXsn() { return pXXsn; }
    arr4_double getpXYsn() { return pXYsn; }
    arr4_double getpXZsn() { return pXZsn; }
    arr4_double getpYYsn() { return pYYsn; }
    arr4_double getpYZsn() { return pYZsn; }
    arr4_double getpZZsn() { return pZZsn; }

    double getpXXsn(int X, int Y, int Z, int is) const { return pXXsn.get(is,X,Y,Z); }
    double getpXYsn(int X, int Y, int Z, int is) const { return pXYsn.get(is,X,Y,Z); }
    double getpXZsn(int X, int Y, int Z, int is) const { return pXZsn.get(is,X,Y,Z); }
    double getpYYsn(int X, int Y, int Z, int is) const { return pYYsn.get(is,X,Y,Z); }
    double getpYZsn(int X, int Y, int Z, int is) const { return pYZsn.get(is,X,Y,Z); }
    double getpZZsn(int X, int Y, int Z, int is) const { return pZZsn.get(is,X,Y,Z); }

    //* Energy Flux Density
    arr4_double getEFxs() { return E_flux_xs; }
    arr4_double getEFys() { return E_flux_ys; }
    arr4_double getEFzs() { return E_flux_zs; }

    double getEFxs(int X, int Y, int Z, int is) const { return E_flux_xs.get(is,X,Y,Z); }
    double getEFys(int X, int Y, int Z, int is) const { return E_flux_ys.get(is,X,Y,Z); }
    double getEFzs(int X, int Y, int Z, int is) const { return E_flux_zs.get(is,X,Y,Z); }
    

    //* Heat Flux Tensor 
    double ****getQxxxs() { return (Qxxxs); }
    double ****getQyyys() { return (Qyyys); }
    double ****getQzzzs() { return (Qzzzs); }
    double ****getQxyzs() { return (Qxyzs); }
    double ****getQxxys() { return (Qxxys); }
    double ****getQxxzs() { return (Qxxzs); }
    double ****getQxyys() { return (Qxyys); }
    double ****getQxzzs() { return (Qxzzs); }
    double ****getQyzzs() { return (Qyzzs); }
    double ****getQyyzs() { return (Qyyzs); }

    double getQxxxs(int X, int Y, int Z, int is) const { return Qxxxs[is][X][Y][Z]; }
    double getQyyys(int X, int Y, int Z, int is) const { return Qyyys[is][X][Y][Z]; }
    double getQzzzs(int X, int Y, int Z, int is) const { return Qzzzs[is][X][Y][Z]; }
    double getQxyzs(int X, int Y, int Z, int is) const { return Qxyzs[is][X][Y][Z]; }
    double getQxxys(int X, int Y, int Z, int is) const { return Qxxys[is][X][Y][Z]; }
    double getQxxzs(int X, int Y, int Z, int is) const { return Qxxzs[is][X][Y][Z]; }
    double getQxyys(int X, int Y, int Z, int is) const { return Qxyys[is][X][Y][Z]; }
    double getQxzzs(int X, int Y, int Z, int is) const { return Qxzzs[is][X][Y][Z]; }
    double getQyzzs(int X, int Y, int Z, int is) const { return Qyzzs[is][X][Y][Z]; }
    double getQyyzs(int X, int Y, int Z, int is) const { return Qyyzs[is][X][Y][Z]; }

    //* Divergences
    arr3_double getDivE() { return divE; }
    arr3_double getDivB() { return divB; }
    arr3_double getDivAverage() { return divE_average; }

    //* Residual of DivE on cell centers *//
    double getResDiv(int X, int Y, int Z, int is) const { return residual_divergence.get(is, X, Y, Z); }

    void divergence_E(double ma);
    void divergence_B();
    void timeAveragedRho(double ma);
    void timeAveragedDivE(double ma);

    double get_E_field_energy();        //* Electric field energy
    double get_Ex_field_energy();       //* Electric field energy along X
    double get_Ey_field_energy();       //* Electric field energy along Y
    double get_Ez_field_energy();       //* Electric field energy along Z
    double get_B_field_energy();        //* Magnetic (internal) field energy along X
    double get_Bx_field_energy();       //* Magnetic (internal) field energy along Y
    double get_By_field_energy();       //* Magnetic (internal) field energy along Z
    double get_Bz_field_energy();       //* Magnetic (internal) field energy
    double get_Bext_energy();           //* External magnetic field energy
    double get_bulk_energy(int is);      //* Bulk kinetic energy

    void setZeroCurrent();
    void setZeroRho();

    double LOG_COSH(double x) 
    {
        double res;
        if (fabs(x) > 18.5) 
            res = fabs(x) - log(2.0);
        else 
            res = log(cosh(x));
      
        return res;
    }

    /*! fetch array for summing moments of thread i */
    Moments10& fetch_moments10Array(int i)
    {
        assert_le(0,i);
        assert_lt(i,sizeMomentsArray);
        return *(moments10Array[i]);
    }

    ECSIM_Moments13& fetch_moments13Array(int i)
    {
        assert_le(0, i);
        assert_lt(i, sizeMomentsArray);
        return *(ecsim_moments13Array[i]);
    }

    int get_sizeMomentsArray() { return sizeMomentsArray; }

    /*! print electromagnetic fields info */
    void print(void) const;
    
    //get MPI Derived Datatype
    MPI_Datatype getYZFacetype(bool isCenterFlag){return isCenterFlag ?yzFacetypeC : yzFacetypeN;}
    MPI_Datatype getXZFacetype(bool isCenterFlag){return isCenterFlag ?xzFacetypeC : xzFacetypeN;}
    MPI_Datatype getXYFacetype(bool isCenterFlag){return isCenterFlag ?xyFacetypeC : xyFacetypeN;}
    MPI_Datatype getXEdgetype(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC : xEdgetypeN;}
    MPI_Datatype getYEdgetype(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC : yEdgetypeN;}
    MPI_Datatype getZEdgetype(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC : zEdgetypeN;}
    MPI_Datatype getXEdgetype2(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC2 : xEdgetypeN2;}
    MPI_Datatype getYEdgetype2(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC2 : yEdgetypeN2;}
    MPI_Datatype getZEdgetype2(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC2 : zEdgetypeN2;}
    MPI_Datatype getCornertype(bool isCenterFlag){return  isCenterFlag ?cornertypeC : cornertypeN;}

    MPI_Datatype getProcview(){return  procview;}
    MPI_Datatype getXYZeType(){return xyzcomp;}
    MPI_Datatype getProcviewXYZ(){return  procviewXYZ;}
    MPI_Datatype getGhostType(){return  ghosttype;}

    void freeDataType();
    bool isLittleEndian(){return lEndFlag;};

public: // accessors
    const Collective& get_col()const{return _col;}
    const Grid& get_grid()const{return _grid;};
    const VirtualTopology3D& get_vct()const{return _vct;}

    //* Number of ghost cell layers per face (1 = legacy CIC, 2 = TSC).
    //* Cached from grid->getNGhost() at construction.
    int getNGhost()const{ return n_ghost_; }

#ifdef USE_PETSC
    // Mass matrix components (for PETSc preconditioner assembly)
    const_arr4_double getMxx() const { return Mxx; }
    const_arr4_double getMxy() const { return Mxy; }
    const_arr4_double getMxz() const { return Mxz; }
    const_arr4_double getMyx() const { return Myx; }
    const_arr4_double getMyy() const { return Myy; }
    const_arr4_double getMyz() const { return Myz; }
    const_arr4_double getMzx() const { return Mzx; }
    const_arr4_double getMzy() const { return Mzy; }
    const_arr4_double getMzz() const { return Mzz; }

    // Grid dimensions and physics constants (for PETSc preconditioner)
    int    get_nxn()    const { return nxn; }
    int    get_nyn()    const { return nyn; }
    int    get_nzn()    const { return nzn; }
    double get_dx()     const { return dx; }
    double get_dy()     const { return dy; }
    double get_dz()     const { return dz; }
    double get_c()      const { return c; }
    double get_th()     const { return th; }
    double get_dt()     const { return dt; }
    double get_FourPI() const { return FourPI; }
    double get_invVOL() const { return invVOL; }
#endif

    //* ********************************* VARIABLES ********************************* *//
    
private:
    //? access to global data
    const Collective& _col;
    const Grid& _grid;
    const VirtualTopology3D&_vct;

    //* Step 38: per-stage MaxwellImage dump bookkeeping.
    //* mi_dump_target_cycle_ is set from the main loop (currently 1); the first
    //* call to MaxwellImage during that cycle dumps, later calls don't.
    int mi_dump_target_cycle_ = -1;
    int mi_matvec_count_      = 0;
    
    /*! light speed */
    double c;
    /* 4*PI for normalization */
    double FourPI;
    /*! time step */
    double dt;
    /*! decentering parameter */
    double th;
    
    /*! Smoothing value */
    bool Smooth;
    int smooth_cycle;         // Frequency of smoothing (after every "smooth_cycle" time cycles)
    int num_smoothings;       // Number of times of smoothing at a given time cycle

    //* Custom input parameters
    double *input_param; int nparam;
    
    int zeroCurrent; double delt; int ns; double delta;
    
    //* Magnetic field
    double B0x, B0y, B0z, B1x, B1y, B1z;
    
    //* Charge to mass ratio
    double *qom;
    
    //* Boundary electron speed
    double ue0, ve0, we0;

    //! Mass matrix
    double *mass_matrix;

    //* Number of cells including 2 (guard cells)
    int nxc, nxn, nyc, nyn, nzc, nzn;

    //* Local grid boundaries coordinate
    double xStart, xEnd, yStart, yEnd, zStart, zEnd;

    //* Grid spacing
    double dx, dy, dz, invVOL;
    
    //* Simulation box length
    double Lx, Ly, Lz;
    
    //* Source
    double x_center, y_center, z_center, L_square;

    //* Particle/grid shape function order: 1 = linear (CIC), 2 = quadratic (TSC)
    //* Read from Collective::StencilOrder; default 1 (legacy byte-identical path).
    //* Declared early so it is initialized BEFORE the mass-matrix arrays below.
    const int stencil_order_;

    //* Number of ghost cell layers per face (cached from grid->getNGhost()).
    //* Linear -> 1 (legacy literal). Quadratic -> 2 (TSC mass-matrix product cube needs +/- 2 reach).
    const int n_ghost_;

    //* Number of distinct +/- offset groups in the mass-matrix product cube.
    //*   stencil_order_ = 1  ->  ne_mass_ = 14   (3x3x3 cube)
    //*   stencil_order_ = 2  ->  ne_mass_ = 63   (5x5x5 cube)
    const int ne_mass_;

    //? Electric field component used to move particles organized for rapid access
    //! This is the information transferred from cluster to booster
    array4_pfloat fieldForPcls;

    //? Electric field (defined at nodes)
    array3_double Ex, Ey, Ez;

    //? Implicit electric field (defined at nodes)
    array3_double Exth, Eyth, Ezth;
    //* Step 33: stashed MaxwellSource RHS in physical (node) space for the
    //* cross-code cycle-N byte diff. Populated right after MaxwellSource and
    //* before GMRES/PETSc so MaxwellImage iterations don't clobber them.
    array3_double bXn, bYn, bZn;
    //* Step 33: direct operator diagnostic — result of applying MaxwellImage
    //* to (bXn, bYn, bZn). If `b` matches across codes but `A·b` differs, the
    //* operator is the source of the E_θ divergence.
    array3_double AbXn, AbYn, AbZn;

    //? Magnetic field (defined at nodes)
    array3_double Bxn, Byn, Bzn;

    //? Magnetic field (defined at cell centres)
    array3_double Bxc, Byc, Bzc;

    //? Current density (defined at nodes)
    array3_double Jx, Jy, Jz;

    //? Implicit current density 
    array3_double Jxh, Jyh, Jzh;

    //? External magnetic field (defined at cell centres)
    array3_double Bxc_ext, Byc_ext, Bzc_ext;

    //? External magnetic field (defined at nodes)
    array3_double Bx_ext, By_ext, Bz_ext;

    //? Total magnetic field (defined at nodes)
    array3_double Bx_tot, By_tot, Bz_tot;

    //? External current (defined at nodes)
    array3_double Jx_ext, Jy_ext, Jz_ext;

    //? External electric field (defined at nodes)
    array3_double Ex_ext, Ey_ext, Ez_ext;

    //! Mass matrix components (defined at nodes)
    array4_double Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz;

    //! Step 68b: Kahan-compensated gather companion arrays.
    //* Populated only while `KahanGather=true`; sized identically to their
    //* primary partner. Each gathered field (ρ_s, Jxh_s, Jyh_s, Jzh_s, and
    //* the 9-component mass matrix) gets a per-node compensation scalar used
    //* by Neumaier's summation. Allocated unconditionally for layout
    //* simplicity; the zero-fill pass is cheap and the runtime cost when the
    //* flag is off is only the memory footprint.
    array4_double Mxx_c, Mxy_c, Mxz_c, Myx_c, Myy_c, Myz_c, Mzx_c, Mzy_c, Mzz_c;

    //? Density for each species (defined at nodes and centres, respectively)
    array4_double rhons, rhocs;

    array3_double rhoc_avg;             //* Time averaged density (defined at cell centres)
    array4_double rhocs_avg;            //* Time averaged density for each species (defined at cell centres)

    //? Current densities for each species (defined at nodes)
    array4_double Jxs, Jys, Jzs, Jxhs, Jyhs, Jzhs;

    //? Step 68b: Kahan companions for species-level gather accumulators.
    array4_double rhons_c, Jxhs_c, Jyhs_c, Jzhs_c;
    
    //! Supplementary moments
    //? Energy flux for each species (defined at nodes)
    array4_double E_flux_xs, E_flux_ys, E_flux_zs;

    //? Number of particles for each species (defined at nodes)
    array4_double Nns;

    bool SaveHeatFluxTensor;

    //? Heat Flux Tensor (defined at nodes)
    double**** Qxxxs; double**** Qyyys; double**** Qzzzs;  
    double**** Qxxys; double**** Qxzzs; double**** Qxyys; double**** Qxyzs; 
    double**** Qyzzs; double**** Qyyzs; double**** Qxxzs;

    //? Pressure Tensor (defined at nodes)
    array4_double pXXsn, pXYsn, pXZsn, pYYsn, pYZsn, pZZsn;

    //? Density (defined at nodes and centres, respectively)
    array3_double rhon, rhoc;            
    
    //? Implicit density (defined at cell centres)
    // array3_double rhoh;

    //? Electric potential (defined at cell centres)
    array3_double PHI, Phic;  

    //? Used in computing divergence
    array3_double divC, divB, divE, divE_average;
    array4_double residual_divergence;

    //? ***************** TEMPORARY ARRAYS ***************** ?//

    //* Used in MaxwellSource()
    array3_double tempC, tempXC, tempYC, tempZC, tempXC2, tempYC2, tempZC2;
    array3_double tempX, tempY, tempZ, temp2X, temp2Y, temp2Z, temp3X, temp3Y, temp3Z, smooth_temp;
    array3_double tempXN, tempYN, tempZN;

    //* Temporary arrays for MaxwellImage() *//
    array3_double imageX, imageY, imageZ, Dx, Dy, Dz, vectX, vectY, vectZ;

    //* Arrays for summing moments *//
    int sizeMomentsArray;
    Moments10 **moments10Array;
    ECSIM_Moments13 **ecsim_moments13Array;

    //* Object of class to handle which nodes have to be computed when the mass matrix is calculated
    NeighbouringNodes NeNo;

    /*! Field Boundary Condition
      0 = Dirichlet Boundary Condition: specifies the
          value on the boundary of the domain
      1 = Neumann Boundary Condition: specifies the value of
          derivative on the boundary of the domain
      2 = Periodic boundary condition */

    // boundary conditions for electrostatic potential
    //
    int bcPHIfaceXright;
    int bcPHIfaceXleft;
    int bcPHIfaceYright;
    int bcPHIfaceYleft;
    int bcPHIfaceZright;
    int bcPHIfaceZleft;

    /*! Boundary condition for electric field 0 = perfect conductor 1 = magnetic mirror */
    // boundary conditions for EM field
    int bcEMfaceXright;
    int bcEMfaceXleft;
    int bcEMfaceYright;
    int bcEMfaceYleft;
    int bcEMfaceZright;
    int bcEMfaceZleft;

    /*! GEM Challenge background ion */
    double *rhoINIT;
    /*! Drift of the species */
    bool *DriftSpecies;
    
    int restart_status;

    double GMREStol;                            //* GMRES tolerance criterium for stopping iterations

    double Fext;
    double Fzro;
    bool Fpext;
    
    double u_bulk, v_bulk, w_bulk;              //* Bulk velocity along X, Y, Z, respectively for the Lagrangian reference frame

    //MPI Derived Datatype for Center Halo Exchange
    MPI_Datatype yzFacetypeC;
    MPI_Datatype xzFacetypeC;
    MPI_Datatype xyFacetypeC;
    MPI_Datatype xEdgetypeC;
    MPI_Datatype yEdgetypeC;
    MPI_Datatype zEdgetypeC;
    MPI_Datatype xEdgetypeC2;
    MPI_Datatype yEdgetypeC2;
    MPI_Datatype zEdgetypeC2;
    MPI_Datatype cornertypeC;

    //MPI Derived Datatype for Node Halo Exchange
    MPI_Datatype yzFacetypeN;
    MPI_Datatype xzFacetypeN;
    MPI_Datatype xyFacetypeN;
    MPI_Datatype xEdgetypeN;
    MPI_Datatype yEdgetypeN;
    MPI_Datatype zEdgetypeN;
    MPI_Datatype xEdgetypeN2;
    MPI_Datatype yEdgetypeN2;
    MPI_Datatype zEdgetypeN2;
    MPI_Datatype cornertypeN;
    
    //for VTK output
    MPI_Datatype  procviewXYZ,xyzcomp,procview,ghosttype;
    bool lEndFlag;
    
    void OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz);
    void OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int nx, int ny, int nz);
    void OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                                 const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                                 int nx, int ny, int nz);

#ifdef USE_PETSC
    PetscSolver *petscSolver_ = nullptr;
#endif
};

//* Add an amount of charge density to charge density field at node X,Y,Z
// See note above add_Rho_node for the atomic-update rationale.
inline void EMfields3D::add_Rho(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                #pragma omp atomic update
                rhons[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
            }
}

//* Add an amount of current density to current density field at node X,Y,Z
inline void EMfields3D::add_Jxh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                #pragma omp atomic update
                Jxhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
            }
}
inline void EMfields3D::add_Jyh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                #pragma omp atomic update
                Jyhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
            }
}
inline void EMfields3D::add_Jzh(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                #pragma omp atomic update
                Jzhs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
            }
}

//* Step 68b: Kahan-compensated CIC (8-corner) deposit variants.
inline void EMfields3D::add_Rho_kahan(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                const int xi = X - i, yi = Y - j, zi = Z - k;
                kahan_add(rhons[is][xi][yi][zi], rhons_c[is][xi][yi][zi],
                          weight[i * 4 + j * 2 + k] * invVOL);
            }
}
inline void EMfields3D::add_Jxh_kahan(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                const int xi = X - i, yi = Y - j, zi = Z - k;
                kahan_add(Jxhs[is][xi][yi][zi], Jxhs_c[is][xi][yi][zi],
                          weight[i * 4 + j * 2 + k]);
            }
}
inline void EMfields3D::add_Jyh_kahan(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                const int xi = X - i, yi = Y - j, zi = Z - k;
                kahan_add(Jyhs[is][xi][yi][zi], Jyhs_c[is][xi][yi][zi],
                          weight[i * 4 + j * 2 + k]);
            }
}
inline void EMfields3D::add_Jzh_kahan(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                const int xi = X - i, yi = Y - j, zi = Z - k;
                kahan_add(Jzhs[is][xi][yi][zi], Jzhs_c[is][xi][yi][zi],
                          weight[i * 4 + j * 2 + k]);
            }
}

inline void EMfields3D::add_Jx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jxs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Jy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Jz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Jzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_N(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                Nns[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k];
}

inline void EMfields3D::add_Pxx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXXsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pxy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXYsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pxz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pXZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pYYsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pyz(double weight[8], int X, int Y, int Z, int is)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pYZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Pzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                pZZsn[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_x(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_xs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_y(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_ys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_E_flux_z(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                E_flux_zs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxx(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxxs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyyys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qzzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qzzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxyz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxyzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxxz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxxzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxyy(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxyys[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qxzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qxzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyzz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyzzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

inline void EMfields3D::add_Qyyz(double weight[8], int X, int Y, int Z, int is) 
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) 
                Qyyzs[is][X - i][Y - j][Z - k] += weight[i * 4 + j * 2 + k] * invVOL;
}

//* Add an amount of current density to mass matrix field at node X,Y *//
// See note above add_Rho_node for the atomic-update rationale. add_Mass is called
// once per node-pair in the 2x2x2 x 14-group mass-matrix assembly (Particles3D.cpp
// ~1801), so it's the hottest atomic site; measured ~30% gather overhead vs no-atomic.
inline void EMfields3D::add_Mass(double value[3][3], int X, int Y, int Z, int ind)
{
    #pragma omp atomic update
    Mxx[ind][X][Y][Z] += value[0][0];
    #pragma omp atomic update
    Mxy[ind][X][Y][Z] += value[0][1];
    #pragma omp atomic update
    Mxz[ind][X][Y][Z] += value[0][2];
    #pragma omp atomic update
    Myx[ind][X][Y][Z] += value[1][0];
    #pragma omp atomic update
    Myy[ind][X][Y][Z] += value[1][1];
    #pragma omp atomic update
    Myz[ind][X][Y][Z] += value[1][2];
    #pragma omp atomic update
    Mzx[ind][X][Y][Z] += value[2][0];
    #pragma omp atomic update
    Mzy[ind][X][Y][Z] += value[2][1];
    #pragma omp atomic update
    Mzz[ind][X][Y][Z] += value[2][2];
}

//* Step 68b: Kahan-compensated mass matrix accumulator.
inline void EMfields3D::add_Mass_kahan(double value[3][3], int X, int Y, int Z, int ind)
{
    kahan_add(Mxx[ind][X][Y][Z], Mxx_c[ind][X][Y][Z], value[0][0]);
    kahan_add(Mxy[ind][X][Y][Z], Mxy_c[ind][X][Y][Z], value[0][1]);
    kahan_add(Mxz[ind][X][Y][Z], Mxz_c[ind][X][Y][Z], value[0][2]);
    kahan_add(Myx[ind][X][Y][Z], Myx_c[ind][X][Y][Z], value[1][0]);
    kahan_add(Myy[ind][X][Y][Z], Myy_c[ind][X][Y][Z], value[1][1]);
    kahan_add(Myz[ind][X][Y][Z], Myz_c[ind][X][Y][Z], value[1][2]);
    kahan_add(Mzx[ind][X][Y][Z], Mzx_c[ind][X][Y][Z], value[2][0]);
    kahan_add(Mzy[ind][X][Y][Z], Mzy_c[ind][X][Y][Z], value[2][1]);
    kahan_add(Mzz[ind][X][Y][Z], Mzz_c[ind][X][Y][Z], value[2][2]);
}

inline void get_field_components_for_cell(const double* field_components[8], const_arr4_double fieldForPcls, int cx, int cy, int cz)
{
    // interface to the right of cell
    const int ix = cx + 1;
    const int iy = cy + 1;
    const int iz = cz + 1;

    arr3_double_get field0 = fieldForPcls[ix];
    arr3_double_get field1 = fieldForPcls[cx];
    arr2_double_get field00 = field0[iy];
    arr2_double_get field01 = field0[cy];
    arr2_double_get field10 = field1[iy];
    arr2_double_get field11 = field1[cy];
    field_components[0] = field00[iz]; // field000 
    field_components[1] = field00[cz]; // field001 
    field_components[2] = field01[iz]; // field010 
    field_components[3] = field01[cz]; // field011 
    field_components[4] = field10[iz]; // field100 
    field_components[5] = field10[cz]; // field101 
    field_components[6] = field11[iz]; // field110 
    field_components[7] = field11[cz]; // field111 
}

typedef EMfields3D Field;

#endif
