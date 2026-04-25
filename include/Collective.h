/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
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

/***************************************************************************
  Collective.h  -  Stefano Markidis, Giovanni Lapenta
  -------------------------------------------------------------------------- */


/*! Collective properties. Input physical parameters for the simulation.  Use ConfigFile to parse the input file @date Wed Jun 8 2011 @par Copyright: (C) 2011 K.U. LEUVEN @author Pierre Henri, Stefano Markidis @version 1.0 */

#ifndef Collective_H
#define Collective_H

#ifdef BATSRUS
#include "InterfaceFluid.h"
#endif
#include <iostream>
#include <string>
#include <memory>
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "aligned_vector.h"

class ConfigFile;
using namespace std;

class Collective
#ifdef BATSRUS
: public InterfaceFluid
#endif
{
  private:
    enum Enum{
      thedefault=0,
      initial,
      final,
      // used by ImplSusceptMode
      explPredict,
      implPredict,
      NUMBER_OF_ENUMS, // this must be last
      INVALID_ENUM
    };
    int read_enum_parameter(const char* option_name, const char* default_value,
      const ConfigFile& config);
  public:
    static const char* get_name_of_enum(int in);
  public:
    /*! constructor: initialize physical parameters with values */
    Collective(int argc, char **argv);
    /*! read input file */
    void ReadInput(string inputfile);
    /*! read the restart input file from HDF5 */
    int ReadRestart(string inputfile);

    void trim_conserved_quantities_file(const std::string& filename, int restart_cycle);

    void read_field_restart(const VCtopology3D* vct,const Grid* grid, arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
                            arr3_double Bxc, arr3_double Byc, arr3_double Bzc,
    						arr3_double Ex, arr3_double Ey, arr3_double Ez,
                            arr3_double rhoc_avg, arr3_double divE_avg, int ns)const;

    void read_particles_restart(const VCtopology3D* vct,int species_number,vector_double& u,vector_double& v,vector_double& w,
    							vector_double& q,vector_double& x,vector_double& y,vector_double& z,vector_double& t)const;

    void init_derived_parameters();
    /*! Print physical parameters */
    void Print();
    /*! save setting in a file */
    void save();

    // accessors
    int getDim()                        const{ return (dim); }
    double getLx()                      const{ return (Lx); }
    double getLy()                      const{ return (Ly); }
    double getLz()                      const{ return (Lz); }
    double getx_center()                const{ return (x_center); }
    double gety_center()                const{ return (y_center); }
    double getz_center()                const{ return (z_center); }
    double getL_square()                const{ return (L_square); }
    int getNxc()                        const{ return (nxc); }
    int getNyc()                        const{ return (nyc); }
    int getNzc()                        const{ return (nzc); }
    int getXLEN()                       const{ return (XLEN); }
    int getYLEN()                       const{ return (YLEN); }
    int getZLEN()                       const{ return (ZLEN); }
    bool getPERIODICX()                 const{ return (PERIODICX); }
    bool getPERIODICY()                 const{ return (PERIODICY); }
    bool getPERIODICZ()                 const{ return (PERIODICZ); }
    bool getPERIODICX_P()               const{ return (PERIODICX_P); }
    bool getPERIODICY_P()               const{ return (PERIODICY_P); }
    bool getPERIODICZ_P()               const{ return (PERIODICZ_P); }
    double getDx()                      const{ return (dx); }
    double getDy()                      const{ return (dy); }
    double getDz()                      const{ return (dz); }
    double getC()                       const{ return (c); }
    double getDt()                      const{ return (dt); }
    double getTh()                      const{ return (th); }    
    double getSmooth()                  const{ return (Smooth); }
    int getNcycles()                    const{ return (ncycles); }

    int getNs()                         const{ return (ns); }
    int getNsTestPart()                 const{ return (nstestpart); }
    int getNpcel(int nspecies)          const{ return (npcel[nspecies]); }
    int getNpcelx(int nspecies)         const{ return (npcelx[nspecies]); }
    int getNpcely(int nspecies)         const{ return (npcely[nspecies]); }
    int getNpcelz(int nspecies)         const{ return (npcelz[nspecies]); }

    double getNpMaxNpRatio()            const{ return (NpMaxNpRatio); }
    double getQOM(int nspecies)         const{ return (qom[nspecies]); }
    double getRHOinit(int nspecies)     const{ return (rhoINIT[nspecies]); }
    double getRHOinject(int nspecies)   const{ return(rhoINJECT[nspecies]); }
    double getUth(int nspecies)         const{ return (uth[nspecies]); }
    double getVth(int nspecies)         const{ return (vth[nspecies]); }
    double getWth(int nspecies)         const{ return (wth[nspecies]); }
    double getU0(int nspecies)          const{ return (u0[nspecies]); }
    double getV0(int nspecies)          const{ return (v0[nspecies]); }
    double getW0(int nspecies)          const{ return (w0[nspecies]); }

    double getPitchAngle(int nspecies)  const{ return (pitch_angle[nspecies]); }
    double getEnergy(int nspecies)      const{ return (energy[nspecies]); }

    //? Nonperiodic boundaries
    int getBcPfaceXright()              const{ return (bcPfaceXright); }
    int getBcPfaceXleft()               const{ return (bcPfaceXleft); }
    int getBcPfaceYright()              const{ return (bcPfaceYright); }
    int getBcPfaceYleft()               const{ return (bcPfaceYleft); }
    int getBcPfaceZright()              const{ return (bcPfaceZright); }
    int getBcPfaceZleft()               const{ return (bcPfaceZleft); }
    int getBcPHIfaceXright()            const{ return (bcPHIfaceXright); }
    int getBcPHIfaceXleft()             const{ return (bcPHIfaceXleft); }
    int getBcPHIfaceYright()            const{ return (bcPHIfaceYright); }
    int getBcPHIfaceYleft()             const{ return (bcPHIfaceYleft); }
    int getBcPHIfaceZright()            const{ return (bcPHIfaceZright); }
    int getBcPHIfaceZleft()             const{ return (bcPHIfaceZleft); }
    int getBcEMfaceXright()             const{ return (bcEMfaceXright); }
    int getBcEMfaceXleft()              const{ return (bcEMfaceXleft); }
    int getBcEMfaceYright()             const{ return (bcEMfaceYright); }
    int getBcEMfaceYleft()              const{ return (bcEMfaceYleft); }
    int getBcEMfaceZright()             const{ return (bcEMfaceZright); }
    int getBcEMfaceZleft()              const{ return (bcEMfaceZleft); }

    double getB0x()                     const{ return (B0x); }
    double getB0y()                     const{ return (B0y); }
    double getB0z()                     const{ return (B0z); }
    double getB1x()                     const{ return (B1x); }
    double getB1y()                     const{ return (B1y); }
    double getB1z()                     const{ return (B1z); }

    int getRestart_status()             const{ return (restart_status); }
    string getSaveDirName()             const{ return (SaveDirName); }
    string getRestartDirName()          const{ return (RestartDirName); }
    string getinputfile()               const{ return (inputfile); }
    string getCase()                    const{ return (Case); }
    string getSimName()                 const{ return (SimName); }
    string getWriteMethod()             const{ return (wmethod); }
    string getFieldOutputTag()          const{ return FieldOutputTag;}
    string getMomentsOutputTag()        const{ return MomentsOutputTag;}
    string getPclOutputTag()            const{ return ParticlesOutputTag;}
    string getPclDSOutputTag()          const{ return ParticlesDownsampleOutputTag;}
    string get_output_data_precision()  const{ return output_data_precision; }

    double getVinj()                    const{ return (Vinj); }
    double getGMREStol()                const{ return (GMREStol); }
    int    getNiterGMRES()              const{ return (NiterGMRES); }
    string getSolverType()              const{ return (SolverType); }
    //* Step 11: MaxwellImage operator choice ("curl_curl" legacy | "lap_graddiv" self-adjoint).
    string getMaxwellOperator()         const{ return (MaxwellOperator); }
    bool getPrecMatrix()                const{ return (PrecMatrix); }
    bool getPrecDiagnostics()           const{ return (PrecDiagnostics); }
    string getPrecType()                const{ return (PrecType); }
    string getStencilOrder()            const{ return (StencilOrder); }
    int    getStencilOrderInt()         const{ return (stencilOrderInt); }
    int getLast_cycle()                 const{ return (last_cycle); }
    int getNiterMover()                 const{ return (NiterMover); }
    int getFieldOutputCycle()           const{ return (FieldOutputCycle); }
    int getParticlesOutputCycle()       const{ return (ParticlesOutputCycle); }
    int getTestParticlesOutputCycle()   const{ return (TestParticlesOutputCycle); }
    int getRestartOutputCycle()         const{ return (RestartOutputCycle); }
    int getDiagnosticsOutputCycle()     const{ return (DiagnosticsOutputCycle); }
    int getParticlesDownsampleOutputCycle()    const{ return (ParticlesDownsampleOutputCycle); }
    int getParticlesDownsampleFactor()  const{ return (ParticlesDownsampleFactor); }
    
    bool getCallFinalize()              const{ return (CallFinalize); }
    bool particle_output_is_off()       const;
    bool DS_particle_output_is_off()    const;
    bool testparticle_output_is_off()   const;
    bool field_output_is_off()          const;
    bool getSaveHeatFluxTensor()        const { return (SaveHeatFluxTensor); }

    bool getAddExternalCurlB()          const { return AddExternalCurlB; }
    bool getAddExternalCurlE()          const { return AddExternalCurlE; }
    int getZeroCurrent()                      { return zeroCurrent; }

    int getSmoothCycle()                const { return SmoothCycle; }
    int getNumSmoothings()              const { return num_smoothings; }
    string getSmoothKernel()            const { return SmoothKernel; }
    int    getSmoothKernelInt()         const { return smoothKernelInt; }
    double getHelmholtzAlpha()          const { return HelmholtzAlpha; }
    int    getHelmholtzNiter()          const { return HelmholtzNiter; }
    bool   getPostSolveHelmholtz()      const { return PostSolveHelmholtz; }
    bool   getVerifyAdjoint()           const { return VerifyAdjoint; }
    bool   getVerifySmoothSymmetry()    const { return VerifySmoothSymmetry; }
    bool   getVerifySubspacePreservation() const { return VerifySubspacePreservation; }
    bool   getDumpCycleIdentity()       const { return DumpCycleIdentity; }
    bool   getDumpParticlesInit()       const { return DumpParticlesInit; }
    bool   getLoadParticlesInit()       const { return LoadParticlesInit; }
    const string& getParticlesInitDir() const { return ParticlesInitDir; }
    bool   getDumpCycle1Fields()        const { return DumpCycle1Fields; }
    bool   getDumpMaxwellImageStages()  const { return DumpMaxwellImageStages; }
    bool   getSubcycleMover()           const { return SubcycleMover; }
    bool   getDeterministicMPIReductions()   const { return DeterministicMPIReductions; }
    bool   getDeterministicThreadMoments()   const { return DeterministicThreadMoments; }
    bool   getDumpAlphaBothPaths()           const { return DumpAlphaBothPaths; }
    bool   getDeterministicParticleComm()    const { return DeterministicParticleComm; }
    bool   getDumpParticlesGlobal()          const { return DumpParticlesGlobal; }
    bool   getLoadParticlesGlobal()          const { return LoadParticlesGlobal; }
    bool   getKahanParticleSums()            const { return KahanParticleSums; }
    bool   getKahanGather()                  const { return KahanGather; }
    bool   getKahanFieldEnergy()             const { return KahanFieldEnergy; }
    bool   getKahanHalo()                    const { return KahanHalo; }
    int getCurrentCycle()               const { return CurrentCycle; }
    void setCurrentCycle(int cycle)           { CurrentCycle = cycle; }

    double getPoissonMAdiv()                  { return PoissonMAdiv; }
    double getPoissonMAres()                  { return PoissonMAres; }
    double getPoissonMArho()                  { return PoissonMArho; }

    //* Particle distribution parameters
    int getParticleDistOutputCycle()          { return ParticleDistOutputCycle; }
    int getParticleDistBins()                 { return ParticleDistBins; }
    double getParticleDistMinVelocity()       { return ParticleDistMinVelocity; }
    double getParticleDistMaxVelocity()       { return ParticleDistMaxVelocity; }

    bool getRelativistic()              const { return Relativistic; }
    string getRelativisticPusher()      const { return Relativistic_pusher; }   

    int getNparam()                     const { return (nparam); }
    double getInputParam(int ip)        const { return (input_param[ip]); }
    
    /*! Boundary condition selection for BCFace for the electric field components */
    int bcEx[6], bcEy[6], bcEz[6];
    /*! Boundary condition selection for BCFace for the magnetic field components */
    int bcBx[6], bcBy[6], bcBz[6];
    string getParaviewScriptPath()      const { return ParaviewScriptPath; }

  private:
    /*! inputfile */
    string inputfile;
    string ParaviewScriptPath;

    double c; double fourpi; double dt; double th; //! second-order for th=1/2; stable for 1/2 <= th <= 1

    int ImplSusceptMode; // "initial" (default), "explPredict", "implPredict"
    
    //* Smoothing parameters
    double Smooth; int num_smoothings; int SmoothCycle;

    //* Phase 10i: choice of binomial smoother kernel applied to E/J in energy_conserve_smooth_direction.
    //*   "binomial"  (default) -> (1,2,1)/4 per-dim tensor product, 27-point 3D, 1-cell half-width.
    //*   "binomial5"            -> (1,4,6,4,1)/16 per-dim tensor product, 125-point 3D, 2-cell half-width.
    //*                             Needs n_ghost >= 2 (already the TSC default).
    //* Phase 10k: low-k–aware Helmholtz filter:
    //*   "helmholtz" -> (I - α ∇²) S_new = S_old, solved per-call by a few CG iterations.
    //*                  α has units of length²; default 0 -> auto = (max(Lx,Ly,Lz)/(2π))²
    //*                  which puts the half-power point at the domain fundamental k = 2π/L.
    //* Phase 10l: bin5 decomposed into bin × halo refresh × bin (Finding 26 follow-up).
    //*   "binomial5_refresh" -> per pass: (1,2,1)/4 → communicateNodeBC → (1,2,1)/4.
    //*                          Equals binomial sm=2N bit-for-bit if Finding 26 is right
    //*                          (the missing piece in plain binomial5 is the inter-pass halo refresh).
    string SmoothKernel;
    int    smoothKernelInt;     // 0 = binomial (3-pt), 1 = binomial5 (5-pt), 2 = helmholtz, 3 = binomial5_refresh

    //* Phase 10k Helmholtz smoother knobs
    double HelmholtzAlpha;      // 0 (default) -> auto from box size; >0 -> explicit α
    int    HelmholtzNiter;      // inner CG iterations per smoother call (default 12)

    //* Phase 10m: enable a once-per-cycle Helmholtz low-pass filter applied to (Ex, Ey, Ez)
    //* AFTER `calculateE()` returns and BEFORE `calculateB()` and the particle push.
    //* Decoupled from `MaxwellImage`'s S·M·S, so the implicit operator structure
    //* (which Phase 10k showed is fragile to long-range filters) is unchanged.
    bool   PostSolveHelmholtz;

    //* Step 34b: programmatic self-adjointness probe for MaxwellImage. When true,
    //* at cycle 1 (after MaxwellSource, before the Krylov solve) computes
    //* <A·u, v> − <u, A·v> on two deterministic pseudo-random Krylov vectors
    //* and prints the absolute/relative gap. Off by default.
    bool   VerifyAdjoint;

    //* Smoothing-operator symmetry probe. When true, at cycle 1 applies S
    //* (component-wise `energy_conserve_smooth_direction`) to two deterministic
    //* pseudo-random Krylov vectors u, v and prints <S·u, v> − <u, S·v>.
    //* Verifies Lapenta-2023 condition that the smoothing matrix is symmetric,
    //* which is required for exact energy conservation when smoothing fires.
    //* Off by default.
    bool   VerifySmoothSymmetry;

    //* Subspace-preservation probe. When true, at cycle 1 applies MaxwellImage
    //* to a deterministic input that has been pre-projected onto the
    //* consistent-periodic subspace (duplicate node pairs equal by construction)
    //* and reports the max |output[duplicate1] - output[duplicate2]| across all
    //* periodic-self axes. Distinguishes "matvec drifts off subspace at FP-ε"
    //* from "matvec is bit-exact on subspace, GMRES iterates drift instead".
    //* Off by default.
    bool   VerifySubspacePreservation;


    //* Step 25: cycle-1 identity decomposition print. When true, calculateE prints
    //* I_J = dt · <Eth, Jxh>_unique and I_M = dt · <Eth, M·Eth>_unique so external
    //* post-processing can compute R_part = ΔKE − (I_J + I_M) and R_field =
    //* ΔUE + ΔUB + (I_J + I_M). Cheap (~two extra matvec passes) but gated off.
    bool   DumpCycleIdentity;

    //* Step 3: enable ECSIM-style combined velocity+position mover with adaptive
    //* sub-cycling (dt_sub = π·c/(4·|qom|·B)), `NiterMover` inner midpoint iterations,
    //* midpoint-velocity position update. Port of ecsim/particles/Particles3D.cpp:4209.
    //* Default false — opt-in to compare with the legacy ECSIM_velocity+ECSIM_position path.
    bool   SubcycleMover;

    //* Step 31: particle-state dump / load for cross-code (iPIC3D ↔ ECSIM) byte diff.
    //* `DumpParticlesInit`: after species init and before cycle 1, write each species'
    //* particles to `{ParticlesInitDir}/particles_init_s{ns}_r{rank}.txt` — 17-digit
    //* space-separated `x y z u v w q`, one line per particle, no ordering guarantee.
    //* `LoadParticlesInit`: after the case's init has run (harmless filler), clear
    //* `_pcls` and repopulate from the file. Default both off.
    bool   DumpParticlesInit;
    bool   LoadParticlesInit;
    string ParticlesInitDir;

    //* Step 32: after cycle 1 (the earliest point at which Jxh, M, and E_th have
    //* all been updated from initial state), write raw-binary dumps of all node
    //* fields into `{SaveDirName}/fields_cycle1_{name}.bin` — one IEEE-754 double
    //* array per file, row-major C order with k (z) fastest. A companion
    //* `fields_cycle1.meta.txt` lists names + shapes. Designed so ECSIM can
    //* emit the same filename+layout convention for a trivial Python byte diff.
    bool   DumpCycle1Fields;

    //* Step 38: inside MaxwellImage, at cycle 1, first matvec only, dump tempX/Y/Z,
    //* imageX/Y/Z, temp2X/Y/Z at six composition stages (post-input-halo, post-curl²
    //* assembly, pre-M·E, raw M·E, post-outer-smooth×invVOL, final A·E). Used with
    //* `scripts/diff_maxwell_stages.py` for cross-code operator-interior byte diff.
    bool   DumpMaxwellImageStages;

    //* Step 62: force bit-deterministic scalar MPI_SUM allreduce (rank-order gather→
    //* sum→broadcast) for GMRES dot products / norms and energy diagnostics. Baseline
    //* MPI_Allreduce trees can reassociate summation depending on process layout, so
    //* two identical np>1 runs drift by ±1 ULP. Opt-in: adds 2·log(p) extra comms per
    //* reduction but guarantees cross-run bit-reproducibility. Sets the module-level
    //* `g_deterministic_mpi_reductions` flag in `utility/Basic.cpp` during init.
    bool   DeterministicMPIReductions;

    //* Step 63: serialize `computeMoments`' OpenMP parallel region (num_threads=1).
    //* The ECSIM gather writes to shared grid nodes via `#pragma omp atomic update`;
    //* atomics protect against races but the landing order of thread writes still
    //* varies run to run, which drifts the result by ±1 ULP at the same node. Opt-in
    //* flag that gives cross-OMP-thread-count bit-reproducibility at the cost of
    //* losing OpenMP parallelism inside the gather. MPI parallelism is unaffected.
    bool   DeterministicThreadMoments;

    //* Step 64: mover-gather α-parity audit. When true, both `computeMoments`
    //* (gather) and `ECSIM_velocity` (mover) emit per-particle α tensors to
    //* binary files `{SaveDirName}/alpha_{gather|mover}_cyc{N}_s{s}_r{r}.bin`.
    //* Layout per particle: 16 doubles — [pidx_as_double, x, y, z, Bx, By, Bz,
    //* α00..α22 (row-major)]. Post-process with `scripts/diff_alpha_paths.py`
    //* to confirm ECSIM condition #4 (mover-α ≡ gather-α).
    bool   DumpAlphaBothPaths;

    //* Step 66: process incoming particle blocks in fixed direction order
    //* [XDN, XUP, YDN, YUP, ZDN, ZUP] instead of the OS-scheduled completion
    //* order returned by `MPI_Waitany`. Each direction is drained fully
    //* (via `MPI_Wait` on the streaming block communicator) before moving
    //* to the next. Removes the last source of run-to-run non-determinism
    //* in the particle pipeline at np>1: with `MPI_Waitany` the order that
    //* particles are appended to `_pcls` varies per run, so subsequent serial
    //* sums (kinetic energy, gather) walk the list in a different FP order
    //* and drift by ±1 ULP even with `DeterministicMPIReductions` on.
    //* Opt-in (default false) because switching to per-direction sequential
    //* drain is a behavioural change even when the outputs are mathematically
    //* equivalent. Combine with `DeterministicMPIReductions=1` and
    //* `DeterministicThreadMoments=1` for full same-config bit reproducibility
    //* at np>1.
    bool   DeterministicParticleComm;

    //* Step 68: global-to-local particle dump/load so an np=1 reference state
    //* can be consumed by an np>1 run (and vice versa). Unlike
    //* DumpParticlesInit/LoadParticlesInit — which write one file per
    //* (species, rank) and therefore round-trip only at matched decomposition
    //* — the "global" variants aggregate to rank 0 via `MPI_Gatherv` and emit
    //* one file per species. On load, every rank reads the full file and
    //* keeps only particles whose position falls in its local subdomain
    //* (`Grid3DCU::get{X,Y,Z}{start,end}`), so the decomposition can change.
    //* Opt-in. Cost is only the extra communication/disk during Init.
    bool   DumpParticlesGlobal;
    bool   LoadParticlesGlobal;

    //* Step 68: Kahan-compensated accumulation inside the serial per-particle
    //* sums in `Particles3Dcomm::get_kinetic_energy`, `get_total_charge`,
    //* `get_momentum`. Plain accumulation is FP-non-associative, so the
    //* single-accumulator result at np=1 drifts from the per-rank-partial
    //* result at np>1 by ~1 ULP per cycle even with matched particles and
    //* `DeterministicMPIReductions`. Kahan reduces the per-addition error to
    //* O(ε²) which is below IEEE 754's resolution, so per-rank partial sums
    //* match the np=1 single accumulator to bit identity. Opt-in.
    bool   KahanParticleSums;

    //* Step 68b: Kahan-compensated accumulation inside the particle→grid
    //* gather (`Particles3D::computeMoments`). The default path accumulates
    //* ρ, Jxh, Jyh, Jzh, and the 9-component mass matrix via `#pragma omp
    //* atomic update`, so threads race on the same node and the landing
    //* order is schedule-dependent. Under `KahanGather=true`, a companion
    //* compensation array is allocated for each deposited field and the
    //* accumulators do Neumaier-compensated adds (atomics replaced by
    //* sequential single-thread sums — the flag forces `num_threads=1` in
    //* the gather region). Before halo exchange, compensation is folded
    //* back into the main field and zeroed. Result: each rank's local sum
    //* is ε²-accurate regardless of particle ordering, and combined with
    //* `DeterministicParticleComm` the per-cycle np=1 vs np=4 drift of ~3e-9
    //* collapses toward the halo-exchange floor. Memory cost: one extra
    //* companion array per accumulated field (4 × ns + 9 × ne_mass_ node
    //* arrays). Opt-in.
    bool   KahanGather;

    //* Step 68c: Kahan-compensated grid field-energy reductions
    //* (`get_E_field_energy`, `get_B_field_energy` and their per-axis
    //* variants). Serial grid sums followed by `MPI_Allreduce`; at
    //* different decompositions the per-rank partial walks different
    //* nodes in different orders, drifting by O(ε) per node. Kahan brings
    //* per-rank sums to ε², so `DeterministicMPIReductions` can combine
    //* them cross-rank without adding noise. Opt-in, default off, zero
    //* runtime cost when off (hoisted const branch).
    bool   KahanFieldEnergy;

    //* Step 68c: Kahan-aware halo exchange. The sum-on-receive in
    //* `communicateInterp` (`addFace`/`addEdge*`/`addCorner`) plainly
    //* accumulates neighbour ghost values into this rank's interior
    //* boundary, which is the last cross-decomposition FP drift source
    //* after Step 68b. Under `KahanHalo=true`, `NBDerivedHaloComm` takes an
    //* optional companion pointer and dispatches the 5 sum-on-receive
    //* sites to `*_kahan` variants; `communicateGhostP2G_{ecsim,mass_matrix}`
    //* fold compensation once more after the halo to merge halo-add
    //* residuals into the primary. Legacy path byte-identical when off.
    //* Opt-in.
    bool   KahanHalo;

    int CurrentCycle;
    int zeroCurrent;

    /*! number of time cycles */
    int ncycles;
    /*! physical space dimensions */
    int dim;
    /*! simulation box length - X direction */
    double Lx;
    /*! simulation box length - Y direction */
    double Ly;
    /*! simulation box length - Z direction */
    double Lz;
    /*! object center - X direction */
    double x_center;
    /*! object center - Y direction */
    double y_center;
    /*! object center - Z direction */
    double z_center;
    /*! object size - assuming a cubic box */
    double L_square;
    // number of cells per direction of problem domain
    int nxc;
    int nyc;
    int nzc;
    /*! grid spacing - X direction */
    double dx;
    /*! grid spacing - Y direction */
    double dy;
    /*! grid spacing - Z direction */
    double dz;
    /*! number of MPI subdomains in each direction */
    int XLEN;
    int YLEN;
    int ZLEN;
    /*! periodicity in each direction */
    bool PERIODICX;
    bool PERIODICY;
    bool PERIODICZ;
    /*! Particle periodicity in each direction */
    bool PERIODICX_P;
    bool PERIODICY_P;
    bool PERIODICZ_P;

    /*! number of species */
    int ns;
    /*! number of test particle species */
    int nstestpart;
    /*! number of particles per cell */
    std::unique_ptr<int[]> npcel;
    /*! number of particles per cell - X direction */
    std::unique_ptr<int[]> npcelx;
    /*! number of particles per cell - Y direction */
    std::unique_ptr<int[]> npcely;
    /*! number of particles per cell - Z direction */
    std::unique_ptr<int[]> npcelz;
    // either make these of longid type or do not declare them.
    //int *np; /*! number of particles array for different species */
    //int *npMax; /*! maximum number of particles array for different species */
    /*! max number of particles */
    double NpMaxNpRatio;
    /*! charge to mass ratio array for different species */
    std::unique_ptr<double[]> qom;
    /*! charge to mass ratio array for different species */
    std::unique_ptr<double[]> rhoINIT;
    /*! density of injection */
    std::unique_ptr<double[]> rhoINJECT;
    /*! thermal velocity - Direction X */
    std::unique_ptr<double[]> uth;
    /*! thermal velocity - Direction Y */
    std::unique_ptr<double[]> vth;
    /*! thermal velocity - Direction Z */
    std::unique_ptr<double[]> wth;
    /*! Drift velocity - Direction X */
    std::unique_ptr<double[]> u0;
    /*! Drift velocity - Direction Y */
    std::unique_ptr<double[]> v0;
    /*! Drift velocity - Direction Z */
    std::unique_ptr<double[]> w0;

    /*! Pitch Angle for Test Particles */
    std::unique_ptr<double[]> pitch_angle;
    /*! Energy for Test Particles */
    std::unique_ptr<double[]> energy;

    /*! Case type */
    string Case;
    /*! Output writing method */
    string wmethod;
    /*! Simulation name */
    string SimName;

    /*! TrackParticleID */
    //bool *TrackParticleID;
    /*! SaveDirName */
    string SaveDirName;
    /*! RestartDirName */
    string RestartDirName;
    /*! restart_status 0 --> no restart; 1--> restart, create new; 2--> restart, append; */
    int restart_status;
    /*! last cycle */
    int last_cycle;

    /*! Boundary condition on particles 0 = exit 1 = perfect mirror 2 = riemission */
    /*! Boundary Condition Particles: FaceXright */
    int bcPfaceXright;
    /*! Boundary Condition Particles: FaceXleft */
    int bcPfaceXleft;
    /*! Boundary Condition Particles: FaceYright */
    int bcPfaceYright;
    /*! Boundary Condition Particles: FaceYleft */
    int bcPfaceYleft;
    /*! Boundary Condition Particles: FaceYright */
    int bcPfaceZright;
    /*! Boundary Condition Particles: FaceYleft */
    int bcPfaceZleft;


    /*! Field Boundary Condition 0 = Dirichlet Boundary Condition: specifies the valueto take pn the boundary of the domain 1 = Neumann Boundary Condition: specifies the value of derivative to take on the boundary of the domain 2 = Periodic Condition */
    /*! Boundary Condition Electrostatic Potential: FaceXright */
    int bcPHIfaceXright;
    /*! Boundary Condition Electrostatic Potential:FaceXleft */
    int bcPHIfaceXleft;
    /*! Boundary Condition Electrostatic Potential:FaceYright */
    int bcPHIfaceYright;
    /*! Boundary Condition Electrostatic Potential:FaceYleft */
    int bcPHIfaceYleft;
    /*! Boundary Condition Electrostatic Potential:FaceZright */
    int bcPHIfaceZright;
    /*! Boundary Condition Electrostatic Potential:FaceZleft */
    int bcPHIfaceZleft;

    /*! Boundary Condition EM Field: FaceXright */
    int bcEMfaceXright;
    /*! Boundary Condition EM Field: FaceXleft */
    int bcEMfaceXleft;
    /*! Boundary Condition EM Field: FaceYright */
    int bcEMfaceYright;
    /*! Boundary Condition EM Field: FaceYleft */
    int bcEMfaceYleft;
    /*! Boundary Condition EM Field: FaceZright */
    int bcEMfaceZright;
    /*! Boundary Condition EM Field: FaceZleft */
    int bcEMfaceZleft;

    /*! GEM Challenge parameters */

    /* Amplitude of the field */
    double B0x;
    double B0y;
    double B0z;
    double B1x;
    double B1y;
    double B1z;

    //* RESTART simulations
    bool RESTART1;
    
    //* Restart cycle
    int RestartOutputCycle;

    //* Velocity of the injection from the wall
    double Vinj;

    //* Tolerance for GMRES solver
    double GMREStol;

    //* Krylov restart length for the built-in GMRES (Phase 10f bisection knob).
    //* -1 (default) -> legacy hardcoded m=40, max_iter=25 (up to 1000 Krylov steps).
    //*  N > 0       -> force m=N, max_iter=1 (exactly N Krylov steps, no restart).
    int NiterGMRES;

    //* Field solver type: "GMRES" (default) or "PETSc"
    string SolverType;

    //* MaxwellImage curl² operator: "curl_curl" (legacy) or "lap_graddiv" (self-adjoint).
    //* See plan-energy-conservation.md Step 11.
    string MaxwellOperator;

    //* Use explicit preconditioner matrix for PETSc solver
    bool PrecMatrix;

    //* Print preconditioner diagnostics and dump matrix to file
    bool PrecDiagnostics;

    //* Preconditioner type: "None" (default), "Matrix" (explicit P), "Smooth" (PCShell with smoothing)
    string PrecType;

    //* Particle/grid shape function order: "Linear" (default, 8-node CIC) or "Quadratic" (27-node TSC)
    string StencilOrder;
    int stencilOrderInt;        // 1 = Linear, 2 = Quadratic

    //* mover predictor correction iteration (not needed for ECSIM)
    int NiterMover;

    //* Output for field
    int FieldOutputCycle;
    string FieldOutputTag;
    string MomentsOutputTag;

    //* Output for particles
    int ParticlesOutputCycle;
    string ParticlesOutputTag;
    string ParticlesDownsampleOutputTag;

    //* Downsamplimg of particles
    int ParticlesDownsampleOutputCycle; int ParticlesDownsampleFactor;

    //* Output for test particles
    int TestParticlesOutputCycle;
    
    //* Test particles are flushed to disk every testPartFlushCycle
    int testPartFlushCycle;
    
    //* Output for diagnostics
    int DiagnosticsOutputCycle;
    
    /*! Call Finalize() at end of program execution (true by default) */
    bool CallFinalize;

    //* Add External CurlB and External CurlE
    bool AddExternalCurlB;
    bool AddExternalCurlE;

    //* RelSIM
    bool Relativistic;

    //* Relativistic particle pusher
    string Relativistic_pusher;

    //* Write heat flux tensor to files
    bool SaveHeatFluxTensor;

    //* Moving average value for rho density (Poisson correction)
    double PoissonMArho, PoissonMAdiv, PoissonMAres;

    //* Custom input parameters
    double *input_param; int nparam;

    //* Output data precision
    string output_data_precision;

    //* Particle distribution parameters
    int ParticleDistOutputCycle; int ParticleDistBins;
    double ParticleDistMinVelocity; double ParticleDistMaxVelocity;
};
typedef Collective CollectiveIO;

#endif
