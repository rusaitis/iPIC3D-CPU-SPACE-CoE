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
  iPIC3D.cpp  -  Main file for 3D simulation
  -------------------
 ************************************************************************** */

#ifndef _IPIC3D_H_
#define _IPIC3D_H_

class Timing;

#ifndef NO_MPI
#include "mpi.h"
#endif
#include "ipicfwd.h"
#include "assert.h"
#include <string>
using std::string;
#ifndef NO_HDF5
class OutputWrapperFPP;
#endif
#ifdef USE_PETSC
class PetscSolver;
#endif
namespace iPic3D {

  class c_Solver {

  public:
    ~c_Solver();
    c_Solver():
      col(0),
      vct(0),
      grid(0),
      EMf(0),
#ifdef USE_PETSC
      petscSolver(0),
#endif
      particles(0),
#ifndef NO_HDF5
      outputWrapperFPP(0),
#endif
      kinetic_energy_species(0),
      bulk_energy_species(0),
      momentum_species(0),
      Qremoved(0),
      my_clock(0)
    {}


    int Init(int argc, char **argv);
    void CalculateMoments();
    void ComputeE(int cycle);
    bool ParticlesMover();
    void CalculateB();
    void ComputeEMFields(int cycle);

    //* Step 32: cycle 0 (post-init, pre-solve) or cycle 1 (post-solve-1) dump
    //* of all node fields + stored M as raw IEEE-754 doubles for cross-code
    //* byte diff. No-op if DumpCycle1Fields flag is off.
    void DumpCycleFields(int cycle);

    //* Step 34c: gather/scatter transpose duality probe. Tests condition #3 of
    //* the six ECSIM-exact energy-identity conditions: ⟨gather(f), q⟩_particle
    //* should equal ⟨f, scatter(q)⟩_grid bit-for-bit when the gather and scatter
    //* use the same weights. Opt-in via VerifyAdjoint flag (shared with 34b);
    //* runs once at cycle 1 between CalculateMoments and ComputeEMFields so the
    //* particle state is at x_n — the same position the gather/scatter kernels
    //* just consumed.
    void ProbeGatherScatterDuality(int cycle);
    void SupplementaryMoments();

    void WriteRestart(int cycle);
    void WriteConserved(int cycle);
    void WriteVelocityDistribution(int cycle);
    void WriteVirtualSatelliteTraces();
    void WriteFields(int cycle);
    void WriteParticles(int cycle);
    void WriteDSParticles(int cycle);
    void WriteTestParticles(int cycle);
    void WriteOutput(int cycle);
    void Finalize();

    int FirstCycle() { return (first_cycle); }
    int get_myrank() { return (myrank); }
    int LastCycle();

  private:
    void pad_particle_capacities();
    void convertParticlesToSoA();
    void convertParticlesToAoS();
    void convertParticlesToSynched();
    void sortParticles();

  private:
    //static MPIdata * mpi;
    Collective    *col; // the input parameters
    VCtopology3D  *vct; // mpi topology 
    Grid3DCU      *grid; // 3d cartesion grid, local grid
    EMfields3D    *EMf; //
#ifdef USE_PETSC
    PetscSolver   *petscSolver;
#endif
    Particles3D   *particles;
    Particles3D   *testpart;
    double        *kinetic_energy_species;  //* kinetic energy of each species, the normal one, added up
    double        *bulk_energy_species; // bulk kinetic energy of each species, consider the bulk motion
    double        *momentum_species; // an array of doubles, total momentum of all particle species
    int           *num_particles_species;
    double        *charge_species;
    double        *Qremoved; // array of double, with species length, removed charges from the depopulation area
    Timing        *my_clock;

#ifndef NO_HDF5
    OutputWrapperFPP& fetch_outputWrapperFPP(){
      assert(outputWrapperFPP);
      return *outputWrapperFPP;}
    OutputWrapperFPP *outputWrapperFPP;
#endif

    string SaveDirName;
    string RestartDirName;
    string cqsat;
    string cq;
    string cqs;
    string ds;
    string num_proc_str;
    int restart_cycle;
    int restart_status;
    int first_cycle;
    int ns;
    int nstestpart;
    int nprocs;
    int myrank;
    int nsat;
    int nDistributionBins;
    double Eenergy;
    double Benergy;
    double TOTenergy;
    double TOTmomentum;
    double initial_total_energy;

    //the below used for IO
    MPI_Request *headerReq;
    MPI_Request *dataReq;
    MPI_Request *footReq;
    float *testpclPos;
    int    pclbuffersize;
    float *testpclVel;
    MPI_File fh;
  	MPI_Status*  status;
  	float**** fieldwritebuffer;
	MPI_Request fieldreqArr[4];//E+B+Je+Ji
	MPI_File    fieldfhArr[4];
	MPI_Status  fieldstsArr[4];
	int fieldreqcounter;

  	float*** momentwritebuffer;
	MPI_Request momentreqArr[14];//rho+PXX+PXY+PXZ++PYY+PYZ+PZZ for species0,1
	MPI_File    momentfhArr[14];
	MPI_Status  momentstsArr[14];
	int momentreqcounter;

  };

}

#endif
