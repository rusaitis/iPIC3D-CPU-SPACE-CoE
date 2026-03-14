/***************************************************************************
  PETSC.h  -  PETSc KSP solver for the implicit Maxwell system
 ***************************************************************************/

#ifndef PETSC_H
#define PETSC_H
#ifdef USE_PETSC

#include <petscksp.h>
#include <petscmat.h>
#include "Neighbouring_Nodes.h"

class EMfields3D;
class VCtopology3D;

struct PetscSolverContext {
    EMfields3D *emf;
};

class PetscSolver {
public:
    PetscSolver(int localSize, EMfields3D *emf, const VCtopology3D *vct,
                double tol, bool usePrecMatrix);
    ~PetscSolver();
    PetscSolver(const PetscSolver&) = delete;
    PetscSolver& operator=(const PetscSolver&) = delete;

    void solve(double *x, int size, const double *b);

private:
    void assembleP();
    void computeCurlCurlStencil();
    void buildMassGroupLookup();
    void setupNearNullSpace();
    PetscInt nodeToGlobalBlock(int ni, int nj, int nk) const;

    int localSize_;
    PetscSolverContext ctx_;
    Mat A_;
    KSP ksp_;
    Vec petsc_x_;   // persistent Vec wrappers (zero-copy, reused across solves)
    Vec petsc_b_;

    // Preconditioner matrix (27-point stencil: I + curl-curl + mass matrix)
    bool usePrecMatrix_;
    Mat P_;                 // explicit preconditioner matrix (MATMPIAIJ, block size 3)
    EMfields3D *emf_;       // same pointer as ctx_.emf; cached for direct use in assembleP()
    PetscInt globalRowOffset_;  // local→global index mapping

    // Curl-curl stencil: precomputed tensor-product form
    // curlCurlStencil_[eq_comp][in_comp][di+1][dj+1][dk+1]
    // Encodes the composed curlC2N(curlN2C(E)) discrete operator
    double curlCurlStencil_[3][3][3][3][3];

    // Offset→mass-group lookup for unified 27-offset assembly
    struct MassGroupEntry {
        int g;        // NeighbouringNodes group (0..13), -1 if no mass entry
        bool forward; // true: M[g][i][j][k]; false: M[g][i+oi][j+oj][k+ok]
    };
    MassGroupEntry massGroupLookup_[3][3][3];  // indexed by [oi+1][oj+1][ok+1]

    // Near-null space for AMG (3 constant vectors, one per E-component)
    MatNullSpace nsp_;

    // MPI topology (for mapping ghost nodes to global block indices)
    const VCtopology3D *vct_;
    NeighbouringNodes neNo_;
    int coords_[3], dims_[3];
    bool periodic_[3];
    int niX_, niY_, niZ_;       // interior node counts (nxn-2, etc.)
    PetscInt nprocsBlocks_;     // blocks per rank (uniform decomposition)

    // Cached physics parameters (extracted from EMfields3D in constructor)
    int nxn_, nyn_, nzn_;
    double dx_, dy_, dz_;
    double c_, th_, dt_, FourPI_, invVOL_;
};

PetscErrorCode PetscMaxwellMatMult(Mat A, Vec x, Vec y);

#endif // USE_PETSC
#endif // PETSC_H
