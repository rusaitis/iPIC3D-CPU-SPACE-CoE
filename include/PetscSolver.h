/***************************************************************************
  PetscSolver.h  -  PETSc KSP solver for the implicit Maxwell system
 ***************************************************************************/

#ifndef PETSCSOLVER_H
#define PETSCSOLVER_H
#ifdef USE_PETSC

#include <petscksp.h>
#include <petscmat.h>

class EMfields3D;

struct PetscSolverContext {
    EMfields3D *emf;
};

class PetscSolver {
public:
    PetscSolver(int localSize, EMfields3D *emf, double tol);
    ~PetscSolver();
    PetscSolver(const PetscSolver&) = delete;
    PetscSolver& operator=(const PetscSolver&) = delete;

    void solve(double *x, int size, const double *b);

private:
    int localSize_;
    PetscSolverContext ctx_;
    Mat A_;
    KSP ksp_;
};

PetscErrorCode PetscMaxwellMatMult(Mat A, Vec x, Vec y);

#endif // USE_PETSC
#endif // PETSCSOLVER_H
