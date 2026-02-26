/***************************************************************************
  PETSC.cpp  -  PETSc KSP solver for the implicit Maxwell system
  -------------------
  Matrix-free method: uses EMfields3D::MaxwellImage() as the A*x product,
  identical to the built-in GMRES solver.
 ***************************************************************************/

#ifdef USE_PETSC

#include "PETSC.h"
#include "EMfields3D.h"

/*****************************/
/*  Matrix-free A*x callback */
/*****************************/
PetscErrorCode PetscMaxwellMatMult(Mat A, Vec x, Vec y)
{
    PetscSolverContext *ctx;
    MatShellGetContext(A, &ctx);

    const double *d_x;
    double *d_y;
    VecGetArrayRead(x, &d_x);
    VecGetArray(y, &d_y);

    // MaxwellImage takes (double*, double*) but does not modify the input vector.
    // const_cast is safe here.
    ctx->emf->MaxwellImage(d_y, const_cast<double*>(d_x));

    VecRestoreArrayRead(x, &d_x);
    VecRestoreArray(y, &d_y);

    return 0;
}

/*****************************/
/*  Constructor              */
/*****************************/
PetscSolver::PetscSolver(int localSize, EMfields3D *emf, double tol)
    : localSize_(localSize)
{
    ctx_.emf = emf;

    // Create matrix-free shell matrix
    MatCreateShell(PETSC_COMM_WORLD, localSize_, localSize_,
                   PETSC_DETERMINE, PETSC_DETERMINE, &ctx_, &A_);
    MatShellSetOperation(A_, MATOP_MULT, (void(*)(void))PetscMaxwellMatMult);
    MatSetFromOptions(A_);
    MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);

    // Create KSP solver
    KSPCreate(PETSC_COMM_WORLD, &ksp_);
    KSPSetType(ksp_, KSPGMRES);
    KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE);
    KSPSetOperators(ksp_, A_, A_);
    KSPSetTolerances(ksp_, tol, 1e-17, 10000, 1000);
    KSPSetFromOptions(ksp_);  // allows runtime override via -ksp_type, -pc_type, etc.
    KSPSetUp(ksp_);
}

/*****************************/
/*  Destructor               */
/*****************************/
PetscSolver::~PetscSolver()
{
    KSPDestroy(&ksp_);
    MatDestroy(&A_);
}

/*****************************/
/*  Solve                    */
/*****************************/
void PetscSolver::solve(double *x, int size, const double *b)
{
    // Wrap user arrays as PETSc Vecs (zero-copy)
    Vec petsc_x, petsc_b;
    VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, size, PETSC_DECIDE, x, &petsc_x);
    VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, size, PETSC_DECIDE, b, &petsc_b);

    // Solve
    KSPSolve(ksp_, petsc_b, petsc_x);

    // Report convergence
    PetscInt iter;
    PetscReal residual;
    KSPGetIterationNumber(ksp_, &iter);
    KSPGetResidualNorm(ksp_, &residual);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp_, &reason);
    if (reason > 0) {
        PetscPrintf(PETSC_COMM_WORLD, "PETSc KSP converged in %d iterations, residual = %e\n",
                    (int)iter, (double)residual);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "WARNING: PETSc KSP did NOT converge (reason=%d) after %d iterations, residual = %e\n",
                    (int)reason, (int)iter, (double)residual);
    }

    // Destroy temporary Vec wrappers (does NOT free the underlying arrays)
    VecDestroy(&petsc_x);
    VecDestroy(&petsc_b);
}

#endif // USE_PETSC
