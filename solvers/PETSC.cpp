/***************************************************************************
  PETSC.cpp  -  PETSc KSP solver for the implicit Maxwell system
  -------------------
  Matrix-free method: uses EMfields3D::MaxwellImage() as the A*x product,
  identical to the built-in GMRES solver.
 ***************************************************************************/

#ifdef USE_PETSC

#include "PETSC.h"
#include "EMfields3D.h"

/*  Matrix-free A*x callback: computes y = A*x.
 *  PETSc calls this whenever the KSP solver needs a matrix-vector product.
 *
 *  A   — shell matrix (carries our context, not actual matrix entries)
 *  x   — input vector  (Krylov basis vector)
 *  y   — output vector (result of A*x)
 */
PetscErrorCode PetscMaxwellMatMult(Mat A, Vec x, Vec y)
{
    PetscFunctionBeginUser;

    PetscSolverContext *ctx;
    PetscCall(MatShellGetContext(A, &ctx));

    const double *d_x;  // raw read-only pointer into x's data
    double       *d_y;  // raw writable   pointer into y's data
    PetscCall(VecGetArrayRead(x, &d_x));
    PetscCall(VecGetArray(y, &d_y));

    // const_cast is safe: MaxwellImage's signature lacks const, but it only reads the input.
    ctx->emf->MaxwellImage(d_y, const_cast<double*>(d_x));

    PetscCall(VecRestoreArrayRead(x, &d_x));
    PetscCall(VecRestoreArray(y, &d_y));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*  Constructor
 *
 *  localSize — number of unknowns on this MPI rank (3 E-field components per interior node)
 *  emf       — the EMfields3D object whose MaxwellImage() provides the A*x product
 *  tol       — relative convergence tolerance (same GMREStol used by the built-in GMRES)
 */
PetscSolver::PetscSolver(int localSize, EMfields3D *emf, double tol)
    : localSize_(localSize), petsc_x_(nullptr), petsc_b_(nullptr)
{
    ctx_.emf = emf;

    // Matrix-free shell: PETSc calls PetscMaxwellMatMult for A*x products
    PetscCallAbort(PETSC_COMM_WORLD,
        MatCreateShell(PETSC_COMM_WORLD, localSize_, localSize_,
                       PETSC_DETERMINE, PETSC_DETERMINE, &ctx_, &A_));
    MatShellSetOperation(A_, MATOP_MULT, (void(*)(void))PetscMaxwellMatMult);

    // KSP solver setup
    // Match the built-in GMRES defaults: restart=20, max 50 restarts (= 1000 total iterations)
    PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp_));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPGMRES));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(ksp_, 20));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, A_, A_));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetTolerances(ksp_, tol, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp_));  // allows runtime override via -ksp_type, -ksp_rtol, etc.
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(ksp_));

    // Pre-allocate PETSc Vec wrappers with no underlying array yet.
    // In solve(), VecPlaceArray/VecResetArray swap in the caller's arrays (zero-copy).
    // petsc_x_ — solution vector (E-field unknowns, read/write)
    // petsc_b_ — right-hand side vector (Maxwell source, read-only)
    PetscCallAbort(PETSC_COMM_WORLD,
        VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, localSize_, PETSC_DECIDE, nullptr, &petsc_x_));
    PetscCallAbort(PETSC_COMM_WORLD,
        VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, localSize_, PETSC_DECIDE, nullptr, &petsc_b_));
}

/*  Destructor  */
PetscSolver::~PetscSolver()
{
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&petsc_x_));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&petsc_b_));
    PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp_));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&A_));
}

/*  Solve  */
void PetscSolver::solve(double *x, int size, const double *b)
{
    // Point the pre-allocated Vecs at the caller's arrays (zero-copy)
    PetscCallAbort(PETSC_COMM_WORLD, VecPlaceArray(petsc_x_, x));
    PetscCallAbort(PETSC_COMM_WORLD, VecPlaceArray(petsc_b_, b));

    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp_, petsc_b_, petsc_x_));

    // Report convergence
    PetscInt iter;
    PetscReal residual;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetIterationNumber(ksp_, &iter));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetResidualNorm(ksp_, &residual));

    KSPConvergedReason reason;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetConvergedReason(ksp_, &reason));
    if (reason > 0) {
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscPrintf(PETSC_COMM_WORLD, "PETSc KSP converged in %d iterations, residual = %e\n",
                        (int)iter, (double)residual));
    } else {
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscPrintf(PETSC_COMM_WORLD, "WARNING: PETSc KSP did NOT converge (reason=%d) after %d iterations, residual = %e\n",
                        (int)reason, (int)iter, (double)residual));
    }

    // Release the caller's arrays from the Vecs (does NOT free them)
    PetscCallAbort(PETSC_COMM_WORLD, VecResetArray(petsc_x_));
    PetscCallAbort(PETSC_COMM_WORLD, VecResetArray(petsc_b_));
}

#endif // USE_PETSC
