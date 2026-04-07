/***************************************************************************
  PETSC.cpp  -  PETSc KSP solver for the implicit Maxwell system
  -------------------
  Matrix-free method: uses EMfields3D::MaxwellImage() as the A*x product,
  identical to the built-in GMRES solver.

  Optional preconditioner matrix P: when usePrecMatrix_ is true, assembles
  the full 27-point stencil (identity + curl-curl + mass matrix) and passes
  it to KSPSetOperators(ksp_, A_shell, P_explicit).  Three constant
  near-null space vectors (one per E-component) are attached to P for
  AMG coarsening (GAMG smoothed aggregation, HYPRE nodal coarsening).
 ***************************************************************************/

#ifdef USE_PETSC

#include "PETSC.h"
#include "EMfields3D.h"
#include "VCtopology3D.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <filesystem>

// Free functions defined in EMfields3D.cpp (no header declaration).
// The trailing int is the ghost-cell layer count (1 = legacy CIC; 2 = TSC).
void solver2phys(arr3_double, arr3_double, arr3_double, double*, int, int, int, int n_ghost = 1);
void phys2solver(double*, const arr3_double, const arr3_double, const arr3_double, int, int, int, int n_ghost = 1);

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
 *  localSize     — number of unknowns on this MPI rank (3 E-field components per interior node)
 *  emf           — the EMfields3D object whose MaxwellImage() provides the A*x product
 *  vct           — MPI Cartesian topology (for ghost→global index mapping in preconditioner)
 *  tol           — relative convergence tolerance (same GMREStol used by the built-in GMRES)
 *  precType      — "None", "Matrix" (explicit P for AMG), or "Smooth" (PCShell with smoothing)
 */
PetscSolver::PetscSolver(int localSize, EMfields3D *emf, const VCtopology3D *vct,
                         double tol, const std::string &precType, bool diagnostics,
                         const std::string &simName, const std::string &saveDir)
    : localSize_(localSize), petsc_x_(nullptr), petsc_b_(nullptr),
      usePrecMatrix_(precType == "Matrix"), needsReassembly_(false), diagnostics_(diagnostics),
      simName_(simName), saveDir_(saveDir),
      P_(nullptr), emf_(emf), globalBlockOffset_(0),
      nsp_(nullptr),
      vct_(vct), nprocsBlocks_(0),
      nxn_(0), nyn_(0), nzn_(0), niX_(0), niY_(0), niZ_(0),
      dx_(0), dy_(0), dz_(0),
      c_(0), th_(0), dt_(0), FourPI_(0), invVOL_(0),
      useSmoothPC_(precType == "Smooth"),
      pcWorkX_(nullptr), pcWorkY_(nullptr), pcWorkZ_(nullptr),
      diagInv_(nullptr)
{
    coords_[0] = coords_[1] = coords_[2] = 0;
    dims_[0] = dims_[1] = dims_[2] = 0;
    periodic_[0] = periodic_[1] = periodic_[2] = false;
    ctx_.emf = emf;

    // Matrix-free shell: PETSc calls PetscMaxwellMatMult for A*x products
    PetscCallAbort(PETSC_COMM_WORLD,
        MatCreateShell(PETSC_COMM_WORLD, localSize_, localSize_,
                       PETSC_DETERMINE, PETSC_DETERMINE, &ctx_, &A_));
    MatShellSetOperation(A_, MATOP_MULT, (void(*)(void))PetscMaxwellMatMult);

    // Cache physics parameters (needed by both Matrix and Smooth preconditioner paths)
    if (usePrecMatrix_ || useSmoothPC_) {
        nxn_ = emf->get_nxn();
        nyn_ = emf->get_nyn();
        nzn_ = emf->get_nzn();
        dx_  = emf->get_dx();
        dy_  = emf->get_dy();
        dz_  = emf->get_dz();
        c_   = emf->get_c();
        th_  = emf->get_th();
        dt_  = emf->get_dt();
        FourPI_ = emf->get_FourPI();
        invVOL_ = emf->get_invVOL();

        // Cache MPI topology for ghost→global mapping
        niX_ = nxn_ - 2;
        niY_ = nyn_ - 2;
        niZ_ = nzn_ - 2;
        nprocsBlocks_ = niX_ * niY_ * niZ_;
        for (int d = 0; d < 3; d++)
            coords_[d] = vct_->getCoordinates(d);
        dims_[0] = vct_->getXLEN();
        dims_[1] = vct_->getYLEN();
        dims_[2] = vct_->getZLEN();
        periodic_[0] = vct_->getPERIODICX();
        periodic_[1] = vct_->getPERIODICY();
        periodic_[2] = vct_->getPERIODICZ();

        // Precompute curl-curl stencil (needed by both Matrix assembly and Smooth diagonal blocks)
        computeCurlCurlStencil();
        buildMassGroupLookup();

        // Verify curl-curl stencil against known analytical values
        {
            const double invdx2 = 1.0 / (dx_ * dx_);
            const double invdy2 = 1.0 / (dy_ * dy_);
            const double invdz2 = 1.0 / (dz_ * dz_);
            assert(std::abs(curlCurlStencil_[0][0][1][1][1] - 0.5 * (invdy2 + invdz2)) < 1e-12);
            assert(std::abs(curlCurlStencil_[1][1][1][1][1] - 0.5 * (invdx2 + invdz2)) < 1e-12);
            assert(std::abs(curlCurlStencil_[2][2][1][1][1] - 0.5 * (invdx2 + invdy2)) < 1e-12);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    if (a != b) assert(curlCurlStencil_[a][b][1][1][1] == 0.0);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    double sum = 0.0;
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                            for (int dk = 0; dk < 3; dk++)
                                sum += curlCurlStencil_[a][b][di][dj][dk];
                    assert(std::abs(sum) < 1e-12);
                }
        }
    }

    // Explicit preconditioner matrix P (for AMG-type preconditioners)
    if (usePrecMatrix_) {
        // Compute global row offset via MPI_Exscan (prefix sum of local sizes)
        PetscInt localBlockCount = localSize_ / 3;
        PetscInt offset = 0;
        MPI_Exscan(&localBlockCount, &offset, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (rank == 0) offset = 0;
        globalBlockOffset_ = offset;

        // Create MATMPIAIJ with block size 3
        PetscCallAbort(PETSC_COMM_WORLD,
            MatCreate(PETSC_COMM_WORLD, &P_));
        PetscCallAbort(PETSC_COMM_WORLD,
            MatSetSizes(P_, localSize_, localSize_, PETSC_DETERMINE, PETSC_DETERMINE));
        PetscCallAbort(PETSC_COMM_WORLD,
            MatSetType(P_, MATMPIAIJ));
        PetscCallAbort(PETSC_COMM_WORLD,
            MatSetBlockSize(P_, 3));
        PetscCallAbort(PETSC_COMM_WORLD,
            MatMPIAIJSetPreallocation(P_, 81, nullptr, 81, nullptr));
        PetscCallAbort(PETSC_COMM_WORLD,
            MatSetOption(P_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

        assembleP();
        setupNearNullSpace();
        needsReassembly_ = true;

        if (diagnostics_) {
            printDiagnostics(0);
            std::filesystem::create_directories(saveDir_);
            std::string dumpPath = saveDir_ + "/" + simName_ + "_P_cycle0.bin";
            dumpP(dumpPath.c_str());
        }
    }

    // KSP solver setup
    // Match the built-in GMRES defaults: restart=20, max 50 restarts (= 1000 total iterations)
    PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp_));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPGMRES));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(ksp_, 20));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE));

    if (usePrecMatrix_) {
        PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, A_, P_));
        PC pc;
        PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp_, &pc));
        PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCNONE));
    } else if (useSmoothPC_) {
        PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, A_, A_));
        setupSmoothPC();
    } else {
        PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp_, A_, A_));
    }

    PetscCallAbort(PETSC_COMM_WORLD, KSPSetTolerances(ksp_, tol, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp_));  // allows runtime override via -ksp_type, -ksp_rtol, etc.
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetUp(ksp_));

    // Pre-allocate PETSc Vec wrappers with no underlying array yet.
    // In solve(), VecPlaceArray/VecResetArray swap in the caller's arrays (zero-copy).
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
    if (usePrecMatrix_) {
        PetscCallAbort(PETSC_COMM_WORLD, MatNullSpaceDestroy(&nsp_));
        PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&P_));
    }
    if (useSmoothPC_) {
        delete pcWorkX_;
        delete pcWorkY_;
        delete pcWorkZ_;
        delete[] diagInv_;
    }
}

/*  Map local node index (i,j,k) — possibly a ghost — to global PETSc block index.
 *  Returns -1 for non-periodic boundary ghosts (no valid global node).
 *
 *  Interior nodes (1..niX, 1..niY, 1..niZ) map directly.
 *  Ghost nodes (0 or nxn-1 etc.) are resolved to the owning MPI rank via the
 *  Cartesian communicator, exploiting uniform decomposition (all ranks have
 *  identical niX*niY*niZ interior nodes).
 */
PetscInt PetscSolver::nodeToGlobalBlock(int ni, int nj, int nk) const
{
    // Interior node on this rank — fast path
    if (ni >= 1 && ni <= niX_ && nj >= 1 && nj <= niY_ && nk >= 1 && nk <= niZ_)
        return globalBlockOffset_ + (ni - 1) * niY_ * niZ_ + (nj - 1) * niZ_ + (nk - 1);

    // Ghost node — determine owning rank and local index there
    int neighborCoords[3] = {coords_[0], coords_[1], coords_[2]};
    int localI = ni, localJ = nj, localK = nk;

    // Ghost at index 0 → rank-1 in that dimension, local index niX on that rank
    // Ghost at index nxn-1 (= niX+1) → rank+1, local index 1
    if (ni == 0)        { neighborCoords[0]--; localI = niX_; }
    else if (ni > niX_) { neighborCoords[0]++; localI = 1; }
    if (nj == 0)        { neighborCoords[1]--; localJ = niY_; }
    else if (nj > niY_) { neighborCoords[1]++; localJ = 1; }
    if (nk == 0)        { neighborCoords[2]--; localK = niZ_; }
    else if (nk > niZ_) { neighborCoords[2]++; localK = 1; }

    // Handle periodicity / boundary
    for (int d = 0; d < 3; d++) {
        if (neighborCoords[d] < 0) {
            if (!periodic_[d]) return -1;
            neighborCoords[d] += dims_[d];
        } else if (neighborCoords[d] >= dims_[d]) {
            if (!periodic_[d]) return -1;
            neighborCoords[d] -= dims_[d];
        }
    }

    int neighborRank;
    MPI_Cart_rank(vct_->getFieldComm(), neighborCoords, &neighborRank);

    PetscInt neighborOffset = (PetscInt)neighborRank * nprocsBlocks_;
    return neighborOffset + (localI - 1) * niY_ * niZ_ + (localJ - 1) * niZ_ + (localK - 1);
}

/*  Precompute the curl-curl stencil in tensor-product form.
 *
 *  The composed curlC2N(curlN2C(E)) operator has a 3×3×3 stencil determined
 *  by three weight vectors:
 *    w_avg = [1, 2, 1]  — averaging (non-differentiated direction)
 *    w_lap = [1,-2, 1]  — second derivative
 *    w_der = [-1, 0, 1] — first derivative (mixed terms)
 *
 *  The 1/16 prefactor comes from (0.25)² across the two curl operations,
 *  each averaging over 4 cell-face corners.
 *
 *  For same-component (a==b): negative transverse Laplacian
 *    CC[a][a] = -1/16 · Σ_{q≠a} w_avg(a) ⊗ w_lap(q)/dq² ⊗ w_avg(r)
 *
 *  For cross-component (a≠b): mixed second derivative
 *    CC[a][b] = +1/16 · w_der(a)/da ⊗ w_der(b)/db ⊗ w_avg(r)
 */
void PetscSolver::computeCurlCurlStencil()
{
    const double w_avg[3] = {1.0, 2.0, 1.0};
    const double w_lap[3] = {1.0, -2.0, 1.0};
    const double w_der[3] = {-1.0, 0.0, 1.0};

    const double invd[3] = {1.0 / dx_, 1.0 / dy_, 1.0 / dz_};

    std::memset(curlCurlStencil_, 0, sizeof(curlCurlStencil_));

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            for (int n0 = 0; n0 < 3; n0++)
                for (int n1 = 0; n1 < 3; n1++)
                    for (int n2 = 0; n2 < 3; n2++)
                    {
                        const int n[3] = {n0, n1, n2};
                        double val = 0.0;

                        if (a == b) {
                            // -d²/dq² for each q ≠ a
                            for (int q = 0; q < 3; q++) {
                                if (q == a) continue;
                                double prod = 1.0;
                                for (int d = 0; d < 3; d++) {
                                    if (d == q)
                                        prod *= w_lap[n[d]] * invd[q] * invd[q];
                                    else
                                        prod *= w_avg[n[d]];
                                }
                                val -= prod / 16.0;
                            }
                        } else {
                            // +d²/(da·db)
                            double prod = 1.0;
                            for (int d = 0; d < 3; d++) {
                                if (d == a || d == b)
                                    prod *= w_der[n[d]] * invd[d];
                                else
                                    prod *= w_avg[n[d]];
                            }
                            val = prod / 16.0;
                        }

                        curlCurlStencil_[a][b][n0][n1][n2] = val;
                    }
        }
    }
}

/*  Build lookup table mapping (di,dj,dk) offsets to NeighbouringNodes groups.
 *
 *  The 14 NeighbouringNodes groups (g=0..13) cover all 27 offsets:
 *    g=0: center (0,0,0)  — forward only
 *    g=1..13: forward (+offset) and backward (-offset)
 *
 *  For forward entries, mass matrix is indexed at M[g][i][j][k] (row node).
 *  For backward entries, mass matrix is indexed at M[g][i+oi][j+oj][k+ok] (col node).
 */
void PetscSolver::buildMassGroupLookup()
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 3; c++)
                massGroupLookup_[a][b][c] = {-1, true};

    // g=0: center node
    massGroupLookup_[1][1][1] = {0, true};

    // g=1..13: forward and backward offsets
    for (int g = 1; g < 14; g++) {
        int di = neNo_.getX(g);
        int dj = neNo_.getY(g);
        int dk = neNo_.getZ(g);
        massGroupLookup_[di + 1][dj + 1][dk + 1] = {g, true};
        massGroupLookup_[-di + 1][-dj + 1][-dk + 1] = {g, false};
    }
}

/*  Attach 3 near-null space vectors to P for AMG preconditioners.
 *
 *  The DOF ordering is interleaved: phys2solver packs as [Ex,Ey,Ez, Ex,Ey,Ez, ...]
 *  per node.  Vector c has 1.0 at every index ≡ c (mod 3), representing a constant
 *  field in component c.  These are (near-)null vectors of the curl-curl part
 *  (curl of constant = 0); the mass matrix breaks exact nullity, hence "near."
 *
 *  GAMG uses these for smoothed aggregation coarsening; HYPRE uses them for
 *  nodal/unknown-based coarsening.  Without them, GAMG produces 250-500 iterations.
 */
void PetscSolver::setupNearNullSpace()
{
    Vec vecs[3];
    for (int c = 0; c < 3; c++) {
        PetscCallAbort(PETSC_COMM_WORLD,
            VecCreate(PETSC_COMM_WORLD, &vecs[c]));
        PetscCallAbort(PETSC_COMM_WORLD,
            VecSetSizes(vecs[c], localSize_, PETSC_DETERMINE));
        PetscCallAbort(PETSC_COMM_WORLD,
            VecSetFromOptions(vecs[c]));
        PetscCallAbort(PETSC_COMM_WORLD,
            VecZeroEntries(vecs[c]));

        PetscInt lo, hi;
        PetscCallAbort(PETSC_COMM_WORLD,
            VecGetOwnershipRange(vecs[c], &lo, &hi));
        for (PetscInt idx = lo + c; idx < hi; idx += 3)
            PetscCallAbort(PETSC_COMM_WORLD,
                VecSetValue(vecs[c], idx, 1.0, INSERT_VALUES));

        PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(vecs[c]));
        PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(vecs[c]));

        PetscReal norm;
        PetscCallAbort(PETSC_COMM_WORLD,
            VecNorm(vecs[c], NORM_2, &norm));
        PetscCallAbort(PETSC_COMM_WORLD,
            VecScale(vecs[c], 1.0 / norm));
    }

    PetscCallAbort(PETSC_COMM_WORLD,
        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 3, vecs, &nsp_));
    PetscCallAbort(PETSC_COMM_WORLD,
        MatSetNearNullSpace(P_, nsp_));

    for (int c = 0; c < 3; c++)
        PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&vecs[c]));
}

/*  Assemble preconditioner matrix P with full 27-point stencil.
 *
 *  Each interior node (i,j,k) gets a 3×3 block for each of the 27 neighbor
 *  offsets (including self).  Each block combines three contributions:
 *
 *    1. Curl-curl (constant, precomputed): (c·θ·Δt)² · CC[a][b][offset]
 *    2. Identity (self offset only): δ_{ab}
 *    3. Mass matrix (from lookup table): scale · M_ab[g][source_node]
 *
 *  Ghost mass matrix values are valid (communicateGhostP2G_mass_matrix() is
 *  called before calculateE()).
 */
void PetscSolver::assembleP()
{
    const double cthdt  = c_ * th_ * dt_;
    const double cthdt2 = cthdt * cthdt;
    const double scale  = dt_ * th_ * FourPI_ * invVOL_;  // implicit current: dt·θ·4π/V_cell (from Ampère + linearized J)

    // Access mass matrix arrays from EMfields3D
    const_arr4_double Mxx = emf_->getMxx();
    const_arr4_double Mxy = emf_->getMxy();
    const_arr4_double Mxz = emf_->getMxz();
    const_arr4_double Myx = emf_->getMyx();
    const_arr4_double Myy = emf_->getMyy();
    const_arr4_double Myz = emf_->getMyz();
    const_arr4_double Mzx = emf_->getMzx();
    const_arr4_double Mzy = emf_->getMzy();
    const_arr4_double Mzz = emf_->getMzz();

    for (int i = 1; i <= niX_; i++)
        for (int j = 1; j <= niY_; j++)
            for (int k = 1; k <= niZ_; k++)
            {
                PetscInt grow = globalBlockOffset_
                              + (i - 1) * niY_ * niZ_ + (j - 1) * niZ_ + (k - 1);

                for (int oi = -1; oi <= 1; oi++)
                    for (int oj = -1; oj <= 1; oj++)
                        for (int ok = -1; ok <= 1; ok++)
                        {
                            PetscInt gcol = nodeToGlobalBlock(i + oi, j + oj, k + ok);
                            if (gcol < 0) continue;

                            double block[9] = {0};

                            // 1. Curl-curl (constant, precomputed)
                            for (int a = 0; a < 3; a++)
                                for (int b = 0; b < 3; b++)
                                    block[a * 3 + b] += cthdt2 * curlCurlStencil_[a][b][oi + 1][oj + 1][ok + 1];

                            // 2. Identity (self only)
                            if (oi == 0 && oj == 0 && ok == 0) {
                                block[0] += 1.0;
                                block[4] += 1.0;
                                block[8] += 1.0;
                            }

                            // 3. Mass matrix (from offset→group lookup)
                            //    block[row*3 + col] layout — matches mass_matrix_times_vector():
                            //           col:  Ex     Ey     Ez
                            //    row Ex:     Mxx    Myx    Mzx     [0,1,2]
                            //    row Ey:     Mxy    Myy    Mzy     [3,4,5]
                            //    row Ez:     Mxz    Myz    Mzz     [6,7,8]
                            const auto &entry = massGroupLookup_[oi + 1][oj + 1][ok + 1];
                            if (entry.g >= 0) {
                                int g = entry.g;
                                int mi, mj, mk;
                                if (entry.forward) {
                                    mi = i; mj = j; mk = k;
                                } else {
                                    mi = i + oi; mj = j + oj; mk = k + ok;
                                }
                                block[0] += scale * Mxx[g][mi][mj][mk];
                                block[1] += scale * Myx[g][mi][mj][mk];
                                block[2] += scale * Mzx[g][mi][mj][mk];
                                block[3] += scale * Mxy[g][mi][mj][mk];
                                block[4] += scale * Myy[g][mi][mj][mk];
                                block[5] += scale * Mzy[g][mi][mj][mk];
                                block[6] += scale * Mxz[g][mi][mj][mk];
                                block[7] += scale * Myz[g][mi][mj][mk];
                                block[8] += scale * Mzz[g][mi][mj][mk];
                            }

                            PetscCallAbort(PETSC_COMM_WORLD,
                                MatSetValuesBlocked(P_, 1, &grow, 1, &gcol, block, INSERT_VALUES));
                        }
            }

    PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(P_, MAT_FINAL_ASSEMBLY));
    PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(P_, MAT_FINAL_ASSEMBLY));
}

/*  Print preconditioner matrix diagnostics: norms, deviation from identity, physics scale factors.  */
void PetscSolver::printDiagnostics(int cycle) const
{
    if (!usePrecMatrix_ || P_ == nullptr) return;

    PetscReal normP;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(P_, NORM_FROBENIUS, &normP));

    // ||P - I||_F: duplicate P, shift diagonal by -1
    Mat PmI;
    PetscCallAbort(PETSC_COMM_WORLD, MatDuplicate(P_, MAT_COPY_VALUES, &PmI));
    PetscCallAbort(PETSC_COMM_WORLD, MatShift(PmI, -1.0));
    PetscReal normPmI;
    PetscCallAbort(PETSC_COMM_WORLD, MatNorm(PmI, NORM_FROBENIUS, &normPmI));
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&PmI));

    // ||I||_F = sqrt(N) for N×N identity
    PetscInt M, N;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(P_, &M, &N));
    PetscReal normI = PetscSqrtReal((PetscReal)M);

    MatInfo info;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetInfo(P_, MAT_GLOBAL_SUM, &info));

    double cthdt2 = c_ * th_ * dt_ * c_ * th_ * dt_;
    double scale  = dt_ * th_ * FourPI_ * invVOL_;

    PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD,
        "\n=== Preconditioner P diagnostics (cycle %d) ===\n"
        "  Matrix size:      %d x %d  (%.0f nonzeros)\n"
        "  ||P||_F           = %.6e\n"
        "  ||P - I||_F       = %.6e\n"
        "  ||P - I|| / ||I|| = %.6e\n"
        "  Scale factors:\n"
        "    (c*th*dt)^2     = %.6e   (curl-curl weight)\n"
        "    dt*th*4pi/V     = %.6e   (mass matrix weight)\n"
        "================================================\n\n",
        cycle, (int)M, (int)N, info.nz_used,
        (double)normP, (double)normPmI, (double)(normPmI / normI),
        cthdt2, scale));
}

/*  Dump preconditioner matrix P to PETSc binary file for Python visualization.  */
void PetscSolver::dumpP(const char *filename) const
{
    if (!usePrecMatrix_ || P_ == nullptr) return;

    PetscViewer viewer;
    PetscCallAbort(PETSC_COMM_WORLD,
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
    PetscCallAbort(PETSC_COMM_WORLD, MatView(P_, viewer));
    PetscCallAbort(PETSC_COMM_WORLD, PetscViewerDestroy(&viewer));

    PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD,
        "Preconditioner matrix P dumped to %s\n", filename));
}

/*  Analytically invert a 3×3 matrix stored in row-major order.
 *  Uses the cofactor (adjugate) method: Dinv = adj(D) / det(D).
 */
void PetscSolver::invert3x3(const double D[9], double Dinv[9])
{
    // Cofactors
    double c00 = D[4]*D[8] - D[5]*D[7];
    double c01 = D[5]*D[6] - D[3]*D[8];
    double c02 = D[3]*D[7] - D[4]*D[6];
    double c10 = D[2]*D[7] - D[1]*D[8];
    double c11 = D[0]*D[8] - D[2]*D[6];
    double c12 = D[1]*D[6] - D[0]*D[7];
    double c20 = D[1]*D[5] - D[2]*D[4];
    double c21 = D[2]*D[3] - D[0]*D[5];
    double c22 = D[0]*D[4] - D[1]*D[3];

    double det = D[0]*c00 + D[1]*c01 + D[2]*c02;
    double invdet = 1.0 / det;

    Dinv[0] = c00 * invdet;  Dinv[1] = c10 * invdet;  Dinv[2] = c20 * invdet;
    Dinv[3] = c01 * invdet;  Dinv[4] = c11 * invdet;  Dinv[5] = c21 * invdet;
    Dinv[6] = c02 * invdet;  Dinv[7] = c12 * invdet;  Dinv[8] = c22 * invdet;
}

/*  Precompute D⁻¹ at every interior node.
 *  D(i,j,k) = I + cthdt² * diag(CC) + scale * M[g=0](i,j,k)
 *  where CC_diag is the curl-curl self-coupling (diagonal only, cross-component = 0).
 */
void PetscSolver::updateDiagInv()
{
    const double cthdt2 = c_ * th_ * dt_ * c_ * th_ * dt_;
    const double scale  = dt_ * th_ * FourPI_ * invVOL_;

    const_arr4_double Mxx = emf_->getMxx();
    const_arr4_double Mxy = emf_->getMxy();
    const_arr4_double Mxz = emf_->getMxz();
    const_arr4_double Myx = emf_->getMyx();
    const_arr4_double Myy = emf_->getMyy();
    const_arr4_double Myz = emf_->getMyz();
    const_arr4_double Mzx = emf_->getMzx();
    const_arr4_double Mzy = emf_->getMzy();
    const_arr4_double Mzz = emf_->getMzz();

    for (int i = 1; i < nxn_ - 1; i++)
        for (int j = 1; j < nyn_ - 1; j++)
            for (int k = 1; k < nzn_ - 1; k++)
            {
                int idx = i * nyn_ * nzn_ + j * nzn_ + k;
                double D[9];

                // Diagonal: I + cthdt² * CC_diag (cross-component CC diag is zero)
                D[0] = 1.0 + cthdt2 * ccDiag_[0] + scale * Mxx.get(0, i, j, k);
                D[1] =                              scale * Myx.get(0, i, j, k);
                D[2] =                              scale * Mzx.get(0, i, j, k);
                D[3] =                              scale * Mxy.get(0, i, j, k);
                D[4] = 1.0 + cthdt2 * ccDiag_[1] + scale * Myy.get(0, i, j, k);
                D[5] =                              scale * Mzy.get(0, i, j, k);
                D[6] =                              scale * Mxz.get(0, i, j, k);
                D[7] =                              scale * Myz.get(0, i, j, k);
                D[8] = 1.0 + cthdt2 * ccDiag_[2] + scale * Mzz.get(0, i, j, k);

                invert3x3(D, &diagInv_[idx * 9]);
            }
}

/*  Set up the PCShell preconditioner: allocate work arrays, precompute D⁻¹,
 *  register the PCApply callback, and switch KSP to FGMRES (flexible GMRES
 *  is required because the preconditioner involves MPI communication).
 */
void PetscSolver::setupSmoothPC()
{
    // Allocate work arrays for PCApply (cannot reuse EMfields3D's temp arrays)
    pcWorkX_ = new array3_double(nxn_, nyn_, nzn_);
    pcWorkY_ = new array3_double(nxn_, nyn_, nzn_);
    pcWorkZ_ = new array3_double(nxn_, nyn_, nzn_);

    // Cache curl-curl diagonal entries (per E-component)
    ccDiag_[0] = curlCurlStencil_[0][0][1][1][1];  // 0.5*(1/dy² + 1/dz²)
    ccDiag_[1] = curlCurlStencil_[1][1][1][1][1];  // 0.5*(1/dx² + 1/dz²)
    ccDiag_[2] = curlCurlStencil_[2][2][1][1][1];  // 0.5*(1/dx² + 1/dy²)

    // Allocate and compute D⁻¹
    diagInv_ = new double[nxn_ * nyn_ * nzn_ * 9]();
    updateDiagInv();

    // Switch to FGMRES (PCShell involves MPI comms, making it "variable")
    PetscCallAbort(PETSC_COMM_WORLD, KSPSetType(ksp_, KSPFGMRES));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGMRESSetRestart(ksp_, 20));

    // Register PCShell
    PC pc;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp_, &pc));
    PetscCallAbort(PETSC_COMM_WORLD, PCSetType(pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetContext(pc, this));
    PetscCallAbort(PETSC_COMM_WORLD, PCShellSetApply(pc, PetscSmoothPCApply));

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0)
        printf("PCShell with smoothing preconditioner enabled (FGMRES + S·D⁻¹·S)\n");
}

/*  PCShell apply callback: y = D⁻¹ · x
 *
 *  D⁻¹ = pre-inverted 3×3 block-diagonal of A at each node.
 *  D captures the identity, curl-curl diagonal, and center mass block.
 *
 *  ─── How to experiment with new preconditioners ───
 *  This function is the place to modify.  It receives a PETSc Vec x (the
 *  residual) and must write the preconditioned result to Vec y.  The pattern is:
 *    1. solver2phys() to unpack interleaved Krylov vector → 3 separate 3D fields
 *    2. Apply your preconditioner to the 3D fields
 *    3. phys2solver() to pack back → interleaved Krylov vector
 *
 *  Available tools:
 *    - emf->energy_conserve_smooth_direction(field, nx, ny, nz, dir) — 4-pass 27-pt filter
 *    - diagInv_[] — pre-inverted 3×3 blocks at each node (updated every cycle)
 *    - emf->mass_matrix_times_vector() — full 27-point mass matrix multiply
 *    - Work arrays pcWorkX_/Y_/Z_ — allocated, same size as E-field arrays
 */
PetscErrorCode PetscSmoothPCApply(PC pc, Vec x, Vec y)
{
    PetscFunctionBeginUser;

    void *ctx;
    PetscCall(PCShellGetContext(pc, &ctx));
    auto *solver = static_cast<PetscSolver*>(ctx);

    EMfields3D *emf = solver->emf_;
    int nxn = solver->nxn_, nyn = solver->nyn_, nzn = solver->nzn_;

    // Get raw arrays from PETSc Vecs
    const double *xarr;
    double *yarr;
    PetscCall(VecGetArrayRead(x, &xarr));
    PetscCall(VecGetArray(y, &yarr));

    // Unpack from interleaved Krylov vector to 3 separate 3D fields
    array3_double &wX = *solver->pcWorkX_;
    array3_double &wY = *solver->pcWorkY_;
    array3_double &wZ = *solver->pcWorkZ_;
    solver2phys(wX, wY, wZ, const_cast<double*>(xarr), nxn, nyn, nzn);

    // Apply block-diagonal D⁻¹ at each interior node
    // D = I + cthdt²·CC_diag + scale·M[g=0] captures the local operator structure
    for (int i = 1; i < nxn - 1; i++)
        for (int j = 1; j < nyn - 1; j++)
            for (int k = 1; k < nzn - 1; k++)
            {
                int idx = i * nyn * nzn + j * nzn + k;
                const double *Dinv = &solver->diagInv_[idx * 9];
                double ex = wX[i][j][k], ey = wY[i][j][k], ez = wZ[i][j][k];

                wX[i][j][k] = Dinv[0]*ex + Dinv[1]*ey + Dinv[2]*ez;
                wY[i][j][k] = Dinv[3]*ex + Dinv[4]*ey + Dinv[5]*ez;
                wZ[i][j][k] = Dinv[6]*ex + Dinv[7]*ey + Dinv[8]*ez;
            }

    // Pack back to interleaved Krylov vector
    phys2solver(yarr, wX, wY, wZ, nxn, nyn, nzn);

    PetscCall(VecRestoreArrayRead(x, &xarr));
    PetscCall(VecRestoreArray(y, &yarr));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/*  Solve  */
void PetscSolver::solve(double *x, int size, const double *b, int cycle)
{
    assert(size == localSize_);
    // Re-assemble P if it contains cycle-dependent terms (e.g. mass matrix).
    if (usePrecMatrix_ && needsReassembly_)
        assembleP();
    // Update block-diagonal D⁻¹ for PCShell (mass matrix changes every cycle)
    if (useSmoothPC_)
        updateDiagInv();

    // Point the pre-allocated Vecs at the caller's arrays (zero-copy)
    PetscCallAbort(PETSC_COMM_WORLD, VecPlaceArray(petsc_x_, x));
    PetscCallAbort(PETSC_COMM_WORLD, VecPlaceArray(petsc_b_, b));

    PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp_, petsc_b_, petsc_x_));

    // Report convergence (structured format for post-processing parser)
    PetscInt iter;
    PetscReal residual;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetIterationNumber(ksp_, &iter));
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetResidualNorm(ksp_, &residual));

    KSPConvergedReason reason;
    PetscCallAbort(PETSC_COMM_WORLD, KSPGetConvergedReason(ksp_, &reason));
    if (reason > 0) {
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscPrintf(PETSC_COMM_WORLD, "PETSc KSP converged: cycle=%d iterations=%d residual=%e\n",
                        cycle, (int)iter, (double)residual));
    } else {
        PetscCallAbort(PETSC_COMM_WORLD,
            PetscPrintf(PETSC_COMM_WORLD, "WARNING: PETSc KSP did NOT converge: cycle=%d reason=%d iterations=%d residual=%e\n",
                        cycle, (int)reason, (int)iter, (double)residual));
    }

    // Release the caller's arrays from the Vecs (does NOT free them)
    PetscCallAbort(PETSC_COMM_WORLD, VecResetArray(petsc_x_));
    PetscCallAbort(PETSC_COMM_WORLD, VecResetArray(petsc_b_));
}

#endif // USE_PETSC
