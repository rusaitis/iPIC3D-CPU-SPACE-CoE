/***************************************************************************
  PETSC.h  -  PETSc KSP solver for the implicit Maxwell system
 ***************************************************************************/

#ifndef PETSC_H
#define PETSC_H
#ifdef USE_PETSC

#include <petscksp.h>
#include <petscmat.h>
#include <string>
#include "Neighbouring_Nodes.h"
#include "arraysfwd.h"

class EMfields3D;
class VCtopology3D;

struct PetscSolverContext {
    EMfields3D *emf;
};

class PetscSolver {
    friend PetscErrorCode PetscSmoothPCApply(PC pc, Vec x, Vec y);
    friend PetscErrorCode PetscHelmholtzPCApply(PC pc, Vec x, Vec y);
public:
    PetscSolver(int localSize, EMfields3D *emf, const VCtopology3D *vct,
                double tol, const std::string &precType, bool diagnostics = false,
                const std::string &simName = "",
                const std::string &saveDir = "output",
                const std::string &helmInner = "AMG",
                const std::string &helmAMGType = "gamg",
                int helmVcycles = 2, int helmSweeps = 4, bool helmMassShift = true);
    ~PetscSolver();
    PetscSolver(const PetscSolver&) = delete;
    PetscSolver& operator=(const PetscSolver&) = delete;

    void solve(double *x, int size, const double *b, int cycle = -1);
    void printDiagnostics(int cycle) const;
    void dumpP(const char *filename) const;

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
    bool needsReassembly_;  // true if P contains cycle-dependent terms (e.g. mass matrix)
    bool diagnostics_;      // print norms and dump matrix to file
    std::string simName_;   // simulation name for output file prefixes
    std::string saveDir_;   // output directory (from input file SaveDirName)
    Mat P_;                 // explicit preconditioner matrix (MATMPIAIJ, block size 3)
    EMfields3D *emf_;       // same pointer as ctx_.emf; cached for direct use in assembleP()
    PetscInt globalBlockOffset_;  // local→global block index offset

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

    // ── Experimental PCShell preconditioner ──────────────────────────────
    //
    // Activated by input parameter PrecType = Smooth (default: None).
    // Uses FGMRES with a matrix-free PCApply callback.
    //
    // Current implementation: block-diagonal D⁻¹ at each node, where
    //   D(i,j,k) = I + (cθdt)²·CC_diag + dt·θ·4π·invVOL · M[g=0](i,j,k)
    //
    // This captures the local mass + curl-curl structure but NOT the 27-point
    // off-diagonal coupling. Tested approaches and results:
    //   - S(D⁻¹(S(r))): over-smooths, stalls at residual ~0.12
    //   - D⁻¹ only:      1.6–2× more iterations than PCNONE
    //
    // To experiment with new preconditioners:
    //   1. Modify PetscSmoothPCApply() in PETSC.cpp — that's the PCApply callback
    //   2. updateDiagInv() runs each cycle to refresh D⁻¹ from the current mass matrix
    //   3. energy_conserve_smooth_direction() is available for smoothing (public in EMfields3D)
    //   4. FGMRES calls PCApply and MatMult sequentially — no MPI re-entrancy issues
    //
    bool useSmoothPC_;
    void setupSmoothPC();
    void updateDiagInv();
    static void invert3x3(const double D[9], double Dinv[9]);
    array3_double *pcWorkX_, *pcWorkY_, *pcWorkZ_;  // work arrays for PCApply
    double ccDiag_[3];       // curl-curl diagonal per E-component
    double *diagInv_;        // pre-inverted 3×3 blocks, nxn*nyn*nzn*9 doubles

    // ── Physics-based scalar-Helmholtz preconditioner ────────────────────
    //
    // Activated by PrecType = Helmholtz.  Approximates the vector Maxwell
    // operator A by a SCALAR Helmholtz operator applied independently to each
    // E-component (auxiliary-space / Hiptmair–Xu style reduction):
    //
    //   H = (1 + s·m̄)·I  −  α·∇²       α = (cθΔt)²,  s = θΔt·4π/V,  m̄ = ⅓tr M[g=0]
    //
    // On the stiff solenoidal modes (∇·E≈0) the curl-curl reduces to −∇², so H
    // captures the hard part of A while being SPD with a bounded condition
    // number and NO null space (the +1 and plasma shift s·m̄ ~ (ωₚΔt)² lift it).
    // Unlike the vector curl-curl operator (which defeats AMG via its gradient
    // null space), H is exactly what AMG solves well.
    //
    // Both inner engines run on the same explicit scalar matrix H_ (the
    // "matrix-free Chebyshev" variant is realized as PETSc's KSPCHEBYSHEV+PCJACOBI,
    // which needs no hand-rolled matvec or halo exchange):
    //   - AMG:       nested KSPRICHARDSON + PCGAMG/PCHYPRE, hVcycles_ sweeps
    //   - Chebyshev: nested KSPCHEBYSHEV  + PCJACOBI,        hSweeps_  iterations
    // Outer KSP is FGMRES (the nested solve is a variable preconditioner).
    bool useHelmholtzPC_;
    enum class HInner { AMG, Chebyshev };
    HInner hInner_;
    std::string hAMGType_;     // "gamg" | "hypre"
    int  hVcycles_, hSweeps_;  // nested KSP max_it (AMG / Chebyshev)
    bool hMassShift_;          // include s·m̄ plasma shift
    double scalarLapStencil_[3][3][3];  // discrete node Laplacian ∇² (tensor-product, ≡ lapN2N on uniform grid)
    Mat  H_;                   // scalar Helmholtz matrix (MATMPIAIJ, block size 1)
    KSP  helmKSP_;             // nested inner solver for H·ψ = r
    Vec  helmR_, helmPsi_;     // scalar RHS / solution work Vecs
    Vec  helmDiag_;            // refreshed diagonal (−α·∇²_self + 1 + s·m̄) for MatDiagonalSet
    void setupHelmholtzPC();
    void assembleHelmholtz();          // build H_ off-diagonals + initial diagonal
    void updateHelmholtzDiagonal();    // refresh (1 + s·m̄) diagonal each cycle
    void computeScalarLaplacianStencil();
};

PetscErrorCode PetscMaxwellMatMult(Mat A, Vec x, Vec y);
PetscErrorCode PetscSmoothPCApply(PC pc, Vec x, Vec y);
PetscErrorCode PetscHelmholtzPCApply(PC pc, Vec x, Vec y);

#endif // USE_PETSC
#endif // PETSC_H
