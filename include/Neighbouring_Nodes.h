#ifndef NEIGHBOURINGNODES_H
#define NEIGHBOURINGNODES_H

#include <cstdlib>

//!  Class to handle which nodes have to be computed when the mass matrix is calculated !//
//
// The mass matrix product cube has support determined by the particle shape function:
//   order = 1 (linear / CIC, 8-node particle support)
//       -> 3x3x3 product cube, 27 unique offsets
//       -> stored as center + 13 forward halves (NE_MASS = 14, ±-symmetric)
//   order = 2 (quadratic / TSC, 27-node particle support)
//       -> 5x5x5 product cube, 125 unique offsets
//       -> stored as center + 62 forward halves (NE_MASS = 63, ±-symmetric)
//
// In both cases, getX/Y/Z(g) for g in [0, ne_mass) returns the forward offset
// of the g-th group; the corresponding backward offset is the negation. The
// mass-matrix-times-vector code (and the per-particle assembly code) loops
// over forward groups and applies +/- symmetry.

class NeighbouringNodes
{
    public:

    //* Default ctor: linear (CIC), preserves the legacy fixed table.
    NeighbouringNodes()
    {
        build_linear();
        order_ = 1;
        ne_mass_ = 14;
    };

    //* Order-parameterised ctor:
    //*   order = 1 -> linear  (NE_MASS = 14, 3x3x3 cube)
    //*   order = 2 -> quadratic / TSC (NE_MASS = 63, 5x5x5 cube)
    explicit NeighbouringNodes(int order)
    {
        order_   = order;
        if (order == 1) {
            build_linear();
            ne_mass_ = 14;
        } else if (order == 2) {
            build_quadratic();
            ne_mass_ = 63;
        } else {
            // Caller's responsibility to validate; abort hard if it slips through.
            std::abort();
        }
    }

    int order()   const { return order_; }
    int ne_mass() const { return ne_mass_; }

    //* Forward offset of group `ind` (0 = center, 1..ne_mass-1 = neighbours).
    //* For backwards compatibility with legacy callers that pass ind > ne_mass-1
    //* (the linear table used to allow ind in [0, 27) and return -i[ind-13]),
    //* we keep that fallback for order = 1 only.
    int getX(int ind) const
    {
        if (ind < ne_mass_) return i[ind];
        if (order_ == 1)    return -i[ind - 13];
        std::abort();
    }
    int getY(int ind) const
    {
        if (ind < ne_mass_) return j[ind];
        if (order_ == 1)    return -j[ind - 13];
        std::abort();
    }
    int getZ(int ind) const
    {
        if (ind < ne_mass_) return k[ind];
        if (order_ == 1)    return -k[ind - 13];
        std::abort();
    }


    private:
    static const int MAX_NE_MASS = 63;
    int i[MAX_NE_MASS];
    int j[MAX_NE_MASS];
    int k[MAX_NE_MASS];
    int order_;
    int ne_mass_;

    void build_linear()
    {
        // 14 distinct +/- groups covering the 3x3x3 cube. Layout matches the
        // legacy hand-rolled table — do not reorder, the assembly code in
        // Particles3D::computeMoments() relies on this exact ordering for the
        // (i, j, k) overlap test.
        i[0 ] = 0; j[0 ] = 0; k[0 ] = 0;  // center

        i[1 ] = 1; j[1 ] = 0; k[1 ] = 0;
        i[2 ] = 0; j[2 ] = 1; k[2 ] = 0;
        i[3 ] = 0; j[3 ] = 0; k[3 ] = 1;

        i[4 ] = 1; j[4 ] = 1; k[4 ] = 0;
        i[5 ] = 1; j[5 ] =-1; k[5 ] = 0;
        i[6 ] = 1; j[6 ] = 0; k[6 ] = 1;
        i[7 ] = 1; j[7 ] = 0; k[7 ] =-1;
        i[8 ] = 0; j[8 ] = 1; k[8 ] = 1;
        i[9 ] = 0; j[9 ] = 1; k[9 ] =-1;

        i[10] = 1; j[10] =-1; k[10] = 1;
        i[11] = 1; j[11] = 1; k[11] =-1;
        i[12] =-1; j[12] = 1; k[12] = 1;

        i[13] = 1; j[13] = 1; k[13] = 1;
    }

    void build_quadratic()
    {
        // 63 distinct +/- groups covering the 5x5x5 cube (TSC product support).
        // Group 0 is the center; groups 1..62 are one half of the 124 non-center
        // offsets, picked by the lexicographic-positive convention:
        //   (di > 0)  ||  (di == 0 && dj > 0)  ||  (di == 0 && dj == 0 && dk > 0)
        // The reverse offsets are then -group, exactly as in the linear table.
        int g = 0;
        i[g] = 0; j[g] = 0; k[g] = 0;
        ++g;
        for (int di = -2; di <= 2; ++di)
            for (int dj = -2; dj <= 2; ++dj)
                for (int dk = -2; dk <= 2; ++dk)
                {
                    if (di == 0 && dj == 0 && dk == 0) continue;
                    const bool positive_half =
                        (di > 0) ||
                        (di == 0 && dj > 0) ||
                        (di == 0 && dj == 0 && dk > 0);
                    if (!positive_half) continue;
                    i[g] = di; j[g] = dj; k[g] = dk;
                    ++g;
                }
        // Sanity: the loop must produce exactly 62 forward groups -> g == 63.
        if (g != 63) std::abort();
    }
};

#endif
