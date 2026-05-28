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

#include "Com3DNonblk.h"
#include "EMfields3D.h"
#include "Collective.h"

//* Halo call kinds. Each value names the public-API wrapper it serves and
//* maps to a fixed combination of the four behavioural flags decoded by
//* `to_flags`. Internal: only used inside this TU.
enum class HaloKind {
    NodeBC,                  // node + full halo + copy + field comm
    NodeBC_P,                // node + full halo + copy + particle comm
    NodeBoxStencilBC,        // node + face-only + copy + field comm
    NodeBoxStencilBC_P,      // node + face-only + copy + particle comm
    CenterBC,                // center + full halo + copy + field comm
    CenterBC_P,              // center + full halo + copy + particle comm
    CenterBoxStencilBC,      // center + face-only + copy + field comm
    CenterBoxStencilBC_P,    // center + face-only + copy + particle comm
    Interp,                  // center + full halo + interp (sum-on-receive) + particle comm
};

struct HaloFlagBits {
    bool is_center;
    bool face_only;
    bool need_interp;
    bool is_particle;
};

constexpr HaloFlagBits to_flags(HaloKind k) {
    switch (k) {
        case HaloKind::NodeBC:                 return {false, false, false, false};
        case HaloKind::NodeBC_P:               return {false, false, false, true };
        case HaloKind::NodeBoxStencilBC:       return {false, true,  false, false};
        case HaloKind::NodeBoxStencilBC_P:     return {false, true,  false, true };
        case HaloKind::CenterBC:               return {true,  false, false, false};
        case HaloKind::CenterBC_P:             return {true,  false, false, true };
        case HaloKind::CenterBoxStencilBC:     return {true,  true,  false, false};
        case HaloKind::CenterBoxStencilBC_P:   return {true,  true,  false, true };
        case HaloKind::Interp:                 return {true,  false, true,  true };
    }
    return {false, false, false, false};  //* unreachable; quiets non-exhaustive-switch warnings
}

//* Single source of truth for the entire halo exchange. `vectors` is an
//* array of `n_fields` pointers to 3D field arrays sharing the same
//* (nx, ny, nz) extents and MPI topology. All N fields are exchanged in
//* one batched MPI message per direction.
//*
//*   kind           Which behavioural flag combination to apply (see
//*                  HaloKind / to_flags above).
//*   unify_ps_dups  At periodic-self axes, fold the LO=HI strict
//*                  duplicate before the cross-rank pack so neighbours
//*                  inherit post-unify strict (mass-matrix / ECSIM).
static void NBDerivedHaloCommN(int nx, int ny, int nz,
                                int n_fields, double ****vectors,
                                const VirtualTopology3D * vct, EMfields3D *EMf,
                                HaloKind kind,
                                bool unify_ps_dups = false);


// Halo state computed once per NBDerivedHaloCommN call, shared by the stage helpers.
namespace {

struct HaloContext {
    int n_ghost;
    MPI_Comm comm;
    int myrank;

    //* Neighbour ranks, indexed: 0=Xleft 1=Xright 2=Yleft 3=Yright 4=Zleft 5=Zright.
    int neighbours[6];
    int communicationCnt[6];

    //* Cartesian topology cache (single MPI_Cart_get).
    int my_coords[3];
    int dims[3];
    int periods[3];
    bool any_xrank_periodic;

    bool isCenterDim;
    //* face_send_offset: depth-from-boundary of the cell packed for cross-rank
    //* sends. image_offset: depth used in periodic-image stride formulas. Both
    //* are (isCenterFlag ? 0 : 1) at n_ghost==1, (isCenterDim ? 0 : 1) at n_ghost>1.
    int  face_send_offset;
    int  image_offset;

    //* Periodic-self flags: axis is periodic and decomposition along axis = 1.
    bool ps_x_self, ps_y_self, ps_z_self;

    //* Periodic-self ghost-source remap, no-op at n_ghost=1.
    int g_src(int g) const { return n_ghost - 1 - g; }

    //* Cart-neighbour lookup with periodic wrap. Returns MPI_PROC_NULL on
    //* non-periodic out-of-range, myrank when the neighbour wraps to self.
    int neighbour_at(int sx, int sy, int sz) const {
        int nbr[3] = {my_coords[0] + sx, my_coords[1] + sy, my_coords[2] + sz};
        for (int d = 0; d < 3; ++d) {
            if (nbr[d] < 0 || nbr[d] >= dims[d]) {
                if (!periods[d]) return MPI_PROC_NULL;
                nbr[d] = (nbr[d] % dims[d] + dims[d]) % dims[d];
            }
        }
        int r;
        MPI_Cart_rank(comm, nbr, &r);
        return r;
    }
};

static HaloContext make_halo_context(EMfields3D *EMf,
                                      const VirtualTopology3D *vct,
                                      bool isCenterFlag, bool needInterp,
                                      bool isParticle)
{
    HaloContext ctx;
    ctx.n_ghost = EMf->getNGhost();
    ctx.comm    = isParticle ? vct->getParticleComm() : vct->getFieldComm();
    ctx.myrank  = vct->getCartesian_rank();

    ctx.neighbours[0] = isParticle ? vct->getXleft_neighbor_P()  : vct->getXleft_neighbor();
    ctx.neighbours[1] = isParticle ? vct->getXright_neighbor_P() : vct->getXright_neighbor();
    ctx.neighbours[2] = isParticle ? vct->getYleft_neighbor_P()  : vct->getYleft_neighbor();
    ctx.neighbours[3] = isParticle ? vct->getYright_neighbor_P() : vct->getYright_neighbor();
    ctx.neighbours[4] = isParticle ? vct->getZleft_neighbor_P()  : vct->getZleft_neighbor();
    ctx.neighbours[5] = isParticle ? vct->getZright_neighbor_P() : vct->getZright_neighbor();

    for (int i = 0; i < 6; ++i) {
        ctx.communicationCnt[i] =
            (ctx.neighbours[i] != MPI_PROC_NULL && ctx.neighbours[i] != ctx.myrank) ? 1 : 0;
    }

    MPI_Cart_get(ctx.comm, 3, ctx.dims, ctx.periods, ctx.my_coords);
    ctx.any_xrank_periodic =
        (ctx.periods[0] && ctx.dims[0] > 1) ||
        (ctx.periods[1] && ctx.dims[1] > 1) ||
        (ctx.periods[2] && ctx.dims[2] > 1);

    ctx.isCenterDim = (isCenterFlag && !needInterp);
    if (ctx.n_ghost == 1) {
        ctx.face_send_offset = isCenterFlag ? 0 : 1;
        ctx.image_offset     = isCenterFlag ? 0 : 1;
    } else {
        ctx.face_send_offset = ctx.isCenterDim ? 0 : 1;
        ctx.image_offset     = ctx.isCenterDim ? 0 : 1;
    }

    ctx.ps_x_self = (ctx.neighbours[0] == ctx.myrank && ctx.neighbours[1] == ctx.myrank);
    ctx.ps_y_self = (ctx.neighbours[2] == ctx.myrank && ctx.neighbours[3] == ctx.myrank);
    ctx.ps_z_self = (ctx.neighbours[4] == ctx.myrank && ctx.neighbours[5] == ctx.myrank);

    return ctx;
}

}  // namespace

//* Copies the (sa,sb) diagonal corner from the diagonal Cart neighbour, which
//* holds the correct cell — the single-axis edge phase would route it through
//* the wrong rank at a 2-axis cross-rank corner, so this overrides it (copy
//* semantics, c over strict+dup). No-op unless any_xrank_periodic; 3-axis
//* 8-rank corners are not closed here.
static void do_diagonal_edge_copy(int nx, int ny, int nz,
                                   int n_fields, double ****vectors,
                                   const HaloContext &ctx)
{
    if (!ctx.any_xrank_periodic) return;

    const int       n_ghost          = ctx.n_ghost;
    const int       face_send_offset = ctx.face_send_offset;
    const MPI_Comm  comm             = ctx.comm;
    const int       myrank           = ctx.myrank;
    const int* const dims            = ctx.dims;
    const int* const periods         = ctx.periods;

    const bool xrank[3] = {
        periods[0] && dims[0] > 1,
        periods[1] && dims[1] > 1,
        periods[2] && dims[2] > 1
    };
    const int  nax[3]   = {nx, ny, nz};

    //* Tag scheme: 500 + (axis_a*3 + axis_b)*16 + ((sa+1)/2)*8 + ((sb+1)/2)*4.
    //* Range fits in [500, 547]. Disjoint from face tags (1..6), the 26-slot
    //* cross-rank SOR (200..226), the corner-completion pass (300..335), and
    //* the face-cell completion pass (400..443).
    auto detag = [](int axis_a, int axis_b, int sa, int sb) {
        return 500 + (axis_a * 3 + axis_b) * 16
                    + ((sa + 1) / 2) * 8
                    + ((sb + 1) / 2) * 4;
    };

    struct DSlot {
        int axis_a, axis_b, axis_c;
        int sa, sb;
        int len_c;
        int c_lo;
        int nbr;
        std::vector<double> send;
        std::vector<double> recv;
    };
    std::vector<DSlot> dslots;
    dslots.reserve(12);

    //* Unordered pairs (b > a) — each pair handles all 4 (sa, sb) corners.
    for (int a = 0; a < 3; ++a) {
        if (!xrank[a]) continue;
        for (int b = a + 1; b < 3; ++b) {
            if (!xrank[b]) continue;
            const int c = 3 - a - b;
            //* c-perp range covers c-strict + c-dup so stencil reaches into
            //* c are valid. c-ghost is filled separately by FACE PHASE.
            const int len_c = nax[c] - 2;  // [1..nax-2]: strict + dup
            const int c_lo  = 1;
            if (len_c <= 0) continue;

            for (int sa : {-1, +1}) {
                for (int sb : {-1, +1}) {
                    int sxyz[3] = {0, 0, 0};
                    sxyz[a] = sa;
                    sxyz[b] = sb;
                    const int nbr = ctx.neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                    if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

                    dslots.push_back({a, b, c, sa, sb, len_c, c_lo,
                                      nbr, {}, {}});
                    DSlot &s = dslots.back();
                    const int slab = n_ghost * n_ghost * len_c;
                    s.send.resize(n_fields * slab);
                    s.recv.resize(n_fields * slab);

                    //* Pack at (sa, sb) strict-near-bdry of self.
                    //* g_src(ga) = n_ghost-1-ga.
                    int ijk[3];
                    for (int f = 0; f < n_fields; ++f) {
                        for (int ga = 0; ga < n_ghost; ++ga) {
                            ijk[a] = (sa == +1)
                                     ? (nax[a] - 1 - n_ghost - face_send_offset
                                        - (n_ghost - 1 - ga))
                                     : (n_ghost + face_send_offset
                                        + (n_ghost - 1 - ga));
                            for (int gb = 0; gb < n_ghost; ++gb) {
                                ijk[b] = (sb == +1)
                                         ? (nax[b] - 1 - n_ghost - face_send_offset
                                            - (n_ghost - 1 - gb))
                                         : (n_ghost + face_send_offset
                                            + (n_ghost - 1 - gb));
                                for (int hc = 0; hc < len_c; ++hc) {
                                    ijk[c] = c_lo + hc;
                                    s.send[f * slab + (ga * n_ghost + gb) * len_c + hc] =
                                        vectors[f][ijk[0]][ijk[1]][ijk[2]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<MPI_Request> drqs;
    drqs.reserve(2 * dslots.size());
    for (DSlot &s : dslots) {
        const int n = n_fields * n_ghost * n_ghost * s.len_c;
        const int tag_send = detag(s.axis_a, s.axis_b,  s.sa,  s.sb);
        const int tag_recv = detag(s.axis_a, s.axis_b, -s.sa, -s.sb);
        drqs.emplace_back();
        MPI_Irecv(s.recv.data(), n, MPI_DOUBLE, s.nbr,
                  tag_recv, comm, &drqs.back());
        drqs.emplace_back();
        MPI_Isend(s.send.data(), n, MPI_DOUBLE, s.nbr,
                  tag_send, comm, &drqs.back());
    }
    if (!drqs.empty())
        MPI_Waitall(drqs.size(), drqs.data(), MPI_STATUSES_IGNORE);

    //* Apply: COPY (=) into (sa, sb) ghost corner. Override the wrong value
    //* written by the standard EDGE PHASE.
    for (const DSlot &s : dslots) {
        const int slab = n_ghost * n_ghost * s.len_c;
        int ijk[3];
        for (int f = 0; f < n_fields; ++f) {
            for (int ga = 0; ga < n_ghost; ++ga) {
                ijk[s.axis_a] = (s.sa == -1) ? ga : (nax[s.axis_a] - 1 - ga);
                for (int gb = 0; gb < n_ghost; ++gb) {
                    ijk[s.axis_b] = (s.sb == -1) ? gb : (nax[s.axis_b] - 1 - gb);
                    for (int hc = 0; hc < s.len_c; ++hc) {
                        ijk[s.axis_c] = s.c_lo + hc;
                        vectors[f][ijk[0]][ijk[1]][ijk[2]] =
                            s.recv[f * slab + (ga * n_ghost + gb) * s.len_c + hc];
                    }
                }
            }
        }
    }
}

//* Closes the cross-rank moment halo at TSC width with three sum passes sharing
//* the src_idx/dst_idx helpers: a 26-slot ghost-slab exchange, a corner pass
//* that completes 4-rank corners the diagonal slot only pairs 2 ranks for, and
//* a face-cell pass for the 4th rank's contribution at face-strict cells.
//* No-op unless (needInterp && n_ghost > 1 && any_xrank_periodic).
static void do_xrank_completion(int nx, int ny, int nz,
                                 int n_fields, double ****vectors,
                                 const HaloContext &ctx, bool needInterp)
{
    //* TSC-only: at n_ghost=1 dst_idx degenerates; CIC unifies cross-rank dups
    //* via face_send_offset=0 + addFace(skip_self=false) instead.
    if (!(needInterp && ctx.n_ghost > 1 && ctx.any_xrank_periodic)) return;

    const int       n_ghost      = ctx.n_ghost;
    const int       image_offset = ctx.image_offset;
    const MPI_Comm  comm         = ctx.comm;
    const int       myrank       = ctx.myrank;
    const int* const dims        = ctx.dims;
    const int* const periods     = ctx.periods;

    //* Per-axis (n, s) → source index, and (n, s) → destination index.
    //* h is the layer counter:
    //*   sd = -1: h ∈ [0..ng]  (h=0..ng-1 ghost outer→inner; h=ng LO dup at i=ng)
    //*   sd = +1: h ∈ [0..ng]  (h=0..ng-1 ghost outer→inner; h=ng HI dup)
    //*   sd =  0: h ∈ [0..len-1] over strict interior [ng+1..n-2-ng]
    auto src_idx = [&](int sd, int n, int h) -> int {
        if (sd == -1) return h;
        if (sd == +1) return n - 1 - h;
        return n_ghost + 1 + h;
    };
    auto dst_idx = [&](int sd, int n, int h) -> int {
        if (sd == -1) {
            return (h < n_ghost) ? (h + n - 2 * n_ghost - image_offset) : (n - 1 - n_ghost);
        }
        if (sd == +1) {
            return (h < n_ghost) ? ((2 * n_ghost - 1 + image_offset) - h) : n_ghost;
        }
        return n_ghost + 1 + h;
    };

    //* 26-slot ghost-slab SOR pre-pass.
    {
        const int strict_x = nx - 2 * n_ghost - 2;
        const int strict_y = ny - 2 * n_ghost - 2;
        const int strict_z = nz - 2 * n_ghost - 2;
        const int dup_n    = n_ghost + 1;

        //* Tags: encode (sx,sy,sz) as a base-3 number in [0..26]; offset 200
        //* keeps it disjoint from standard exchange tags 1..6.
        auto encode_tag = [](int sx, int sy, int sz) {
            return 200 + (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
        };

        struct ExchSlot {
            int sx, sy, sz;
            int nbr;
            int lx, ly, lz;
            std::vector<double> send;
            std::vector<double> recv;
        };
        std::vector<ExchSlot> slots;
        slots.reserve(26);

        for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
        for (int sz = -1; sz <= 1; ++sz) {
            if (sx == 0 && sy == 0 && sz == 0) continue;
            const int nbr = ctx.neighbour_at(sx, sy, sz);
            if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

            const int lx = (sx != 0) ? dup_n : strict_x;
            const int ly = (sy != 0) ? dup_n : strict_y;
            const int lz = (sz != 0) ? dup_n : strict_z;
            if (lx <= 0 || ly <= 0 || lz <= 0) continue;

            slots.push_back({sx, sy, sz, nbr, lx, ly, lz, {}, {}});
            ExchSlot &s = slots.back();
            const int slab = lx * ly * lz;
            s.send.resize(n_fields * slab);
            s.recv.resize(n_fields * slab);
            //* Field-major pack: buf[f*slab + (hx*ly+hy)*lz+hz].
            for (int f = 0; f < n_fields; ++f) {
                for (int hx = 0; hx < lx; ++hx) {
                    const int ix = src_idx(sx, nx, hx);
                    for (int hy = 0; hy < ly; ++hy) {
                        const int iy = src_idx(sy, ny, hy);
                        for (int hz = 0; hz < lz; ++hz) {
                            const int iz = src_idx(sz, nz, hz);
                            s.send[f * slab + (hx * ly + hy) * lz + hz] = vectors[f][ix][iy][iz];
                        }
                    }
                }
            }
        }

        std::vector<MPI_Request> rqs;
        rqs.reserve(2 * slots.size());
        for (ExchSlot &s : slots) {
            const int slab_total = n_fields * s.lx * s.ly * s.lz;
            const int tag_send = encode_tag( s.sx,  s.sy,  s.sz);
            const int tag_recv = encode_tag(-s.sx, -s.sy, -s.sz);
            rqs.emplace_back();
            MPI_Irecv(s.recv.data(), slab_total, MPI_DOUBLE, s.nbr, tag_recv, comm, &rqs.back());
            rqs.emplace_back();
            MPI_Isend(s.send.data(), slab_total, MPI_DOUBLE, s.nbr, tag_send, comm, &rqs.back());
        }
        if (!rqs.empty()) MPI_Waitall(rqs.size(), rqs.data(), MPI_STATUSES_IGNORE);

        //* SOR received slabs into the matching destination region. Each
        //* (sx,sy,sz) slot's recv buffer holds the neighbour's (-sx,-sy,-sz)
        //* ghost slab — physically the periodic-image of OUR (-sx,-sy,-sz)
        //* destination strict + dup.
        for (const ExchSlot &s : slots) {
            const int slab = s.lx * s.ly * s.lz;
            for (int f = 0; f < n_fields; ++f) {
                for (int hx = 0; hx < s.lx; ++hx) {
                    const int ix = dst_idx(-s.sx, nx, hx);
                    for (int hy = 0; hy < s.ly; ++hy) {
                        const int iy = dst_idx(-s.sy, ny, hy);
                        for (int hz = 0; hz < s.lz; ++hz) {
                            const int iz = dst_idx(-s.sz, nz, hz);
                            vectors[f][ix][iy][iz] += s.recv[f * slab + (hx * s.ly + hy) * s.lz + hz];
                        }
                    }
                }
            }
        }
    }

    //* Multi-axis corner completion.
    {
        const bool xrank[3] = {
            periods[0] && dims[0] > 1,
            periods[1] && dims[1] > 1,
            periods[2] && dims[2] > 1
        };
        const int  nax[3]   = {nx, ny, nz};

        //* Tag scheme: 300 + axis_a*12 + ((sa+1)/2)*6 + axis_b*2 + dup_side.
        //* Range [300, 335], disjoint from face tags (1..6) and the SOR pass (200..226).
        auto ctag = [](int axis_a, int sa, int axis_b, int dup_side) {
            return 300 + axis_a * 12 + ((sa + 1) / 2) * 6 + axis_b * 2 + dup_side;
        };

        struct CSlot {
            int axis_a, axis_b, axis_c;
            int sa;
            int dup_side;       //* 0 = LO (i_b = ng), 1 = HI (i_b = n_b-ng-1)
            int dup_b;
            int len_c;
            int c_lo;           //* c-axis start: 0 (full) or ng+1 (strict).
            int nbr;
            std::vector<double> send;
            std::vector<double> recv;
        };
        std::vector<CSlot> cslots;
        cslots.reserve(24);

        //* Unordered pairs (b > a). Adding (b,a) sends would over-count.
        for (int a = 0; a < 3; ++a) {
            if (!xrank[a]) continue;
            for (int b = a + 1; b < 3; ++b) {
                if (!xrank[b]) continue;
                const int c = 3 - a - b;
                //* c-perp: strict-c when c is cross-rank; full (0..nc-1) when
                //* c is single-rank periodic / non-periodic.
                const bool c_full = !xrank[c];
                const int  len_c  = c_full ? nax[c] : (nax[c] - 2 * n_ghost - 2);
                const int  c_lo   = c_full ? 0 : (n_ghost + 1);
                if (len_c <= 0) continue;
                for (int sa : {-1, +1}) {
                    int sxyz[3] = {0, 0, 0};
                    sxyz[a] = sa;
                    const int nbr = ctx.neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                    if (nbr == MPI_PROC_NULL || nbr == myrank) continue;
                    for (int dup_side = 0; dup_side < 2; ++dup_side) {
                        const int dup_b = (dup_side == 0)
                                          ? n_ghost
                                          : (nax[b] - n_ghost - 1);
                        cslots.push_back({a, b, c, sa, dup_side, dup_b,
                                          len_c, c_lo, nbr, {}, {}});
                        CSlot &s = cslots.back();
                        s.send.resize(n_fields * len_c);
                        s.recv.resize(n_fields * len_c);
                        const int idx_a = (sa == -1)
                                          ? n_ghost
                                          : (nax[a] - n_ghost - 1);
                        int ijk[3];
                        ijk[a] = idx_a;
                        ijk[b] = dup_b;
                        for (int f = 0; f < n_fields; ++f) {
                            for (int hc = 0; hc < len_c; ++hc) {
                                ijk[c] = c_lo + hc;
                                s.send[f * len_c + hc] = vectors[f][ijk[0]][ijk[1]][ijk[2]];
                            }
                        }
                    }
                }
            }
        }

        std::vector<MPI_Request> crqs;
        crqs.reserve(2 * cslots.size());
        for (CSlot &s : cslots) {
            const int n = n_fields * s.len_c;
            const int tag_send = ctag(s.axis_a,  s.sa, s.axis_b, s.dup_side);
            const int tag_recv = ctag(s.axis_a, -s.sa, s.axis_b, s.dup_side);
            crqs.emplace_back();
            MPI_Irecv(s.recv.data(), n, MPI_DOUBLE, s.nbr,
                      tag_recv, comm, &crqs.back());
            crqs.emplace_back();
            MPI_Isend(s.send.data(), n, MPI_DOUBLE, s.nbr,
                      tag_send, comm, &crqs.back());
        }
        if (!crqs.empty())
            MPI_Waitall(crqs.size(), crqs.data(), MPI_STATUSES_IGNORE);

        //* Apply: receiver sums into the SAME i_a dup position it sent from.
        for (const CSlot &s : cslots) {
            const int idx_a = (s.sa == -1)
                              ? n_ghost
                              : (nax[s.axis_a] - n_ghost - 1);
            int ijk[3];
            ijk[s.axis_a] = idx_a;
            ijk[s.axis_b] = s.dup_b;
            for (int f = 0; f < n_fields; ++f) {
                for (int hc = 0; hc < s.len_c; ++hc) {
                    ijk[s.axis_c] = s.c_lo + hc;
                    vectors[f][ijk[0]][ijk[1]][ijk[2]] += s.recv[f * s.len_c + hc];
                }
            }
        }
    }

    //* Face-cell completion.
    {
        const bool xrank[3] = {
            periods[0] && dims[0] > 1,
            periods[1] && dims[1] > 1,
            periods[2] && dims[2] > 1
        };
        const int  nax[3]   = {nx, ny, nz};

        //* Tag: 400 + axis_a*16 + axis_b*4 + sa_side*2 + (sb+1)/2.
        //* Range [400, 443], disjoint from face (1..6), the 26-slot SOR
        //* (200..226), and the corner-completion pass (300..335).
        auto ftag = [](int axis_a, int axis_b, int sa_side, int sb) {
            return 400 + axis_a * 16 + axis_b * 4 + sa_side * 2 + (sb + 1) / 2;
        };

        struct FSlot {
            int axis_a, axis_b, axis_c;
            int sa_side;       //* 0 = LO (i_a = ng), 1 = HI (i_a = nax-ng-1)
            int sb;            //* -1 or +1
            int len_c;
            int c_lo;
            int nbr;
            std::vector<double> send;
            std::vector<double> recv;
        };
        std::vector<FSlot> fslots;
        fslots.reserve(24);

        //* ORDERED pairs (a != b). Both (a,b) and (b,a) cover different cells
        //* and are required.
        for (int a = 0; a < 3; ++a) {
            if (!xrank[a]) continue;
            for (int b = 0; b < 3; ++b) {
                if (b == a || !xrank[b]) continue;
                const int c = 3 - a - b;
                const bool c_full = !xrank[c];
                const int  len_c  = c_full ? nax[c] : (nax[c] - 2 * n_ghost - 2);
                const int  c_lo   = c_full ? 0 : (n_ghost + 1);
                if (len_c <= 0) continue;

                for (int sa_side = 0; sa_side < 2; ++sa_side) {
                    const int idx_a = (sa_side == 0)
                                      ? n_ghost
                                      : (nax[a] - n_ghost - 1);
                    for (int sb : {-1, +1}) {
                        int sxyz[3] = {0, 0, 0};
                        sxyz[b] = sb;
                        const int nbr = ctx.neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                        if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

                        fslots.push_back({a, b, c, sa_side, sb,
                                          len_c, c_lo, nbr, {}, {}});
                        FSlot &s = fslots.back();
                        const int slab = n_ghost * len_c;
                        s.send.resize(n_fields * slab);
                        s.recv.resize(n_fields * slab);
                        int ijk[3];
                        ijk[a] = idx_a;
                        for (int f = 0; f < n_fields; ++f) {
                            for (int h_b = 0; h_b < n_ghost; ++h_b) {
                                ijk[b] = src_idx(sb, nax[b], h_b);
                                for (int hc = 0; hc < len_c; ++hc) {
                                    ijk[c] = c_lo + hc;
                                    s.send[f * slab + h_b * len_c + hc] =
                                        vectors[f][ijk[0]][ijk[1]][ijk[2]];
                                }
                            }
                        }
                    }
                }
            }
        }

        std::vector<MPI_Request> frqs;
        frqs.reserve(2 * fslots.size());
        for (FSlot &s : fslots) {
            const int n = n_fields * n_ghost * s.len_c;
            const int tag_send = ftag(s.axis_a, s.axis_b, s.sa_side,  s.sb);
            const int tag_recv = ftag(s.axis_a, s.axis_b, s.sa_side, -s.sb);
            frqs.emplace_back();
            MPI_Irecv(s.recv.data(), n, MPI_DOUBLE, s.nbr,
                      tag_recv, comm, &frqs.back());
            frqs.emplace_back();
            MPI_Isend(s.send.data(), n, MPI_DOUBLE, s.nbr,
                      tag_send, comm, &frqs.back());
        }
        if (!frqs.empty())
            MPI_Waitall(frqs.size(), frqs.data(), MPI_STATUSES_IGNORE);

        //* Apply: matching neighbour's slot has -sb. Receiver applies at
        //* idx_a (same dup-column position) and at j_b = dst_idx(-s.sb,...).
        for (const FSlot &s : fslots) {
            const int idx_a = (s.sa_side == 0)
                              ? n_ghost
                              : (nax[s.axis_a] - n_ghost - 1);
            const int slab = n_ghost * s.len_c;
            int ijk[3];
            ijk[s.axis_a] = idx_a;
            for (int f = 0; f < n_fields; ++f) {
                for (int h_b = 0; h_b < n_ghost; ++h_b) {
                    ijk[s.axis_b] = dst_idx(-s.sb, nax[s.axis_b], h_b);
                    for (int hc = 0; hc < s.len_c; ++hc) {
                        ijk[s.axis_c] = s.c_lo + hc;
                        vectors[f][ijk[0]][ijk[1]][ijk[2]] +=
                            s.recv[f * slab + h_b * s.len_c + hc];
                    }
                }
            }
        }
    }
}

//* Edge exchange (manual pack/unpack, n_ghost >= 1). Each axis has 4 edges; for
//* n_ghost > 1 each expands to an n_ghost x n_ghost block. Routing: yEdge → X
//* neighbours, zEdge → Y, xEdge → Z, each side batching its 2 perpendicular
//* corners into one message. Self-copies run before Waitall+unpack, so unpack
//* wins where both target the same cell.
static void do_edge_phase(int nx, int ny, int nz,
                           int n_fields, double ****vectors,
                           const HaloContext &ctx)
{
    const int        n_ghost          = ctx.n_ghost;
    const int        face_send_offset = ctx.face_send_offset;
    const int        image_offset     = ctx.image_offset;
    const MPI_Comm   comm             = ctx.comm;
    const int        myrank  = ctx.myrank;
    const int* const cnt     = ctx.communicationCnt;
    const int        lnX     = ctx.neighbours[0], rnX = ctx.neighbours[1];
    const int        lnY     = ctx.neighbours[2], rnY = ctx.neighbours[3];
    const int        lnZ     = ctx.neighbours[4], rnZ = ctx.neighbours[5];
    const int tag_XL=1, tag_YL=2, tag_ZL=3, tag_XR=4, tag_YR=5, tag_ZR=6;
    const int neighbours_arr[6] = { lnX, rnX, lnY, rnY, lnZ, rnZ };
    const int recv_tags[6] = { tag_XR, tag_XL, tag_YR, tag_YL, tag_ZR, tag_ZL };
    const int send_tags[6] = { tag_XL, tag_XR, tag_YL, tag_YR, tag_ZL, tag_ZR };

    const int n_ghost_sq = n_ghost * n_ghost;
    const int yedge_xs   = ny - 2;
    const int zedge_xs   = nz - 2;
    const int xedge_xs   = nx - 2;

    //* Per Cart-direction d ∈ [0..5]: which edge axis routes through and
    //* which two perp commCnt sides feed the 2 corners.
    //*   d=0 X-LO / d=1 X-HI : yEdge, perp = Z (sides 4,5), xs len = ny-2
    //*   d=2 Y-LO / d=3 Y-HI : zEdge, perp = X (sides 0,1), xs len = nz-2
    //*   d=4 Z-LO / d=5 Z-HI : xEdge, perp = Y (sides 2,3), xs len = nx-2
    const int edge_perp_lo[6] = {4, 4, 0, 0, 2, 2};
    const int edge_perp_hi[6] = {5, 5, 1, 1, 3, 3};
    const int edge_xs[6]      = {yedge_xs, yedge_xs, zedge_xs, zedge_xs, xedge_xs, xedge_xs};

    //* Per-direction batched buffer = n_fields × active_perp × n_ghost^2 × edge_xs.
    //* Layout: field-major outer (f) → perp-side block (LO before HI) → cross-section.
    std::vector<double> edge_send_bufs[6];
    std::vector<double> edge_recv_bufs[6];
    for (int d = 0; d < 6; ++d) {
        if (!cnt[d]) continue;
        const int active_perp = cnt[edge_perp_lo[d]] + cnt[edge_perp_hi[d]];
        if (active_perp == 0) continue;
        const int total = n_fields * active_perp * n_ghost_sq * edge_xs[d];
        edge_send_bufs[d].resize(total);
        edge_recv_bufs[d].resize(total);
    }

    //* Per-edge pack/unpack helpers. Each maps a (field, route-side, perp-side)
    //* triple to its n_ghost^2 cross-section block in a contiguous buffer.
    auto pack_yedge = [&](int f, int x_side, int z_side, double* buf) {
        int idx = 0;
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int sx = ctx.g_src(gx);
            const int ix = (x_side == 0) ? (n_ghost + face_send_offset + sx)
                                         : (nx - 1 - n_ghost - face_send_offset - sx);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int sz = ctx.g_src(gz);
                const int iz = (z_side == 0) ? (n_ghost + face_send_offset + sz)
                                             : (nz - 1 - n_ghost - face_send_offset - sz);
                for (int iy = 1; iy <= ny - 2; ++iy)
                    buf[idx++] = vectors[f][ix][iy][iz];
            }
        }
    };
    auto unpack_yedge = [&](int f, int x_side, int z_side, const double* buf) {
        int idx = 0;
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int ix = (x_side == 0) ? gx : (nx - 1 - gx);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int iz = (z_side == 0) ? gz : (nz - 1 - gz);
                for (int iy = 1; iy <= ny - 2; ++iy)
                    vectors[f][ix][iy][iz] = buf[idx++];
            }
        }
    };
    auto pack_zedge = [&](int f, int y_side, int x_side, double* buf) {
        int idx = 0;
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int sx = ctx.g_src(gx);
            const int ix = (x_side == 0) ? (n_ghost + face_send_offset + sx)
                                         : (nx - 1 - n_ghost - face_send_offset - sx);
            for (int gy = 0; gy < n_ghost; ++gy) {
                const int sy = ctx.g_src(gy);
                const int iy = (y_side == 0) ? (n_ghost + face_send_offset + sy)
                                             : (ny - 1 - n_ghost - face_send_offset - sy);
                for (int iz = 1; iz <= nz - 2; ++iz)
                    buf[idx++] = vectors[f][ix][iy][iz];
            }
        }
    };
    auto unpack_zedge = [&](int f, int y_side, int x_side, const double* buf) {
        int idx = 0;
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int ix = (x_side == 0) ? gx : (nx - 1 - gx);
            for (int gy = 0; gy < n_ghost; ++gy) {
                const int iy = (y_side == 0) ? gy : (ny - 1 - gy);
                for (int iz = 1; iz <= nz - 2; ++iz)
                    vectors[f][ix][iy][iz] = buf[idx++];
            }
        }
    };
    auto pack_xedge = [&](int f, int z_side, int y_side, double* buf) {
        int idx = 0;
        for (int gy = 0; gy < n_ghost; ++gy) {
            const int sy = ctx.g_src(gy);
            const int iy = (y_side == 0) ? (n_ghost + face_send_offset + sy)
                                         : (ny - 1 - n_ghost - face_send_offset - sy);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int sz = ctx.g_src(gz);
                const int iz = (z_side == 0) ? (n_ghost + face_send_offset + sz)
                                             : (nz - 1 - n_ghost - face_send_offset - sz);
                for (int ix = 1; ix <= nx - 2; ++ix)
                    buf[idx++] = vectors[f][ix][iy][iz];
            }
        }
    };
    auto unpack_xedge = [&](int f, int z_side, int y_side, const double* buf) {
        int idx = 0;
        for (int gy = 0; gy < n_ghost; ++gy) {
            const int iy = (y_side == 0) ? gy : (ny - 1 - gy);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int iz = (z_side == 0) ? gz : (nz - 1 - gz);
                for (int ix = 1; ix <= nx - 2; ++ix)
                    vectors[f][ix][iy][iz] = buf[idx++];
            }
        }
    };

    const int yedge_block = n_ghost_sq * yedge_xs;
    const int zedge_block = n_ghost_sq * zedge_xs;
    const int xedge_block = n_ghost_sq * xedge_xs;

    //* Pack send buffers. Field-major outer (f), perp LO before perp HI.
    for (int f = 0; f < n_fields; ++f) {
        if (cnt[0]) {
            int off = f * (cnt[4] + cnt[5]) * yedge_block;
            if (cnt[4]) { pack_yedge(f, 0, 0, edge_send_bufs[0].data() + off); off += yedge_block; }
            if (cnt[5]) { pack_yedge(f, 0, 1, edge_send_bufs[0].data() + off); off += yedge_block; }
        }
        if (cnt[1]) {
            int off = f * (cnt[4] + cnt[5]) * yedge_block;
            if (cnt[4]) { pack_yedge(f, 1, 0, edge_send_bufs[1].data() + off); off += yedge_block; }
            if (cnt[5]) { pack_yedge(f, 1, 1, edge_send_bufs[1].data() + off); off += yedge_block; }
        }
        if (cnt[2]) {
            int off = f * (cnt[0] + cnt[1]) * zedge_block;
            if (cnt[0]) { pack_zedge(f, 0, 0, edge_send_bufs[2].data() + off); off += zedge_block; }
            if (cnt[1]) { pack_zedge(f, 0, 1, edge_send_bufs[2].data() + off); off += zedge_block; }
        }
        if (cnt[3]) {
            int off = f * (cnt[0] + cnt[1]) * zedge_block;
            if (cnt[0]) { pack_zedge(f, 1, 0, edge_send_bufs[3].data() + off); off += zedge_block; }
            if (cnt[1]) { pack_zedge(f, 1, 1, edge_send_bufs[3].data() + off); off += zedge_block; }
        }
        if (cnt[4]) {
            int off = f * (cnt[2] + cnt[3]) * xedge_block;
            if (cnt[2]) { pack_xedge(f, 0, 0, edge_send_bufs[4].data() + off); off += xedge_block; }
            if (cnt[3]) { pack_xedge(f, 0, 1, edge_send_bufs[4].data() + off); off += xedge_block; }
        }
        if (cnt[5]) {
            int off = f * (cnt[2] + cnt[3]) * xedge_block;
            if (cnt[2]) { pack_xedge(f, 1, 0, edge_send_bufs[5].data() + off); off += xedge_block; }
            if (cnt[3]) { pack_xedge(f, 1, 1, edge_send_bufs[5].data() + off); off += xedge_block; }
        }
    }

    //* Post Irecv (into recv_bufs) + Isend (from send_bufs).
    MPI_Request reqs[12];
    int rcv = 0;
    for (int d = 0; d < 6; ++d) {
        if (edge_recv_bufs[d].empty()) continue;
        MPI_Irecv(edge_recv_bufs[d].data(), (int)edge_recv_bufs[d].size(),
                  MPI_DOUBLE, neighbours_arr[d], recv_tags[d], comm, &reqs[rcv++]);
    }
    int snd = rcv;
    for (int d = 0; d < 6; ++d) {
        if (edge_send_bufs[d].empty()) continue;
        MPI_Isend(edge_send_bufs[d].data(), (int)edge_send_bufs[d].size(),
                  MPI_DOUBLE, neighbours_arr[d], send_tags[d], comm, &reqs[snd++]);
    }

    //* Unpack helper.
    auto unpack_edge_recvs = [&]() {
        for (int f = 0; f < n_fields; ++f) {
            if (cnt[0]) {
                int off = f * (cnt[4] + cnt[5]) * yedge_block;
                if (cnt[4]) { unpack_yedge(f, 0, 0, edge_recv_bufs[0].data() + off); off += yedge_block; }
                if (cnt[5]) { unpack_yedge(f, 0, 1, edge_recv_bufs[0].data() + off); off += yedge_block; }
            }
            if (cnt[1]) {
                int off = f * (cnt[4] + cnt[5]) * yedge_block;
                if (cnt[4]) { unpack_yedge(f, 1, 0, edge_recv_bufs[1].data() + off); off += yedge_block; }
                if (cnt[5]) { unpack_yedge(f, 1, 1, edge_recv_bufs[1].data() + off); off += yedge_block; }
            }
            if (cnt[2]) {
                int off = f * (cnt[0] + cnt[1]) * zedge_block;
                if (cnt[0]) { unpack_zedge(f, 0, 0, edge_recv_bufs[2].data() + off); off += zedge_block; }
                if (cnt[1]) { unpack_zedge(f, 0, 1, edge_recv_bufs[2].data() + off); off += zedge_block; }
            }
            if (cnt[3]) {
                int off = f * (cnt[0] + cnt[1]) * zedge_block;
                if (cnt[0]) { unpack_zedge(f, 1, 0, edge_recv_bufs[3].data() + off); off += zedge_block; }
                if (cnt[1]) { unpack_zedge(f, 1, 1, edge_recv_bufs[3].data() + off); off += zedge_block; }
            }
            if (cnt[4]) {
                int off = f * (cnt[2] + cnt[3]) * xedge_block;
                if (cnt[2]) { unpack_xedge(f, 0, 0, edge_recv_bufs[4].data() + off); off += xedge_block; }
                if (cnt[3]) { unpack_xedge(f, 0, 1, edge_recv_bufs[4].data() + off); off += xedge_block; }
            }
            if (cnt[5]) {
                int off = f * (cnt[2] + cnt[3]) * xedge_block;
                if (cnt[2]) { unpack_xedge(f, 1, 0, edge_recv_bufs[5].data() + off); off += xedge_block; }
                if (cnt[3]) { unpack_xedge(f, 1, 1, edge_recv_bufs[5].data() + off); off += xedge_block; }
            }
        }
    };

    //* Periodic single-rank edge self-copies. For each self-periodic axis,
    //* copy the axis-ghost cells at perpendicular outermost-ghost positions.
    //* Reads only from face-exchange-filled cells. The widened face types
    //* cover inner ghost rows, so only outermost (index 0 / n-1) edges need
    //* this pass. At n_ghost == 1 the g loop runs once.
    for (int f = 0; f < n_fields; ++f) {
        if (rnX == myrank && lnX == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nx - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                if (rnZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vectors[f][g][iy][nz-1]         = vectors[f][sL][iy][nz-1];
                        vectors[f][nx-1-g][iy][nz-1]    = vectors[f][sR][iy][nz-1];
                    }
                }
                if (lnZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vectors[f][g][iy][0]         = vectors[f][sL][iy][0];
                        vectors[f][nx-1-g][iy][0]    = vectors[f][sR][iy][0];
                    }
                }
                if (rnY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vectors[f][g][ny-1][iz]         = vectors[f][sL][ny-1][iz];
                        vectors[f][nx-1-g][ny-1][iz]    = vectors[f][sR][ny-1][iz];
                    }
                }
                if (lnY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vectors[f][g][0][iz]         = vectors[f][sL][0][iz];
                        vectors[f][nx-1-g][0][iz]    = vectors[f][sR][0][iz];
                    }
                }
            }
        }
        if (rnY == myrank && lnY == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = ny - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                if (rnX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vectors[f][nx-1][g][iz]         = vectors[f][nx-1][sL][iz];
                        vectors[f][nx-1][ny-1-g][iz]    = vectors[f][nx-1][sR][iz];
                    }
                }
                if (lnX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vectors[f][0][g][iz]         = vectors[f][0][sL][iz];
                        vectors[f][0][ny-1-g][iz]    = vectors[f][0][sR][iz];
                    }
                }
                if (rnZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vectors[f][ix][g][nz-1]         = vectors[f][ix][sL][nz-1];
                        vectors[f][ix][ny-1-g][nz-1]    = vectors[f][ix][sR][nz-1];
                    }
                }
                if (lnZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vectors[f][ix][g][0]         = vectors[f][ix][sL][0];
                        vectors[f][ix][ny-1-g][0]    = vectors[f][ix][sR][0];
                    }
                }
            }
        }
        if (rnZ == myrank && lnZ == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nz - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                if (rnY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vectors[f][ix][ny-1][g]         = vectors[f][ix][ny-1][sL];
                        vectors[f][ix][ny-1][nz-1-g]    = vectors[f][ix][ny-1][sR];
                    }
                }
                if (lnY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vectors[f][ix][0][g]         = vectors[f][ix][0][sL];
                        vectors[f][ix][0][nz-1-g]    = vectors[f][ix][0][sR];
                    }
                }
                if (rnX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vectors[f][nx-1][iy][g]         = vectors[f][nx-1][iy][sL];
                        vectors[f][nx-1][iy][nz-1-g]    = vectors[f][nx-1][iy][sR];
                    }
                }
                if (lnX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vectors[f][0][iy][g]         = vectors[f][0][iy][sL];
                        vectors[f][0][iy][nz-1-g]    = vectors[f][0][iy][sR];
                    }
                }
            }
        }
    }

    if (snd > 0) MPI_Waitall(snd, reqs, MPI_STATUSES_IGNORE);

    //* Unpack edge recv buffers AFTER edge self-copies — where both target the
    //* same ghost-corner, unpack runs second and wins.
    unpack_edge_recvs();
}

//* Periodic-self fold: sums TSC ghost-cell deposits at periodic-self axes into
//* the matching strict-interior cells (n_ghost deep) so following cross-rank
//* packs carry post-fold strict. No-op outside periodic-self; gated on needInterp.
static void do_periodic_self_fold(int nx, int ny, int nz,
                                   int n_fields, double ****vectors,
                                   const HaloContext &ctx)
{
    const int n_ghost      = ctx.n_ghost;
    //* image_offset only — fold dst is a periodic-image stride.
    const int image_offset = ctx.image_offset;
    const bool ps_x        = ctx.ps_x_self;
    const bool ps_y        = ctx.ps_y_self;
    const bool ps_z        = ctx.ps_z_self;

    if (!(ps_x || ps_y || ps_z)) return;

    for (int f = 0; f < n_fields; ++f) {
        if (ps_x) {
            for (int g = 0; g < n_ghost; g++) {
                const int dst_lo = g + nx - 2 * n_ghost - image_offset;
                const int dst_hi = (2 * n_ghost - 1 + image_offset) - g;
                for (int iy = 1; iy <= ny - 2; iy++)
                    for (int iz = 1; iz <= nz - 2; iz++) {
                        vectors[f][dst_lo][iy][iz] += vectors[f][g][iy][iz];
                        vectors[f][dst_hi][iy][iz] += vectors[f][nx - 1 - g][iy][iz];
                    }
            }
        }
        if (ps_y) {
            for (int g = 0; g < n_ghost; g++) {
                const int dst_lo = g + ny - 2 * n_ghost - image_offset;
                const int dst_hi = (2 * n_ghost - 1 + image_offset) - g;
                for (int ix = 1; ix <= nx - 2; ix++)
                    for (int iz = 1; iz <= nz - 2; iz++) {
                        vectors[f][ix][dst_lo][iz] += vectors[f][ix][g][iz];
                        vectors[f][ix][dst_hi][iz] += vectors[f][ix][ny - 1 - g][iz];
                    }
            }
        }
        if (ps_z) {
            for (int g = 0; g < n_ghost; g++) {
                const int dst_lo = g + nz - 2 * n_ghost - image_offset;
                const int dst_hi = (2 * n_ghost - 1 + image_offset) - g;
                for (int ix = 1; ix <= nx - 2; ix++)
                    for (int iy = 1; iy <= ny - 2; iy++) {
                        vectors[f][ix][iy][dst_lo] += vectors[f][ix][iy][g];
                        vectors[f][ix][iy][dst_hi] += vectors[f][ix][iy][nz - 1 - g];
                    }
            }
        }
    }
}

//* Unify the LO=HI periodic-self strict duplicates (same physical node at
//* i=n_ghost and i=nx-n_ghost-1) that an asymmetric gather can split: SUM at
//* n_ghost>1 (TSC, each carries half), AVG at n_ghost=1 (CIC, no-op when already
//* equal). Runs before the cross-rank pack so neighbour ghosts inherit it.
static void do_unify_ps_dups(int nx, int ny, int nz,
                              int n_fields, double ****vectors,
                              const HaloContext &ctx)
{
    const int  n_ghost   = ctx.n_ghost;
    const bool ps_x      = ctx.ps_x_self;
    const bool ps_y      = ctx.ps_y_self;
    const bool ps_z      = ctx.ps_z_self;
    const bool unify_sum = (n_ghost > 1);

    if (!(ps_x || ps_y || ps_z)) return;

    for (int f = 0; f < n_fields; ++f) {
        if (ps_x) {
            const int ilo = n_ghost;
            const int ihi = nx - n_ghost - 1;
            if (ihi > ilo)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k) {
                        const double s = vectors[f][ilo][j][k] + vectors[f][ihi][j][k];
                        const double v = unify_sum ? s : 0.5 * s;
                        vectors[f][ilo][j][k] = v;
                        vectors[f][ihi][j][k] = v;
                    }
        }
        if (ps_y) {
            const int jlo = n_ghost;
            const int jhi = ny - n_ghost - 1;
            if (jhi > jlo)
                for (int i = 0; i < nx; ++i)
                    for (int k = 0; k < nz; ++k) {
                        const double s = vectors[f][i][jlo][k] + vectors[f][i][jhi][k];
                        const double v = unify_sum ? s : 0.5 * s;
                        vectors[f][i][jlo][k] = v;
                        vectors[f][i][jhi][k] = v;
                    }
        }
        if (ps_z) {
            const int klo = n_ghost;
            const int khi = nz - n_ghost - 1;
            if (khi > klo)
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) {
                        const double s = vectors[f][i][j][klo] + vectors[f][i][j][khi];
                        const double v = unify_sum ? s : 0.5 * s;
                        vectors[f][i][j][klo] = v;
                        vectors[f][i][j][khi] = v;
                    }
        }
    }
}

//* Face exchange (manual pack/unpack, n_ghost >= 1): each of 6 directions sends
//* n_ghost slab layers batched into one message. Irecv targets a separate recv
//* buffer, so self-copies / sum-on-receive read vector untouched until unpack.
//* Ordering: needInterp=true runs self-copies BEFORE Waitall+unpack (reads land
//* on pre-Irecv local-gather values); needInterp=false unpacks first so the
//* self-copy consumes the received halo.
static void do_face_phase(int nx, int ny, int nz,
                           int n_fields, double ****vectors,
                           const HaloContext &ctx, bool needInterp)
{
    const int        n_ghost          = ctx.n_ghost;
    const int        face_send_offset = ctx.face_send_offset;
    const int        image_offset     = ctx.image_offset;
    const MPI_Comm   comm             = ctx.comm;
    const int        myrank           = ctx.myrank;
    const int* const cnt              = ctx.communicationCnt;
    const int        lnX = ctx.neighbours[0], rnX = ctx.neighbours[1];
    const int        lnY = ctx.neighbours[2], rnY = ctx.neighbours[3];
    const int        lnZ = ctx.neighbours[4], rnZ = ctx.neighbours[5];
    const int tag_XL=1, tag_YL=2, tag_ZL=3, tag_XR=4, tag_YR=5, tag_ZR=6;

    const int yz_per_layer = (ny - 2) * (nz - 2);
    const int xz_per_layer = (nx - 2) * (nz - 2);
    const int xy_per_layer = (nx - 2) * (ny - 2);
    const int per_layer[6] = {
        yz_per_layer, yz_per_layer,
        xz_per_layer, xz_per_layer,
        xy_per_layer, xy_per_layer
    };
    const int neighbours_arr[6] = { lnX, rnX, lnY, rnY, lnZ, rnZ };
    const int recv_tags[6] = { tag_XR, tag_XL, tag_YR, tag_YL, tag_ZR, tag_ZL };
    const int send_tags[6] = { tag_XL, tag_XR, tag_YL, tag_YR, tag_ZL, tag_ZR };

    //* Per-direction batched buffer = n_fields × n_ghost × per_layer doubles.
    //* Layout: field-major outer (f) → layer (g) → cell (j,k or i,k or i,j).
    std::vector<double> face_send_bufs[6];
    std::vector<double> face_recv_bufs[6];
    for (int d = 0; d < 6; ++d) {
        if (!cnt[d]) continue;
        face_send_bufs[d].resize(n_fields * n_ghost * per_layer[d]);
        face_recv_bufs[d].resize(n_fields * n_ghost * per_layer[d]);
    }

    //* Pack a yz / xz / xy face slab (fixed coord on the named axis,
    //* strict-interior on the other two) into a contiguous buffer for one
    //* field.
    auto pack_yz = [&](int f, int ix, double* buf) {
        int idx = 0;
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 1; k <= nz - 2; ++k)
                buf[idx++] = vectors[f][ix][j][k];
    };
    auto unpack_yz = [&](int f, int ix, const double* buf) {
        int idx = 0;
        for (int j = 1; j <= ny - 2; ++j)
            for (int k = 1; k <= nz - 2; ++k)
                vectors[f][ix][j][k] = buf[idx++];
    };
    auto pack_xz = [&](int f, int iy, double* buf) {
        int idx = 0;
        for (int i = 1; i <= nx - 2; ++i)
            for (int k = 1; k <= nz - 2; ++k)
                buf[idx++] = vectors[f][i][iy][k];
    };
    auto unpack_xz = [&](int f, int iy, const double* buf) {
        int idx = 0;
        for (int i = 1; i <= nx - 2; ++i)
            for (int k = 1; k <= nz - 2; ++k)
                vectors[f][i][iy][k] = buf[idx++];
    };
    auto pack_xy = [&](int f, int iz, double* buf) {
        int idx = 0;
        for (int i = 1; i <= nx - 2; ++i)
            for (int j = 1; j <= ny - 2; ++j)
                buf[idx++] = vectors[f][i][j][iz];
    };
    auto unpack_xy = [&](int f, int iz, const double* buf) {
        int idx = 0;
        for (int i = 1; i <= nx - 2; ++i)
            for (int j = 1; j <= ny - 2; ++j)
                vectors[f][i][j][iz] = buf[idx++];
    };

    //* Pack send buffers from strict-interior cells (post-fold and, when
    //* requested, post-unify). g_src(g) = (n_ghost-1-g) reverses ghost depth.
    for (int f = 0; f < n_fields; ++f) {
        for (int g = 0; g < n_ghost; ++g) {
            const int s = ctx.g_src(g);
            if (cnt[0]) pack_yz(f, n_ghost + face_send_offset + s,            face_send_bufs[0].data() + (f * n_ghost + g) * yz_per_layer);
            if (cnt[1]) pack_yz(f, nx - 1 - n_ghost - face_send_offset - s,   face_send_bufs[1].data() + (f * n_ghost + g) * yz_per_layer);
            if (cnt[2]) pack_xz(f, n_ghost + face_send_offset + s,            face_send_bufs[2].data() + (f * n_ghost + g) * xz_per_layer);
            if (cnt[3]) pack_xz(f, ny - 1 - n_ghost - face_send_offset - s,   face_send_bufs[3].data() + (f * n_ghost + g) * xz_per_layer);
            if (cnt[4]) pack_xy(f, n_ghost + face_send_offset + s,            face_send_bufs[4].data() + (f * n_ghost + g) * xy_per_layer);
            if (cnt[5]) pack_xy(f, nz - 1 - n_ghost - face_send_offset - s,   face_send_bufs[5].data() + (f * n_ghost + g) * xy_per_layer);
        }
    }

    //* Post Irecv (into recv_bufs) + Isend (from send_bufs).
    MPI_Request reqs[12];
    int rcv = 0;
    for (int d = 0; d < 6; ++d) {
        if (!cnt[d]) continue;
        const int n = n_fields * n_ghost * per_layer[d];
        MPI_Irecv(face_recv_bufs[d].data(), n, MPI_DOUBLE, neighbours_arr[d],
                  recv_tags[d], comm, &reqs[rcv++]);
    }
    int snd = rcv;
    for (int d = 0; d < 6; ++d) {
        if (!cnt[d]) continue;
        const int n = n_fields * n_ghost * per_layer[d];
        MPI_Isend(face_send_bufs[d].data(), n, MPI_DOUBLE, neighbours_arr[d],
                  send_tags[d], comm, &reqs[snd++]);
    }

    //* Unpack recv buffers into LO/HI ghost slabs.
    auto unpack_face_recvs = [&]() {
        for (int f = 0; f < n_fields; ++f) {
            for (int g = 0; g < n_ghost; ++g) {
                if (cnt[0]) unpack_yz(f, g,            face_recv_bufs[0].data() + (f * n_ghost + g) * yz_per_layer);
                if (cnt[1]) unpack_yz(f, nx - 1 - g,   face_recv_bufs[1].data() + (f * n_ghost + g) * yz_per_layer);
                if (cnt[2]) unpack_xz(f, g,            face_recv_bufs[2].data() + (f * n_ghost + g) * xz_per_layer);
                if (cnt[3]) unpack_xz(f, ny - 1 - g,   face_recv_bufs[3].data() + (f * n_ghost + g) * xz_per_layer);
                if (cnt[4]) unpack_xy(f, g,            face_recv_bufs[4].data() + (f * n_ghost + g) * xy_per_layer);
                if (cnt[5]) unpack_xy(f, nz - 1 - g,   face_recv_bufs[5].data() + (f * n_ghost + g) * xy_per_layer);
            }
        }
    };

    //* Periodic single-rank self-copies (X/Y/Z). Source indices match the
    //  (n_ghost + offset + g) depth convention used by the MPI sends, with
    //  modular wrapping for thin dimensions.
    auto periodic_self_copies = [&]() {
        for (int f = 0; f < n_fields; ++f) {
            if (rnX == myrank && lnX == myrank) {
                const int stride = nx - 2 * n_ghost - image_offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = nx - 1 - n_ghost - image_offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + image_offset + ctx.g_src(g);
                    if (sR > nx - 1 - n_ghost) sR -= stride;
                    for (int iy = 1; iy < ny - 1; iy++)
                        for (int iz = 1; iz < nz - 1; iz++) {
                            vectors[f][g][iy][iz]            = vectors[f][sL][iy][iz];
                            vectors[f][nx - 1 - g][iy][iz]   = vectors[f][sR][iy][iz];
                        }
                }
            }
            if (rnY == myrank && lnY == myrank) {
                const int stride = ny - 2 * n_ghost - image_offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = ny - 1 - n_ghost - image_offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + image_offset + ctx.g_src(g);
                    if (sR > ny - 1 - n_ghost) sR -= stride;
                    for (int ix = 1; ix < nx - 1; ix++)
                        for (int iz = 1; iz < nz - 1; iz++) {
                            vectors[f][ix][g][iz]            = vectors[f][ix][sL][iz];
                            vectors[f][ix][ny - 1 - g][iz]   = vectors[f][ix][sR][iz];
                        }
                }
            }
            if (rnZ == myrank && lnZ == myrank) {
                const int stride = nz - 2 * n_ghost - image_offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = nz - 1 - n_ghost - image_offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + image_offset + ctx.g_src(g);
                    if (sR > nz - 1 - n_ghost) sR -= stride;
                    for (int ix = 1; ix < nx - 1; ix++)
                        for (int iy = 1; iy < ny - 1; iy++) {
                            vectors[f][ix][iy][g]            = vectors[f][ix][iy][sL];
                            vectors[f][ix][iy][nz - 1 - g]   = vectors[f][ix][iy][sR];
                        }
                }
            }
        }
    };

    //* Waitall (unconditional — manual pack/unpack closes the buffer-reuse race).
    if (snd > 0) MPI_Waitall(snd, reqs, MPI_STATUSES_IGNORE);

    //* Unpack neighbour values into ghost slabs.
    unpack_face_recvs();

    //* Periodic-self self-copies run AFTER cross-rank unpack (separate recv
    //* buffers decouple vectors[sL] from the Irecv) so periodic-self ghosts
    //* inherit post-exchange strict. The (cross-rank ghost) ∩ (periodic-self
    //* ghost) columns that face unpack cannot reach (it writes only k ∈ [1, nz-2])
    //* need this for mass-matrix |offset|=2 correctness when unify_ps_dups=true
    //* and no trailing Node_P_multi runs.
    periodic_self_copies();
}

//* Corner exchange (manual pack/unpack, n_ghost >= 1). Each X-LO/HI Cart side
//* carries all 4 of its perpendicular Y/Z corners (each an n_ghost^3 cube)
//* batched into one message. Only fires when both Y and Z have a cross-rank
//* face neighbour; at np=1 along Y or Z the cube is filled by the periodic-self
//* self-copies below.
static void do_corner_phase(int nx, int ny, int nz,
                             int n_fields, double ****vectors,
                             const HaloContext &ctx)
{
    const int       n_ghost          = ctx.n_ghost;
    //* face_send_offset only — corner pack source is a cross-rank send.
    const int       face_send_offset = ctx.face_send_offset;
    const MPI_Comm  comm             = ctx.comm;
    const int* const cnt             = ctx.communicationCnt;
    const int       lnX              = ctx.neighbours[0];
    const int       rnX              = ctx.neighbours[1];
    const int       tag_XL           = 1, tag_XR = 4;

    if (!((cnt[2] == 1 || cnt[3] == 1) && (cnt[4] == 1 || cnt[5] == 1)))
        return;

    const int corner_per_field = 4 * n_ghost * n_ghost * n_ghost;
    const int corner_total     = n_fields * corner_per_field;

    std::vector<double> corner_send_bufs[2];
    std::vector<double> corner_recv_bufs[2];
    for (int side = 0; side < 2; ++side) {
        if (!cnt[side]) continue;
        corner_send_bufs[side].resize(corner_total);
        corner_recv_bufs[side].resize(corner_total);
    }

    //* Pack send buffers. Source cells = strict-interior cube at depth
    //* (n_ghost+face_send_offset+g_src) from the active X face. For each (gx,gy,gz)
    //* the 4 perp Y/Z corners pack in canonical order (Y-LO,Z-LO),
    //* (Y-LO,Z-HI), (Y-HI,Z-LO), (Y-HI,Z-HI). Both ranks pack the same
    //* canonical order so the receiver's unpack lands matching doubles.
    for (int f = 0; f < n_fields; ++f) {
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int sx   = ctx.g_src(gx);
            const int xs_l = n_ghost + face_send_offset + sx;
            const int xs_r = nx - 1 - n_ghost - face_send_offset - sx;
            for (int gy = 0; gy < n_ghost; ++gy) {
                const int sy   = ctx.g_src(gy);
                const int ys_l = n_ghost + face_send_offset + sy;
                const int ys_r = ny - 1 - n_ghost - face_send_offset - sy;
                for (int gz = 0; gz < n_ghost; ++gz) {
                    const int sz   = ctx.g_src(gz);
                    const int zs_l = n_ghost + face_send_offset + sz;
                    const int zs_r = nz - 1 - n_ghost - face_send_offset - sz;
                    const int base = f * corner_per_field
                                   + 4 * ((gx * n_ghost + gy) * n_ghost + gz);
                    if (cnt[0]) {
                        corner_send_bufs[0][base + 0] = vectors[f][xs_l][ys_l][zs_l];
                        corner_send_bufs[0][base + 1] = vectors[f][xs_l][ys_l][zs_r];
                        corner_send_bufs[0][base + 2] = vectors[f][xs_l][ys_r][zs_l];
                        corner_send_bufs[0][base + 3] = vectors[f][xs_l][ys_r][zs_r];
                    }
                    if (cnt[1]) {
                        corner_send_bufs[1][base + 0] = vectors[f][xs_r][ys_l][zs_l];
                        corner_send_bufs[1][base + 1] = vectors[f][xs_r][ys_l][zs_r];
                        corner_send_bufs[1][base + 2] = vectors[f][xs_r][ys_r][zs_l];
                        corner_send_bufs[1][base + 3] = vectors[f][xs_r][ys_r][zs_r];
                    }
                }
            }
        }
    }

    MPI_Request reqs[4];
    int rcv = 0;
    if (cnt[0]) MPI_Irecv(corner_recv_bufs[0].data(), corner_total, MPI_DOUBLE, lnX, tag_XR, comm, &reqs[rcv++]);
    if (cnt[1]) MPI_Irecv(corner_recv_bufs[1].data(), corner_total, MPI_DOUBLE, rnX, tag_XL, comm, &reqs[rcv++]);
    int snd = rcv;
    if (cnt[0]) MPI_Isend(corner_send_bufs[0].data(), corner_total, MPI_DOUBLE, lnX, tag_XL, comm, &reqs[snd++]);
    if (cnt[1]) MPI_Isend(corner_send_bufs[1].data(), corner_total, MPI_DOUBLE, rnX, tag_XR, comm, &reqs[snd++]);
    if (snd > 0) MPI_Waitall(snd, reqs, MPI_STATUSES_IGNORE);

    //* Unpack into ghost cells in same canonical (Y-LO/HI × Z-LO/HI) order.
    if (corner_recv_bufs[0].empty() && corner_recv_bufs[1].empty()) return;
    for (int f = 0; f < n_fields; ++f) {
        for (int gx = 0; gx < n_ghost; ++gx) {
            for (int gy = 0; gy < n_ghost; ++gy) {
                for (int gz = 0; gz < n_ghost; ++gz) {
                    const int base = f * corner_per_field
                                   + 4 * ((gx * n_ghost + gy) * n_ghost + gz);
                    if (!corner_recv_bufs[0].empty()) {
                        vectors[f][gx][gy][gz]                = corner_recv_bufs[0][base + 0];
                        vectors[f][gx][gy][nz-1-gz]           = corner_recv_bufs[0][base + 1];
                        vectors[f][gx][ny-1-gy][gz]           = corner_recv_bufs[0][base + 2];
                        vectors[f][gx][ny-1-gy][nz-1-gz]      = corner_recv_bufs[0][base + 3];
                    }
                    if (!corner_recv_bufs[1].empty()) {
                        vectors[f][nx-1-gx][gy][gz]           = corner_recv_bufs[1][base + 0];
                        vectors[f][nx-1-gx][gy][nz-1-gz]      = corner_recv_bufs[1][base + 1];
                        vectors[f][nx-1-gx][ny-1-gy][gz]      = corner_recv_bufs[1][base + 2];
                        vectors[f][nx-1-gx][ny-1-gy][nz-1-gz] = corner_recv_bufs[1][base + 3];
                    }
                }
            }
        }
    }
}

//* Periodic single-rank corner self-copies. The else-if cascade lets the first
//* branch (X-self) fill all 8 corner cubes at np=1 via face → edge → corner;
//* source cells were filled by the edge self-copies in do_edge_phase.
static void do_corner_periodic_self(int nx, int ny, int nz,
                                     int n_fields, double ****vectors,
                                     const HaloContext &ctx)
{
    const int n_ghost      = ctx.n_ghost;
    //* image_offset only — corner self-copy sources are periodic-image strides.
    const int image_offset = ctx.image_offset;
    const int myrank       = ctx.myrank;
    const int lnX          = ctx.neighbours[0], rnX = ctx.neighbours[1];
    const int lnY          = ctx.neighbours[2], rnY = ctx.neighbours[3];
    const int lnZ          = ctx.neighbours[4], rnZ = ctx.neighbours[5];

    for (int f = 0; f < n_fields; ++f) {
        if (lnX == myrank && rnX == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nx - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                for (int gy = 0; gy < n_ghost; gy++)
                for (int gz = 0; gz < n_ghost; gz++) {
                    if (lnY != MPI_PROC_NULL && lnZ != MPI_PROC_NULL) {
                        vectors[f][g][gy][gz]                   = vectors[f][sL][gy][gz];
                        vectors[f][nx-1-g][gy][gz]              = vectors[f][sR][gy][gz];
                    }
                    if (lnY != MPI_PROC_NULL && rnZ != MPI_PROC_NULL) {
                        vectors[f][g][gy][nz-1-gz]              = vectors[f][sL][gy][nz-1-gz];
                        vectors[f][nx-1-g][gy][nz-1-gz]         = vectors[f][sR][gy][nz-1-gz];
                    }
                    if (rnY != MPI_PROC_NULL && lnZ != MPI_PROC_NULL) {
                        vectors[f][g][ny-1-gy][gz]              = vectors[f][sL][ny-1-gy][gz];
                        vectors[f][nx-1-g][ny-1-gy][gz]         = vectors[f][sR][ny-1-gy][gz];
                    }
                    if (rnY != MPI_PROC_NULL && rnZ != MPI_PROC_NULL) {
                        vectors[f][g][ny-1-gy][nz-1-gz]         = vectors[f][sL][ny-1-gy][nz-1-gz];
                        vectors[f][nx-1-g][ny-1-gy][nz-1-gz]    = vectors[f][sR][ny-1-gy][nz-1-gz];
                    }
                }
            }
        }
        else if (lnY == myrank && rnY == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = ny - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                for (int gx = 0; gx < n_ghost; gx++)
                for (int gz = 0; gz < n_ghost; gz++) {
                    if (lnX != MPI_PROC_NULL && lnZ != MPI_PROC_NULL) {
                        vectors[f][gx][g][gz]                   = vectors[f][gx][sL][gz];
                        vectors[f][gx][ny-1-g][gz]              = vectors[f][gx][sR][gz];
                    }
                    if (lnX != MPI_PROC_NULL && rnZ != MPI_PROC_NULL) {
                        vectors[f][gx][g][nz-1-gz]              = vectors[f][gx][sL][nz-1-gz];
                        vectors[f][gx][ny-1-g][nz-1-gz]         = vectors[f][gx][sR][nz-1-gz];
                    }
                    if (rnX != MPI_PROC_NULL && lnZ != MPI_PROC_NULL) {
                        vectors[f][nx-1-gx][g][gz]              = vectors[f][nx-1-gx][sL][gz];
                        vectors[f][nx-1-gx][ny-1-g][gz]         = vectors[f][nx-1-gx][sR][gz];
                    }
                    if (rnX != MPI_PROC_NULL && rnZ != MPI_PROC_NULL) {
                        vectors[f][nx-1-gx][g][nz-1-gz]         = vectors[f][nx-1-gx][sL][nz-1-gz];
                        vectors[f][nx-1-gx][ny-1-g][nz-1-gz]    = vectors[f][nx-1-gx][sR][nz-1-gz];
                    }
                }
            }
        }
        else if (lnZ == myrank && rnZ == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nz - 1 - n_ghost - image_offset - ctx.g_src(g);
                const int sR = n_ghost + image_offset + ctx.g_src(g);
                for (int gx = 0; gx < n_ghost; gx++)
                for (int gy = 0; gy < n_ghost; gy++) {
                    if (lnX != MPI_PROC_NULL && lnY != MPI_PROC_NULL) {
                        vectors[f][gx][gy][g]                   = vectors[f][gx][gy][sL];
                        vectors[f][gx][gy][nz-1-g]              = vectors[f][gx][gy][sR];
                    }
                    if (lnX != MPI_PROC_NULL && rnY != MPI_PROC_NULL) {
                        vectors[f][gx][ny-1-gy][g]              = vectors[f][gx][ny-1-gy][sL];
                        vectors[f][gx][ny-1-gy][nz-1-g]         = vectors[f][gx][ny-1-gy][sR];
                    }
                    if (rnX != MPI_PROC_NULL && lnY != MPI_PROC_NULL) {
                        vectors[f][nx-1-gx][gy][g]              = vectors[f][nx-1-gx][gy][sL];
                        vectors[f][nx-1-gx][gy][nz-1-g]         = vectors[f][nx-1-gx][gy][sR];
                    }
                    if (rnX != MPI_PROC_NULL && rnY != MPI_PROC_NULL) {
                        vectors[f][nx-1-gx][ny-1-gy][g]         = vectors[f][nx-1-gx][ny-1-gy][sL];
                        vectors[f][nx-1-gx][ny-1-gy][nz-1-g]    = vectors[f][nx-1-gx][ny-1-gy][sR];
                    }
                }
            }
        }
    }
}

static void NBDerivedHaloCommN(int nx, int ny, int nz,
                                int n_fields, double ****vectors,
                                const VirtualTopology3D * vct, EMfields3D *EMf,
                                HaloKind kind,
                                bool unify_ps_dups)
{
    const auto [isCenterFlag, isFaceOnlyFlag, needInterp, isParticle] = to_flags(kind);
    //* Single-source-of-truth halo state shared across all phase helpers.
    const HaloContext ctx = make_halo_context(EMf, vct, isCenterFlag, needInterp, isParticle);
    #ifdef DEBUG
        MPI_Errhandler_set(ctx.comm, MPI_ERRORS_RETURN);
    #endif

    //* Stage order (each no-ops when its precondition fails):
    //*  1. Cross-rank moment completion (TSC, any_xrank_periodic) — must precede
    //*     FACE, whose copy semantics would overwrite the ghosts it sends.
    //*  2. Periodic-self fold — skipped at cic_interp, where the trailing
    //*     addFace(skip_self=false) does the dup-pair unify (folding here double-counts).
    //*  3. Periodic-self LO=HI dup unify, when the caller sets unify_ps_dups.
    //*  4. FACE, then 5. EDGE/CORNER/diagonal/corner-self (skipped if isFaceOnlyFlag).
    //*  6. Trailing addFace family at n_ghost==1 (CIC interp).
    //*  7. At cic_interp + unify_ps_dups, re-run FACE/EDGE/CORNER with a Node-style
    //*     ctx (offsets=1) to replace the dup ghosts left by step 6 with strict-near.
    //* Thin self-periodic axes double-count when n_cells_per_rank < 2*n_ghost;
    //* require nxc_r >= 2*n_ghost - 1.
    const bool cic_interp = (ctx.n_ghost == 1) && needInterp;
    do_xrank_completion(nx, ny, nz, n_fields, vectors, ctx, needInterp);
    if (needInterp && !cic_interp)
                       do_periodic_self_fold(nx, ny, nz, n_fields, vectors, ctx);
    if (unify_ps_dups) do_unify_ps_dups     (nx, ny, nz, n_fields, vectors, ctx);
    do_face_phase      (nx, ny, nz, n_fields, vectors, ctx, needInterp);
    if (!isFaceOnlyFlag) {
        do_edge_phase           (nx, ny, nz, n_fields, vectors, ctx);
        do_corner_phase         (nx, ny, nz, n_fields, vectors, ctx);
        do_diagonal_edge_copy   (nx, ny, nz, n_fields, vectors, ctx);
        do_corner_periodic_self (nx, ny, nz, n_fields, vectors, ctx);
    }

    //* CIC moments-interp trailing addFace family — sum each ghost layer
    //* into the matching interior layer. skip_self=false so periodic-self
    //* axes get their dup-pair unify here (see step-2 invariant above).
    if (cic_interp) {
        const bool skip_self = false;
        for (int f = 0; f < n_fields; ++f) {
            addFace  (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self);
            addEdgeZ (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self);
            addEdgeY (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self);
            addEdgeX (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self);
            addCorner(nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self);
        }
    }

    //* Node-style ghost refresh: at cic_interp + unify_ps_dups the
    //* trailing addFace leaves ghosts at dup values, but downstream
    //* Maxwell stencils expect strict-near. Clone ctx with offsets=1
    //* and re-run face/edge/corner to overwrite ghost with strict-near.
    if (cic_interp && unify_ps_dups) {
        HaloContext ctx_node = ctx;
        ctx_node.face_send_offset = 1;
        ctx_node.image_offset     = 1;
        do_face_phase           (nx, ny, nz, n_fields, vectors, ctx_node, /*needInterp=*/false);
        if (!isFaceOnlyFlag) {
            do_edge_phase           (nx, ny, nz, n_fields, vectors, ctx_node);
            do_corner_phase         (nx, ny, nz, n_fields, vectors, ctx_node);
            do_diagonal_edge_copy   (nx, ny, nz, n_fields, vectors, ctx_node);
            do_corner_periodic_self (nx, ny, nz, n_fields, vectors, ctx_node);
        }
    }
}


//* communicate*BC* wrappers all follow the same shape: NBDerivedHaloCommN
//* with one of four (isCenter, faceOnly) × (P / non-P) flag bits, then
//* a BCface (or BCface_P) finalisation. The double*** overload carries the
//* logic; the arr3_double overload just unpacks via fetch_arr3().
void communicateNodeBC(int nx, int ny, int nz, double*** vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::NodeBC);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateNodeBC(nx, ny, nz, _vector.fetch_arr3(),
                       bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct, EMf);
}

void communicateNodeBoxStencilBC(int nx, int ny, int nz, arr3_double _vector,
                                  int bcFaceXrght, int bcFaceXleft,
                                  int bcFaceYrght, int bcFaceYleft,
                                  int bcFaceZrght, int bcFaceZleft,
                                  const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::NodeBoxStencilBC);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::NodeBoxStencilBC_P);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_P(int nx, int ny, int nz, arr3_double _vector,
                          int bcFaceXrght, int bcFaceXleft,
                          int bcFaceYrght, int bcFaceYleft,
                          int bcFaceZrght, int bcFaceZleft,
                          const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::NodeBC_P);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC(int nx, int ny, int nz, double*** vector,
                          int bcFaceXrght, int bcFaceXleft,
                          int bcFaceYrght, int bcFaceYleft,
                          int bcFaceZrght, int bcFaceZleft,
                          const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::CenterBC);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC(int nx, int ny, int nz, arr3_double _vector,
                          int bcFaceXrght, int bcFaceXleft,
                          int bcFaceYrght, int bcFaceYleft,
                          int bcFaceZrght, int bcFaceZleft,
                          const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateCenterBC(nx, ny, nz, _vector.fetch_arr3(),
                         bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct, EMf);
}

void communicateCenterBC_P(int nx, int ny, int nz, arr3_double _vector,
                            int bcFaceXrght, int bcFaceXleft,
                            int bcFaceYrght, int bcFaceYleft,
                            int bcFaceZrght, int bcFaceZleft,
                            const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::CenterBC_P);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::CenterBoxStencilBC);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                      int bcFaceXrght, int bcFaceXleft,
                                      int bcFaceYrght, int bcFaceYleft,
                                      int bcFaceZrght, int bcFaceZleft,
                                      const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::CenterBoxStencilBC_P);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}


/** add the values of ghost cells faces to the 3D physical vector */
//  Ghost layer g (from the outermost) is summed into the interior layer at
//  depth g from the boundary node:
//      left:  vector[n_ghost + g]    += vector[g]
//      right: vector[nx-1-n_ghost-g] += vector[nx-1-g]
//  Each (src, dst) pair is independent (src in ghost slab, dst strictly interior).
//* Per-axis periodic-self skip: when skip_self_periodic is set and an axis's
//* left/right neighbours are both myrank, the upstream fold+copy already handled
//* it, so the add* helpers skip it to avoid double-counting.
struct AxisSkip { bool x, y, z; };

static inline AxisSkip axis_skip(const VirtualTopology3D * vct, bool skip_self_periodic)
{
    if (!skip_self_periodic) return { false, false, false };
    const int r = vct->getCartesian_rank();
    return {
        vct->getXleft_neighbor_P() == r && vct->getXright_neighbor_P() == r,
        vct->getYleft_neighbor_P() == r && vct->getYright_neighbor_P() == r,
        vct->getZleft_neighbor_P() == r && vct->getZright_neighbor_P() == r
    };
}

//* Sum-on-receive helper: dst += src. Not hot (boundary slabs only).
static inline void halo_add(double ***v,
                            int di, int dj, int dk,
                            int si, int sj, int sk)
{
    v[di][dj][dk] += v[si][sj][sk];
}

void addFace(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    //* Perpendicular ranges widened to [1, n-2] to match the widened MPI face
    //  types. At n_ghost==1 this is [1, n-2] — unchanged. At n_ghost==2 it
    //  captures inner-ghost-row moment contributions that were previously missed.
    const AxisSkip skip = axis_skip(vct, skip_self_periodic);
    const bool do_xR = vct->hasXrghtNeighbor_P() && !skip.x;
    const bool do_xL = vct->hasXleftNeighbor_P() && !skip.x;
    const bool do_yR = vct->hasYrghtNeighbor_P() && !skip.y;
    const bool do_yL = vct->hasYleftNeighbor_P() && !skip.y;
    const bool do_zR = vct->hasZrghtNeighbor_P() && !skip.z;
    const bool do_zL = vct->hasZleftNeighbor_P() && !skip.z;

    for (int g = 0; g < n_ghost; g++) {
        if (do_xR)
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector,nx - 1 - n_ghost - g, j, k,   nx - 1 - g, j, k);
        if (do_xL)
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector,n_ghost + g, j, k,   g, j, k);
        if (do_yR)
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector,i, ny - 1 - n_ghost - g, k,   i, ny - 1 - g, k);
        if (do_yL)
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector,i, n_ghost + g, k,   i, g, k);
        if (do_zR)
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    halo_add(vector,i, j, nz - 1 - n_ghost - g,   i, j, nz - 1 - g);
        if (do_zL)
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    halo_add(vector,i, j, n_ghost + g,   i, j, g);
    }
}

/** insert the ghost cells Edge Z in the 3D physical vector */
//  Z-aligned edge: ghost block lives in the (x, y) cross-section.
//  Same shift convention as addFace: (src ghost, dst interior) pairs are
//  independent — no chained accumulation.
void addEdgeZ(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    //* For an edge to need addEdge, BOTH cross-section axes must contribute a
    //* halo overlap. If either axis is periodic-self, the upstream periodic-self
    //* fold+copy already merged that pair (the corresponding 2-D sum lives at the
    //* cross-section corner of the duplicates).
    const AxisSkip skip = axis_skip(vct, skip_self_periodic);
    if (skip.x || skip.y) return;
    const bool do_RR = vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P();
    const bool do_LL = vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P();
    const bool do_RL = vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P();
    const bool do_LR = vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P();

    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    {
        if (do_RR)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector,nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, i,   nx - 1 - gx, ny - 1 - gy, i);
        if (do_LL)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector,n_ghost + gx, n_ghost + gy, i,   gx, gy, i);
        if (do_RL)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector,nx - 1 - n_ghost - gx, n_ghost + gy, i,   nx - 1 - gx, gy, i);
        if (do_LR)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector,n_ghost + gx, ny - 1 - n_ghost - gy, i,   gx, ny - 1 - gy, i);
    }
}
/** add the ghost cell values Edge Y to the 3D physical vector */
//  Y-aligned edge: ghost block lives in the (x, z) cross-section.
void addEdgeY(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const AxisSkip skip = axis_skip(vct, skip_self_periodic);
    if (skip.x || skip.z) return;
    const bool do_RR = vct->hasXrghtNeighbor_P() && vct->hasZrghtNeighbor_P();
    const bool do_LL = vct->hasXleftNeighbor_P() && vct->hasZleftNeighbor_P();
    const bool do_LR = vct->hasXleftNeighbor_P() && vct->hasZrghtNeighbor_P();
    const bool do_RL = vct->hasXrghtNeighbor_P() && vct->hasZleftNeighbor_P();

    for (int gx = 0; gx < n_ghost; gx++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (do_RR)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector,nx - 1 - n_ghost - gx, i, nz - 1 - n_ghost - gz,   nx - 1 - gx, i, nz - 1 - gz);
        if (do_LL)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector,n_ghost + gx, i, n_ghost + gz,   gx, i, gz);
        if (do_LR)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector,n_ghost + gx, i, nz - 1 - n_ghost - gz,   gx, i, nz - 1 - gz);
        if (do_RL)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector,nx - 1 - n_ghost - gx, i, n_ghost + gz,   nx - 1 - gx, i, gz);
    }
}

/** add the ghost values Edge X to the 3D physical vector */
//  X-aligned edge: ghost block lives in the (y, z) cross-section.
void addEdgeX(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const AxisSkip skip = axis_skip(vct, skip_self_periodic);
    if (skip.y || skip.z) return;
    const bool do_RR = vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P();
    const bool do_LL = vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P();
    const bool do_LR = vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P();
    const bool do_RL = vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P();

    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (do_RR)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector,i, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   i, ny - 1 - gy, nz - 1 - gz);
        if (do_LL)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector,i, n_ghost + gy, n_ghost + gz,   i, gy, gz);
        if (do_LR)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector,i, n_ghost + gy, nz - 1 - n_ghost - gz,   i, gy, nz - 1 - gz);
        if (do_RL)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector,i, ny - 1 - n_ghost - gy, n_ghost + gz,   i, ny - 1 - gy, gz);
    }
}

/** add ghost cells values Corners in the 3D physical vector */
//  Each corner is a single node for n_ghost == 1; for n_ghost > 1 it expands
//  to a (gx, gy, gz) cube of nodes that are summed back into the matching
//  inner interior corner.
void addCorner(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    //* A corner participates only when ALL three axes contribute halo overlap;
    //* if any of those axes is periodic-self, the upstream fold + copy already
    //* merged that triple-overlap.
    const AxisSkip skip = axis_skip(vct, skip_self_periodic);
    if (skip.x || skip.y || skip.z) return;

    const bool xR = vct->hasXrghtNeighbor_P(), xL = vct->hasXleftNeighbor_P();
    const bool yR = vct->hasYrghtNeighbor_P(), yL = vct->hasYleftNeighbor_P();
    const bool zR = vct->hasZrghtNeighbor_P(), zL = vct->hasZleftNeighbor_P();
    const bool do_RRR = xR && yR && zR;
    const bool do_LRR = xL && yR && zR;
    const bool do_RLR = xR && yL && zR;
    const bool do_LLR = xL && yL && zR;
    const bool do_RRL = xR && yR && zL;
    const bool do_LRL = xL && yR && zL;
    const bool do_RLL = xR && yL && zL;
    const bool do_LLL = xL && yL && zL;

    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (do_RRR)
            halo_add(vector,nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, ny - 1 - gy, nz - 1 - gz);
        if (do_LRR)
            halo_add(vector,n_ghost + gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   gx, ny - 1 - gy, nz - 1 - gz);
        if (do_RLR)
            halo_add(vector,nx - 1 - n_ghost - gx, n_ghost + gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, gy, nz - 1 - gz);
        if (do_LLR)
            halo_add(vector,n_ghost + gx, n_ghost + gy, nz - 1 - n_ghost - gz,   gx, gy, nz - 1 - gz);
        if (do_RRL)
            halo_add(vector,nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, n_ghost + gz,   nx - 1 - gx, ny - 1 - gy, gz);
        if (do_LRL)
            halo_add(vector,n_ghost + gx, ny - 1 - n_ghost - gy, n_ghost + gz,   gx, ny - 1 - gy, gz);
        if (do_RLL)
            halo_add(vector,nx - 1 - n_ghost - gx, n_ghost + gy, n_ghost + gz,   nx - 1 - gx, gy, gz);
        if (do_LLL)
            halo_add(vector,n_ghost + gx, n_ghost + gy, n_ghost + gz,   gx, gy, gz);
    }
}

/** communicate and sum shared ghost cells */

//? Used for communicating moments
//* Moment interpolation halo (sum-on-receive).
void communicateInterp(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::Interp);
}

void communicateInterp(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateInterp(nx, ny, nz, _vector.fetch_arr3(), vct, EMf);
}

void communicateNode_P(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double*** vectors[1] = {vector};
    NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf, HaloKind::NodeBC_P);
}

void communicateNode_P(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateNode_P(nx, ny, nz, _vector.fetch_arr3(), vct, EMf);
}

//* Multi-field call-site wrappers: batch N >= 1 fields sharing the same extents
//* and topology into one exchange per direction, cutting MPI message count N×.
//* All forward to NBDerivedHaloCommN.

//* Multi-field interpolation halo. With unify_ps_dups=true the halo also does
//* the periodic-self LO=HI unify before the cross-rank pack, so no trailing
//* Node_P call is needed at the call site.
void communicateInterp_multi(int nx, int ny, int nz, int n_fields, double ****vectors,
                              const VirtualTopology3D *vct, EMfields3D *EMf,
                              bool unify_ps_dups)
{
    NBDerivedHaloCommN(nx, ny, nz, n_fields, vectors, vct, EMf,
                       HaloKind::Interp, unify_ps_dups);
}

void communicateNode_P_multi(int nx, int ny, int nz, int n_fields, double ****vectors,
                              const VirtualTopology3D *vct, EMfields3D *EMf)
{
    NBDerivedHaloCommN(nx, ny, nz, n_fields, vectors, vct, EMf, HaloKind::NodeBC_P);
}

