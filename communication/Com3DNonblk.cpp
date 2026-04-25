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

//* Forward declaration of the n_ghost > 1 helper (definition further down).
static void NBDerivedHaloCommN(int nx, int ny, int nz, double ***vector,
                                const VirtualTopology3D * vct, EMfields3D *EMf,
                                bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                                double ***vector_c = nullptr);

//* Step 68c: forward declarations of the Kahan-compensated sum-on-receive
//* helpers (definitions near the legacy `addFace`/`addEdge*`/`addCorner`
//* further down). Same signature as the legacy helpers plus a companion
//* `vector_c` compensation array.
void addFace_kahan  (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeX_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeY_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeZ_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addCorner_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);

//! isCenterFlag: 1 = communicateCenter; 0 = communicateNode
//! Step 68c: `vector_c` is the Kahan-compensation companion array. `= nullptr`
//! is the legacy behaviour (byte-identical); a non-null pointer switches the
//! sum-on-receive step (`addFace`/`addEdge*`/`addCorner`) to the Neumaier-
//! compensated variants that land residuals in `vector_c`.
void NBDerivedHaloComm(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, EMfields3D *EMf,
                        bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                        double ***vector_c = nullptr)
{
    //* Phase A.3b: dispatch wider ghost-slab exchanges to the loop-based helper.
    //  For n_ghost == 1 we fall through to the legacy fast path below, which
    //  uses merged edge/corner MPI datatypes and remains byte-identical.
    if (EMf->getNGhost() > 1) {
        NBDerivedHaloCommN(nx, ny, nz, vector, vct, EMf,
                           isCenterFlag, isFaceOnlyFlag, needInterp, isParticle, vector_c);
        return;
    }

    const MPI_Comm comm       = isParticle ?vct->getParticleComm()      :vct->getFieldComm();
    #ifdef DEBUG
	    MPI_Errhandler_set(comm,MPI_ERRORS_RETURN);
    #endif
    MPI_Status  stat[12];
    MPI_Request reqList[12];				  //at most 6 requests x 2 (send recv)
    int communicationCnt[6] = {0,0,0,0,0,0};  //1 if there communication on that dir
    int recvcnt = 0,sendcnt = 0;              //request counter

    const int tag_XL=1,tag_YL=2,tag_ZL=3,tag_XR=4,tag_YR=5,tag_ZR=6;//To address same rank as left and right neighbour in periodic case
    const int myrank    	  = vct->getCartesian_rank();
    const int right_neighborX = isParticle ?vct->getXright_neighbor_P() :vct->getXright_neighbor();
    const int left_neighborX  = isParticle ?vct->getXleft_neighbor_P()  :vct->getXleft_neighbor();
    const int right_neighborY = isParticle ?vct->getYright_neighbor_P() :vct->getYright_neighbor();
    const int left_neighborY  = isParticle ?vct->getYleft_neighbor_P()  :vct->getYleft_neighbor();
    const int right_neighborZ = isParticle ?vct->getZright_neighbor_P() :vct->getZright_neighbor();
    const int left_neighborZ  = isParticle ?vct->getZleft_neighbor_P()  :vct->getZleft_neighbor();
    
    bool isCenterDim = (isCenterFlag&&!needInterp);//This is to address moment interp use nodes dimension but exchange center face
    
    const MPI_Datatype yzFacetype = EMf->getYZFacetype(isCenterDim);
    const MPI_Datatype xzFacetype = EMf->getXZFacetype(isCenterDim);
    const MPI_Datatype xyFacetype = EMf->getXYFacetype(isCenterDim);
    const MPI_Datatype xEdgetype = EMf->getXEdgetype(isCenterDim);
    const MPI_Datatype yEdgetype = EMf->getYEdgetype(isCenterDim);
    const MPI_Datatype zEdgetype = EMf->getZEdgetype(isCenterDim);
    const MPI_Datatype xEdgetype2 = EMf->getXEdgetype2(isCenterDim);
    const MPI_Datatype yEdgetype2 = EMf->getYEdgetype2(isCenterDim);
    const MPI_Datatype zEdgetype2 = EMf->getZEdgetype2(isCenterDim);
    const MPI_Datatype cornertype = EMf->getCornertype(isCenterDim);


    //Face Exchange as long as neighbor exits on that direction
    //Tag is based on the sender: XL for sender = XR for receiver
    //Post Recv before Send
    if(left_neighborX != MPI_PROC_NULL && left_neighborX != myrank)
    {
        MPI_Irecv(&vector[0][1][1], 1, yzFacetype, left_neighborX,tag_XR, comm, &reqList[recvcnt++]);
        communicationCnt[0] = 1;
    }
    if(right_neighborX != MPI_PROC_NULL && right_neighborX != myrank)
    {
        MPI_Irecv(&vector[nx-1][1][1], 1, yzFacetype, right_neighborX,tag_XL, comm, &reqList[recvcnt++]);
        communicationCnt[1] = 1;
    }
    if(left_neighborY != MPI_PROC_NULL && left_neighborY != myrank)
    {
        MPI_Irecv(&vector[1][0][1], 1, xzFacetype, left_neighborY,tag_YR, comm, &reqList[recvcnt++]);
        communicationCnt[2] = 1;
    }
    if(right_neighborY != MPI_PROC_NULL && right_neighborY != myrank)
    {
        MPI_Irecv(&vector[1][ny-1][1], 1, xzFacetype, right_neighborY,tag_YL, comm, &reqList[recvcnt++]);
        communicationCnt[3] = 1;
    }
    if(left_neighborZ != MPI_PROC_NULL && left_neighborZ != myrank)
    {
        MPI_Irecv(&vector[1][1][0],    1, xyFacetype, left_neighborZ,tag_ZR, comm, &reqList[recvcnt++]);
        communicationCnt[4] = 1;
    }
    if(right_neighborZ != MPI_PROC_NULL&& right_neighborZ != myrank)
    {
        MPI_Irecv(&vector[1][1][nz-1], 1, xyFacetype, right_neighborZ,tag_ZL, comm, &reqList[recvcnt++]);
        communicationCnt[5] = 1;
    }

    sendcnt = recvcnt;

    int offset = (isCenterFlag ?0:1);

    if(communicationCnt[0] == 1)
        MPI_Isend(&vector[1+offset][1][1],    1, yzFacetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
    if(communicationCnt[1] == 1)
        MPI_Isend(&vector[nx-2-offset][1][1], 1, yzFacetype, right_neighborX,tag_XR, comm, &reqList[sendcnt++]);
    if(communicationCnt[2] == 1)
        MPI_Isend(&vector[1][1+offset][1],    1, xzFacetype, left_neighborY, tag_YL, comm, &reqList[sendcnt++]);
    if(communicationCnt[3] == 1)
        MPI_Isend(&vector[1][ny-2-offset][1], 1, xzFacetype, right_neighborY,tag_YR, comm, &reqList[sendcnt++]);
    if(communicationCnt[4] == 1)
        MPI_Isend(&vector[1][1][1+offset],    1, xyFacetype, left_neighborZ, tag_ZL, comm, &reqList[sendcnt++]);
    if(communicationCnt[5] == 1)
        MPI_Isend(&vector[1][1][nz-2-offset], 1, xyFacetype, right_neighborZ,tag_ZR, comm, &reqList[sendcnt++]);

    assert_eq(recvcnt,sendcnt-recvcnt);

    //Buffer swap if any (done before waiting for receiving msg done to delay sync)
    //* Step 23: NODE-centered periodic arrays have a duplicate at indices {n_ghost, nx-n_ghost-1}
    //* (the two images of the same physical boundary node). The legacy self-swap
    //*   ghost(0) = interior(nx-2),  ghost(nx-1) = interior(1)
    //* maps the ghost to the other-side duplicate of the CENTER node instead of its
    //* geometric offset-by-one neighbour, biasing stencils at the periodic boundary.
    //* When `FixNodePeriodicHalo` is set, swap in the offset-by-one convention for
    //* plain node-copy arrays (E halo, B halo, moment-copy halo). Matches the MPI-send
    //* convention for XLEN>1 subdomains (lines 112-114: sends interior(1+offset)/nx-2-offset).
    //* Also matches ECSIM's `makeNodeFace` (ComParser3D.cpp:34-60), which reads from
    //* interior[2] / interior[nx-3] for the communicateNode_P path.
    //*
    //* Step 38 follow-up (2026-04-23): drop the `!isParticle` exclusion so the
    //* particle-moment ghost copy path (communicateNode_P on Mij, ρ, J with
    //* isParticle=true) also picks up the offset-by-one convention. Previously
    //* the E halo used nx-3/2 while the mass-matrix ghost-copy halo stayed on
    //* legacy nx-2/1, producing ~4e-5 l2_rel divergence in the cross-code
    //* MaxwellImage Stage C (raw M·E) byte diff.
    //*
    //* `needInterp` (the sum-on-receive addFace/Edge/Corner path used by
    //* communicateInterp) is still excluded: ECSIM's `makeCenterFace`
    //* (ComParser3D.cpp:91-116) reads from interior[1] / interior[nx-2]
    //* (legacy convention), so iPIC3D's legacy self-swap already matches there.
    //*
    //* Non-periodic BC safety (audit 2026-04-25). The offset-by-one source
    //* indices below are computed unconditionally but only *read* when the
    //* self-swap blocks fire (lines ~176/185/194), and those gate on
    //* `right_neighborX == myrank == left_neighborX`. For non-periodic axes,
    //* `MPI_Cart_create` with `periods[X]=false` returns `MPI_PROC_NULL` for
    //* off-end neighbours, so the test fails and the offset-by-one is dead
    //* code on that axis. `PERIODICX_P` defaults to `PERIODICX`
    //* (main/Collective.cpp:434), so the particle communicator agrees with
    //* the field one. The only reachable concern is asymmetric
    //* `PERIODICX=0 + PERIODICX_P=1` with `XLEN=1` — no committed input file
    //* uses this. If introduced, switch to per-axis gating with
    //* `vct->getPERIODICX*_P()` / `getPERIODICX()` matched on `isParticle`.
    //* Verified empirically: AuditAllOpen (PERIODICX=Y=Z=0) byte-identical
    //* with flag on/off; Double_Harris 10cyc still at 3.70e-14; 6/6 regression.
    const bool node_halo_fix = (EMf != nullptr) && EMf->get_col().getFixNodePeriodicHalo()
                               && !isCenterFlag && !needInterp;
    const int xlo_src = node_halo_fix ? (nx-3) : (nx-2);
    const int xhi_src = node_halo_fix ? 2      : 1;
    const int ylo_src = node_halo_fix ? (ny-3) : (ny-2);
    const int yhi_src = node_halo_fix ? 2      : 1;
    const int zlo_src = node_halo_fix ? (nz-3) : (nz-2);
    const int zhi_src = node_halo_fix ? 2      : 1;

    if (right_neighborX == myrank &&  left_neighborX== myrank)
    {
        for (int iy = 1; iy < ny-1; iy++)
            for (int iz = 1; iz < nz-1; iz++)
            {
                vector[0][iy][iz] = vector[xlo_src][iy][iz];
                vector[nx-1][iy][iz] = vector[xhi_src][iy][iz];
            }
    }
    if (right_neighborY == myrank &&  left_neighborY == myrank)
    {
        for (int ix = 1; ix < nx-1; ix++)
            for (int iz = 1; iz < nz-1; iz++)
            {
                vector[ix][0][iz] = vector[ix][ylo_src][iz];
                vector[ix][ny-1][iz] = vector[ix][yhi_src][iz];
            }
    }
    if (right_neighborZ == myrank &&  left_neighborZ == myrank)
    {
        for (int ix = 1; ix < nx-1; ix++)
            for (int iy = 1; iy < ny-1; iy++)
            {
                vector[ix][iy][0]    = vector[ix][iy][zlo_src];
                vector[ix][iy][nz-1] = vector[ix][iy][zhi_src];
            }
    }

    //* Need to finish receiving + sending before edge exchange
    if(sendcnt>0)
    {
        MPI_Waitall(sendcnt,  &reqList[0], &stat[0]);
        bool stopFlag = false;
        for(int si=0;si< sendcnt;si++){
            int error_code = stat[si].MPI_ERROR;
            if (error_code != MPI_SUCCESS) 
            {
                stopFlag = true;
                
                #ifdef DEBUG
                    char error_string[100];
                    int length_of_error_string, error_class;

                    MPI_Error_class(error_code, &error_class);
                    MPI_Error_string(error_class, error_string, &length_of_error_string);
                    dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
                #endif
            }
        }
        
        if(stopFlag) exit (EXIT_FAILURE);
    }


	if( !isFaceOnlyFlag ){

		//Exchange yEdge only when Z X neighbours exist
		//if Zleft + Zright, use merged yEdgeType2 to send two edges in one msg
		//Otherwise, only send one yEdge
		recvcnt = 0,sendcnt = 0;
		if(communicationCnt[0] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Irecv(&vector[0][1][0],   1,  yEdgetype2, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Irecv(&vector[0][1][0],   1,  yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Irecv(&vector[0][1][nz-1],1,  yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[1] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Irecv(&vector[nx-1][1][0],  1,  yEdgetype2,right_neighborX,tag_XL, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Irecv(&vector[nx-1][1][0],   1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Irecv(&vector[nx-1][1][nz-1],1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
			}
		}
		//Exchange zEdge only when X Y neighbours exist
		//if Xleft + Xright, use merged zEdgeType2
		//Otherwise, only send one zEdge
		if(communicationCnt[2] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Irecv(&vector[0][0][1],   1, zEdgetype2,left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][0][1],   1, zEdgetype, left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][0][1],1, zEdgetype, left_neighborY, tag_YR,comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[3] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Irecv(&vector[0][ny-1][1],	 1, zEdgetype2,right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][ny-1][1],   1, zEdgetype, right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][ny-1][1],1, zEdgetype, right_neighborY,tag_YL,comm, &reqList[recvcnt++]);
			}
		}
		//Exchange xEdge only when Y Z neighbours exist
		//if Yleft + Yright exist, use merged xEdgeType2
		//Otherwise, only send one xEdge
		if(communicationCnt[4] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][0][0],1,    xEdgetype2,left_neighborZ, tag_ZR,   comm, &reqList[recvcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Irecv(&vector[1][0][0],1,    xEdgetype, left_neighborZ, tag_ZR,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][ny-1][0],1, xEdgetype, left_neighborZ,tag_ZR,comm, &reqList[recvcnt++]);
			}
		}
		if(communicationCnt[5] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][0][nz-1],1,  xEdgetype2,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Irecv(&vector[1][0][nz-1],1,   xEdgetype,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Irecv(&vector[1][ny-1][nz-1],1,xEdgetype,right_neighborZ, tag_ZL,comm, &reqList[recvcnt++]);
			}
		}

		sendcnt = recvcnt;

		//* Step 61 fix: edge/corner MPI sends inherit the face-send offset-by-one
		//* convention so rank-boundary corner ghosts use the same periodic-duplicate
		//* source as face ghosts do. Face sends at lines 113-124 already apply
		//*   base_idx = isCenterFlag ? legacy : 1+offset (= 2 for node-copy)
		//* but edge/corner sends previously hard-coded legacy (idx=1, nx-2, etc.).
		//* That mismatch shows up as 2e-7 abs / 1e-3 l2_rel in Stage C (raw M·E)
		//* at rank-interior corners (local X=1 & nx-2, Y=1 & ny-2) for np>1 runs.
		if(communicationCnt[0] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Isend(&vector[1+offset][1][0],   1,  yEdgetype2,left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Isend(&vector[1+offset][1][0],   1,  yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Isend(&vector[1+offset][1][nz-1],1,  yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[1] == 1){
			if(communicationCnt[4] == 1 && communicationCnt[5] == 1){
				MPI_Isend(&vector[nx-2-offset][1][0],1,   yEdgetype2,right_neighborX, tag_XR, comm,&reqList[sendcnt++]);
			}else if(communicationCnt[4] == 1){
				MPI_Isend(&vector[nx-2-offset][1][0],1,   yEdgetype,right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[5] == 1){
				MPI_Isend(&vector[nx-2-offset][1][nz-1],1,yEdgetype,right_neighborX, tag_XR,comm,&reqList[sendcnt++]);
			}
		}
		if(communicationCnt[2] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Isend(&vector[0][1+offset][1],1,   zEdgetype2, left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Isend(&vector[0][1+offset][1],1,   zEdgetype,  left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-1][1+offset][1],1,zEdgetype,left_neighborY, tag_YL,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[3] == 1){
			if(communicationCnt[0] == 1 && communicationCnt[1] == 1){
				MPI_Isend(&vector[0][ny-2-offset][1],1,   zEdgetype2, right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[0] == 1){
				MPI_Isend(&vector[0][ny-2-offset][1],1,   zEdgetype,  right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-1][ny-2-offset][1],1,zEdgetype,right_neighborY, tag_YR,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[4] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Isend(&vector[1][0][1+offset],1, xEdgetype2, left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Isend(&vector[1][0][1+offset],1, xEdgetype,  left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Isend(&vector[1][ny-1][1+offset],1,xEdgetype,left_neighborZ, tag_ZL,   comm, &reqList[sendcnt++]);
			}
		}
		if(communicationCnt[5] == 1){
			if(communicationCnt[2] == 1 && communicationCnt[3] == 1){
				MPI_Isend(&vector[1][0][nz-2-offset],1,    xEdgetype2, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[2] == 1){
				MPI_Isend(&vector[1][0][nz-2-offset],1,    xEdgetype,  right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}else if(communicationCnt[3] == 1){
				MPI_Isend(&vector[1][ny-1][nz-2-offset],1, xEdgetype,  right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
			}
		}

		assert_eq(recvcnt,sendcnt-recvcnt);

		//Swap Local Edges
		//* Step 34a: edge swaps honour the Step 23 node_halo_fix so the
		//* periodic-self X/Y/Z wrap uses the offset-by-one source convention
		//* (xlo_src/xhi_src etc.) matching the face swaps above. Without this,
		//* corner/edge ghosts inherit the legacy nx-2/1 mapping while face
		//* ghosts have been remapped to nx-3/2 — the mismatch biases the
		//* MaxwellImage stencil at interior-boundary DOFs.
		if(right_neighborX == myrank &&  left_neighborX== myrank){
           if(right_neighborZ != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][nz-1]    = vector[xlo_src][iy][nz-1];
                    vector[nx-1][iy][nz-1] = vector[xhi_src][iy][nz-1];
                }
           }
           if(left_neighborZ != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][0]    = vector[xlo_src][iy][0];
                    vector[nx-1][iy][0] = vector[xhi_src][iy][0];
                }
           }
           if(right_neighborY != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][ny-1][iz]    = vector[xlo_src][ny-1][iz];
                    vector[nx-1][ny-1][iz] = vector[xhi_src][ny-1][iz];
                }
           }
           if(left_neighborY != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][0][iz]       = vector[xlo_src][0][iz];
                    vector[nx-1][0][iz]    = vector[xhi_src][0][iz];
                }
           }
        }
		if(right_neighborY == myrank &&  left_neighborY == myrank){
		   if(right_neighborX != MPI_PROC_NULL ){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[nx-1][0][iz]    = vector[nx-1][ylo_src][iz];
                    vector[nx-1][ny-1][iz] = vector[nx-1][yhi_src][iz];
			}
		  }
          if(left_neighborX != MPI_PROC_NULL){
                for (int iz = 1; iz < nz-1; iz++) {
                    vector[0][0][iz]       = vector[0][ylo_src][iz];
                    vector[0][ny-1][iz]    = vector[0][yhi_src][iz];
                }
		  }
		  if(right_neighborZ != MPI_PROC_NULL){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][nz-1]    = vector[ix][ylo_src][nz-1];
                    vector[ix][ny-1][nz-1] = vector[ix][yhi_src][nz-1];
                }
		  }
		  if(left_neighborZ != MPI_PROC_NULL){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][0]       = vector[ix][ylo_src][0];
                    vector[ix][ny-1][0]    = vector[ix][yhi_src][0];
                }
		  }
		}

		if(right_neighborZ == myrank &&  left_neighborZ == myrank){
            if(right_neighborY != MPI_PROC_NULL ){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][ny-1][0]    = vector[ix][ny-1][zlo_src];
                    vector[ix][ny-1][nz-1] = vector[ix][ny-1][zhi_src];
                }
            }
            if(left_neighborY != MPI_PROC_NULL ){
                for (int ix = 1; ix < nx-1; ix++){
                    vector[ix][0][0]    = vector[ix][0][zlo_src];
                    vector[ix][0][nz-1] = vector[ix][0][zhi_src];
                }
            }
            if(right_neighborX != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[nx-1][iy][0]    = vector[nx-1][iy][zlo_src];
                    vector[nx-1][iy][nz-1] = vector[nx-1][iy][zhi_src];
                }
            }
            if(left_neighborX != MPI_PROC_NULL ){
                for (int iy = 1; iy < ny-1; iy++){
                    vector[0][iy][0]    = vector[0][iy][zlo_src];
                    vector[0][iy][nz-1] = vector[0][iy][zhi_src];
                }
            }
        }
		//Need to finish receiving edges for corner exchange
		if(sendcnt>0){
			MPI_Waitall(sendcnt,&reqList[0],&stat[0]);
			bool stopFlag = false;
			for(int si=0;si< sendcnt;si++){
				int error_code = stat[si].MPI_ERROR;
				if (error_code != MPI_SUCCESS) {
					stopFlag = true;
#ifdef DEBUG
					char error_string[100];
					int length_of_error_string, error_class;

					MPI_Error_class(error_code, &error_class);
					MPI_Error_string(error_class, error_string, &length_of_error_string);
					dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
#endif
				}
			}
			if(stopFlag) exit (EXIT_FAILURE);;
		}


		//Corner Exchange only needed if XYZ neighbours all exist
		//4 corners communicated in one message
		//Assume Non-periodic will be handled in BC
		//Define corner types for X communication
		recvcnt = 0,sendcnt = 0;
		if((communicationCnt[2] == 1 || communicationCnt[3] == 1) && (communicationCnt[4] == 1 || communicationCnt[5] == 1)){
			//if XLeft exists, send 4 corners to XLeft
			if(communicationCnt[0] == 1){
				MPI_Irecv(&vector[0][0][0],1,   cornertype, left_neighborX, tag_XR,comm, &reqList[recvcnt++]);
			}
			//if XRight exist
			if(communicationCnt[1] == 1){
				MPI_Irecv(&vector[nx-1][0][0],1,cornertype, right_neighborX, tag_XL,comm, &reqList[recvcnt++]);
			}

			sendcnt=recvcnt;

			//* Step 61 fix: corner send uses the offset-by-one X source,
			//* matching face/edge sends.
			if(communicationCnt[0] == 1){
				MPI_Isend(&vector[1+offset][0][0],   1,cornertype,left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}
			if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-2-offset][0][0],1,cornertype,right_neighborX, tag_XR,comm, &reqList[sendcnt++]);
			}
		}

		assert_eq(recvcnt,sendcnt-recvcnt);


		//Delay local data copy — Step 34a: corner self-swaps honour the same
		//* node_halo_fix offset as face/edge swaps so all three periodic-wrap
		//* source indices agree.
		if (left_neighborX== myrank && right_neighborX == myrank){
			if( (left_neighborY != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[xlo_src][0][0];
				vector[nx-1][0][0]    = vector[xhi_src][0][0];
			}
			if( (left_neighborY != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][0][nz-1]    = vector[xlo_src][0][nz-1];
				vector[nx-1][0][nz-1] = vector[xhi_src][0][nz-1];
			}
			if( (right_neighborY != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][ny-1][0]    = vector[xlo_src][ny-1][0];
				vector[nx-1][ny-1][0] = vector[xhi_src][ny-1][0];
			}
			if( (right_neighborY != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][ny-1][nz-1]    = vector[xlo_src][ny-1][nz-1];
				vector[nx-1][ny-1][nz-1] = vector[xhi_src][ny-1][nz-1];
			}
		}
		else if (left_neighborY== myrank && right_neighborY == myrank){
			if( (left_neighborX != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[0][ylo_src][0];
				vector[0][ny-1][0]    = vector[0][yhi_src][0];
			}
			if( (left_neighborX != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[0][0][nz-1]    = vector[0][ylo_src][nz-1];
				vector[0][ny-1][nz-1]    = vector[0][yhi_src][nz-1];
			}
			if( (right_neighborX != MPI_PROC_NULL) && (left_neighborZ != MPI_PROC_NULL)){
				vector[nx-1][0][0]    = vector[nx-1][ylo_src][0];
				vector[nx-1][ny-1][0] = vector[nx-1][yhi_src][0];
			}
			if( (right_neighborX != MPI_PROC_NULL) && (right_neighborZ != MPI_PROC_NULL)){
				vector[nx-1][0][nz-1]    = vector[nx-1][ylo_src][nz-1];
				vector[nx-1][ny-1][nz-1] = vector[nx-1][yhi_src][nz-1];
			}
		}
		else if (left_neighborZ== myrank && right_neighborZ == myrank){
			if( (left_neighborY != MPI_PROC_NULL) && (left_neighborX != MPI_PROC_NULL)){
				vector[0][0][0]       = vector[0][0][zlo_src];
				vector[0][0][nz-1]    = vector[0][0][zhi_src];
			}

			if( (left_neighborY != MPI_PROC_NULL) && (right_neighborX != MPI_PROC_NULL)){
				vector[nx-1][0][0]    = vector[nx-1][0][zlo_src];
				vector[nx-1][0][nz-1] = vector[nx-1][0][zhi_src];
			}

			if( (right_neighborY != MPI_PROC_NULL) && (left_neighborX != MPI_PROC_NULL)){
				vector[0][ny-1][0]    = vector[0][ny-1][zlo_src];
				vector[0][ny-1][nz-1] = vector[0][ny-1][zhi_src];
			}

			if( (right_neighborY != MPI_PROC_NULL) && (right_neighborX != MPI_PROC_NULL)){
				vector[nx-1][ny-1][0]    = vector[nx-1][ny-1][zlo_src];
				vector[nx-1][ny-1][nz-1] = vector[nx-1][ny-1][zhi_src];
			}
		}


		if(sendcnt>0){
			MPI_Waitall(sendcnt,  &reqList[0], &stat[0]);
			bool stopFlag = false;
			for(int si=0;si< sendcnt;si++){
				int error_code = stat[si].MPI_ERROR;
				if (error_code != MPI_SUCCESS) {
					stopFlag = true;
#ifdef DEBUG
					char error_string[100];
					int length_of_error_string, error_class;

					MPI_Error_class(error_code, &error_class);
					MPI_Error_string(error_class, error_string, &length_of_error_string);
					dprintf("MPI_Waitall error at %d  %s\n",si, error_string);
#endif
				}
			}
			if(stopFlag) exit (EXIT_FAILURE);
		}
	}

	//if this is NodeInterpolation operation
	if(needInterp){
	    //* Step 68c: Kahan-aware sum-on-receive. Legacy path (`vector_c ==
	    //* nullptr`) falls through to the plain `+=` helpers below, byte-
	    //* identical to pre-Step-68c. When `vector_c` is supplied, the
	    //* receiving interior cell is updated via a Neumaier step with the
	    //* residual landing in `vector_c`; the caller folds `vector_c`
	    //* back into `vector` after the halo exchange.
	    if (vector_c != nullptr) {
	        addFace_kahan  (nx, ny, nz, vector, vector_c, vct, 1);
	        addEdgeZ_kahan (nx, ny, nz, vector, vector_c, vct, 1);
	        addEdgeY_kahan (nx, ny, nz, vector, vector_c, vct, 1);
	        addEdgeX_kahan (nx, ny, nz, vector, vector_c, vct, 1);
	        addCorner_kahan(nx, ny, nz, vector, vector_c, vct, 1);
	    } else {
	        addFace  (nx, ny, nz, vector, vct);
	        addEdgeZ (nx, ny, nz, vector, vct);
	        addEdgeY (nx, ny, nz, vector, vct);
	        addEdgeX (nx, ny, nz, vector, vct);
	        addCorner(nx, ny, nz, vector, vct);
	    }
	}

}


//! ================================================================================
//  NBDerivedHaloCommN: wider ghost-slab variant of NBDerivedHaloComm.
//
//  Phase A.3b helper used only when grid->getNGhost() > 1. Instead of relying
//  on the merged MPI edge/corner datatypes (which hard-code an n_ghost == 1
//  outermost-only geometry), each layer of the n_ghost-thick ghost slab is
//  exchanged via its own MPI call in a loop over layer offsets. The per-layer
//  face/edge types come from EMfields3D.cpp; those were already widened in
//  Phase A.2 so that one layer contains (n - 2*n_ghost)^2 doubles stripped of
//  the inner y/z ghost corners.
//
//  Notes:
//  - Corners fall back to scalar (MPI_DOUBLE) sends since there is no
//    precomputed corner-cube datatype. For n_ghost == 2 that is 8 corners *
//    n_ghost^3 = 64 doubles per X-direction per phase; acceptable as a first
//    implementation.
//  - Periodic-self buffer copies use a "constant extension" fallback that
//    matches the legacy behaviour at n_ghost == 1 and is a safe starting
//    point at n_ghost > 1. Strict periodic wrapping can be refined in Phase B
//    if a periodic single-rank n_ghost > 1 run is added to the regression set.
//  - The 4D legacy moment path (communicateInterp_old / communicateNode_P_old
//    via ComParser3D) is NOT widened here; for n_ghost > 1 the per-species
//    moment halo sum will be incorrect. Phase B adds that path, or routes it
//    through the 3D modern exchange on each species slice.
//! ================================================================================
static void NBDerivedHaloCommN(int nx, int ny, int nz, double ***vector,
                                const VirtualTopology3D * vct, EMfields3D *EMf,
                                bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                                double ***vector_c)
{
    const int n_ghost_ = EMf->getNGhost();

    const MPI_Comm comm = isParticle ? vct->getParticleComm() : vct->getFieldComm();
    #ifdef DEBUG
        MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);
    #endif

    //* Upper bound on in-flight messages per phase. Sized conservatively for
    //  n_ghost up to 3 (the grid constructor currently asserts <= 2).
    static const int MAX_REQS = 512;
    MPI_Status  stat[MAX_REQS];
    MPI_Request reqList[MAX_REQS];
    int communicationCnt[6] = {0,0,0,0,0,0};
    int recvcnt = 0, sendcnt = 0;

    const int tag_XL=1,tag_YL=2,tag_ZL=3,tag_XR=4,tag_YR=5,tag_ZR=6;
    const int myrank = vct->getCartesian_rank();
    const int right_neighborX = isParticle ? vct->getXright_neighbor_P() : vct->getXright_neighbor();
    const int left_neighborX  = isParticle ? vct->getXleft_neighbor_P()  : vct->getXleft_neighbor();
    const int right_neighborY = isParticle ? vct->getYright_neighbor_P() : vct->getYright_neighbor();
    const int left_neighborY  = isParticle ? vct->getYleft_neighbor_P()  : vct->getYleft_neighbor();
    const int right_neighborZ = isParticle ? vct->getZright_neighbor_P() : vct->getZright_neighbor();
    const int left_neighborZ  = isParticle ? vct->getZleft_neighbor_P()  : vct->getZleft_neighbor();

    const bool isCenterDim = (isCenterFlag && !needInterp);

    const MPI_Datatype yzFacetype = EMf->getYZFacetype(isCenterDim);
    const MPI_Datatype xzFacetype = EMf->getXZFacetype(isCenterDim);
    const MPI_Datatype xyFacetype = EMf->getXYFacetype(isCenterDim);
    const MPI_Datatype xEdgetype  = EMf->getXEdgetype(isCenterDim);
    const MPI_Datatype yEdgetype  = EMf->getYEdgetype(isCenterDim);
    const MPI_Datatype zEdgetype  = EMf->getZEdgetype(isCenterDim);

    if (left_neighborX  != MPI_PROC_NULL && left_neighborX  != myrank) communicationCnt[0] = 1;
    if (right_neighborX != MPI_PROC_NULL && right_neighborX != myrank) communicationCnt[1] = 1;
    if (left_neighborY  != MPI_PROC_NULL && left_neighborY  != myrank) communicationCnt[2] = 1;
    if (right_neighborY != MPI_PROC_NULL && right_neighborY != myrank) communicationCnt[3] = 1;
    if (left_neighborZ  != MPI_PROC_NULL && left_neighborZ  != myrank) communicationCnt[4] = 1;
    if (right_neighborZ != MPI_PROC_NULL && right_neighborZ != myrank) communicationCnt[5] = 1;

    //* Interior/boundary offset: centers use offset = 0 (no shared node); nodes
    //  use offset = 1 (ghost receives "one node deeper" than the boundary-shared
    //  node, so the send source is one step further into the interior).
    const int offset = (isCenterFlag ? 0 : 1);

    //! ============================================================
    //  FACE PHASE
    //  Each of the 6 face directions exchanges n_ghost slab layers.
    //! ============================================================
    for (int g = 0; g < n_ghost_; g++) {
        if (communicationCnt[0])
            MPI_Irecv(&vector[g][1][1],       1, yzFacetype, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
        if (communicationCnt[1])
            MPI_Irecv(&vector[nx-1-g][1][1],  1, yzFacetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
        if (communicationCnt[2])
            MPI_Irecv(&vector[1][g][1],       1, xzFacetype, left_neighborY,  tag_YR, comm, &reqList[recvcnt++]);
        if (communicationCnt[3])
            MPI_Irecv(&vector[1][ny-1-g][1],  1, xzFacetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
        if (communicationCnt[4])
            MPI_Irecv(&vector[1][1][g],       1, xyFacetype, left_neighborZ,  tag_ZR, comm, &reqList[recvcnt++]);
        if (communicationCnt[5])
            MPI_Irecv(&vector[1][1][nz-1-g],  1, xyFacetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
    }
    sendcnt = recvcnt;
    for (int g = 0; g < n_ghost_; g++) {
        if (communicationCnt[0])
            MPI_Isend(&vector[n_ghost_+offset+g][1][1],       1, yzFacetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
        if (communicationCnt[1])
            MPI_Isend(&vector[nx-1-n_ghost_-offset-g][1][1],  1, yzFacetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
        if (communicationCnt[2])
            MPI_Isend(&vector[1][n_ghost_+offset+g][1],       1, xzFacetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
        if (communicationCnt[3])
            MPI_Isend(&vector[1][ny-1-n_ghost_-offset-g][1],  1, xzFacetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
        if (communicationCnt[4])
            MPI_Isend(&vector[1][1][n_ghost_+offset+g],       1, xyFacetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
        if (communicationCnt[5])
            MPI_Isend(&vector[1][1][nz-1-n_ghost_-offset-g],  1, xyFacetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
    }
    assert_eq(recvcnt, sendcnt - recvcnt);
    assert_le(sendcnt, MAX_REQS);

    //* Periodic single-rank self-copies (X/Y/Z). Source indices use the same
    //  (n_ghost + offset + g) depth convention as the MPI face sends, with
    //  modular wrapping for thin dimensions where the source would otherwise
    //  fall outside the interior range [n_ghost, n-1-n_ghost].
    //  stride = n - 2*n_ghost - offset = nxc_r (distinct periodic nodes/centers).
    //  For offset=0 (moment/interp path), this reduces to the original formula.
    //  For offset=1 (node field-copy path), this corrects the source to match MPI.
    if (right_neighborX == myrank && left_neighborX == myrank) {
        const int stride_x = nx - 2 * n_ghost_ - offset;
        for (int g = 0; g < n_ghost_; g++) {
            int sL = nx - 1 - n_ghost_ - offset - g;
            if (sL < n_ghost_) sL += stride_x;
            int sR = n_ghost_ + offset + g;
            if (sR > nx - 1 - n_ghost_) sR -= stride_x;
            for (int iy = 1; iy < ny - 1; iy++)
                for (int iz = 1; iz < nz - 1; iz++) {
                    vector[g][iy][iz]            = vector[sL][iy][iz];
                    vector[nx - 1 - g][iy][iz]   = vector[sR][iy][iz];
                }
        }
    }
    if (right_neighborY == myrank && left_neighborY == myrank) {
        const int stride_y = ny - 2 * n_ghost_ - offset;
        for (int g = 0; g < n_ghost_; g++) {
            int sL = ny - 1 - n_ghost_ - offset - g;
            if (sL < n_ghost_) sL += stride_y;
            int sR = n_ghost_ + offset + g;
            if (sR > ny - 1 - n_ghost_) sR -= stride_y;
            for (int ix = 1; ix < nx - 1; ix++)
                for (int iz = 1; iz < nz - 1; iz++) {
                    vector[ix][g][iz]            = vector[ix][sL][iz];
                    vector[ix][ny - 1 - g][iz]   = vector[ix][sR][iz];
                }
        }
    }
    if (right_neighborZ == myrank && left_neighborZ == myrank) {
        const int stride_z = nz - 2 * n_ghost_ - offset;
        for (int g = 0; g < n_ghost_; g++) {
            int sL = nz - 1 - n_ghost_ - offset - g;
            if (sL < n_ghost_) sL += stride_z;
            int sR = n_ghost_ + offset + g;
            if (sR > nz - 1 - n_ghost_) sR -= stride_z;
            for (int ix = 1; ix < nx - 1; ix++)
                for (int iy = 1; iy < ny - 1; iy++) {
                    vector[ix][iy][g]            = vector[ix][iy][sL];
                    vector[ix][iy][nz - 1 - g]   = vector[ix][iy][sR];
                }
        }
    }

    if (sendcnt > 0) {
        MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
        bool stopFlag = false;
        for (int si = 0; si < sendcnt; si++) {
            if (stat[si].MPI_ERROR != MPI_SUCCESS) { stopFlag = true; }
        }
        if (stopFlag) exit(EXIT_FAILURE);
    }

    if (!isFaceOnlyFlag) {
        //! ============================================================
        //  EDGE PHASE
        //  Each axis has 4 edges (corners of the perpendicular face).
        //  For n_ghost > 1 each corner expands to an n_ghost x n_ghost
        //  block in the cross-section perpendicular to the edge direction.
        //! ============================================================
        recvcnt = 0; sendcnt = 0;

        //? yEdge: Y-aligned line at (X, Z) corners. Cross-section = (gx, gz).
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            if (communicationCnt[0]) {
                if (communicationCnt[4]) MPI_Irecv(&vector[gx][1][gz],            1, yEdgetype, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
                if (communicationCnt[5]) MPI_Irecv(&vector[gx][1][nz-1-gz],       1, yEdgetype, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
            }
            if (communicationCnt[1]) {
                if (communicationCnt[4]) MPI_Irecv(&vector[nx-1-gx][1][gz],       1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
                if (communicationCnt[5]) MPI_Irecv(&vector[nx-1-gx][1][nz-1-gz],  1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
            }
        }

        //? zEdge: Z-aligned line at (X, Y) corners. Cross-section = (gx, gy).
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gy = 0; gy < n_ghost_; gy++)
        {
            if (communicationCnt[2]) {
                if (communicationCnt[0]) MPI_Irecv(&vector[gx][gy][1],            1, zEdgetype, left_neighborY,  tag_YR, comm, &reqList[recvcnt++]);
                if (communicationCnt[1]) MPI_Irecv(&vector[nx-1-gx][gy][1],       1, zEdgetype, left_neighborY,  tag_YR, comm, &reqList[recvcnt++]);
            }
            if (communicationCnt[3]) {
                if (communicationCnt[0]) MPI_Irecv(&vector[gx][ny-1-gy][1],       1, zEdgetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
                if (communicationCnt[1]) MPI_Irecv(&vector[nx-1-gx][ny-1-gy][1],  1, zEdgetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
            }
        }

        //? xEdge: X-aligned line at (Y, Z) corners. Cross-section = (gy, gz).
        for (int gy = 0; gy < n_ghost_; gy++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            if (communicationCnt[4]) {
                if (communicationCnt[2]) MPI_Irecv(&vector[1][gy][gz],             1, xEdgetype, left_neighborZ,  tag_ZR, comm, &reqList[recvcnt++]);
                if (communicationCnt[3]) MPI_Irecv(&vector[1][ny-1-gy][gz],        1, xEdgetype, left_neighborZ,  tag_ZR, comm, &reqList[recvcnt++]);
            }
            if (communicationCnt[5]) {
                if (communicationCnt[2]) MPI_Irecv(&vector[1][gy][nz-1-gz],        1, xEdgetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
                if (communicationCnt[3]) MPI_Irecv(&vector[1][ny-1-gy][nz-1-gz],   1, xEdgetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
            }
        }

        sendcnt = recvcnt;

        //? yEdge sends (mirror of recvs; source at interior depth n_ghost + offset + g)
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            if (communicationCnt[0]) {
                if (communicationCnt[4]) MPI_Isend(&vector[n_ghost_+offset+gx][1][n_ghost_+offset+gz],        1, yEdgetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                if (communicationCnt[5]) MPI_Isend(&vector[n_ghost_+offset+gx][1][nz-1-n_ghost_-offset-gz],   1, yEdgetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[1]) {
                if (communicationCnt[4]) MPI_Isend(&vector[nx-1-n_ghost_-offset-gx][1][n_ghost_+offset+gz],       1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                if (communicationCnt[5]) MPI_Isend(&vector[nx-1-n_ghost_-offset-gx][1][nz-1-n_ghost_-offset-gz],  1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
            }
        }

        //? zEdge sends
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gy = 0; gy < n_ghost_; gy++)
        {
            if (communicationCnt[2]) {
                if (communicationCnt[0]) MPI_Isend(&vector[n_ghost_+offset+gx][n_ghost_+offset+gy][1],         1, zEdgetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
                if (communicationCnt[1]) MPI_Isend(&vector[nx-1-n_ghost_-offset-gx][n_ghost_+offset+gy][1],    1, zEdgetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[3]) {
                if (communicationCnt[0]) MPI_Isend(&vector[n_ghost_+offset+gx][ny-1-n_ghost_-offset-gy][1],        1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
                if (communicationCnt[1]) MPI_Isend(&vector[nx-1-n_ghost_-offset-gx][ny-1-n_ghost_-offset-gy][1],   1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
            }
        }

        //? xEdge sends
        for (int gy = 0; gy < n_ghost_; gy++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            if (communicationCnt[4]) {
                if (communicationCnt[2]) MPI_Isend(&vector[1][n_ghost_+offset+gy][n_ghost_+offset+gz],         1, xEdgetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
                if (communicationCnt[3]) MPI_Isend(&vector[1][ny-1-n_ghost_-offset-gy][n_ghost_+offset+gz],    1, xEdgetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[5]) {
                if (communicationCnt[2]) MPI_Isend(&vector[1][n_ghost_+offset+gy][nz-1-n_ghost_-offset-gz],        1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
                if (communicationCnt[3]) MPI_Isend(&vector[1][ny-1-n_ghost_-offset-gy][nz-1-n_ghost_-offset-gz],   1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
            }
        }

        assert_eq(recvcnt, sendcnt - recvcnt);
        assert_le(sendcnt, MAX_REQS);

        //* Periodic single-rank edge self-copies.
        //  Generalized from legacy L308-386: for each self-periodic axis, copy
        //  the axis-ghost cells at perpendicular outermost-ghost positions.
        //  Reads only from face-exchange-filled cells. The widened face types
        //  cover inner ghost rows, so only outermost (index 0 / n-1) edges need
        //  this pass. At n_ghost == 1 the g loop runs once and the indices
        //  reduce to the legacy literals.
        if (right_neighborX == myrank && left_neighborX == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                if (right_neighborZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[g][iy][nz-1]         = vector[nx-1-n_ghost_-offset-g][iy][nz-1];
                        vector[nx-1-g][iy][nz-1]    = vector[n_ghost_+offset+g][iy][nz-1];
                    }
                }
                if (left_neighborZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[g][iy][0]         = vector[nx-1-n_ghost_-offset-g][iy][0];
                        vector[nx-1-g][iy][0]    = vector[n_ghost_+offset+g][iy][0];
                    }
                }
                if (right_neighborY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[g][ny-1][iz]         = vector[nx-1-n_ghost_-offset-g][ny-1][iz];
                        vector[nx-1-g][ny-1][iz]    = vector[n_ghost_+offset+g][ny-1][iz];
                    }
                }
                if (left_neighborY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[g][0][iz]         = vector[nx-1-n_ghost_-offset-g][0][iz];
                        vector[nx-1-g][0][iz]    = vector[n_ghost_+offset+g][0][iz];
                    }
                }
            }
        }
        if (right_neighborY == myrank && left_neighborY == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                if (right_neighborX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[nx-1][g][iz]         = vector[nx-1][ny-1-n_ghost_-offset-g][iz];
                        vector[nx-1][ny-1-g][iz]    = vector[nx-1][n_ghost_+offset+g][iz];
                    }
                }
                if (left_neighborX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[0][g][iz]         = vector[0][ny-1-n_ghost_-offset-g][iz];
                        vector[0][ny-1-g][iz]    = vector[0][n_ghost_+offset+g][iz];
                    }
                }
                if (right_neighborZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][g][nz-1]         = vector[ix][ny-1-n_ghost_-offset-g][nz-1];
                        vector[ix][ny-1-g][nz-1]    = vector[ix][n_ghost_+offset+g][nz-1];
                    }
                }
                if (left_neighborZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][g][0]         = vector[ix][ny-1-n_ghost_-offset-g][0];
                        vector[ix][ny-1-g][0]    = vector[ix][n_ghost_+offset+g][0];
                    }
                }
            }
        }
        if (right_neighborZ == myrank && left_neighborZ == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                if (right_neighborY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][ny-1][g]         = vector[ix][ny-1][nz-1-n_ghost_-offset-g];
                        vector[ix][ny-1][nz-1-g]    = vector[ix][ny-1][n_ghost_+offset+g];
                    }
                }
                if (left_neighborY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][0][g]         = vector[ix][0][nz-1-n_ghost_-offset-g];
                        vector[ix][0][nz-1-g]    = vector[ix][0][n_ghost_+offset+g];
                    }
                }
                if (right_neighborX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[nx-1][iy][g]         = vector[nx-1][iy][nz-1-n_ghost_-offset-g];
                        vector[nx-1][iy][nz-1-g]    = vector[nx-1][iy][n_ghost_+offset+g];
                    }
                }
                if (left_neighborX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[0][iy][g]         = vector[0][iy][nz-1-n_ghost_-offset-g];
                        vector[0][iy][nz-1-g]    = vector[0][iy][n_ghost_+offset+g];
                    }
                }
            }
        }

        if (sendcnt > 0) {
            MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
            bool stopFlag = false;
            for (int si = 0; si < sendcnt; si++) {
                if (stat[si].MPI_ERROR != MPI_SUCCESS) { stopFlag = true; }
            }
            if (stopFlag) exit(EXIT_FAILURE);
        }

        //! ============================================================
        //  CORNER PHASE
        //  Each of the 8 box corners is an n_ghost^3 cube of scalar nodes
        //  for n_ghost > 1, exchanged as MPI_DOUBLE singletons. Aggregated
        //  along the X direction in 2 messages (X-left, X-right) per
        //  (gx, gy, gz) triple, like the cornertype does at n_ghost == 1.
        //! ============================================================
        recvcnt = 0; sendcnt = 0;

        if ((communicationCnt[2] == 1 || communicationCnt[3] == 1) &&
            (communicationCnt[4] == 1 || communicationCnt[5] == 1))
        {
            for (int gx = 0; gx < n_ghost_; gx++)
            for (int gy = 0; gy < n_ghost_; gy++)
            for (int gz = 0; gz < n_ghost_; gz++)
            {
                if (communicationCnt[0]) {
                    MPI_Irecv(&vector[gx][gy][gz],                      1, MPI_DOUBLE, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[gx][gy][nz-1-gz],                 1, MPI_DOUBLE, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[gx][ny-1-gy][gz],                 1, MPI_DOUBLE, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[gx][ny-1-gy][nz-1-gz],            1, MPI_DOUBLE, left_neighborX,  tag_XR, comm, &reqList[recvcnt++]);
                }
                if (communicationCnt[1]) {
                    MPI_Irecv(&vector[nx-1-gx][gy][gz],                 1, MPI_DOUBLE, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[nx-1-gx][gy][nz-1-gz],            1, MPI_DOUBLE, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[nx-1-gx][ny-1-gy][gz],            1, MPI_DOUBLE, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
                    MPI_Irecv(&vector[nx-1-gx][ny-1-gy][nz-1-gz],       1, MPI_DOUBLE, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
                }
            }

            sendcnt = recvcnt;

            for (int gx = 0; gx < n_ghost_; gx++)
            for (int gy = 0; gy < n_ghost_; gy++)
            for (int gz = 0; gz < n_ghost_; gz++)
            {
                const int xs_l = n_ghost_ + offset + gx;            //* X source index for left send
                const int xs_r = nx - 1 - n_ghost_ - offset - gx;   //* X source index for right send
                const int ys_l = n_ghost_ + offset + gy;
                const int ys_r = ny - 1 - n_ghost_ - offset - gy;
                const int zs_l = n_ghost_ + offset + gz;
                const int zs_r = nz - 1 - n_ghost_ - offset - gz;

                if (communicationCnt[0]) {
                    MPI_Isend(&vector[xs_l][ys_l][zs_l],   1, MPI_DOUBLE, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_l][ys_l][zs_r],   1, MPI_DOUBLE, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_l][ys_r][zs_l],   1, MPI_DOUBLE, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_l][ys_r][zs_r],   1, MPI_DOUBLE, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                }
                if (communicationCnt[1]) {
                    MPI_Isend(&vector[xs_r][ys_l][zs_l],   1, MPI_DOUBLE, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_r][ys_l][zs_r],   1, MPI_DOUBLE, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_r][ys_r][zs_l],   1, MPI_DOUBLE, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                    MPI_Isend(&vector[xs_r][ys_r][zs_r],   1, MPI_DOUBLE, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                }
            }
        }

        assert_eq(recvcnt, sendcnt - recvcnt);
        assert_le(sendcnt, MAX_REQS);

        if (sendcnt > 0) {
            MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
            bool stopFlag = false;
            for (int si = 0; si < sendcnt; si++) {
                if (stat[si].MPI_ERROR != MPI_SUCCESS) { stopFlag = true; }
            }
            if (stopFlag) exit(EXIT_FAILURE);
        }

        //* Periodic single-rank corner self-copies.
        //  Generalized from legacy L437-494. Uses else-if like legacy: for np=1
        //  (all periodic) the first branch handles all 8 corner cubes via the
        //  cascade face->edge->corner. Source cells were filled by edge self-copies.
        if (left_neighborX == myrank && right_neighborX == myrank) {
            for (int g = 0; g < n_ghost_; g++)
            for (int gy = 0; gy < n_ghost_; gy++)
            for (int gz = 0; gz < n_ghost_; gz++) {
                if (left_neighborY != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                    vector[g][gy][gz]                   = vector[nx-1-n_ghost_-offset-g][gy][gz];
                    vector[nx-1-g][gy][gz]              = vector[n_ghost_+offset+g][gy][gz];
                }
                if (left_neighborY != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                    vector[g][gy][nz-1-gz]              = vector[nx-1-n_ghost_-offset-g][gy][nz-1-gz];
                    vector[nx-1-g][gy][nz-1-gz]         = vector[n_ghost_+offset+g][gy][nz-1-gz];
                }
                if (right_neighborY != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                    vector[g][ny-1-gy][gz]              = vector[nx-1-n_ghost_-offset-g][ny-1-gy][gz];
                    vector[nx-1-g][ny-1-gy][gz]         = vector[n_ghost_+offset+g][ny-1-gy][gz];
                }
                if (right_neighborY != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                    vector[g][ny-1-gy][nz-1-gz]         = vector[nx-1-n_ghost_-offset-g][ny-1-gy][nz-1-gz];
                    vector[nx-1-g][ny-1-gy][nz-1-gz]    = vector[n_ghost_+offset+g][ny-1-gy][nz-1-gz];
                }
            }
        }
        else if (left_neighborY == myrank && right_neighborY == myrank) {
            for (int g = 0; g < n_ghost_; g++)
            for (int gx = 0; gx < n_ghost_; gx++)
            for (int gz = 0; gz < n_ghost_; gz++) {
                if (left_neighborX != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                    vector[gx][g][gz]                   = vector[gx][ny-1-n_ghost_-offset-g][gz];
                    vector[gx][ny-1-g][gz]              = vector[gx][n_ghost_+offset+g][gz];
                }
                if (left_neighborX != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                    vector[gx][g][nz-1-gz]              = vector[gx][ny-1-n_ghost_-offset-g][nz-1-gz];
                    vector[gx][ny-1-g][nz-1-gz]         = vector[gx][n_ghost_+offset+g][nz-1-gz];
                }
                if (right_neighborX != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                    vector[nx-1-gx][g][gz]              = vector[nx-1-gx][ny-1-n_ghost_-offset-g][gz];
                    vector[nx-1-gx][ny-1-g][gz]         = vector[nx-1-gx][n_ghost_+offset+g][gz];
                }
                if (right_neighborX != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                    vector[nx-1-gx][g][nz-1-gz]         = vector[nx-1-gx][ny-1-n_ghost_-offset-g][nz-1-gz];
                    vector[nx-1-gx][ny-1-g][nz-1-gz]    = vector[nx-1-gx][n_ghost_+offset+g][nz-1-gz];
                }
            }
        }
        else if (left_neighborZ == myrank && right_neighborZ == myrank) {
            for (int g = 0; g < n_ghost_; g++)
            for (int gx = 0; gx < n_ghost_; gx++)
            for (int gy = 0; gy < n_ghost_; gy++) {
                if (left_neighborX != MPI_PROC_NULL && left_neighborY != MPI_PROC_NULL) {
                    vector[gx][gy][g]                   = vector[gx][gy][nz-1-n_ghost_-offset-g];
                    vector[gx][gy][nz-1-g]              = vector[gx][gy][n_ghost_+offset+g];
                }
                if (left_neighborX != MPI_PROC_NULL && right_neighborY != MPI_PROC_NULL) {
                    vector[gx][ny-1-gy][g]              = vector[gx][ny-1-gy][nz-1-n_ghost_-offset-g];
                    vector[gx][ny-1-gy][nz-1-g]         = vector[gx][ny-1-gy][n_ghost_+offset+g];
                }
                if (right_neighborX != MPI_PROC_NULL && left_neighborY != MPI_PROC_NULL) {
                    vector[nx-1-gx][gy][g]              = vector[nx-1-gx][gy][nz-1-n_ghost_-offset-g];
                    vector[nx-1-gx][gy][nz-1-g]         = vector[nx-1-gx][gy][n_ghost_+offset+g];
                }
                if (right_neighborX != MPI_PROC_NULL && right_neighborY != MPI_PROC_NULL) {
                    vector[nx-1-gx][ny-1-gy][g]         = vector[nx-1-gx][ny-1-gy][nz-1-n_ghost_-offset-g];
                    vector[nx-1-gx][ny-1-gy][nz-1-g]    = vector[nx-1-gx][ny-1-gy][n_ghost_+offset+g];
                }
            }
        }
    }

    //* Moment summation: after the halo has filled the ghost slab, sum each
    //  ghost layer back into the matching inner interior layer.
    //
    //  NOTE: for self-periodic thin axes where n_cells_per_rank < 2*n_ghost,
    //  the left-side and right-side addFace destinations overlap, causing
    //  double-counting of periodic-wrap moment contributions. This can NOT
    //  be fixed by clamping addFace alone — the whole discretization (field
    //  copy, stencils, moments) is self-consistently calibrated. Require
    //  nxc_r >= 2*n_ghost - 1 per rank for each self-periodic axis.
    if (needInterp) {
        //* Step 68c: Kahan-aware sum-on-receive, n_ghost > 1 variant.
        if (vector_c != nullptr) {
            addFace_kahan  (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeZ_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeY_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeX_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addCorner_kahan(nx, ny, nz, vector, vector_c, vct, n_ghost_);
        } else {
            addFace  (nx, ny, nz, vector, vct, n_ghost_);
            addEdgeZ (nx, ny, nz, vector, vct, n_ghost_);
            addEdgeY (nx, ny, nz, vector, vct, n_ghost_);
            addEdgeX (nx, ny, nz, vector, vct, n_ghost_);
            addCorner(nx, ny, nz, vector, vct, n_ghost_);
        }
    }
}


void communicateNodeBC(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,false,false,false);
	BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

//* double*** overload — same body as the arr3_double version, used by code paths
//  that hold a raw triple pointer (e.g. energy_conserve_smooth_direction).
void communicateNodeBC(int nx, int ny, int nz, double*** vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,false,false,false);
	BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_old( int nx, int ny, int nz, double ***vector, 
                            int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                            const VirtualTopology3D *vct, EMfields3D *EMf)
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;

    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeNodeFace(nx, ny, nz, vector, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    parseFace(nx, ny, nz, vector, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);

    // prepare ghost cell Edge Y for communication: these are communicate: these are communicated in X direction */
    makeNodeGhostEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    // prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeGhostEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    // prepare ghost cell Edge X for communication: these are communicated in Z direction*/
    makeNodeGhostEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    // communicate twice each direction
    // X-DIRECTION: Z -> X
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor(), vct->getYleft_neighbor(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor(), vct->getZleft_neighbor(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    // parse
    MPI_Barrier(MPI_COMM_WORLD);

    parseEdgeZ(nx, ny, nz, vector, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    parseEdgeY(nx, ny, nz, vector, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    parseEdgeX(nx, ny, nz, vector, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    // apply boundary condition to 8 Ghost Corners and communicate if necessary to 8 processors
    makeNodeGhostCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor(), vct->getXleft_neighbor(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    // parse
    parseCorner(nx, ny, nz, vector, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);

    // APPLY the boundary conditions
    BCface(nx, ny, nz, vector, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, vct);

    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
}

void communicateNodeBoxStencilBC( int nx, int ny, int nz, arr3_double _vector,
								  int bcFaceXrght, int bcFaceXleft,
								  int bcFaceYrght, int bcFaceYleft,
								  int bcFaceZrght, int bcFaceZleft,
								  const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,true,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,true,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_P(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false,false,false,true);
	BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}


void communicateCenterBC(int nx, int ny, int nz, arr3_double _vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC(int nx, int ny, int nz, double*** vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, false, false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBC_P( int nx, int ny, int nz, arr3_double _vector,
                            int bcFaceXrght, int bcFaceXleft,
                            int bcFaceYrght, int bcFaceYleft,
                            int bcFaceZrght, int bcFaceZleft,
                            const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC( int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true,true,false,false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC_P( int nx, int ny, int nz, arr3_double _vector,
									  int bcFaceXrght, int bcFaceXleft,
									  int bcFaceYrght, int bcFaceYleft,
									  int bcFaceZrght, int bcFaceZleft,
									  const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true,true,false,true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}


/** add the values of ghost cells faces to the 3D physical vector */
//  For n_ghost == 1 the loop runs once with g = 0, reproducing the
//  byte-identical literal indices of the original implementation.
//  For n_ghost > 1, ghost layer g (counted from the outermost) is summed into
//  the corresponding interior layer at depth g from the boundary node:
//      left:  vector[n_ghost + g]   += vector[g]
//      right: vector[nx-1-n_ghost-g] += vector[nx-1-g]
//  No chained accumulation: each (src, dst) pair is independent because the
//  src lives in the ghost slab and the dst lives strictly inside the interior.
void addFace(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost)
{
    //* Perpendicular ranges widened to [1, n-2] to match the widened MPI face
    //  types. At n_ghost==1 this is [1, n-2] — unchanged. At n_ghost==2 it
    //  captures inner-ghost-row moment contributions that were previously missed.
    for (int g = 0; g < n_ghost; g++) {
        // Xright
        if (vct->hasXrghtNeighbor_P())
        {
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[nx - 1 - n_ghost - g][j][k] += vector[nx - 1 - g][j][k];
        }
        // XLEFT
        if (vct->hasXleftNeighbor_P())
        {
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[n_ghost + g][j][k] += vector[g][j][k];
        }

        // Yright
        if (vct->hasYrghtNeighbor_P())
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[i][ny - 1 - n_ghost - g][k] += vector[i][ny - 1 - g][k];
        }
        // Yleft
        if (vct->hasYleftNeighbor_P())
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[i][n_ghost + g][k] += vector[i][g][k];
        }
        // Zright
        if (vct->hasZrghtNeighbor_P())
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    vector[i][j][nz - 1 - n_ghost - g] += vector[i][j][nz - 1 - g];
        }
        // ZLEFT
        if (vct->hasZleftNeighbor_P())
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    vector[i][j][n_ghost + g] += vector[i][j][g];
        }
    }
}

/** insert the ghost cells Edge Z in the 3D physical vector */
//  Z-aligned edge: ghost block lives in the (x, y) cross-section.
//  Same shift convention as addFace: (src ghost, dst interior) pairs are
//  independent — no chained accumulation.
void addEdgeZ(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P())
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[nx - 1 - n_ghost - gx][ny - 1 - n_ghost - gy][i] += vector[nx - 1 - gx][ny - 1 - gy][i];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P())
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[n_ghost + gx][n_ghost + gy][i] += vector[gx][gy][i];
        }
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P())
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[nx - 1 - n_ghost - gx][n_ghost + gy][i] += vector[nx - 1 - gx][gy][i];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P())
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[n_ghost + gx][ny - 1 - n_ghost - gy][i] += vector[gx][ny - 1 - gy][i];
        }
    }
}
/** add the ghost cell values Edge Y to the 3D physical vector */
//  Y-aligned edge: ghost block lives in the (x, z) cross-section.
void addEdgeY(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[nx - 1 - n_ghost - gx][i][nz - 1 - n_ghost - gz] += vector[nx - 1 - gx][i][nz - 1 - gz];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasZleftNeighbor_P())
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[n_ghost + gx][i][n_ghost + gz] += vector[gx][i][gz];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasZrghtNeighbor_P())
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[n_ghost + gx][i][nz - 1 - n_ghost - gz] += vector[gx][i][nz - 1 - gz];
        }
        if (vct->hasXrghtNeighbor_P() && vct->hasZleftNeighbor_P())
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[nx - 1 - n_ghost - gx][i][n_ghost + gz] += vector[nx - 1 - gx][i][gz];
        }
    }
}

/** add the ghost values Edge X to the 3D physical vector */
//  X-aligned edge: ghost block lives in the (y, z) cross-section.
void addEdgeX(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost)
{
    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P()) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][ny - 1 - n_ghost - gy][nz - 1 - n_ghost - gz] += vector[i][ny - 1 - gy][nz - 1 - gz];
        }
        if (vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P()) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][n_ghost + gy][n_ghost + gz] += vector[i][gy][gz];
        }
        if (vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P()) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][n_ghost + gy][nz - 1 - n_ghost - gz] += vector[i][gy][nz - 1 - gz];
        }
        if (vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P()) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][ny - 1 - n_ghost - gy][n_ghost + gz] += vector[i][ny - 1 - gy][gz];
        }
    }
}

/** add ghost cells values Corners in the 3D physical vector */
//  Each corner is a single node for n_ghost == 1; for n_ghost > 1 it expands
//  to a (gx, gy, gz) cube of nodes that are summed back into the matching
//  inner interior corner.
void addCorner(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            vector[nx - 1 - n_ghost - gx][ny - 1 - n_ghost - gy][nz - 1 - n_ghost - gz] += vector[nx - 1 - gx][ny - 1 - gy][nz - 1 - gz];
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            vector[n_ghost + gx][ny - 1 - n_ghost - gy][nz - 1 - n_ghost - gz] += vector[gx][ny - 1 - gy][nz - 1 - gz];
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            vector[nx - 1 - n_ghost - gx][n_ghost + gy][nz - 1 - n_ghost - gz] += vector[nx - 1 - gx][gy][nz - 1 - gz];
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            vector[n_ghost + gx][n_ghost + gy][nz - 1 - n_ghost - gz] += vector[gx][gy][nz - 1 - gz];
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            vector[nx - 1 - n_ghost - gx][ny - 1 - n_ghost - gy][n_ghost + gz] += vector[nx - 1 - gx][ny - 1 - gy][gz];
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            vector[n_ghost + gx][ny - 1 - n_ghost - gy][n_ghost + gz] += vector[gx][ny - 1 - gy][gz];
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
            vector[nx - 1 - n_ghost - gx][n_ghost + gy][n_ghost + gz] += vector[nx - 1 - gx][gy][gz];
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
            vector[n_ghost + gx][n_ghost + gy][n_ghost + gz] += vector[gx][gy][gz];
    }
}

//! ================================================================================
//  Step 68c: Kahan-compensated parallel helpers for the sum-on-receive path.
//  One-to-one with the addFace / addEdge{X,Y,Z} / addCorner helpers above, but
//  each `interior += ghost` is replaced by a Neumaier-compensated add that
//  updates the matching interior cell of `vector_c` alongside `vector`. The
//  companion `vector_c` must be zero at entry (callers fold it into `vector`
//  before the halo exchange; after halo, a second fold merges the residuals
//  back into `vector` for the field solver).
//
//  Runs only when `KahanHalo=true` — same hot-loop shape as the legacy
//  helpers, just doing ~3× FLOP per node pair. These touch only interior-
//  boundary nodes so the extra cost is a small fraction of a matvec.
//! ================================================================================

static inline void kahan_add_inline(double& sum, double& comp, double term)
{
    const double t = sum + term;
    if (std::fabs(sum) >= std::fabs(term))
        comp += (sum - t) + term;
    else
        comp += (term - t) + sum;
    sum = t;
}

#define KAHAN_ADD(DST_I, DST_J, DST_K, SRC_I, SRC_J, SRC_K)               \
    kahan_add_inline(vector  [DST_I][DST_J][DST_K],                       \
                     vector_c[DST_I][DST_J][DST_K],                       \
                     vector  [SRC_I][SRC_J][SRC_K])

void addFace_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c,
                   const VirtualTopology3D * vct, int n_ghost)
{
    for (int g = 0; g < n_ghost; g++) {
        if (vct->hasXrghtNeighbor_P())
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    KAHAN_ADD(nx - 1 - n_ghost - g, j, k,   nx - 1 - g, j, k);
        if (vct->hasXleftNeighbor_P())
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    KAHAN_ADD(n_ghost + g, j, k,   g, j, k);
        if (vct->hasYrghtNeighbor_P())
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    KAHAN_ADD(i, ny - 1 - n_ghost - g, k,   i, ny - 1 - g, k);
        if (vct->hasYleftNeighbor_P())
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    KAHAN_ADD(i, n_ghost + g, k,   i, g, k);
        if (vct->hasZrghtNeighbor_P())
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    KAHAN_ADD(i, j, nz - 1 - n_ghost - g,   i, j, nz - 1 - g);
        if (vct->hasZleftNeighbor_P())
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    KAHAN_ADD(i, j, n_ghost + g,   i, j, g);
    }
}

void addEdgeZ_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c,
                    const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P())
            for (int i = 1; i < (nz - 1); i++)
                KAHAN_ADD(nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, i,   nx - 1 - gx, ny - 1 - gy, i);
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P())
            for (int i = 1; i < (nz - 1); i++)
                KAHAN_ADD(n_ghost + gx, n_ghost + gy, i,   gx, gy, i);
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P())
            for (int i = 1; i < (nz - 1); i++)
                KAHAN_ADD(nx - 1 - n_ghost - gx, n_ghost + gy, i,   nx - 1 - gx, gy, i);
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P())
            for (int i = 1; i < (nz - 1); i++)
                KAHAN_ADD(n_ghost + gx, ny - 1 - n_ghost - gy, i,   gx, ny - 1 - gy, i);
    }
}

void addEdgeY_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c,
                    const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            for (int i = 1; i < (ny - 1); i++)
                KAHAN_ADD(nx - 1 - n_ghost - gx, i, nz - 1 - n_ghost - gz,   nx - 1 - gx, i, nz - 1 - gz);
        if (vct->hasXleftNeighbor_P() && vct->hasZleftNeighbor_P())
            for (int i = 1; i < (ny - 1); i++)
                KAHAN_ADD(n_ghost + gx, i, n_ghost + gz,   gx, i, gz);
        if (vct->hasXleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            for (int i = 1; i < (ny - 1); i++)
                KAHAN_ADD(n_ghost + gx, i, nz - 1 - n_ghost - gz,   gx, i, nz - 1 - gz);
        if (vct->hasXrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            for (int i = 1; i < (ny - 1); i++)
                KAHAN_ADD(nx - 1 - n_ghost - gx, i, n_ghost + gz,   nx - 1 - gx, i, gz);
    }
}

void addEdgeX_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c,
                    const VirtualTopology3D * vct, int n_ghost)
{
    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            for (int i = 1; i < (nx - 1); i++)
                KAHAN_ADD(i, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   i, ny - 1 - gy, nz - 1 - gz);
        if (vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
            for (int i = 1; i < (nx - 1); i++)
                KAHAN_ADD(i, n_ghost + gy, n_ghost + gz,   i, gy, gz);
        if (vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            for (int i = 1; i < (nx - 1); i++)
                KAHAN_ADD(i, n_ghost + gy, nz - 1 - n_ghost - gz,   i, gy, nz - 1 - gz);
        if (vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            for (int i = 1; i < (nx - 1); i++)
                KAHAN_ADD(i, ny - 1 - n_ghost - gy, n_ghost + gz,   i, ny - 1 - gy, gz);
    }
}

void addCorner_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c,
                     const VirtualTopology3D * vct, int n_ghost)
{
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            KAHAN_ADD(nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, ny - 1 - gy, nz - 1 - gz);
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P())
            KAHAN_ADD(n_ghost + gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   gx, ny - 1 - gy, nz - 1 - gz);
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            KAHAN_ADD(nx - 1 - n_ghost - gx, n_ghost + gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, gy, nz - 1 - gz);
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P())
            KAHAN_ADD(n_ghost + gx, n_ghost + gy, nz - 1 - n_ghost - gz,   gx, gy, nz - 1 - gz);
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            KAHAN_ADD(nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, n_ghost + gz,   nx - 1 - gx, ny - 1 - gy, gz);
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P())
            KAHAN_ADD(n_ghost + gx, ny - 1 - n_ghost - gy, n_ghost + gz,   gx, ny - 1 - gy, gz);
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
            KAHAN_ADD(nx - 1 - n_ghost - gx, n_ghost + gy, n_ghost + gz,   nx - 1 - gx, gy, gz);
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P())
            KAHAN_ADD(n_ghost + gx, n_ghost + gy, n_ghost + gz,   gx, gy, gz);
    }
}

#undef KAHAN_ADD

/** communicate and sum shared ghost cells */

//? Used for communicating moments
void communicateInterp(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true);
}

void communicateInterp(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true);
}

//* Step 68c: Kahan-compensated variants. Forward the companion array into
//* NBDerivedHaloComm so the sum-on-receive goes through `addFace_kahan` etc.
void communicateInterp_kahan(int nx, int ny, int nz, double*** vector, double*** vector_c,
                             const VirtualTopology3D * vct, EMfields3D *EMf)
{
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true, vector_c);
}

void communicateInterp_kahan(int nx, int ny, int nz, arr3_double _vector, arr3_double _vector_c,
                             const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector   = _vector.fetch_arr3();
    double ***vector_c = _vector_c.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true, vector_c);
}

void communicateNode_P(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
}

void communicateNode_P(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector=_vector.fetch_arr3();
	NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
}

//? Old functions used for communicating 4D vectors (each species of particles in moments)
void communicateNode_P_old(int nx, int ny, int nz, int ns, double ****vector, const VirtualTopology3D *vct, EMfields3D *EMf) 
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;
  
    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeNodeFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    parseFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
  
    /** prepare ghost cell Edge Y for communication: these are communicate: these are communicated in X direction */
    makeNodeGhostEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    /** prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeGhostEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    /** prepare ghost cell Edge X for communication: these are communicated in  Z direction*/
    makeNodeGhostEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);
  
    // communicate twice each direction
    // X-DIRECTION: Z -> X
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    
    // parse
    MPI_Barrier(MPI_COMM_WORLD);
    parseEdgeZ(nx, ny, nz, vector, ns, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
    parseEdgeY(nx, ny, nz, vector, ns, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
    parseEdgeX(nx, ny, nz, vector, ns, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);

    makeNodeGhostCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    
    // parse
    parseCorner(nx, ny, nz, vector, ns, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);

    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
    // timeTasks.addto_communicate(); 
}
  
void communicateInterp_old(int nx, int ny, int nz, int ns, double ****vector,
                           int bcFaceXright, int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright, int bcFaceZleft, 
                           const VirtualTopology3D *vct, EMfields3D *EMf)
{
    // allocate 6 ghost cell Faces
    double *ghostXrightFace = new double[(ny - 2) * (nz - 2)];
    double *ghostXleftFace = new double[(ny - 2) * (nz - 2)];
    double *ghostYrightFace = new double[(nx - 2) * (nz - 2)];
    double *ghostYleftFace = new double[(nx - 2) * (nz - 2)];
    double *ghostZrightFace = new double[(nx - 2) * (ny - 2)];
    double *ghostZleftFace = new double[(nx - 2) * (ny - 2)];
    // allocate 12 ghost cell Edges
    // X EDGE
    double *ghostXsameYleftZleftEdge = new double[nx - 2];
    double *ghostXsameYrightZleftEdge = new double[nx - 2];
    double *ghostXsameYleftZrightEdge = new double[nx - 2];
    double *ghostXsameYrightZrightEdge = new double[nx - 2];
    // Y EDGE
    double *ghostXrightYsameZleftEdge = new double[ny - 2];
    double *ghostXleftYsameZleftEdge = new double[ny - 2];
    double *ghostXrightYsameZrightEdge = new double[ny - 2];
    double *ghostXleftYsameZrightEdge = new double[ny - 2];
    // Z EDGE
    double *ghostXrightYleftZsameEdge = new double[nz - 2];
    double *ghostXrightYrightZsameEdge = new double[nz - 2];
    double *ghostXleftYleftZsameEdge = new double[nz - 2];
    double *ghostXleftYrightZsameEdge = new double[nz - 2];
    // allocate 8 ghost cell corner
    double ghostXrightYrightZrightCorner, ghostXleftYrightZrightCorner, ghostXrightYleftZrightCorner, ghostXleftYleftZrightCorner;
    double ghostXrightYrightZleftCorner, ghostXleftYrightZleftCorner, ghostXrightYleftZleftCorner, ghostXleftYleftZleftCorner;
  
    // apply boundary condition to 6 Ghost Faces and communicate if necessary to 6 processors: along 3 DIRECTIONS
    makeCenterFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace);
    // communication
    communicateGhostFace((ny - 2) * (nz - 2), vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightFace, ghostXleftFace);
    communicateGhostFace((nx - 2) * (nz - 2), vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostYrightFace, ghostYleftFace);
    communicateGhostFace((nx - 2) * (ny - 2), vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostZrightFace, ghostZleftFace);
    addFace(nx, ny, nz, vector, ns, ghostXrightFace, ghostXleftFace, ghostYrightFace, ghostYleftFace, ghostZrightFace, ghostZleftFace, vct);
  /** prepare ghost cell Edge Y for communication: these are communicated in X direction */
    makeNodeEdgeY(nx, ny, nz, ghostZleftFace, ghostZrightFace, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge);
  /** prepare ghost cell Edge Z for communication: these are communicated in Y direction */
    makeNodeEdgeZ(nx, ny, nz, ghostXleftFace, ghostXrightFace, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge);
  /** prepare ghost cell Edge X for communication: these are communicated in  Z direction*/
    makeNodeEdgeX(nx, ny, nz, ghostYleftFace, ghostYrightFace, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge);
  
    // communicate twice each direction
    // X-DIRECTION: Z -> X -> Y
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZleftEdge, ghostXleftYsameZleftEdge);
    communicateGhostFace(ny - 2, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYsameZrightEdge, ghostXleftYsameZrightEdge);
    // Y-DIRECTION: X -> Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXleftYrightZsameEdge, ghostXleftYleftZsameEdge);
    communicateGhostFace(nz - 2, vct->getCartesian_rank(), vct->getYright_neighbor_P(), vct->getYleft_neighbor_P(), 1, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXrightYrightZsameEdge, ghostXrightYleftZsameEdge);
    // Z-DIRECTION: Y -> Z
    MPI_Barrier(MPI_COMM_WORLD);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYleftZrightEdge, ghostXsameYleftZleftEdge);
    communicateGhostFace(nx - 2, vct->getCartesian_rank(), vct->getZright_neighbor_P(), vct->getZleft_neighbor_P(), 2, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), ghostXsameYrightZrightEdge, ghostXsameYrightZleftEdge);
    // parse
    MPI_Barrier(MPI_COMM_WORLD);
    addEdgeZ(nx, ny, nz, vector, ns, ghostXrightYrightZsameEdge, ghostXleftYleftZsameEdge, ghostXrightYleftZsameEdge, ghostXleftYrightZsameEdge, vct);
    addEdgeY(nx, ny, nz, vector, ns, ghostXrightYsameZrightEdge, ghostXleftYsameZleftEdge, ghostXleftYsameZrightEdge, ghostXrightYsameZleftEdge, vct);
    addEdgeX(nx, ny, nz, vector, ns, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, vct);
  
    makeNodeCorner(nx, ny, nz, ghostXsameYrightZrightEdge, ghostXsameYleftZleftEdge, ghostXsameYleftZrightEdge, ghostXsameYrightZleftEdge, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    // communicate only in the X-DIRECTION
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner);
    communicateGhostFace(1, vct->getCartesian_rank(), vct->getXright_neighbor_P(), vct->getXleft_neighbor_P(), 0, vct->getXLEN(), vct->getYLEN(), vct->getZLEN(), &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner);
    // parse
    addCorner(nx, ny, nz, vector, ns, &ghostXrightYrightZrightCorner, &ghostXleftYrightZrightCorner, &ghostXrightYleftZrightCorner, &ghostXleftYleftZrightCorner, &ghostXrightYrightZleftCorner, &ghostXleftYrightZleftCorner, &ghostXrightYleftZleftCorner, &ghostXleftYleftZleftCorner, vct);
  
  
    delete[]ghostXrightFace;
    delete[]ghostXleftFace;
    delete[]ghostYrightFace;
    delete[]ghostYleftFace;
    delete[]ghostZrightFace;
    delete[]ghostZleftFace;
    // X EDGE
    delete[]ghostXsameYleftZleftEdge;
    delete[]ghostXsameYrightZleftEdge;
    delete[]ghostXsameYleftZrightEdge;
    delete[]ghostXsameYrightZrightEdge;
    // Y EDGE
    delete[]ghostXrightYsameZleftEdge;
    delete[]ghostXleftYsameZleftEdge;
    delete[]ghostXrightYsameZrightEdge;
    delete[]ghostXleftYsameZrightEdge;
    // Z EDGE
    delete[]ghostXrightYleftZsameEdge;
    delete[]ghostXrightYrightZsameEdge;
    delete[]ghostXleftYleftZsameEdge;
    delete[]ghostXleftYrightZsameEdge;
}

//* ============================================================================================================================ *//