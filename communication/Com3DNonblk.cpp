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

//* forward declarations of the Kahan-compensated sum-on-receive
//* helpers (definitions near the legacy `addFace`/`addEdge*`/`addCorner`
//* further down). Same signature as the legacy helpers plus a companion
//* `vector_c` compensation array.
void addFace_kahan  (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeX_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeY_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addEdgeZ_kahan (int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);
void addCorner_kahan(int nx, int ny, int nz, double ***vector, double ***vector_c, const VirtualTopology3D * vct, int n_ghost);

//! isCenterFlag: 1 = communicateCenter; 0 = communicateNode
//! `vector_c` is the Kahan-compensation companion array. `= nullptr`
//! is the legacy behaviour (byte-identical); a non-null pointer switches the
//! sum-on-receive step (`addFace`/`addEdge*`/`addCorner`) to the Neumaier-
//! compensated variants that land residuals in `vector_c`.
void NBDerivedHaloComm(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, EMfields3D *EMf,
                        bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                        double ***vector_c = nullptr)
{
    //* Wider ghost-slab path goes through the loop-based helper. n_ghost == 1
    //* falls through to the legacy merged-datatype fast path below.
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
    //* NODE arrays on a self-periodic axis carry duplicates at {n_ghost, nx-n_ghost-1}
    //* (two images of the same physical boundary node). The offset-by-one self-swap
    //* ghost(0)=interior(nx-3), ghost(nx-1)=interior(2) maps each ghost to the proper
    //* periodic neighbour one step inside, matching ECSIM's `makeNodeFace` and the
    //* MPI-send convention for XLEN>1. CENTER halos (`isCenterFlag`) and the
    //* sum-on-receive path (`needInterp`) keep the legacy `nx-2 / 1`. Non-periodic
    //* axes return `MPI_PROC_NULL` so the self-swap gate below is false and the
    //* offset is unread.
    const bool node_halo_fix = !isCenterFlag && !needInterp;
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

		//* edge/corner MPI sends inherit the face-send offset-by-one
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
		//* edge swaps honour the node_halo_fix so the
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

			//* corner send uses the offset-by-one X source,
			//* matching face/edge sends.
			if(communicationCnt[0] == 1){
				MPI_Isend(&vector[1+offset][0][0],   1,cornertype,left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
			}
			if(communicationCnt[1] == 1){
				MPI_Isend(&vector[nx-2-offset][0][0],1,cornertype,right_neighborX, tag_XR,comm, &reqList[sendcnt++]);
			}
		}

		assert_eq(recvcnt,sendcnt-recvcnt);


		//Delay local data copy — corner self-swaps honour the same
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
	    //* Kahan-aware sum-on-receive. Legacy path (`vector_c ==
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
//  Used when grid->getNGhost() > 1. Each layer of the n_ghost-thick ghost slab
//  is exchanged via its own MPI call in a loop over layer offsets, since the
//  merged edge/corner datatypes hard-code n_ghost == 1 outermost-only geometry.
//  Per-layer face/edge types come from EMfields3D.cpp.
//
//  Notes:
//  - Corner cubes fall back to scalar MPI_DOUBLE sends (no precomputed type).
//  - Periodic-self buffer copies use a constant-extension fallback (matches
//    legacy n_ghost == 1 behaviour).
//  - The 4D legacy moment path (communicateInterp_old / communicateNode_P_old)
//    is NOT widened here — for n_ghost > 1 the per-species moment halo sum
//    routes through the 3D modern exchange.
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

    //* Interior/boundary offset: centers use offset = 0 (no shared node);
    //  nodes use offset = 1 (ghost receives "one node deeper" than the
    //  boundary-shared node, so the send source is one step further into
    //  the interior). The moment-interp path (isCenterFlag=true,
    //  needInterp=true) actually carries node data, so we take offset
    //  from isCenterDim — which already filters needInterp out.
    const int offset = isCenterDim ? 0 : 1;

    //* Periodic-self ghost-source remap: substitute g → (n_ghost-1-g) so
    //  the OUTERMOST ghost (g=0) holds the value FURTHEST from the active
    //  interior. No-op at n_ghost=1.
    auto g_src = [&](int g) { return n_ghost_ - 1 - g; };

    //! ============================================================
    //* CROSS-RANK MOMENT SOR PRE-PASS (faces + edges + corners)
    //
    //  At n_ghost > 1 (TSC) for the moments path, particles whose stencil
    //  reaches past the rank boundary deposit partials into the local
    //  GHOST cells. Those partials must fold into the *neighbour's*
    //  strict-interior native at the matching periodic-image phys node.
    //
    //  Without explicit edge/corner exchanges, ghost cells in the corners
    //  of two or three axes (e.g. (X-LO ghost ∩ Y-LO ghost)) only reach
    //  one face neighbour and never the diagonal neighbour they physically
    //  belong to. The unified loop below partitions each rank's ghost
    //  region into 26 disjoint slabs indexed by (sx,sy,sz) ∈ {-1,0,+1}³,
    //  one per Cartesian neighbour; each is sent exactly once to its
    //  destination, where the matching SOR formula folds it into the
    //  receiver's strict-interior + LO/HI duplicate.
    //
    //  - sx = ±1: source range covers ng ghost layers + 1 dup (h ∈ [0..ng]).
    //  - sx =  0: source range covers strict interior excluding dups
    //             (i ∈ [ng+1..nx-2-ng]); destination = same coordinate on
    //             the receiver (no shift along non-involved axis).
    //
    //  The standard MPI face exchange below uses COPY semantics and would
    //  overwrite our ghost cells before they could be sent, so this
    //  pre-pass runs first.
    //
    //  Strict no-op at n_ghost=1
    //  and at periodic-self axes (Cartesian neighbour wraps to self).
    //! ============================================================
    //* Resolve Cartesian topology + neighbour-at helper. Loaded once per
    //* call (was being re-done by both the SOR pre-pass and the
    //* XrankDiagonalEdgeCopy block). periods[] and dims[] make the wrap
    //* explicit so periodic-self axes reduce to "neighbour == myrank"
    //* and get filtered out cleanly.
    int my_coords[3], dims[3], periods[3];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);

    //* Cart-neighbour lookup. Declared at function scope so the SOR
    //* pre-pass AND the XrankDiagonalEdgeCopy block both call the same
    //* lambda.
    auto neighbour_at = [&](int sx, int sy, int sz) -> int {
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
    };

    //* Cross-rank periodic axis check: SOR pre-pass + multi-axis completion
    //* passes are pure no-ops at np=1 (neighbours all == myrank) and at
    //* configs with no cross-rank periodic axis. Skip the bookkeeping when
    //* none exists so 1000+ halo calls per cycle don't pay the iterate-
    //* then-skip cost on small / single-axis decompositions.
    const bool any_xrank_periodic =
        (periods[0] && dims[0] > 1) || (periods[1] && dims[1] > 1)
                                    || (periods[2] && dims[2] > 1);

    if (needInterp && n_ghost_ > 1 && any_xrank_periodic) {

        //* Per-axis (n, s) → source index, and (n, s) → destination index.
        //* h is the layer counter:
        //*   sd = -1: h ∈ [0..ng]  (h=0..ng-1 ghost outer→inner; h=ng LO dup at i=ng)
        //*   sd = +1: h ∈ [0..ng]  (h=0..ng-1 ghost outer→inner; h=ng HI dup)
        //*   sd =  0: h ∈ [0..len-1] over strict interior [ng+1..n-2-ng]
        auto src_idx = [&](int sd, int n, int h) -> int {
            if (sd == -1) return h;
            if (sd == +1) return n - 1 - h;
            return n_ghost_ + 1 + h;
        };
        auto dst_idx = [&](int sd, int n, int h) -> int {
            if (sd == -1) {
                return (h < n_ghost_) ? (h + n - 2 * n_ghost_ - offset) : (n - 1 - n_ghost_);
            }
            if (sd == +1) {
                return (h < n_ghost_) ? ((2 * n_ghost_ - 1 + offset) - h) : n_ghost_;
            }
            return n_ghost_ + 1 + h;
        };

        const int strict_x = nx - 2 * n_ghost_ - 2;
        const int strict_y = ny - 2 * n_ghost_ - 2;
        const int strict_z = nz - 2 * n_ghost_ - 2;
        const int dup_n    = n_ghost_ + 1;

        //* Tags: encode (sx,sy,sz) as a base-3 number in [0..26]; offset 200
        //* keeps it disjoint from the standard exchange tags 1..6 below.
        auto encode_tag = [](int sx, int sy, int sz) {
            return 200 + (sx + 1) * 9 + (sy + 1) * 3 + (sz + 1);
        };

        //* Reuse buffers: we exchange one direction at a time. Worst-case
        //* slab is dup_n^3 (corner) up to (strict_x+strict_y+strict_z) faces.
        //* Posting all 26 in flight at once is ~3-4 MB for typical TSC grids
        //* and avoids 26 sequential round trips; pre-allocate vectors and
        //* batch with one Waitall.
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
            const int nbr = neighbour_at(sx, sy, sz);
            if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

            const int lx = (sx != 0) ? dup_n : strict_x;
            const int ly = (sy != 0) ? dup_n : strict_y;
            const int lz = (sz != 0) ? dup_n : strict_z;
            if (lx <= 0 || ly <= 0 || lz <= 0) continue;

            slots.push_back({sx, sy, sz, nbr, lx, ly, lz, {}, {}});
            ExchSlot &s = slots.back();
            const int slab = lx * ly * lz;
            s.send.resize(slab);
            s.recv.resize(slab);
            for (int hx = 0; hx < lx; ++hx) {
                const int ix = src_idx(sx, nx, hx);
                for (int hy = 0; hy < ly; ++hy) {
                    const int iy = src_idx(sy, ny, hy);
                    for (int hz = 0; hz < lz; ++hz) {
                        const int iz = src_idx(sz, nz, hz);
                        s.send[(hx * ly + hy) * lz + hz] = vector[ix][iy][iz];
                    }
                }
            }
        }

        std::vector<MPI_Request> rqs;
        rqs.reserve(2 * slots.size());
        for (ExchSlot &s : slots) {
            const int slab = s.lx * s.ly * s.lz;
            const int tag_send = encode_tag( s.sx,  s.sy,  s.sz);
            const int tag_recv = encode_tag(-s.sx, -s.sy, -s.sz);
            rqs.emplace_back();
            MPI_Irecv(s.recv.data(), slab, MPI_DOUBLE, s.nbr, tag_recv, comm, &rqs.back());
            rqs.emplace_back();
            MPI_Isend(s.send.data(), slab, MPI_DOUBLE, s.nbr, tag_send, comm, &rqs.back());
        }
        if (!rqs.empty()) MPI_Waitall(rqs.size(), rqs.data(), MPI_STATUSES_IGNORE);

        //* SOR received slabs into the matching destination region.
        //* Each (sx,sy,sz) slot's recv buffer holds the neighbour's
        //* (-sx,-sy,-sz) ghost slab — physically the periodic-image of
        //* OUR (-sx,-sy,-sz) destination strict + dup. dst_idx with
        //* (-s.sx,-s.sy,-s.sz) returns: LO strict + LO dup for sd=-1,
        //* HI strict + HI dup for sd=+1, strict perpendicular for sd=0.
        for (const ExchSlot &s : slots) {
            for (int hx = 0; hx < s.lx; ++hx) {
                const int ix = dst_idx(-s.sx, nx, hx);
                for (int hy = 0; hy < s.ly; ++hy) {
                    const int iy = dst_idx(-s.sy, ny, hy);
                    for (int hz = 0; hz < s.lz; ++hz) {
                        const int iz = dst_idx(-s.sz, nz, hz);
                        vector[ix][iy][iz] += s.recv[(hx * s.ly + hy) * s.lz + hz];
                    }
                }
            }
        }

        //! ============================================================
        //* MULTI-AXIS CORNER COMPLETION (Phase E.14)
        //
        //  At a multi-rank corner cell (intersection of TWO cross-rank
        //  periodic axes), the 26-slot SOR's diagonal-(±1,±1,0) slot only
        //  pairs the cell with its diagonally-opposite rank — a 2-rank
        //  sum where 4-rank consensus is required. Empirically: at np=4
        //  X+Y=2x2, post-SOR Mxx max|Δ| at j ∈ {ng, ny-ng-1} dup rows is
        //  ~5e-4 between paired ranks (face cells at j strict are bit-tight).
        //
        //  Fix: for each pair (a,b) of cross-rank axes, do a single extra
        //  face-a exchange restricted to b's dup positions and c's strict
        //  interior. Sender packs (i_a = dup-d_a-cell, i_b ∈ {ng, n_b-ng-1},
        //  i_c ∈ strict). Receiver applies at periodic image (i_a flipped,
        //  i_b same, i_c same) and sums. Post-pass sums equal the full
        //  cross-rank moment at every multi-rank corner-line.
        //
        //  3-axis (8-rank) corners are NOT closed here — would need an
        //  additional pass restricted to all three dups. Out of scope
        //  until a config exercises it.
        //! ============================================================
        {
            const bool xrank[3] = {
                periods[0] && dims[0] > 1,
                periods[1] && dims[1] > 1,
                periods[2] && dims[2] > 1
            };
            const int  nax[3]    = {nx, ny, nz};

            //* For each axis-pair (a, b), pick the c-perpendicular range:
            //*  - c is cross-rank: send strict-c only. c-ghost is fed by
            //*    Phase E.11's c-axis face slot from c's neighbour and is
            //*    NOT a local raw deposit at this point — including it
            //*    would double-count. (Note: 8-rank corners at c-dup ∩
            //*    a-dup ∩ b-dup are NOT closed by this pass.)
            //*  - c is single-rank periodic OR non-periodic: c-ghost holds
            //*    LOCAL raw partial deposits; subsequent c-fold (line
            //*    ~1060) folds c-ghost into c-strict and c-self-copy
            //*    overwrites c-ghost from c-strict. To make those steps
            //*    produce matched outputs at multi-rank corners, send
            //*    the FULL c range so c-ghost AND c-dup are matched
            //*    pre-fold.

            //* Tag scheme: 300 + axis_a*12 + ((sa+1)/2)*6 + axis_b*2 + dup_side.
            //* axis_a,axis_b ∈ {0,1,2}, sa ∈ {-1,+1}, dup_side ∈ {0=LO,1=HI}.
            //* Range: [300, 335], disjoint from face tags (1..6) and 26-slot
            //* SOR tags (200..226).
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

            //* Unordered pairs only (b > a). Iterating both (a,b) and (b,a)
            //* would double-count: pair (a,b) sa=±1 already updates ALL 4
            //* multi-rank corner cells of each rank for that axis-pair via
            //* the periodic-image flip on the a axis. Adding (b,a) sends
            //* would touch the same physical cells through the b axis,
            //* over-counting by p_off-rank-sum.
            for (int a = 0; a < 3; ++a) {
                if (!xrank[a]) continue;
                for (int b = a + 1; b < 3; ++b) {
                    if (!xrank[b]) continue;
                    const int c = 3 - a - b;
                    //* c-perp range: strict-c when c is cross-rank, full
                    //* (0..nc-1) when c is single-rank periodic / non-periodic.
                    const bool c_full = !xrank[c];
                    const int  len_c  = c_full ? nax[c] : (nax[c] - 2 * n_ghost_ - 2);
                    const int  c_lo   = c_full ? 0 : (n_ghost_ + 1);
                    if (len_c <= 0) continue;
                    for (int sa : {-1, +1}) {
                        int sxyz[3] = {0, 0, 0};
                        sxyz[a] = sa;
                        const int nbr = neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                        if (nbr == MPI_PROC_NULL || nbr == myrank) continue;
                        for (int dup_side = 0; dup_side < 2; ++dup_side) {
                            const int dup_b = (dup_side == 0)
                                              ? n_ghost_
                                              : (nax[b] - n_ghost_ - 1);
                            cslots.push_back({a, b, c, sa, dup_side, dup_b,
                                              len_c, c_lo, nbr, {}, {}});
                            CSlot &s = cslots.back();
                            s.send.resize(len_c);
                            s.recv.resize(len_c);
                            const int idx_a = (sa == -1)
                                              ? n_ghost_
                                              : (nax[a] - n_ghost_ - 1);
                            int ijk[3];
                            ijk[a] = idx_a;
                            ijk[b] = dup_b;
                            for (int hc = 0; hc < len_c; ++hc) {
                                ijk[c] = c_lo + hc;
                                s.send[hc] = vector[ijk[0]][ijk[1]][ijk[2]];
                            }
                        }
                    }
                }
            }

            std::vector<MPI_Request> crqs;
            crqs.reserve(2 * cslots.size());
            for (CSlot &s : cslots) {
                const int tag_send = ctag(s.axis_a,  s.sa, s.axis_b, s.dup_side);
                const int tag_recv = ctag(s.axis_a, -s.sa, s.axis_b, s.dup_side);
                crqs.emplace_back();
                MPI_Irecv(s.recv.data(), s.len_c, MPI_DOUBLE, s.nbr,
                          tag_recv, comm, &crqs.back());
                crqs.emplace_back();
                MPI_Isend(s.send.data(), s.len_c, MPI_DOUBLE, s.nbr,
                          tag_send, comm, &crqs.back());
            }
            if (!crqs.empty())
                MPI_Waitall(crqs.size(), crqs.data(), MPI_STATUSES_IGNORE);

            //* Apply: receiver sums into the SAME i_a dup position it sent
            //* from. The matching send across the rank boundary already
            //* targets the same PHYSICAL cell on our side: for sa=+1 we
            //* sent our HI X dup (i=nax-ng-1) and our X-HI Cart nbr's
            //* matching slot packed its LO X dup (i=ng) — same physical
            //* cell as our HI dup. The recv arrives bearing that
            //* neighbour's value summed at the same position, so we apply
            //* at our same dup-position. Same i_b dup, same i_c range.
            for (const CSlot &s : cslots) {
                const int idx_a = (s.sa == -1)
                                  ? n_ghost_
                                  : (nax[s.axis_a] - n_ghost_ - 1);
                int ijk[3];
                ijk[s.axis_a] = idx_a;
                ijk[s.axis_b] = s.dup_b;
                for (int hc = 0; hc < s.len_c; ++hc) {
                    ijk[s.axis_c] = s.c_lo + hc;
                    vector[ijk[0]][ijk[1]][ijk[2]] += s.recv[hc];
                }
            }
        }

        //! ============================================================
        //* FACE-CELL COMPLETION (Phase E.16)
        //
        //  At face cells (a-dup ∩ b-strict-near-boundary ∩ c) where the
        //  TSC stencil reaches across two cross-rank periodic axes, 4
        //  ranks contribute physically but Phase E.11's 26-slot partition
        //  routes only 3 of them to each rank. Specifically, for r0[i_a-dup,
        //  j_b-near-LO, k_c]: r0 self + r2 (across axis a) + r3 (diagonal)
        //  arrive via Phase E.11; r1's contribution at r1[i_a-dup, j_b-ghost,
        //  k_c] is missing because r1's pure-b slots cover only a-strict
        //  (lx=strict_x), and r1's diagonal slots route to r3, not r0.
        //
        //  Fix: for each ORDERED pair (a, b) of cross-rank axes (both (a,b)
        //  and (b,a) needed — they cover different cells), do an extra
        //  exchange along axis b restricted to (i_a = a-dup, j_b ∈ b-ghost,
        //  k_c strict-or-full). Sender packs at i_a-dup ∩ j_b-ghost; the
        //  matching neighbour-side send arrives via the periodic-image flip
        //  on axis b (dst formula uses -s.sb), landing at i_a-dup ∩
        //  j_b-strict-near-boundary, summing the missing diagonal rank's
        //  contribution.
        //
        //  Pack reads ghost cells (h_b ∈ [0..ng-1], strictly excludes b-dup
        //  at h_b=ng), which are NOT mutated by Phase E.11's apply (E.11
        //  reads ghost+dup, writes strict+dup) or Phase E.14 (writes only
        //  dup-pair cells). So pack timing is flexible — placed here for
        //  locality. Apply lands at strict-near-boundary cells, additively
        //  on top of any Phase E.11 contributions.
        //
        //  3-axis (X+Y+Z all cross-rank) is OUT OF SCOPE: at 8-rank cells
        //  the 6 ordered pairs combined with E.11's 4 routings would total
        //  10, over-counting. Correct only for ≤ 2 cross-rank axes.
        //! ============================================================
        {
            const bool xrank[3] = {
                periods[0] && dims[0] > 1,
                periods[1] && dims[1] > 1,
                periods[2] && dims[2] > 1
            };
            const int  nax[3]    = {nx, ny, nz};

            //* Tag: 400 + axis_a*16 + axis_b*4 + sa_side*2 + (sb+1)/2.
            //* axis_a, axis_b ∈ {0,1,2}, sa_side ∈ {0=LO,1=HI}, sb ∈ {-1,+1}.
            //* Range: [400, 443], disjoint from face tags (1..6),
            //* 26-slot SOR (200..226), and Phase E.14 (300..335).
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

            //* ORDERED pairs (a != b). (a,b) brings axis-b neighbour's
            //* (i_a-dup, j_b-ghost, k) contribution; (b,a) brings the
            //* axis-a neighbour's (i_b-dup, j_a-ghost, k) contribution.
            //* These are different physical cells, so both are required.
            for (int a = 0; a < 3; ++a) {
                if (!xrank[a]) continue;
                for (int b = 0; b < 3; ++b) {
                    if (b == a || !xrank[b]) continue;
                    const int c = 3 - a - b;
                    //* c-perp range: same convention as Phase E.14.
                    const bool c_full = !xrank[c];
                    const int  len_c  = c_full ? nax[c] : (nax[c] - 2 * n_ghost_ - 2);
                    const int  c_lo   = c_full ? 0 : (n_ghost_ + 1);
                    if (len_c <= 0) continue;

                    for (int sa_side = 0; sa_side < 2; ++sa_side) {
                        const int idx_a = (sa_side == 0)
                                          ? n_ghost_
                                          : (nax[a] - n_ghost_ - 1);
                        for (int sb : {-1, +1}) {
                            int sxyz[3] = {0, 0, 0};
                            sxyz[b] = sb;
                            const int nbr = neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                            if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

                            fslots.push_back({a, b, c, sa_side, sb,
                                              len_c, c_lo, nbr, {}, {}});
                            FSlot &s = fslots.back();
                            const int slab = n_ghost_ * len_c;
                            s.send.resize(slab);
                            s.recv.resize(slab);
                            int ijk[3];
                            ijk[a] = idx_a;
                            for (int h_b = 0; h_b < n_ghost_; ++h_b) {
                                ijk[b] = src_idx(sb, nax[b], h_b);
                                for (int hc = 0; hc < len_c; ++hc) {
                                    ijk[c] = c_lo + hc;
                                    s.send[h_b * len_c + hc] =
                                        vector[ijk[0]][ijk[1]][ijk[2]];
                                }
                            }
                        }
                    }
                }
            }

            std::vector<MPI_Request> frqs;
            frqs.reserve(2 * fslots.size());
            for (FSlot &s : fslots) {
                const int slab = n_ghost_ * s.len_c;
                const int tag_send = ftag(s.axis_a, s.axis_b, s.sa_side,  s.sb);
                const int tag_recv = ftag(s.axis_a, s.axis_b, s.sa_side, -s.sb);
                frqs.emplace_back();
                MPI_Irecv(s.recv.data(), slab, MPI_DOUBLE, s.nbr,
                          tag_recv, comm, &frqs.back());
                frqs.emplace_back();
                MPI_Isend(s.send.data(), slab, MPI_DOUBLE, s.nbr,
                          tag_send, comm, &frqs.back());
            }
            if (!frqs.empty())
                MPI_Waitall(frqs.size(), frqs.data(), MPI_STATUSES_IGNORE);

            //* Apply: matching neighbour's slot has -sb (we packed b-ghost
            //* on +sb side, neighbour packed b-ghost on -sb side). Receiver
            //* applies at idx_a (same dup-column position) and at j_b =
            //* dst_idx(-s.sb, nax[b], h_b), which lands on
            //* b-strict-near-boundary on the OPPOSITE side of the rank
            //* (periodic image of the sender's b-ghost).
            for (const FSlot &s : fslots) {
                const int idx_a = (s.sa_side == 0)
                                  ? n_ghost_
                                  : (nax[s.axis_a] - n_ghost_ - 1);
                int ijk[3];
                ijk[s.axis_a] = idx_a;
                for (int h_b = 0; h_b < n_ghost_; ++h_b) {
                    ijk[s.axis_b] = dst_idx(-s.sb, nax[s.axis_b], h_b);
                    for (int hc = 0; hc < s.len_c; ++hc) {
                        ijk[s.axis_c] = s.c_lo + hc;
                        vector[ijk[0]][ijk[1]][ijk[2]] +=
                            s.recv[h_b * s.len_c + hc];
                    }
                }
            }
        }
    }

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
    //* Send-side ghost-layer pairing. Receiver puts incoming layer g into
    //  ghost slot g where g=0 is OUTERMOST. The outermost ghost physically
    //  corresponds to the value FURTHEST from the active interior, so the
    //  matching send must come from depth (n_ghost-1) into the sender's
    //  interior — i.e., the same g_src(g) = (n_ghost-1-g) substitution
    //  Phase B applied to the periodic-self self-copy below. At n_ghost=1
    //  the substitution is a no-op (only one layer); at n_ghost=2 it swaps
    //  the two outgoing layers so receivers get the right depth in the right
    //  ghost slot.
    for (int g = 0; g < n_ghost_; g++) {
        const int s = g_src(g);
        if (communicationCnt[0])
            MPI_Isend(&vector[n_ghost_+offset+s][1][1],       1, yzFacetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
        if (communicationCnt[1])
            MPI_Isend(&vector[nx-1-n_ghost_-offset-s][1][1],  1, yzFacetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
        if (communicationCnt[2])
            MPI_Isend(&vector[1][n_ghost_+offset+s][1],       1, xzFacetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
        if (communicationCnt[3])
            MPI_Isend(&vector[1][ny-1-n_ghost_-offset-s][1],  1, xzFacetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
        if (communicationCnt[4])
            MPI_Isend(&vector[1][1][n_ghost_+offset+s],       1, xyFacetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
        if (communicationCnt[5])
            MPI_Isend(&vector[1][1][nz-1-n_ghost_-offset-s],  1, xyFacetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
    }
    assert_eq(recvcnt, sendcnt - recvcnt);
    assert_le(sendcnt, MAX_REQS);

    //* Sum-on-receive at periodic-self for needInterp=true (moments path).
    //  At TSC width the gather deposits to ghost cells; periodic-image sums
    //  recover those contributions before the overwrite below clobbers them.
    //  At Linear width the CIC stencil never reaches ghosts so this loop adds
    //  zero — strict no-op. The destination is the periodic-image interior:
    //    LO ghost vector[g]      → vector[g + nx - 2*n_ghost - offset]
    //    HI ghost vector[nx-1-g] → vector[(2*n_ghost - 1 + offset) - g]
    //  Verified for nodes (offset=1) and cells (offset=0). For n_ghost=1 this
    //  loop also adds zero on Linear because Linear never deposits to ghosts.
    const bool ps_x_self = (right_neighborX == myrank && left_neighborX == myrank);
    const bool ps_y_self = (right_neighborY == myrank && left_neighborY == myrank);
    const bool ps_z_self = (right_neighborZ == myrank && left_neighborZ == myrank);

    if (needInterp) {
        if (ps_x_self) {
            for (int g = 0; g < n_ghost_; g++) {
                const int dst_lo = g + nx - 2 * n_ghost_ - offset;
                const int dst_hi = (2 * n_ghost_ - 1 + offset) - g;
                for (int iy = 1; iy <= ny - 2; iy++)
                    for (int iz = 1; iz <= nz - 2; iz++) {
                        vector[dst_lo][iy][iz] += vector[g][iy][iz];
                        vector[dst_hi][iy][iz] += vector[nx - 1 - g][iy][iz];
                    }
            }
        }
        if (ps_y_self) {
            for (int g = 0; g < n_ghost_; g++) {
                const int dst_lo = g + ny - 2 * n_ghost_ - offset;
                const int dst_hi = (2 * n_ghost_ - 1 + offset) - g;
                for (int ix = 1; ix <= nx - 2; ix++)
                    for (int iz = 1; iz <= nz - 2; iz++) {
                        vector[ix][dst_lo][iz] += vector[ix][g][iz];
                        vector[ix][dst_hi][iz] += vector[ix][ny - 1 - g][iz];
                    }
            }
        }
        if (ps_z_self) {
            for (int g = 0; g < n_ghost_; g++) {
                const int dst_lo = g + nz - 2 * n_ghost_ - offset;
                const int dst_hi = (2 * n_ghost_ - 1 + offset) - g;
                for (int ix = 1; ix <= nx - 2; ix++)
                    for (int iy = 1; iy <= ny - 2; iy++) {
                        vector[ix][iy][dst_lo] += vector[ix][iy][g];
                        vector[ix][iy][dst_hi] += vector[ix][iy][nz - 1 - g];
                    }
            }
        }
    }

    //* Phase E.20a fix: face MPI Irecvs above wrote into vector[g][1..ny-2][1..nz-2]
    //  etc. The periodic single-rank self-copies below READ FROM those same
    //  buffers (e.g. ix=1..nx-2 includes ix=1 = X-LO inner-ghost slab written
    //  by X-Irecv when commCnt[0]=1). Without a Waitall here we have a race:
    //  whether self-copy reads pre-Irecv (zero) or post-Irecv (valid data)
    //  is non-deterministic across reps and was the dominant source of TSC
    //  np>1 cyc-1 ULP non-determinism.
    //
    //  Only run for COPY semantics (needInterp=false). The moments path
    //  (needInterp=true) intentionally reads pre-Irecv ghost values to do
    //  sum-on-receive folding before face Irecv overwrites them; it relies
    //  on the trailing Waitall further down.
    if (!needInterp && sendcnt > 0) {
        MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
    }

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
            int sL = nx - 1 - n_ghost_ - offset - g_src(g);
            if (sL < n_ghost_) sL += stride_x;
            int sR = n_ghost_ + offset + g_src(g);
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
            int sL = ny - 1 - n_ghost_ - offset - g_src(g);
            if (sL < n_ghost_) sL += stride_y;
            int sR = n_ghost_ + offset + g_src(g);
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
            int sL = nz - 1 - n_ghost_ - offset - g_src(g);
            if (sL < n_ghost_) sL += stride_z;
            int sR = n_ghost_ + offset + g_src(g);
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

        //? yEdge sends (mirror of recvs; source at interior depth n_ghost + offset + g_src(g))
        //  Same g→(n_ghost-1-g) substitution as the FACE sends above so each
        //  receiver ghost slot gets the matching depth from the sender.
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            const int sx = g_src(gx), sz = g_src(gz);
            if (communicationCnt[0]) {
                if (communicationCnt[4]) MPI_Isend(&vector[n_ghost_+offset+sx][1][n_ghost_+offset+sz],        1, yEdgetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
                if (communicationCnt[5]) MPI_Isend(&vector[n_ghost_+offset+sx][1][nz-1-n_ghost_-offset-sz],   1, yEdgetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[1]) {
                if (communicationCnt[4]) MPI_Isend(&vector[nx-1-n_ghost_-offset-sx][1][n_ghost_+offset+sz],       1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
                if (communicationCnt[5]) MPI_Isend(&vector[nx-1-n_ghost_-offset-sx][1][nz-1-n_ghost_-offset-sz],  1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
            }
        }

        //? zEdge sends
        for (int gx = 0; gx < n_ghost_; gx++)
        for (int gy = 0; gy < n_ghost_; gy++)
        {
            const int sx = g_src(gx), sy = g_src(gy);
            if (communicationCnt[2]) {
                if (communicationCnt[0]) MPI_Isend(&vector[n_ghost_+offset+sx][n_ghost_+offset+sy][1],         1, zEdgetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
                if (communicationCnt[1]) MPI_Isend(&vector[nx-1-n_ghost_-offset-sx][n_ghost_+offset+sy][1],    1, zEdgetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[3]) {
                if (communicationCnt[0]) MPI_Isend(&vector[n_ghost_+offset+sx][ny-1-n_ghost_-offset-sy][1],        1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
                if (communicationCnt[1]) MPI_Isend(&vector[nx-1-n_ghost_-offset-sx][ny-1-n_ghost_-offset-sy][1],   1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
            }
        }

        //? xEdge sends
        for (int gy = 0; gy < n_ghost_; gy++)
        for (int gz = 0; gz < n_ghost_; gz++)
        {
            const int sy = g_src(gy), sz = g_src(gz);
            if (communicationCnt[4]) {
                if (communicationCnt[2]) MPI_Isend(&vector[1][n_ghost_+offset+sy][n_ghost_+offset+sz],         1, xEdgetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
                if (communicationCnt[3]) MPI_Isend(&vector[1][ny-1-n_ghost_-offset-sy][n_ghost_+offset+sz],    1, xEdgetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
            }
            if (communicationCnt[5]) {
                if (communicationCnt[2]) MPI_Isend(&vector[1][n_ghost_+offset+sy][nz-1-n_ghost_-offset-sz],        1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
                if (communicationCnt[3]) MPI_Isend(&vector[1][ny-1-n_ghost_-offset-sy][nz-1-n_ghost_-offset-sz],   1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
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
                const int sL = nx - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                if (right_neighborZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[g][iy][nz-1]         = vector[sL][iy][nz-1];
                        vector[nx-1-g][iy][nz-1]    = vector[sR][iy][nz-1];
                    }
                }
                if (left_neighborZ != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[g][iy][0]         = vector[sL][iy][0];
                        vector[nx-1-g][iy][0]    = vector[sR][iy][0];
                    }
                }
                if (right_neighborY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[g][ny-1][iz]         = vector[sL][ny-1][iz];
                        vector[nx-1-g][ny-1][iz]    = vector[sR][ny-1][iz];
                    }
                }
                if (left_neighborY != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[g][0][iz]         = vector[sL][0][iz];
                        vector[nx-1-g][0][iz]    = vector[sR][0][iz];
                    }
                }
            }
        }
        if (right_neighborY == myrank && left_neighborY == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                const int sL = ny - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                if (right_neighborX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[nx-1][g][iz]         = vector[nx-1][sL][iz];
                        vector[nx-1][ny-1-g][iz]    = vector[nx-1][sR][iz];
                    }
                }
                if (left_neighborX != MPI_PROC_NULL) {
                    for (int iz = 1; iz < nz - 1; iz++) {
                        vector[0][g][iz]         = vector[0][sL][iz];
                        vector[0][ny-1-g][iz]    = vector[0][sR][iz];
                    }
                }
                if (right_neighborZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][g][nz-1]         = vector[ix][sL][nz-1];
                        vector[ix][ny-1-g][nz-1]    = vector[ix][sR][nz-1];
                    }
                }
                if (left_neighborZ != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][g][0]         = vector[ix][sL][0];
                        vector[ix][ny-1-g][0]    = vector[ix][sR][0];
                    }
                }
            }
        }
        if (right_neighborZ == myrank && left_neighborZ == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                const int sL = nz - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                if (right_neighborY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][ny-1][g]         = vector[ix][ny-1][sL];
                        vector[ix][ny-1][nz-1-g]    = vector[ix][ny-1][sR];
                    }
                }
                if (left_neighborY != MPI_PROC_NULL) {
                    for (int ix = 1; ix < nx - 1; ix++) {
                        vector[ix][0][g]         = vector[ix][0][sL];
                        vector[ix][0][nz-1-g]    = vector[ix][0][sR];
                    }
                }
                if (right_neighborX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[nx-1][iy][g]         = vector[nx-1][iy][sL];
                        vector[nx-1][iy][nz-1-g]    = vector[nx-1][iy][sR];
                    }
                }
                if (left_neighborX != MPI_PROC_NULL) {
                    for (int iy = 1; iy < ny - 1; iy++) {
                        vector[0][iy][g]         = vector[0][iy][sL];
                        vector[0][iy][nz-1-g]    = vector[0][iy][sR];
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
                //* g→(n_ghost-1-g) substitution applied to send sources so the
                //  outermost-ghost RECV slot lines up with the deepest-interior
                //  source — matches the FACE/EDGE phases above. No-op at n_ghost=1.
                const int sx = g_src(gx), sy = g_src(gy), sz = g_src(gz);
                const int xs_l = n_ghost_ + offset + sx;            //* X source index for left send
                const int xs_r = nx - 1 - n_ghost_ - offset - sx;   //* X source index for right send
                const int ys_l = n_ghost_ + offset + sy;
                const int ys_r = ny - 1 - n_ghost_ - offset - sy;
                const int zs_l = n_ghost_ + offset + sz;
                const int zs_r = nz - 1 - n_ghost_ - offset - sz;

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

        //! ============================================================
        //* DIAGONAL CART EDGE-COPY (Phase E.18)
        //
        //  At a 2-axis cross-rank rank, the standard EDGE PHASE above
        //  routes through a SINGLE-axis Cart neighbour (e.g., zEdge LO X
        //  ∩ LO Y corner gets data from left_neighborY only). The data on
        //  that single-axis neighbour at the read position physically
        //  represents a DIFFERENT cell than what the corner ghost should
        //  hold — leading to ~1e-3 paired-rank inconsistency at the edge
        //  ghost. The mass-matrix kernel reads these edge ghosts when the
        //  TSC |offset|=2 stencil reaches them, propagating the error to
        //  C_raw_ME at face dup pairs (~1e-6).
        //
        //  Fix: for each pair (a, b) of cross-rank periodic axes, exchange
        //  the (sa, sb) corner via the DIAGONAL Cart neighbour — which
        //  geometrically holds the correct physical cell — and OVERRIDE
        //  the standard EDGE PHASE write. COPY semantics (= overwrite).
        //  Range over c (third axis) covers strict + dup so all stencil
        //  reaches are covered; ghost-c excluded since FACE PHASE handles
        //  that direction separately.
        //
        //  3-axis (X+Y+Z all cross-rank) 8-rank corners are NOT closed by
        //  this pass — would need an additional 3-axis CORNER override.
        //! ============================================================
        if (any_xrank_periodic) {
            //* Reuse the topology + neighbour_at lambda from function scope.
            const bool xrank_e18[3] = {
                periods[0] && dims[0] > 1,
                periods[1] && dims[1] > 1,
                periods[2] && dims[2] > 1
            };
            const int  nax_e18[3]    = {nx, ny, nz};

            //* Tag scheme: 500 + axis_a*64 + axis_b*16 + (sa+1)/2*8 + (sb+1)/2*4
            //*             + 0..3 for the 4 (sa, sb) combos (already encoded above).
            //* Pair indices: (a, b) ∈ {(0,1), (0,2), (1,2)} unordered. Use the
            //* compact form: 500 + (axis_a*3 + axis_b) * 16 + ((sa+1)/2)*8 + ((sb+1)/2)*4.
            //* Range fits in [500, 535]: 3 pair entries * 16 each = 48 → max 547.
            //* Disjoint from 1..6 face, 200..226, 300..335, 400..443.
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
                if (!xrank_e18[a]) continue;
                for (int b = a + 1; b < 3; ++b) {
                    if (!xrank_e18[b]) continue;
                    const int c = 3 - a - b;
                    //* c-perp range: cover c-strict + c-dup so the kernel's
                    //* stencil reaches into c are valid. c-ghost is filled
                    //* separately by FACE PHASE for the c axis.
                    const int len_c = nax_e18[c] - 2;  // [1..nax-2]: strict + dup
                    const int c_lo  = 1;
                    if (len_c <= 0) continue;

                    for (int sa : {-1, +1}) {
                        for (int sb : {-1, +1}) {
                            int sxyz[3] = {0, 0, 0};
                            sxyz[a] = sa;
                            sxyz[b] = sb;
                            const int nbr = neighbour_at(sxyz[0], sxyz[1], sxyz[2]);
                            if (nbr == MPI_PROC_NULL || nbr == myrank) continue;

                            dslots.push_back({a, b, c, sa, sb, len_c, c_lo,
                                              nbr, {}, {}});
                            DSlot &s = dslots.back();
                            const int slab = n_ghost_ * n_ghost_ * len_c;
                            s.send.resize(slab);
                            s.recv.resize(slab);

                            //* Pack: at (sa, sb) strict-near-bdry of self.
                            //* For sa=+1: ijk[a] = nax-1-ng-offset-(ng-1-ga).
                            //* For sa=-1: ijk[a] = ng+offset+(ng-1-ga).
                            //* g_src(ga) = ng-1-ga. So ijk[a] = (sa==+1)
                            //* ? nax-1-ng-offset-g_src(ga) : ng+offset+g_src(ga).
                            int ijk[3];
                            for (int ga = 0; ga < n_ghost_; ++ga) {
                                ijk[a] = (sa == +1)
                                         ? (nax_e18[a] - 1 - n_ghost_ - offset
                                            - (n_ghost_ - 1 - ga))
                                         : (n_ghost_ + offset
                                            + (n_ghost_ - 1 - ga));
                                for (int gb = 0; gb < n_ghost_; ++gb) {
                                    ijk[b] = (sb == +1)
                                             ? (nax_e18[b] - 1 - n_ghost_ - offset
                                                - (n_ghost_ - 1 - gb))
                                             : (n_ghost_ + offset
                                                + (n_ghost_ - 1 - gb));
                                    for (int hc = 0; hc < len_c; ++hc) {
                                        ijk[c] = c_lo + hc;
                                        s.send[(ga * n_ghost_ + gb) * len_c + hc] =
                                            vector[ijk[0]][ijk[1]][ijk[2]];
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
                const int slab = n_ghost_ * n_ghost_ * s.len_c;
                const int tag_send = detag(s.axis_a, s.axis_b,  s.sa,  s.sb);
                const int tag_recv = detag(s.axis_a, s.axis_b, -s.sa, -s.sb);
                drqs.emplace_back();
                MPI_Irecv(s.recv.data(), slab, MPI_DOUBLE, s.nbr,
                          tag_recv, comm, &drqs.back());
                drqs.emplace_back();
                MPI_Isend(s.send.data(), slab, MPI_DOUBLE, s.nbr,
                          tag_send, comm, &drqs.back());
            }
            if (!drqs.empty())
                MPI_Waitall(drqs.size(), drqs.data(), MPI_STATUSES_IGNORE);

            //* Apply: COPY (=) into (sa, sb) ghost corner. Override the
            //* wrong value written by the standard EDGE PHASE.
            for (const DSlot &s : dslots) {
                int ijk[3];
                for (int ga = 0; ga < n_ghost_; ++ga) {
                    ijk[s.axis_a] = (s.sa == -1) ? ga : (nax_e18[s.axis_a] - 1 - ga);
                    for (int gb = 0; gb < n_ghost_; ++gb) {
                        ijk[s.axis_b] = (s.sb == -1) ? gb : (nax_e18[s.axis_b] - 1 - gb);
                        for (int hc = 0; hc < s.len_c; ++hc) {
                            ijk[s.axis_c] = s.c_lo + hc;
                            vector[ijk[0]][ijk[1]][ijk[2]] =
                                s.recv[(ga * n_ghost_ + gb) * s.len_c + hc];
                        }
                    }
                }
            }
        }

        //* Periodic single-rank corner self-copies.
        //  Generalized from legacy L437-494. Uses else-if like legacy: for np=1
        //  (all periodic) the first branch handles all 8 corner cubes via the
        //  cascade face->edge->corner. Source cells were filled by edge self-copies.
        if (left_neighborX == myrank && right_neighborX == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                const int sL = nx - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                for (int gy = 0; gy < n_ghost_; gy++)
                for (int gz = 0; gz < n_ghost_; gz++) {
                    if (left_neighborY != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                        vector[g][gy][gz]                   = vector[sL][gy][gz];
                        vector[nx-1-g][gy][gz]              = vector[sR][gy][gz];
                    }
                    if (left_neighborY != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                        vector[g][gy][nz-1-gz]              = vector[sL][gy][nz-1-gz];
                        vector[nx-1-g][gy][nz-1-gz]         = vector[sR][gy][nz-1-gz];
                    }
                    if (right_neighborY != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                        vector[g][ny-1-gy][gz]              = vector[sL][ny-1-gy][gz];
                        vector[nx-1-g][ny-1-gy][gz]         = vector[sR][ny-1-gy][gz];
                    }
                    if (right_neighborY != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                        vector[g][ny-1-gy][nz-1-gz]         = vector[sL][ny-1-gy][nz-1-gz];
                        vector[nx-1-g][ny-1-gy][nz-1-gz]    = vector[sR][ny-1-gy][nz-1-gz];
                    }
                }
            }
        }
        else if (left_neighborY == myrank && right_neighborY == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                const int sL = ny - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                for (int gx = 0; gx < n_ghost_; gx++)
                for (int gz = 0; gz < n_ghost_; gz++) {
                    if (left_neighborX != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                        vector[gx][g][gz]                   = vector[gx][sL][gz];
                        vector[gx][ny-1-g][gz]              = vector[gx][sR][gz];
                    }
                    if (left_neighborX != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                        vector[gx][g][nz-1-gz]              = vector[gx][sL][nz-1-gz];
                        vector[gx][ny-1-g][nz-1-gz]         = vector[gx][sR][nz-1-gz];
                    }
                    if (right_neighborX != MPI_PROC_NULL && left_neighborZ != MPI_PROC_NULL) {
                        vector[nx-1-gx][g][gz]              = vector[nx-1-gx][sL][gz];
                        vector[nx-1-gx][ny-1-g][gz]         = vector[nx-1-gx][sR][gz];
                    }
                    if (right_neighborX != MPI_PROC_NULL && right_neighborZ != MPI_PROC_NULL) {
                        vector[nx-1-gx][g][nz-1-gz]         = vector[nx-1-gx][sL][nz-1-gz];
                        vector[nx-1-gx][ny-1-g][nz-1-gz]    = vector[nx-1-gx][sR][nz-1-gz];
                    }
                }
            }
        }
        else if (left_neighborZ == myrank && right_neighborZ == myrank) {
            for (int g = 0; g < n_ghost_; g++) {
                const int sL = nz - 1 - n_ghost_ - offset - g_src(g);
                const int sR = n_ghost_ + offset + g_src(g);
                for (int gx = 0; gx < n_ghost_; gx++)
                for (int gy = 0; gy < n_ghost_; gy++) {
                    if (left_neighborX != MPI_PROC_NULL && left_neighborY != MPI_PROC_NULL) {
                        vector[gx][gy][g]                   = vector[gx][gy][sL];
                        vector[gx][gy][nz-1-g]              = vector[gx][gy][sR];
                    }
                    if (left_neighborX != MPI_PROC_NULL && right_neighborY != MPI_PROC_NULL) {
                        vector[gx][ny-1-gy][g]              = vector[gx][ny-1-gy][sL];
                        vector[gx][ny-1-gy][nz-1-g]         = vector[gx][ny-1-gy][sR];
                    }
                    if (right_neighborX != MPI_PROC_NULL && left_neighborY != MPI_PROC_NULL) {
                        vector[nx-1-gx][gy][g]              = vector[nx-1-gx][gy][sL];
                        vector[nx-1-gx][gy][nz-1-g]         = vector[nx-1-gx][gy][sR];
                    }
                    if (right_neighborX != MPI_PROC_NULL && right_neighborY != MPI_PROC_NULL) {
                        vector[nx-1-gx][ny-1-gy][g]         = vector[nx-1-gx][ny-1-gy][sL];
                        vector[nx-1-gx][ny-1-gy][nz-1-g]    = vector[nx-1-gx][ny-1-gy][sR];
                    }
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
    //* CIC (n_ghost=1) only — at TSC the cross-rank SOR pre-pass above
    //* already accumulated ghost natives into receivers' strict + dup, so
    //* running the trailing addFace family on top would double-count. The
    //* `skip_self_periodic=true` argument keeps the legacy CIC path from
    //* re-summing periodic-self ghosts that the periodic-self fold/copy
    //* above already handled.
    if (needInterp && n_ghost_ == 1) {
        if (vector_c != nullptr) {
            addFace_kahan  (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeZ_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeY_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addEdgeX_kahan (nx, ny, nz, vector, vector_c, vct, n_ghost_);
            addCorner_kahan(nx, ny, nz, vector, vector_c, vct, n_ghost_);
        } else {
            addFace  (nx, ny, nz, vector, vct, n_ghost_, /*skip_self*/ true);
            addEdgeZ (nx, ny, nz, vector, vct, n_ghost_, /*skip_self*/ true);
            addEdgeY (nx, ny, nz, vector, vct, n_ghost_, /*skip_self*/ true);
            addEdgeX (nx, ny, nz, vector, vct, n_ghost_, /*skip_self*/ true);
            addCorner(nx, ny, nz, vector, vct, n_ghost_, /*skip_self*/ true);
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
void addFace(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    //* Per-axis periodic-self skip. When `skip_self_periodic` is on AND the
    //* axis's left and right neighbours are myrank, the periodic-self fold +
    //* copy upstream already handled that direction's halo sum; running addFace
    //* here would double-count.
    const int myrank = vct->getCartesian_rank();
    const bool skip_x = skip_self_periodic &&
        (vct->getXleft_neighbor_P() == myrank && vct->getXright_neighbor_P() == myrank);
    const bool skip_y = skip_self_periodic &&
        (vct->getYleft_neighbor_P() == myrank && vct->getYright_neighbor_P() == myrank);
    const bool skip_z = skip_self_periodic &&
        (vct->getZleft_neighbor_P() == myrank && vct->getZright_neighbor_P() == myrank);

    //* Perpendicular ranges widened to [1, n-2] to match the widened MPI face
    //  types. At n_ghost==1 this is [1, n-2] — unchanged. At n_ghost==2 it
    //  captures inner-ghost-row moment contributions that were previously missed.
    for (int g = 0; g < n_ghost; g++) {
        // Xright
        if (vct->hasXrghtNeighbor_P() && !skip_x)
        {
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[nx - 1 - n_ghost - g][j][k] += vector[nx - 1 - g][j][k];
        }
        // XLEFT
        if (vct->hasXleftNeighbor_P() && !skip_x)
        {
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[n_ghost + g][j][k] += vector[g][j][k];
        }

        // Yright
        if (vct->hasYrghtNeighbor_P() && !skip_y)
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[i][ny - 1 - n_ghost - g][k] += vector[i][ny - 1 - g][k];
        }
        // Yleft
        if (vct->hasYleftNeighbor_P() && !skip_y)
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    vector[i][n_ghost + g][k] += vector[i][g][k];
        }
        // Zright
        if (vct->hasZrghtNeighbor_P() && !skip_z)
        {
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    vector[i][j][nz - 1 - n_ghost - g] += vector[i][j][nz - 1 - g];
        }
        // ZLEFT
        if (vct->hasZleftNeighbor_P() && !skip_z)
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
void addEdgeZ(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const int myrank = vct->getCartesian_rank();
    const bool skip_x = skip_self_periodic &&
        (vct->getXleft_neighbor_P() == myrank && vct->getXright_neighbor_P() == myrank);
    const bool skip_y = skip_self_periodic &&
        (vct->getYleft_neighbor_P() == myrank && vct->getYright_neighbor_P() == myrank);

    //* For an edge to need addEdge, BOTH cross-section axes must contribute a
    //* halo overlap. If either axis is periodic-self, the upstream periodic-self
    //* fold+copy already merged that pair (the corresponding 2-D sum lives at the
    //* cross-section corner of the duplicates).
    for (int gx = 0; gx < n_ghost; gx++)
    for (int gy = 0; gy < n_ghost; gy++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasYrghtNeighbor_P() && !(skip_x || skip_y))
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[nx - 1 - n_ghost - gx][ny - 1 - n_ghost - gy][i] += vector[nx - 1 - gx][ny - 1 - gy][i];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasYleftNeighbor_P() && !(skip_x || skip_y))
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[n_ghost + gx][n_ghost + gy][i] += vector[gx][gy][i];
        }
        if (vct->hasXrghtNeighbor_P() && vct->hasYleftNeighbor_P() && !(skip_x || skip_y))
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[nx - 1 - n_ghost - gx][n_ghost + gy][i] += vector[nx - 1 - gx][gy][i];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasYrghtNeighbor_P() && !(skip_x || skip_y))
        {
            for (int i = 1; i < (nz - 1); i++)
                vector[n_ghost + gx][ny - 1 - n_ghost - gy][i] += vector[gx][ny - 1 - gy][i];
        }
    }
}
/** add the ghost cell values Edge Y to the 3D physical vector */
//  Y-aligned edge: ghost block lives in the (x, z) cross-section.
void addEdgeY(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const int myrank = vct->getCartesian_rank();
    const bool skip_x = skip_self_periodic &&
        (vct->getXleft_neighbor_P() == myrank && vct->getXright_neighbor_P() == myrank);
    const bool skip_z = skip_self_periodic &&
        (vct->getZleft_neighbor_P() == myrank && vct->getZright_neighbor_P() == myrank);

    for (int gx = 0; gx < n_ghost; gx++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasXrghtNeighbor_P() && vct->hasZrghtNeighbor_P() && !(skip_x || skip_z))
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[nx - 1 - n_ghost - gx][i][nz - 1 - n_ghost - gz] += vector[nx - 1 - gx][i][nz - 1 - gz];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasZleftNeighbor_P() && !(skip_x || skip_z))
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[n_ghost + gx][i][n_ghost + gz] += vector[gx][i][gz];
        }
        if (vct->hasXleftNeighbor_P() && vct->hasZrghtNeighbor_P() && !(skip_x || skip_z))
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[n_ghost + gx][i][nz - 1 - n_ghost - gz] += vector[gx][i][nz - 1 - gz];
        }
        if (vct->hasXrghtNeighbor_P() && vct->hasZleftNeighbor_P() && !(skip_x || skip_z))
        {
            for (int i = 1; i < (ny - 1); i++)
                vector[nx - 1 - n_ghost - gx][i][n_ghost + gz] += vector[nx - 1 - gx][i][gz];
        }
    }
}

/** add the ghost values Edge X to the 3D physical vector */
//  X-aligned edge: ghost block lives in the (y, z) cross-section.
void addEdgeX(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const int myrank = vct->getCartesian_rank();
    const bool skip_y = skip_self_periodic &&
        (vct->getYleft_neighbor_P() == myrank && vct->getYright_neighbor_P() == myrank);
    const bool skip_z = skip_self_periodic &&
        (vct->getZleft_neighbor_P() == myrank && vct->getZright_neighbor_P() == myrank);

    for (int gy = 0; gy < n_ghost; gy++)
    for (int gz = 0; gz < n_ghost; gz++)
    {
        if (vct->hasYrghtNeighbor_P() && vct->hasZrghtNeighbor_P() && !(skip_y || skip_z)) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][ny - 1 - n_ghost - gy][nz - 1 - n_ghost - gz] += vector[i][ny - 1 - gy][nz - 1 - gz];
        }
        if (vct->hasYleftNeighbor_P() && vct->hasZleftNeighbor_P() && !(skip_y || skip_z)) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][n_ghost + gy][n_ghost + gz] += vector[i][gy][gz];
        }
        if (vct->hasYleftNeighbor_P() && vct->hasZrghtNeighbor_P() && !(skip_y || skip_z)) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][n_ghost + gy][nz - 1 - n_ghost - gz] += vector[i][gy][nz - 1 - gz];
        }
        if (vct->hasYrghtNeighbor_P() && vct->hasZleftNeighbor_P() && !(skip_y || skip_z)) {
            for (int i = 1; i < (nx - 1); i++)
                vector[i][ny - 1 - n_ghost - gy][n_ghost + gz] += vector[i][ny - 1 - gy][gz];
        }
    }
}

/** add ghost cells values Corners in the 3D physical vector */
//  Each corner is a single node for n_ghost == 1; for n_ghost > 1 it expands
//  to a (gx, gy, gz) cube of nodes that are summed back into the matching
//  inner interior corner.
void addCorner(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic)
{
    const int myrank = vct->getCartesian_rank();
    const bool skip_x = skip_self_periodic &&
        (vct->getXleft_neighbor_P() == myrank && vct->getXright_neighbor_P() == myrank);
    const bool skip_y = skip_self_periodic &&
        (vct->getYleft_neighbor_P() == myrank && vct->getYright_neighbor_P() == myrank);
    const bool skip_z = skip_self_periodic &&
        (vct->getZleft_neighbor_P() == myrank && vct->getZright_neighbor_P() == myrank);

    //* A corner participates only when ALL three axes contribute halo overlap;
    //* if any of those axes is periodic-self, the upstream fold + copy already
    //* merged that triple-overlap.
    const bool any_skip = skip_x || skip_y || skip_z;
    if (any_skip) return;

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
//  Kahan-compensated parallel helpers for the sum-on-receive path.
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

//* Kahan-compensated variants. Forward the companion array into
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
