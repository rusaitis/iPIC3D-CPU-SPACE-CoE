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
//* Multi-field signature: `vectors` is an array of `n_fields` pointers to
//* 3D field arrays sharing the same (nx, ny, nz) extents and same MPI
//* topology. All N fields are exchanged in one batched MPI message per
//* direction (slab × N_fields doubles). At n_fields=1 this is bit-
//* identical to the legacy single-field path. Optional `vectors_c`
//* (Kahan companions) follows the same shape — pass nullptr for legacy
//* COPY semantics, or a non-null array of N companion pointers to
//* enable Neumaier compensation in the trailing addFace at n_ghost=1.
static void NBDerivedHaloCommN(int nx, int ny, int nz,
                                int n_fields, double ****vectors,
                                const VirtualTopology3D * vct, EMfields3D *EMf,
                                bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                                double ****vectors_c = nullptr);

//! isCenterFlag: 1 = communicateCenter; 0 = communicateNode
//! `vector_c` is the optional Kahan-compensation companion array. `= nullptr`
//! is the legacy behaviour (byte-identical); a non-null pointer routes the
//! sum-on-receive step (`addFace`/`addEdge*`/`addCorner`) through a Neumaier-
//! compensated update that lands residuals in `vector_c`.
void NBDerivedHaloComm(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, EMfields3D *EMf,
                        bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                        double ***vector_c = nullptr)
{
    //* Wider ghost-slab path goes through the loop-based helper. n_ghost == 1
    //* falls through to the legacy merged-datatype fast path below.
    //* Wrap the single-field pointer in a 1-element array so the
    //* multi-field helper (NBDerivedHaloCommN) sees vectors[0] = vector.
    if (EMf->getNGhost() > 1) {
        double*** vectors[1] = {vector};
        if (vector_c != nullptr) {
            double*** vectors_c[1] = {vector_c};
            NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf,
                               isCenterFlag, isFaceOnlyFlag, needInterp, isParticle, vectors_c);
        } else {
            NBDerivedHaloCommN(nx, ny, nz, 1, vectors, vct, EMf,
                               isCenterFlag, isFaceOnlyFlag, needInterp, isParticle, nullptr);
        }
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
	    //* Kahan-aware sum-on-receive. Legacy path (`vector_c == nullptr`)
	    //* falls through to plain `+=`, byte-identical to pre-Step-68c.
	    //* When `vector_c` is supplied, each receiving interior cell is
	    //* updated via a Neumaier step with the residual landing in
	    //* `vector_c`; the caller folds `vector_c` back into `vector`
	    //* after the halo exchange.
	    addFace  (nx, ny, nz, vector, vct, /*n_ghost=*/1, /*skip_self=*/false, vector_c);
	    addEdgeZ (nx, ny, nz, vector, vct, /*n_ghost=*/1, /*skip_self=*/false, vector_c);
	    addEdgeY (nx, ny, nz, vector, vct, /*n_ghost=*/1, /*skip_self=*/false, vector_c);
	    addEdgeX (nx, ny, nz, vector, vct, /*n_ghost=*/1, /*skip_self=*/false, vector_c);
	    addCorner(nx, ny, nz, vector, vct, /*n_ghost=*/1, /*skip_self=*/false, vector_c);
	}

}


//! ================================================================================
//* HaloContext: state computed once per NBDerivedHaloCommN call, shared by the
//* (future) multi-field variant. Holds n_ghost, the MPI communicator, the
//* per-axis neighbour ranks, the Cartesian topology coords/dims/periods,
//* and the offset/isCenterDim derived from the call's flags. Carries the
//* two helpers (g_src and neighbour_at) as methods so the multi-field
//* path can reuse them.
//! ================================================================================
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

    //* Field semantics.
    bool isCenterDim;
    int  offset;

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
    ctx.offset      = ctx.isCenterDim ? 0 : 1;

    ctx.ps_x_self = (ctx.neighbours[0] == ctx.myrank && ctx.neighbours[1] == ctx.myrank);
    ctx.ps_y_self = (ctx.neighbours[2] == ctx.myrank && ctx.neighbours[3] == ctx.myrank);
    ctx.ps_z_self = (ctx.neighbours[4] == ctx.myrank && ctx.neighbours[5] == ctx.myrank);

    return ctx;
}

}  // namespace

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
//! ================================================================================

//* ============================================================================
//* Phase E.18 — DIAGONAL CART EDGE-COPY
//*
//* At a 2-axis cross-rank rank, the standard EDGE PHASE routes each edge
//* through a SINGLE-axis Cart neighbour, which physically holds the WRONG cell
//* for diagonal corners. This pass exchanges the (sa, sb) corner via the
//* DIAGONAL Cart neighbour (which geometrically holds the correct cell) and
//* OVERRIDES the standard EDGE PHASE write with COPY semantics. Range over c
//* covers strict + dup so all stencil reaches are valid; ghost-c is excluded
//* (FACE PHASE handles the c-axis separately). 3-axis (X+Y+Z) 8-rank corners
//* are NOT closed here — would need an additional 3-axis CORNER override.
//*
//* No-op unless `any_xrank_periodic`. Tag base 500.
//* ============================================================================
static void do_diagonal_edge_copy(int nx, int ny, int nz,
                                   int n_fields, double ****vectors,
                                   const HaloContext &ctx)
{
    if (!ctx.any_xrank_periodic) return;

    const int       n_ghost  = ctx.n_ghost;
    const int       offset   = ctx.offset;
    const MPI_Comm  comm     = ctx.comm;
    const int       myrank   = ctx.myrank;
    const int* const dims    = ctx.dims;
    const int* const periods = ctx.periods;

    const bool xrank[3] = {
        periods[0] && dims[0] > 1,
        periods[1] && dims[1] > 1,
        periods[2] && dims[2] > 1
    };
    const int  nax[3]   = {nx, ny, nz};

    //* Tag scheme: 500 + (axis_a*3 + axis_b)*16 + ((sa+1)/2)*8 + ((sb+1)/2)*4.
    //* Range fits in [500, 547]. Disjoint from face tags (1..6), 26-slot SOR
    //* (200..226), Phase E.14 (300..335), and Phase E.16 (400..443).
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
                                     ? (nax[a] - 1 - n_ghost - offset
                                        - (n_ghost - 1 - ga))
                                     : (n_ghost + offset
                                        + (n_ghost - 1 - ga));
                            for (int gb = 0; gb < n_ghost; ++gb) {
                                ijk[b] = (sb == +1)
                                         ? (nax[b] - 1 - n_ghost - offset
                                            - (n_ghost - 1 - gb))
                                         : (n_ghost + offset
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

//* ============================================================================
//* CROSS-RANK MOMENT COMPLETION (Phases A + B + C)
//*
//* Three nested passes that close the cross-rank moment-halo at TSC width:
//*   A) 26-slot SOR pre-pass — partitions each rank's ghost region into 26
//*      disjoint slabs indexed by (sx,sy,sz) ∈ {-1,0,+1}³ and sends each to
//*      its Cartesian neighbour. Receiver SORs (sums) the slab into the
//*      matching destination strict-interior + LO/HI duplicate. Tag base 200.
//*   B) Phase E.14 multi-axis corner completion — for each unordered pair
//*      (a,b) of cross-rank axes, an extra face-a exchange restricted to
//*      b-dup positions closes the 4-rank corner where SOR's diagonal slot
//*      only paired 2 ranks. Tag base 300.
//*   C) Phase E.16 face-cell completion — for each ORDERED pair (a,b) of
//*      cross-rank axes, an extra exchange along axis b restricted to
//*      (i_a = a-dup, j_b ∈ b-ghost) brings the 4th rank's contribution
//*      to the face-strict-near-bdry cell. Tag base 400.
//*
//* All three phases are no-ops outside (needInterp && n_ghost > 1 &&
//* any_xrank_periodic). They share src_idx/dst_idx index helpers, hence the
//* combined helper.
//* ============================================================================
static void do_xrank_completion(int nx, int ny, int nz,
                                 int n_fields, double ****vectors,
                                 const HaloContext &ctx, bool needInterp)
{
    if (!(needInterp && ctx.n_ghost > 1 && ctx.any_xrank_periodic)) return;

    const int       n_ghost  = ctx.n_ghost;
    const int       offset   = ctx.offset;
    const MPI_Comm  comm     = ctx.comm;
    const int       myrank   = ctx.myrank;
    const int* const dims    = ctx.dims;
    const int* const periods = ctx.periods;

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
            return (h < n_ghost) ? (h + n - 2 * n_ghost - offset) : (n - 1 - n_ghost);
        }
        if (sd == +1) {
            return (h < n_ghost) ? ((2 * n_ghost - 1 + offset) - h) : n_ghost;
        }
        return n_ghost + 1 + h;
    };

    //! ============================================================
    //* Phase A — 26-slot CROSS-RANK MOMENT SOR PRE-PASS
    //! ============================================================
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

    //! ============================================================
    //* Phase E.14 — MULTI-AXIS CORNER COMPLETION
    //! ============================================================
    {
        const bool xrank[3] = {
            periods[0] && dims[0] > 1,
            periods[1] && dims[1] > 1,
            periods[2] && dims[2] > 1
        };
        const int  nax[3]   = {nx, ny, nz};

        //* Tag scheme: 300 + axis_a*12 + ((sa+1)/2)*6 + axis_b*2 + dup_side.
        //* Range [300, 335], disjoint from face tags (1..6) and Phase A (200..226).
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

    //! ============================================================
    //* Phase E.16 — FACE-CELL COMPLETION
    //! ============================================================
    {
        const bool xrank[3] = {
            periods[0] && dims[0] > 1,
            periods[1] && dims[1] > 1,
            periods[2] && dims[2] > 1
        };
        const int  nax[3]   = {nx, ny, nz};

        //* Tag: 400 + axis_a*16 + axis_b*4 + sa_side*2 + (sb+1)/2.
        //* Range [400, 443], disjoint from face (1..6), Phase A (200..226),
        //* Phase E.14 (300..335).
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

//* ============================================================================
//* EDGE PHASE — manual pack/unpack at N=1+
//*
//* Each axis has 4 edges (corners of the perpendicular face). For n_ghost > 1
//* each corner expands to an n_ghost x n_ghost block in the cross-section
//* perpendicular to the edge direction. Routing: yEdge → X neighbours,
//* zEdge → Y neighbours, xEdge → Z neighbours; each Cart side carries its
//* 2 perp corners (e.g., yEdge X-LO carries Z-LO + Z-HI corners) batched
//* into one MPI message. Replaces the strided yEdgetype / zEdgetype /
//* xEdgetype custom datatypes.
//*
//* Bit-identicality vs the legacy MPI-types path: legacy ran edge self-copies
//* BEFORE the EDGE Waitall (Irecvs in flight, pre-progress reads). Manual
//* pack/unpack mirrors that — self-copies run BEFORE Waitall+unpack so
//* vector[sL] reads see the same FACE-post-unpack-but-EDGE-pre-Irecv state.
//* Where self-copy and unpack target the same cell, unpack runs second and
//* wins, matching the legacy "Irecv data overwrites self-copy" end state.
//* ============================================================================
static void do_edge_phase(int nx, int ny, int nz,
                           int n_fields, double ****vectors,
                           const HaloContext &ctx)
{
    const int        n_ghost = ctx.n_ghost;
    const int        offset  = ctx.offset;
    const MPI_Comm   comm    = ctx.comm;
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
            const int ix = (x_side == 0) ? (n_ghost + offset + sx)
                                         : (nx - 1 - n_ghost - offset - sx);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int sz = ctx.g_src(gz);
                const int iz = (z_side == 0) ? (n_ghost + offset + sz)
                                             : (nz - 1 - n_ghost - offset - sz);
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
            const int ix = (x_side == 0) ? (n_ghost + offset + sx)
                                         : (nx - 1 - n_ghost - offset - sx);
            for (int gy = 0; gy < n_ghost; ++gy) {
                const int sy = ctx.g_src(gy);
                const int iy = (y_side == 0) ? (n_ghost + offset + sy)
                                             : (ny - 1 - n_ghost - offset - sy);
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
            const int iy = (y_side == 0) ? (n_ghost + offset + sy)
                                         : (ny - 1 - n_ghost - offset - sy);
            for (int gz = 0; gz < n_ghost; ++gz) {
                const int sz = ctx.g_src(gz);
                const int iz = (z_side == 0) ? (n_ghost + offset + sz)
                                             : (nz - 1 - n_ghost - offset - sz);
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
    //* this pass. At n_ghost == 1 the g loop runs once and reduces to legacy.
    for (int f = 0; f < n_fields; ++f) {
        if (rnX == myrank && lnX == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nx - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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
                const int sL = ny - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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
                const int sL = nz - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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

    //* Unpack edge recv buffers AFTER edge self-copies — at cells where
    //* self-copy and unpack target the same ghost-corner, unpack runs
    //* second and wins, matching the legacy "in-flight Irecv overwrites
    //* the self-copy write" end state.
    unpack_edge_recvs();
}

//* ============================================================================
//* FACE PHASE — manual pack/unpack at N=1+
//*
//* Each of the 6 face directions exchanges n_ghost slab layers, batched into
//* one MPI message per direction (vs n_ghost messages with the strided
//* yzFacetype/xzFacetype/xyFacetype custom types this block used to call).
//* Bit-identical end state: same doubles in same destination cells. Manual
//* pack/unpack also closes the Phase E.20 race window structurally — Irecv
//* targets a contiguous recv buffer, so periodic-self self-copies and sum-
//* on-receive read vector untouched until the Waitall + unpack pair below.
//*
//* Ordering (preserves legacy bit-identicality):
//*   - needInterp=true:  self-copies BEFORE Waitall+unpack — vector[sL]
//*                        reads land on pre-Irecv (== local-gather) values,
//*                        matching legacy when Irecvs were in flight.
//*   - needInterp=false: Waitall+unpack BEFORE self-copies — vector[sL]
//*                        reads see post-Irecv neighbour values, matching
//*                        legacy E.20-conditional Waitall-then-self-copy.
//* ============================================================================
static void do_face_phase(int nx, int ny, int nz,
                           int n_fields, double ****vectors,
                           const HaloContext &ctx, bool needInterp)
{
    const int        n_ghost = ctx.n_ghost;
    const int        offset  = ctx.offset;
    const MPI_Comm   comm    = ctx.comm;
    const int        myrank  = ctx.myrank;
    const int* const cnt     = ctx.communicationCnt;
    const int        lnX     = ctx.neighbours[0], rnX = ctx.neighbours[1];
    const int        lnY     = ctx.neighbours[2], rnY = ctx.neighbours[3];
    const int        lnZ     = ctx.neighbours[4], rnZ = ctx.neighbours[5];
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
    //* field. Iteration order matches the strided MPI types.
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

    //* Pack send buffers from strict-interior cells. g_src(g) = (n_ghost-1-g)
    //* source-depth swap matches the legacy MPI-types path.
    for (int f = 0; f < n_fields; ++f) {
        for (int g = 0; g < n_ghost; ++g) {
            const int s = ctx.g_src(g);
            if (cnt[0]) pack_yz(f, n_ghost + offset + s,            face_send_bufs[0].data() + (f * n_ghost + g) * yz_per_layer);
            if (cnt[1]) pack_yz(f, nx - 1 - n_ghost - offset - s,   face_send_bufs[1].data() + (f * n_ghost + g) * yz_per_layer);
            if (cnt[2]) pack_xz(f, n_ghost + offset + s,            face_send_bufs[2].data() + (f * n_ghost + g) * xz_per_layer);
            if (cnt[3]) pack_xz(f, ny - 1 - n_ghost - offset - s,   face_send_bufs[3].data() + (f * n_ghost + g) * xz_per_layer);
            if (cnt[4]) pack_xy(f, n_ghost + offset + s,            face_send_bufs[4].data() + (f * n_ghost + g) * xy_per_layer);
            if (cnt[5]) pack_xy(f, nz - 1 - n_ghost - offset - s,   face_send_bufs[5].data() + (f * n_ghost + g) * xy_per_layer);
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

    //* Sum-on-receive at periodic-self for needInterp=true (moments path).
    //  At TSC width the gather deposits to ghost cells; periodic-image sums
    //  recover those contributions before the overwrite below clobbers them.
    //  Reads vector[g] which is untouched by Irecv (recv goes to recv_bufs).
    const bool ps_x_self = (rnX == myrank && lnX == myrank);
    const bool ps_y_self = (rnY == myrank && lnY == myrank);
    const bool ps_z_self = (rnZ == myrank && lnZ == myrank);

    if (needInterp) {
        for (int f = 0; f < n_fields; ++f) {
            if (ps_x_self) {
                for (int g = 0; g < n_ghost; g++) {
                    const int dst_lo = g + nx - 2 * n_ghost - offset;
                    const int dst_hi = (2 * n_ghost - 1 + offset) - g;
                    for (int iy = 1; iy <= ny - 2; iy++)
                        for (int iz = 1; iz <= nz - 2; iz++) {
                            vectors[f][dst_lo][iy][iz] += vectors[f][g][iy][iz];
                            vectors[f][dst_hi][iy][iz] += vectors[f][nx - 1 - g][iy][iz];
                        }
                }
            }
            if (ps_y_self) {
                for (int g = 0; g < n_ghost; g++) {
                    const int dst_lo = g + ny - 2 * n_ghost - offset;
                    const int dst_hi = (2 * n_ghost - 1 + offset) - g;
                    for (int ix = 1; ix <= nx - 2; ix++)
                        for (int iz = 1; iz <= nz - 2; iz++) {
                            vectors[f][ix][dst_lo][iz] += vectors[f][ix][g][iz];
                            vectors[f][ix][dst_hi][iz] += vectors[f][ix][ny - 1 - g][iz];
                        }
                }
            }
            if (ps_z_self) {
                for (int g = 0; g < n_ghost; g++) {
                    const int dst_lo = g + nz - 2 * n_ghost - offset;
                    const int dst_hi = (2 * n_ghost - 1 + offset) - g;
                    for (int ix = 1; ix <= nx - 2; ix++)
                        for (int iy = 1; iy <= ny - 2; iy++) {
                            vectors[f][ix][iy][dst_lo] += vectors[f][ix][iy][g];
                            vectors[f][ix][iy][dst_hi] += vectors[f][ix][iy][nz - 1 - g];
                        }
                }
            }
        }
    }

    //* Periodic single-rank self-copies (X/Y/Z). Source indices match the
    //  (n_ghost + offset + g) depth convention used by the MPI sends, with
    //  modular wrapping for thin dimensions.
    auto periodic_self_copies = [&]() {
        for (int f = 0; f < n_fields; ++f) {
            if (rnX == myrank && lnX == myrank) {
                const int stride = nx - 2 * n_ghost - offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = nx - 1 - n_ghost - offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + offset + ctx.g_src(g);
                    if (sR > nx - 1 - n_ghost) sR -= stride;
                    for (int iy = 1; iy < ny - 1; iy++)
                        for (int iz = 1; iz < nz - 1; iz++) {
                            vectors[f][g][iy][iz]            = vectors[f][sL][iy][iz];
                            vectors[f][nx - 1 - g][iy][iz]   = vectors[f][sR][iy][iz];
                        }
                }
            }
            if (rnY == myrank && lnY == myrank) {
                const int stride = ny - 2 * n_ghost - offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = ny - 1 - n_ghost - offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + offset + ctx.g_src(g);
                    if (sR > ny - 1 - n_ghost) sR -= stride;
                    for (int ix = 1; ix < nx - 1; ix++)
                        for (int iz = 1; iz < nz - 1; iz++) {
                            vectors[f][ix][g][iz]            = vectors[f][ix][sL][iz];
                            vectors[f][ix][ny - 1 - g][iz]   = vectors[f][ix][sR][iz];
                        }
                }
            }
            if (rnZ == myrank && lnZ == myrank) {
                const int stride = nz - 2 * n_ghost - offset;
                for (int g = 0; g < n_ghost; g++) {
                    int sL = nz - 1 - n_ghost - offset - ctx.g_src(g);
                    if (sL < n_ghost) sL += stride;
                    int sR = n_ghost + offset + ctx.g_src(g);
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

    //* needInterp=true: self-copies first, vector[sL] reads pre-Irecv state.
    if (needInterp) periodic_self_copies();

    //* Waitall (unconditional — manual pack/unpack closes the E.20 race).
    if (snd > 0) MPI_Waitall(snd, reqs, MPI_STATUSES_IGNORE);

    //* Unpack neighbour values into ghost slabs.
    unpack_face_recvs();

    //* !needInterp: self-copies after unpack, matching legacy E.20-conditional
    //  Waitall-then-self-copy ordering.
    if (!needInterp) periodic_self_copies();
}

//* ============================================================================
//* CORNER PHASE — manual pack / unpack at N=1+
//*
//* Each X-LO/HI Cart side carries all 4 of its perp Y/Z corners (Y-LO/HI ×
//* Z-LO/HI), each an n_ghost^3 cube, batched into one MPI message of
//* 4 × n_ghost^3 × n_fields doubles. Replaces the legacy 32-scalar-MPI-
//* message-per-side path. Only fires when both Y and Z directions have at
//* least one cross-rank face neighbour — at np=1 along Y or Z, the corner
//* cube is filled by the periodic-self self-copies further down.
//* ============================================================================
static void do_corner_phase(int nx, int ny, int nz,
                             int n_fields, double ****vectors,
                             const HaloContext &ctx)
{
    const int       n_ghost  = ctx.n_ghost;
    const int       offset   = ctx.offset;
    const MPI_Comm  comm     = ctx.comm;
    const int* const cnt     = ctx.communicationCnt;
    const int       lnX      = ctx.neighbours[0];
    const int       rnX      = ctx.neighbours[1];
    const int       tag_XL   = 1, tag_XR = 4;

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
    //* (n_ghost+offset+g_src) from the active X face. For each (gx,gy,gz)
    //* the 4 perp Y/Z corners pack in canonical order (Y-LO,Z-LO),
    //* (Y-LO,Z-HI), (Y-HI,Z-LO), (Y-HI,Z-HI). Both ranks pack the same
    //* canonical order so the receiver's unpack lands matching doubles.
    for (int f = 0; f < n_fields; ++f) {
        for (int gx = 0; gx < n_ghost; ++gx) {
            const int sx   = ctx.g_src(gx);
            const int xs_l = n_ghost + offset + sx;
            const int xs_r = nx - 1 - n_ghost - offset - sx;
            for (int gy = 0; gy < n_ghost; ++gy) {
                const int sy   = ctx.g_src(gy);
                const int ys_l = n_ghost + offset + sy;
                const int ys_r = ny - 1 - n_ghost - offset - sy;
                for (int gz = 0; gz < n_ghost; ++gz) {
                    const int sz   = ctx.g_src(gz);
                    const int zs_l = n_ghost + offset + sz;
                    const int zs_r = nz - 1 - n_ghost - offset - sz;
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

//* ============================================================================
//* Periodic single-rank corner self-copies. Generalized from legacy CIC code:
//* uses else-if cascade — for np=1 (all periodic) the first branch (X-self)
//* handles all 8 corner cubes via the cascade face → edge → corner. Source
//* cells were filled by edge self-copies in the EDGE PHASE.
//* ============================================================================
static void do_corner_periodic_self(int nx, int ny, int nz,
                                     int n_fields, double ****vectors,
                                     const HaloContext &ctx)
{
    const int n_ghost = ctx.n_ghost;
    const int offset  = ctx.offset;
    const int myrank  = ctx.myrank;
    const int lnX     = ctx.neighbours[0], rnX = ctx.neighbours[1];
    const int lnY     = ctx.neighbours[2], rnY = ctx.neighbours[3];
    const int lnZ     = ctx.neighbours[4], rnZ = ctx.neighbours[5];

    for (int f = 0; f < n_fields; ++f) {
        if (lnX == myrank && rnX == myrank) {
            for (int g = 0; g < n_ghost; g++) {
                const int sL = nx - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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
                const int sL = ny - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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
                const int sL = nz - 1 - n_ghost - offset - ctx.g_src(g);
                const int sR = n_ghost + offset + ctx.g_src(g);
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
                                bool isCenterFlag, bool isFaceOnlyFlag, bool needInterp, bool isParticle,
                                double ****vectors_c)
{
    //* Single-source-of-truth halo state shared across all phase helpers.
    const HaloContext ctx = make_halo_context(EMf, vct, isCenterFlag, needInterp, isParticle);
    #ifdef DEBUG
        MPI_Errhandler_set(ctx.comm, MPI_ERRORS_RETURN);
    #endif

    //! ============================================================
    //  Phase order:
    //   1. Cross-rank moment completion (A SOR pre-pass + E.14 corner +
    //      E.16 face-cell). No-op outside (needInterp && n_ghost > 1 &&
    //      any_xrank_periodic). Must precede the FACE PHASE since its
    //      COPY semantics would overwrite the ghost values the SOR
    //      pre-pass needs to send.
    //   2. FACE phase (manual pack/unpack at N=1+).
    //   3. EDGE / CORNER / E.18 diagonal Cart edge-copy / corner periodic-
    //      self self-copies — only when !isFaceOnlyFlag.
    //   4. Trailing addFace family at n_ghost==1 (CIC moments interp).
    //! ============================================================
    do_xrank_completion(nx, ny, nz, n_fields, vectors, ctx, needInterp);
    do_face_phase      (nx, ny, nz, n_fields, vectors, ctx, needInterp);
    if (!isFaceOnlyFlag) {
        do_edge_phase           (nx, ny, nz, n_fields, vectors, ctx);
        do_corner_phase         (nx, ny, nz, n_fields, vectors, ctx);
        do_diagonal_edge_copy   (nx, ny, nz, n_fields, vectors, ctx);
        do_corner_periodic_self (nx, ny, nz, n_fields, vectors, ctx);
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
    if (needInterp && ctx.n_ghost == 1) {
        //* Legacy +=  when vectors_c is null; Neumaier-compensated update
        //* lands residual in vectors_c[f] when supplied. Note: the Kahan
        //* path here historically did NOT skip periodic-self at the multi-
        //* field site (only the legacy plain path did), so preserve that
        //* asymmetry by keeping skip_self gated on vectors_c.
        const bool skip_self = (vectors_c == nullptr);
        for (int f = 0; f < n_fields; ++f) {
            double ***vc = (vectors_c != nullptr) ? vectors_c[f] : nullptr;
            addFace  (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self, vc);
            addEdgeZ (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self, vc);
            addEdgeY (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self, vc);
            addEdgeX (nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self, vc);
            addCorner(nx, ny, nz, vectors[f], vct, ctx.n_ghost, skip_self, vc);
        }
    }
}


//* communicate*BC* wrappers all follow the same shape: NBDerivedHaloComm
//* with one of four (isCenter, faceOnly) × (P / non-P) flag bits, then
//* a BCface (or BCface_P) finalisation. The double*** overload carries the
//* logic; the arr3_double overload just unpacks via fetch_arr3().
void communicateNodeBC(int nx, int ny, int nz, double*** vector,
                        int bcFaceXrght, int bcFaceXleft,
                        int bcFaceYrght, int bcFaceYleft,
                        int bcFaceZrght, int bcFaceZleft,
                        const VirtualTopology3D * vct, EMfields3D *EMf)
{
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, false);
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
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, true, false, false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, true, false, true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateNodeBC_P(int nx, int ny, int nz, arr3_double _vector,
                          int bcFaceXrght, int bcFaceXleft,
                          int bcFaceYrght, int bcFaceYleft,
                          int bcFaceZrght, int bcFaceZleft,
                          const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
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
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, false, true);
    BCface_P(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC(int nx, int ny, int nz, arr3_double _vector,
                                    int bcFaceXrght, int bcFaceXleft,
                                    int bcFaceYrght, int bcFaceYleft,
                                    int bcFaceZrght, int bcFaceZleft,
                                    const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, true, false, false);
    BCface(nx, ny, nz, vector, bcFaceXrght, bcFaceXleft, bcFaceYrght, bcFaceYleft, bcFaceZrght, bcFaceZleft, vct);
}

void communicateCenterBoxStencilBC_P(int nx, int ny, int nz, arr3_double _vector,
                                      int bcFaceXrght, int bcFaceXleft,
                                      int bcFaceYrght, int bcFaceYleft,
                                      int bcFaceZrght, int bcFaceZleft,
                                      const VirtualTopology3D * vct, EMfields3D *EMf)
{
    double ***vector = _vector.fetch_arr3();
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, true, false, true);
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
//* Per-axis periodic-self skip flags. When `skip_self_periodic` is on
//* AND an axis's left and right neighbours are myrank, the periodic-self
//* fold + copy upstream already handled that axis; running the add*
//* helpers on it would double-count. All five add* functions begin with
//* this same computation — hoist it.
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

//* Sum-on-receive helper. When `vc == nullptr` the update is a plain
//* `dst += term`; when `vc` is supplied, a Neumaier-compensated step
//* lands the round-off residual in `vc[di][dj][dk]`. The caller folds
//* `vc` back into `v` after the halo exchange. Branch on the constant
//* `vc` pointer is hoisted by the compiler — these slabs are not hot.
static inline void halo_add(double ***v, double ***vc,
                            int di, int dj, int dk,
                            int si, int sj, int sk)
{
    const double term = v[si][sj][sk];
    if (vc != nullptr) {
        double &sum  = v [di][dj][dk];
        double &comp = vc[di][dj][dk];
        const double t = sum + term;
        if (std::fabs(sum) >= std::fabs(term))
            comp += (sum - t) + term;
        else
            comp += (term - t) + sum;
        sum = t;
    } else {
        v[di][dj][dk] += term;
    }
}

void addFace(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic, double ***vector_c)
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
                    halo_add(vector, vector_c, nx - 1 - n_ghost - g, j, k,   nx - 1 - g, j, k);
        if (do_xL)
            for (int j = 1; j <= ny - 2; j++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector, vector_c, n_ghost + g, j, k,   g, j, k);
        if (do_yR)
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector, vector_c, i, ny - 1 - n_ghost - g, k,   i, ny - 1 - g, k);
        if (do_yL)
            for (int i = 1; i <= nx - 2; i++)
                for (int k = 1; k <= nz - 2; k++)
                    halo_add(vector, vector_c, i, n_ghost + g, k,   i, g, k);
        if (do_zR)
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    halo_add(vector, vector_c, i, j, nz - 1 - n_ghost - g,   i, j, nz - 1 - g);
        if (do_zL)
            for (int i = 1; i <= nx - 2; i++)
                for (int j = 1; j <= ny - 2; j++)
                    halo_add(vector, vector_c, i, j, n_ghost + g,   i, j, g);
    }
}

/** insert the ghost cells Edge Z in the 3D physical vector */
//  Z-aligned edge: ghost block lives in the (x, y) cross-section.
//  Same shift convention as addFace: (src ghost, dst interior) pairs are
//  independent — no chained accumulation.
void addEdgeZ(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic, double ***vector_c)
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
                halo_add(vector, vector_c, nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, i,   nx - 1 - gx, ny - 1 - gy, i);
        if (do_LL)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector, vector_c, n_ghost + gx, n_ghost + gy, i,   gx, gy, i);
        if (do_RL)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector, vector_c, nx - 1 - n_ghost - gx, n_ghost + gy, i,   nx - 1 - gx, gy, i);
        if (do_LR)
            for (int i = 1; i < (nz - 1); i++)
                halo_add(vector, vector_c, n_ghost + gx, ny - 1 - n_ghost - gy, i,   gx, ny - 1 - gy, i);
    }
}
/** add the ghost cell values Edge Y to the 3D physical vector */
//  Y-aligned edge: ghost block lives in the (x, z) cross-section.
void addEdgeY(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic, double ***vector_c)
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
                halo_add(vector, vector_c, nx - 1 - n_ghost - gx, i, nz - 1 - n_ghost - gz,   nx - 1 - gx, i, nz - 1 - gz);
        if (do_LL)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector, vector_c, n_ghost + gx, i, n_ghost + gz,   gx, i, gz);
        if (do_LR)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector, vector_c, n_ghost + gx, i, nz - 1 - n_ghost - gz,   gx, i, nz - 1 - gz);
        if (do_RL)
            for (int i = 1; i < (ny - 1); i++)
                halo_add(vector, vector_c, nx - 1 - n_ghost - gx, i, n_ghost + gz,   nx - 1 - gx, i, gz);
    }
}

/** add the ghost values Edge X to the 3D physical vector */
//  X-aligned edge: ghost block lives in the (y, z) cross-section.
void addEdgeX(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic, double ***vector_c)
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
                halo_add(vector, vector_c, i, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   i, ny - 1 - gy, nz - 1 - gz);
        if (do_LL)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector, vector_c, i, n_ghost + gy, n_ghost + gz,   i, gy, gz);
        if (do_LR)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector, vector_c, i, n_ghost + gy, nz - 1 - n_ghost - gz,   i, gy, nz - 1 - gz);
        if (do_RL)
            for (int i = 1; i < (nx - 1); i++)
                halo_add(vector, vector_c, i, ny - 1 - n_ghost - gy, n_ghost + gz,   i, ny - 1 - gy, gz);
    }
}

/** add ghost cells values Corners in the 3D physical vector */
//  Each corner is a single node for n_ghost == 1; for n_ghost > 1 it expands
//  to a (gx, gy, gz) cube of nodes that are summed back into the matching
//  inner interior corner.
void addCorner(int nx, int ny, int nz, double ***vector, const VirtualTopology3D * vct, int n_ghost, bool skip_self_periodic, double ***vector_c)
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
            halo_add(vector, vector_c, nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, ny - 1 - gy, nz - 1 - gz);
        if (do_LRR)
            halo_add(vector, vector_c, n_ghost + gx, ny - 1 - n_ghost - gy, nz - 1 - n_ghost - gz,   gx, ny - 1 - gy, nz - 1 - gz);
        if (do_RLR)
            halo_add(vector, vector_c, nx - 1 - n_ghost - gx, n_ghost + gy, nz - 1 - n_ghost - gz,   nx - 1 - gx, gy, nz - 1 - gz);
        if (do_LLR)
            halo_add(vector, vector_c, n_ghost + gx, n_ghost + gy, nz - 1 - n_ghost - gz,   gx, gy, nz - 1 - gz);
        if (do_RRL)
            halo_add(vector, vector_c, nx - 1 - n_ghost - gx, ny - 1 - n_ghost - gy, n_ghost + gz,   nx - 1 - gx, ny - 1 - gy, gz);
        if (do_LRL)
            halo_add(vector, vector_c, n_ghost + gx, ny - 1 - n_ghost - gy, n_ghost + gz,   gx, ny - 1 - gy, gz);
        if (do_RLL)
            halo_add(vector, vector_c, nx - 1 - n_ghost - gx, n_ghost + gy, n_ghost + gz,   nx - 1 - gx, gy, gz);
        if (do_LLL)
            halo_add(vector, vector_c, n_ghost + gx, n_ghost + gy, n_ghost + gz,   gx, gy, gz);
    }
}

/** communicate and sum shared ghost cells */

//? Used for communicating moments
//* Moment interpolation halo (sum-on-receive). Optional `vector_c`
//* companion routes the receive through Neumaier-compensated adds —
//* same idiom as the unified `addFace`/`addEdge*`/`addCorner` helpers.
void communicateInterp(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf, double*** vector_c)
{
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, true, false, true, true, vector_c);
}

void communicateInterp(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateInterp(nx, ny, nz, _vector.fetch_arr3(), vct, EMf);
}

void communicateInterp(int nx, int ny, int nz, arr3_double _vector, arr3_double _vector_c, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateInterp(nx, ny, nz, _vector.fetch_arr3(), vct, EMf, _vector_c.fetch_arr3());
}

void communicateNode_P(int nx, int ny, int nz, double*** vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    NBDerivedHaloComm(nx, ny, nz, vector, vct, EMf, false, false, false, true);
}

void communicateNode_P(int nx, int ny, int nz, arr3_double _vector, const VirtualTopology3D * vct, EMfields3D *EMf)
{
    communicateNode_P(nx, ny, nz, _vector.fetch_arr3(), vct, EMf);
}

//* ================================================================================
//  Multi-field dispatcher and call-site wrappers.
//
//  Batches N ≥ 1 fields sharing the same (nx, ny, nz) extents and the same
//  MPI Cartesian topology into one halo exchange per direction. At the
//  EMfields3D mass-matrix call sites, replacing N sequential single-field
//  calls with one batched call cuts MPI message count by N× and amortises
//  per-message latency over an N×-larger payload.
//
//  Path selection mirrors the single-field NBDerivedHaloComm:
//   - n_ghost > 1 (TSC moments path): forward to the multi-field
//     NBDerivedHaloCommN. One MPI message per direction carries
//     n_fields × slab_size doubles.
//   - n_ghost == 1 (CIC fast path): the legacy merged-MPI-types fast path
//     in NBDerivedHaloComm hard-codes single-field strided datatypes.
//     Multi-field at this width just loops, calling the legacy function
//     N times. Halo cost is small at CIC (1 layer) so per-message latency
//     doesn't dominate.
//* ================================================================================
static void NBDerivedHaloComm_multi(int nx, int ny, int nz,
                                     int n_fields, double ****vectors,
                                     const VirtualTopology3D *vct, EMfields3D *EMf,
                                     bool isCenterFlag, bool isFaceOnlyFlag,
                                     bool needInterp, bool isParticle,
                                     double ****vectors_c = nullptr)
{
    if (EMf->getNGhost() > 1) {
        NBDerivedHaloCommN(nx, ny, nz, n_fields, vectors, vct, EMf,
                           isCenterFlag, isFaceOnlyFlag, needInterp, isParticle, vectors_c);
        return;
    }
    //* CIC fallback: loop N times through the legacy single-field fast path.
    for (int f = 0; f < n_fields; ++f) {
        double ***vc = (vectors_c != nullptr) ? vectors_c[f] : nullptr;
        NBDerivedHaloComm(nx, ny, nz, vectors[f], vct, EMf,
                          isCenterFlag, isFaceOnlyFlag, needInterp, isParticle, vc);
    }
}

//* Multi-field interpolation halo. Optional `vectors_c` companion array
//* (parallel layout to `vectors`) routes the receive through Neumaier
//* compensation per field.
void communicateInterp_multi(int nx, int ny, int nz, int n_fields, double ****vectors,
                              const VirtualTopology3D *vct, EMfields3D *EMf,
                              double ****vectors_c)
{
    NBDerivedHaloComm_multi(nx, ny, nz, n_fields, vectors, vct, EMf,
                            /*isCenterFlag=*/true, /*isFaceOnlyFlag=*/false,
                            /*needInterp=*/true, /*isParticle=*/true, vectors_c);
}

void communicateNode_P_multi(int nx, int ny, int nz, int n_fields, double ****vectors,
                              const VirtualTopology3D *vct, EMfields3D *EMf)
{
    NBDerivedHaloComm_multi(nx, ny, nz, n_fields, vectors, vct, EMf,
                            /*isCenterFlag=*/false, /*isFaceOnlyFlag=*/false,
                            /*needInterp=*/false, /*isParticle=*/true);
}

