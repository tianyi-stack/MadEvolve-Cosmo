import random
cimport cython
cimport openmp
import numpy as np
from libc.stdio cimport printf
from cython.parallel import prange,parallel
from libc.stdlib cimport abort, malloc, free, rand, srand, RAND_MAX
from libc.time cimport time
cdef extern from "math.h":
    int floor(float arg) nogil
    float fabs(float arg) nogil
DTYPE = np.intc

@cython.boundscheck(False) 
@cython.wraparound(False)
def CICPaint(float[:, :] Position, int NMesh, int BoxSize, unsigned long int Size):
    """
    Fast Loop of Cloud in Cell Painting
    """
    H = np.float32(BoxSize)/np.float32(NMesh)
    Output = np.zeros([NMesh, NMesh, NMesh], dtype = np.float32)
    cdef float[:, :, :] Output_view = Output
    cdef int I, J, K, I1, J1, K1
    cdef unsigned long int i, j
    cdef float H_view = H
      
    cdef float k1, k2
    k1 = 1/np.float32(H_view)
    k2 = np.float32(NMesh)**3/(np.float32(Size))
    cdef float k2_view = k2
    
    with nogil, parallel(num_threads=5):
        N_view = <int *> malloc(sizeof(int) * 6)
        Np = <int *> malloc(sizeof(int) * 3)
        xp = <float *> malloc(sizeof(float) * 3)
        W = <float *> malloc(sizeof(float) * 3)
        TempMemo = <float *> malloc(sizeof(float) * NMesh**3)
        for i in prange(Size):
            for j in range(3):
                N_view[j] = floor(Position[i, j]*k1)
                N_view[j + 3] = N_view[j] + 1
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        Np[0] = N_view[3*I]
                        Np[1] = N_view[3*J+1]
                        Np[2] = N_view[3*K+2]
                        xp[0] = Np[0]*H_view
                        xp[1] = Np[1]*H_view
                        xp[2] = Np[2]*H_view
                        for j in range(3):
                            W[j] = 1 - fabs(Position[i, j] - xp[j])*k1
                        for j in range(3):
                            if Np[j] < 0:
                                Np[j] += NMesh
                            if Np[j] >= NMesh:
                                Np[j] -= NMesh
                        TempMemo[Np[0]*NMesh**2 + Np[1]*NMesh + Np[2]] += W[0]*W[1]*W[2]*k2_view                   
                        
        with gil:
            for I1 in range(NMesh):
                for J1 in range(NMesh):
                    for K1 in range(NMesh):
                        Output_view[I1, J1, K1] += TempMemo[I1*NMesh**2 + J1*NMesh + K1]

        
    return Output