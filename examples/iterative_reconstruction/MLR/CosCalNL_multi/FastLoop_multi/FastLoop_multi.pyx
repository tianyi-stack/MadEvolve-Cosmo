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
def CICPaint_multi(float[:, :] Position, int NMesh, int BoxSize, unsigned long int Size):
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

@cython.boundscheck(False) 
@cython.wraparound(False)
def Shift_multi(float[:, :] Position, float[:, :, :, :] Dis, int NMesh, int BoxSize, unsigned long int Size):
    """
    Fast Loop to interpolate the displacement of each particle
    """
    H = np.float32(BoxSize)/np.float32(NMesh)
    Position_new = np.array(Position)
    
    cdef float[:, :] Position_new_view = Position_new
    cdef int I, J, K
    cdef unsigned long int i, j
    cdef float H_view = H
    cdef float k1
    k1 = 1/np.float32(H_view)
    
    with nogil, parallel():
        N_view = <int *> malloc(sizeof(int) * 6)
        W = <float *> malloc(sizeof(float) * 3)
        Np = <int *> malloc(sizeof(int) * 6)
    # Loop over each particle
        for i in prange(Size):
            for j in range(3):
                    N_view[j] = floor(Position[i, j]*k1)
                    N_view[j + 3] = N_view[j] + 1
                    Np[j] = N_view[j] 
                    Np[j + 3] = N_view[j + 3]
                    
            for j in range(6):
                if Np[j] > NMesh - 1 :
                    Np[j] = Np[j] - NMesh

        # Loop over 8 cubic points
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        W[0] = 1 - fabs(Position[i, 0]*k1 - N_view[3*I])
                        W[1] = 1 - fabs(Position[i, 1]*k1 - N_view[3*J+1])
                        W[2] = 1 - fabs(Position[i, 2]*k1 - N_view[3*K+2])
                        for j in range(3):
                            Position_new_view[i, j] += W[0]*W[1]*W[2]*Dis[Np[3*I], Np[3*J+1], Np[3*K+2], j]
             
        # Make sure every particle is inside the box
            for j in range(3):
                if Position_new_view[i, j] > BoxSize:
                    Position_new_view[i, j] = Position_new_view[i, j] - BoxSize
                if Position_new_view[i, j] < 0 :
                    Position_new_view[i, j] = Position_new_view[i, j] + BoxSize
    
    return Position_new

@cython.boundscheck(False) 
@cython.wraparound(False)
def DisInter(float[:, :] Position0, float[:, :] Position, float[:, :, :] Density, int NMesh, int BoxSize, unsigned long int Size):
    """
    Fast Loop to interpolate displacement field on the mesh
    """
    H = np.float32(BoxSize)/np.float32(NMesh)
    Output = np.zeros([NMesh, NMesh, NMesh, 3], dtype = np.float32)
    cdef float[:, :, :, :] Output_view = Output
    cdef int I, J, K
    cdef unsigned long int i, j
    cdef float H_view = H    
    cdef float k1, k2
    k1 = 1/np.float32(H_view)
    k2 = np.float32(NMesh)**3/(np.float32(Size))
    cdef float k2_view = k2
    
    with nogil, parallel():
        srand(1234)
        N_view = <int *> malloc(sizeof(int) * 6)
        Np = <int *> malloc(sizeof(int) * 3)
        xp = <float *> malloc(sizeof(float) * 3)
        W = <float *> malloc(sizeof(float) * 3)
        Delta = <float *> malloc(sizeof(float) * 3)
        IJK = <int *> malloc(sizeof(int) * 3)
        
    # Loop over every particle
        for i in prange(Size):
            
        # Calculate the displacement of the particle
            for j in range(3):
                Delta[j] = Position[i, j] - Position0[i, j]
            # Make sure the displacement is in [-500, 500]
                if Delta[j] > 0.5*BoxSize:
                    Delta[j] = Delta[j] - BoxSize
                if Delta[j] < -0.5*BoxSize:
                    Delta[j] = Delta[j] + BoxSize
        # Get the cubic point index
            for j in range(3):
                N_view[j] = floor(Position[i, j]*k1)
                N_view[j + 3] = N_view[j] + 1
        # Loop over every cubic point
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        Np[0] = N_view[3*I]
                        Np[1] = N_view[3*J+1]
                        Np[2] = N_view[3*K+2]
                        xp[0] = Np[0]*H_view
                        xp[1] = Np[1]*H_view
                        xp[2] = Np[2]*H_view
                    # Calculate the weight
                        for j in range(3):
                            W[j] = 1 - fabs(Position[i, j] - xp[j])*k1
                    # Make sure index is inside the box
                        for j in range(3):
                            if Np[j] < 0:
                                Np[j] += NMesh
                            if Np[j] >= NMesh:
                                Np[j] -= NMesh
                        for j in range(3):                            
                            Output_view[Np[0], Np[1], Np[2], j] += Delta[j]*W[0]*W[1]*W[2]*k2_view

    # Weight
        for I in prange(NMesh):
            for J in range(NMesh):
                for K in range(NMesh):
                    if Density[I, J, K] > 1E-12:
                        for j in range(3):
                            Output_view[I, J, K, j] = Output_view[I, J, K, j]/Density[I, J, K]

    # Randomly allocate value to zero-weight pixel from its sorrounding pixels
        for I in prange(NMesh):
            for J in range(NMesh):
                for K in range(NMesh):
                    if Density[I, J, K] <= 1E-12:
                        IJK[0] = I
                        IJK[1] = J
                        IJK[2] = K
                        while Density[IJK[0]%NMesh, IJK[1]%NMesh, IJK[2]%NMesh] <=1E-12:
                            IJK[rand()%3] += (rand()%2)*2 - 1
                        for j in range(3):
                            Output_view[I, J, K, j] = Output_view[IJK[0]%NMesh, IJK[1]%NMesh, IJK[2]%NMesh, j]
    
    return Output

@cython.boundscheck(False) 
@cython.wraparound(False)
def Truncate(float complex [:, :, :] Input, fn, int NMesh, float kmax):
    
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)    
    cdef float[:, :, :] k_ind_view = k_ind
    cdef int I, J, K
    
    with nogil, parallel():
        for I in prange(NMesh):
            for J in range(NMesh):
                for K in range(NMesh):
                    if k_ind_view[I, J, K] > kmax:
                        Input[I, J, K] = 0                   
    return np.array(Input)