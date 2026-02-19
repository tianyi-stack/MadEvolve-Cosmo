import numpy as np
# cimport numpy as np
cdef extern from "math.h":
    double floor(double arg)
    double fabs(double arg)
DTYPE = np.intc
# ctypedef np.npy_uint32 UINT32_t 

def NGPPaint(double[:, :] Position, int NMesh, int BoxSize, unsigned long int Size):
    """
    Fast Loop of NGP Painting
    """
    print('NGP Paint')
    H = np.double(BoxSize)/np.double(NMesh)
    Output = np.zeros([NMesh, NMesh, NMesh])
    cdef double[:, :, :] Output_view = Output
    cdef int I, J, K
    cdef unsigned long int i
    cdef double H_view = H
    cdef int[3] N_view
    cdef double[:] position
    cdef double[3] W
    
    cdef double k1, k2
    k1 = 1/np.double(H_view)
    k2 = np.double(NMesh)**3/np.double(Size)
    cdef double k2_view = k2
    
    
    for i in range(Size):
#         print(i)
        position = Position[i, :]
        for j in range(3):
            N_view[j] = int(floor(position[j]*k1))
            if N_view[j] < 0:
                N_view[j] += NMesh
            if N_view[j] >= NMesh:
                N_view[j] -= NMesh
        Output_view[N_view[0], N_view[1], N_view[2]] += k2_view
    
    return Output

def VelocityPaint(double[:, :] Position, double[:] Velocity, int NMesh, int BoxSize, unsigned long int Size):
    """
    Fast Loop of Velocity NGP Painting
    """
    print('Velocity NGP Paint')
    H = np.double(BoxSize)/np.double(NMesh)
    Output = np.zeros([NMesh, NMesh, NMesh])
    Outputbin = np.zeros([NMesh, NMesh, NMesh])
    cdef double[:, :, :] Output_view = Output
    cdef double[:, :, :] Outputbin_view = Outputbin
    cdef unsigned long int i 
    cdef int I, J, K
    cdef int[3] N_view
    cdef double[:] position
    cdef double H_view = H
    cdef double k1
    k1 = 1/np.double(H_view)
    cdef double k1_view = k1
    
    for i in range(Size):
        position = Position[i, :]
        velocity = Velocity[i]
        for j in range(3):
            N_view[j] = int(floor(position[j]*k1))
        Output_view[N_view[0], N_view[1], N_view[2]] += velocity
        Outputbin_view[N_view[0], N_view[1], N_view[2]] += 1
        
    for I in range(NMesh):
        for J in range(NMesh):
            for K in range(NMesh):
                if Outputbin_view[I, J, K] > 0:
                    Output_view[I, J, K] = Output_view[I, J, K]/Outputbin_view[I, J, K]
                    
    return Output