import numpy as np
cdef extern from "math.h":
    int floor(double arg)
    
cdef extern from "complex.h":
    pass
    
DTYPE = np.intc
# dd

def UniDimension(float[:, :, :] Input, fn, int NMesh):
    print('UniDimension')
    cdef int binnum = int(NMesh/2)
    
    Output = np.zeros(binnum)
    Output_bin = np.zeros(binnum)
    k_bin = np.zeros(binnum)
    Outk = np.zeros(binnum)
    n_bin = np.zeros(binnum)
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    
    cdef double[:] Output_view = Output
    cdef double[:] Output_bin_view = Output_bin
    cdef double[:] k_bin_view = k_bin
    cdef double[:] Outk_view = Outk
    cdef double[:] n_bin_view = n_bin
    cdef double[:, :, :] k_ind_view = k_ind
    cdef int I
    cdef double temp
    
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                temp = k_ind_view[i,j,k]
                I = int(floor(temp))
                if I < binnum:
                    Output_bin_view[I] += Input[i,j,k]
                    k_bin_view[I] += temp
                    n_bin_view[I] += 1
    
    for i in range(binnum):
        Outk_view[i] += k_bin_view[i]/n_bin_view[i]
        Output_view[i] += Output_bin_view[i]/n_bin_view[i]
    
    return n_bin, Outk, Output

def BiDimension(float[:, :, :] Input, fn, int NMesh):
    print('BiDimension')
    cdef int binnum = int(NMesh/2)
    
    Output = np.zeros([binnum, binnum])
    Output_bin = np.zeros([binnum, binnum])
    k_perp_bin = np.zeros([binnum, binnum])
    k_para_bin = np.zeros([binnum, binnum])
    Outk_perp = np.zeros([binnum, binnum])
    Outk_para = np.zeros([binnum, binnum])
    n_bin = np.zeros([binnum, binnum])
    
    k_perp = (np.zeros(NMesh)[:, None, None]**2.+ np.zeros(NMesh)[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    k_para = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ np.zeros(NMesh)[None, None, :]**2)**(0.5)
    
    cdef double[:, :] Output_view = Output
    cdef double[:, :] Output_bin_view = Output_bin
    cdef double[:, :] k_perp_bin_view = k_perp_bin
    cdef double[:, :] k_para_bin_view = k_para_bin
    cdef double[:, :] Outk_perp_view = Outk_perp
    cdef double[:, :] Outk_para_view = Outk_para
    cdef double[:, :] n_bin_view = n_bin
    cdef double[:, :, :] k_perp_view = k_perp
    cdef double[:, :, :] k_para_view = k_para
    cdef int I, J
    cdef double temp1, temp2
    
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                temp1 = k_perp_view[i,j,k]
                temp2 = k_para_view[i,j,k]
                I = int(floor(temp1))
                J = int(floor(temp2))
                if I <= binnum-1 and J <= binnum-1 and I >= 0 and J>= 0:
                    k_perp_bin_view[I, J] += temp1
                    k_para_bin_view[I, J] += temp2
                    Output_bin_view[I, J] += Input[i,j,k]
                    n_bin_view[I, J] += 1
                    
    for i in range(binnum):
        for j in range(binnum):
            Outk_perp_view[i,j] = k_perp_bin_view[i,j]/n_bin_view[i,j]
            Outk_para_view[i,j] = k_para_bin_view[i,j]/n_bin_view[i,j]
            Output_view[i,j] = Output_bin_view[i,j]/n_bin_view[i,j]
            
    return n_bin, Outk_perp, Outk_para, Output

def BiDimensionBin(float[:, :, :] Input, fn, int NMesh, int Nmu):
    cdef int binnum = int(NMesh/2)
    
    z_ind = (np.zeros(NMesh)[:, None, None]**2.+ np.zeros(NMesh)[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5) 
    k_ind[0,0,0] = 1
    mu_ind = z_ind*Nmu*(1 - 1E-10)/k_ind
    
    Output = np.zeros([binnum, Nmu])
    Output_bin = np.zeros([binnum, Nmu])
    k_bin = np.zeros([binnum, Nmu])
    Outk = np.zeros([binnum, Nmu])
    n_bin = np.zeros([binnum, Nmu])
    
    cdef double[:, :] Output_view = Output
    cdef double[:, :] Output_bin_view = Output_bin
    cdef double[:, :] k_bin_view = k_bin
    cdef double[:, :] Outk_view = Outk
    cdef double[:, :] n_bin_view = n_bin
    cdef double[:, :, :] k_ind_view = k_ind
    cdef double[:, :, :] mu_ind_view = mu_ind
    cdef int I, J
    cdef double temp1, temp2
    
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                temp1 = k_ind_view[i,j,k]
                temp2 = mu_ind_view[i,j,k]
                I = int(floor(temp1))
                J = int(floor(temp2))
                if I < binnum and I > 0:
                    Output_bin_view[I, J] += Input[i,j,k]
                    k_bin_view[I, J] += temp1
                    n_bin_view[I, J] += 1
                    
    for i in range(binnum):
        for j in range(Nmu):
            if n_bin_view[i,j]>0:
                Outk_view[i,j] = k_bin_view[i,j]/n_bin_view[i,j]
                Output_view[i,j] = Output_bin_view[i,j]/n_bin_view[i,j]
            else:
                Outk[i,j] = np.nan
                Output[i,j] = np.nan
            
    return n_bin, Outk, Output

def Three2FlatOne(float[:, :, :] Input, fn, int NMesh):
    print('Three2FlatOne')
    cdef int binnum = int(NMesh/2)
    
    Output = np.zeros(binnum)
    Output_bin = np.zeros(binnum)
    k_bin = np.zeros(binnum)
    Outk = np.zeros(binnum)
    n_bin = np.zeros(binnum)
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    
    cdef double[:] Output_view = Output
    cdef double[:] Output_bin_view = Output_bin
    cdef double[:] k_bin_view = k_bin
    cdef double[:] Outk_view = Outk
    cdef double[:] n_bin_view = n_bin
    cdef double[:, :, :] k_ind_view = k_ind
    cdef int I
    cdef double temp
    
    for i in range(NMesh):
        for j in range(NMesh):
            temp = k_ind_view[i,j,0]
            I = int(floor(temp))
            if I < binnum:
                Output_bin_view[I] += Input[i,j,0]
                k_bin_view[I] += temp
                n_bin_view[I] += 1
    
    for i in range(binnum):
        Outk_view[i] += k_bin_view[i]/n_bin_view[i]
        Output_view[i] += Output_bin_view[i]/n_bin_view[i]
    
    return n_bin, Outk, Output

def One2Three(float complex [:, :, :] Input, fn, int NMesh):
    print('OneDim2ThreeDim')
    cdef int binnum = int(NMesh)
    
    Output = np.zeros(binnum).astype(np.complex128)
    Output_bin = np.zeros(binnum).astype(np.complex128)
    n_bin = np.zeros(binnum)
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    ThreeDimOut = np.zeros([binnum, binnum, binnum]).astype(np.complex128)
    
    cdef double complex[:] Output_view = Output
    cdef double complex[:] Output_bin_view = Output_bin
    cdef double[:] n_bin_view = n_bin
    cdef double[:, :, :] k_ind_view = k_ind
    cdef double complex[:, :, :] ThreeDimOut_view = ThreeDimOut
    cdef int I
    cdef double temp
    
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                temp = k_ind_view[i,j,k]
                I = int(floor(temp))
                Output_bin_view[I] += Input[i,j,k]
                n_bin_view[I] += 1
    
    for i in range(binnum):
        if n_bin_view[i] > 0:
            Output_view[i] += Output_bin_view[i]/n_bin_view[i]
            
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                temp = k_ind_view[i,j,k]
                I = int(floor(temp))
                ThreeDimOut_view[i,j,k] = Output_view[I]
    
    return ThreeDimOut

def WF13(float[:] R1D, fn, int NMesh):
    
    k_ind = (fn[:, None, None]**2.+ fn[None, :, None]**2.+ fn[None, None, :]**2)**(0.5)
    WF = np.zeros([NMesh, NMesh, NMesh]).astype(np.float32)
    cdef float[:, :, :] WF_view = WF
    cdef int i, j, k, k1D
        
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                k1D = floor(k_ind[i, j, k])
                if k1D >= (NMesh/2):
                    WF_view[i, j, k] = 0
                else:
                    WF_view[i, j, k] = R1D[k1D]
                    
    return WF

def Preview(double[:, :, :] Input, int NMesh):
    
    Output = np.zeros([NMesh, NMesh])
    
    cdef int i, j, k
    cdef double[:, :]Output_view = Output
    
    for i in range(NMesh):
        for j in range(NMesh):
            for k in range(NMesh):
                Output_view[i, j] += Input[i,j,k]/NMesh
                
    return Output
    
    