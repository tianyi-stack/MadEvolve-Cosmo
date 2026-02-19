import numpy as np
cdef extern from "math.h":
    double floor(double arg)
cdef extern from "complex.h":
    pass
        
DTYPE = np.intc

def UniDimension2D(double[:, :] Input, int NMesh, double[:, :] Mode):
    print('UniDimension2D')
    cdef int binnum = int(NMesh/2)
    
    Output = np.zeros(binnum)
    Output_bin = np.zeros(binnum)
    k_bin = np.zeros(binnum)
    Outk = np.zeros(binnum)
    n_bin = np.zeros(binnum)
    fn = np.array(range(binnum))
    k_ind = (fn[:, None]**2. + fn[None, :]**2.)**0.5
    
    cdef double[:] Output_view = Output
    cdef double[:] Output_bin_view = Output_bin
    cdef double[:] k_bin_view = k_bin
    cdef double[:] Outk_view = Outk
    cdef double[:] n_bin_view = n_bin
    cdef double[:, :] k_ind_view = k_ind
    cdef int I
    cdef double temp
    
    for i in range(binnum):
        for j in range(binnum):
            temp = k_ind_view[i, j]
            I = int(floor(temp) )
            if I < binnum:
                Output_bin_view[I] += Input[i, j]*Mode[i, j]
                k_bin_view[I] += temp*Mode[i, j]
                n_bin_view[I] += Mode[i, j]
                
    for i in range(binnum):
        Outk_view[i] += k_bin_view[i]/n_bin_view[i]
        Output_view[i] += Output_bin_view[i]/n_bin_view[i]
        
    return n_bin, Outk, Output

def BiDimension2DBin(double[:, :] Input, int NMesh, int Nmu, double[:, :] Mode):
    cdef int binnum = int(NMesh/2)
    
    fn = np.array(range(binnum))
    z_ind = (np.zeros(binnum)[None, :]**2.+ fn[:, None]**2)**(0.5)
    k_ind = (fn[:, None]**2.+ fn[None, :]**2.)**(0.5) 
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
    cdef double[:, :] k_ind_view = k_ind
    cdef double[:, :] mu_ind_view = mu_ind
    cdef int I, J
    cdef double temp1, temp2
    
    for i in range(binnum):
        for j in range(binnum):
            temp1 = k_ind_view[i,j]
            temp2 = mu_ind_view[i,j]
            I = int(floor(temp1))
            J = int(floor(temp2))
            if I < binnum and I > 0:
                Output_bin_view[I, J] += Input[i,j]*Mode[i, j]
                k_bin_view[I, J] += temp1*Mode[i, j]
                n_bin_view[I, J] += Mode[i, j]
                    
    for i in range(binnum):
        for j in range(Nmu):
            if n_bin_view[i,j]>0:
                Outk_view[i,j] = k_bin_view[i,j]/n_bin_view[i,j]
                Output_view[i,j] = Output_bin_view[i,j]/n_bin_view[i,j]
            else:
                Outk[i,j] = np.nan
                Output[i,j] = np.nan
            
    return n_bin, Outk, Output