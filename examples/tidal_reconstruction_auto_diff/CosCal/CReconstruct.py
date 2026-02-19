import math
import time
import numpy as np
from numpy import fft
from numpy.fft import fftn,ifftn
from CosCal.CPower import CPower
# from pandas import Series, DataFrame
# import matplotlib.pyplot as plt
from CosCal.FastLoop import Dimension

class CReconstruct(CPower):
    """ Reconstruct Element class """

    def __init__(self):
        super().__init__()

    def GetdeltaXre(self):
        return self.deltaX_re
    
    def Gete1X(self):
        return self.e1X

    def Filter(self, R):
        """
        Gaussian filter exp(-k^2R^2/2)
        R is different according to different bin
        """
        start = time.time()
        print('Filter')
        # Wiener Filter
#         window = np.fromfile('/mnt/scratch-lustre/zangsh/Data_Reconstruction/Average/Ng1536_NEW/Wiener.bin', dtype = np.float32).reshape([self.NMesh, self.NMesh, self.NMesh])
        
        # sigma of gaussian filter
#         self.k_ind2_aniso = ((self.fn[:, None, None]**2.
#           + self.fn[None, :, None]**2.
#           + 0.95**2*self.fn[None, None, :]**2.)**0.5).astype(np.float32)
#         window = np.exp(-0.5 * (self.k_ind2_aniso * self.Kf) ** 2. * R**2.).astype(np.float32)
        window = np.exp(-0.5 * (self.k_ind * self.Kf) ** 2. * R**2.).astype(np.float32)
        self.deltaK *= window
        del window
        
        # Apply cut
#         self.deltaK[self.k_ind < 48] = 0
        
        temp2 = (self.Kf*1j*self.deltaK).astype(np.complex64)
       
        
        self.deltaK1 = ((self.fn[:, None, None]
                  + np.zeros(self.NMesh)[None, :, None]
                  + np.zeros(self.NMesh)[None, None, :])* temp2).astype(np.complex64)
        self.deltaK1[0,0,0] = 0
        self.deltaK2 = ((np.zeros(self.NMesh)[:, None, None]
                  + self.fn[None, :, None] 
                  + np.zeros(self.NMesh)[None, None, :]) * temp2).astype(np.complex64)
        self.deltaK2[0,0,0] = 0
        self.deltaK3 = ((np.zeros(self.NMesh)[:, None, None]
                  + np.zeros(self.NMesh)[None, :, None]
                  + self.fn[None, None, :])* temp2).astype(np.complex64)
        self.deltaK3[0,0,0] = 0
        end = time.time()
        print("Time Consumed in Filter is ", end - start)

    def Reconstruct(self):
        print('Reconstruct')
        start = time.time()
        """
        Perform reconstruction
        calculate e1, e2, ex, ey, ez
        """
        del self.Data, self.deltaX
        deltaX1 = ifftn(self.deltaK1).real.astype(np.float32)
        deltaX2 = ifftn(self.deltaK2).real.astype(np.float32)
        deltaX3 = ifftn(self.deltaK3).real.astype(np.float32)
        del self.deltaK1, self.deltaK2, self.deltaK3
        # e1K = fftn((deltaX1*deltaX1 - deltaX2*deltaX2)*0.5)/((self.NMesh)**3)
        # e2K = fftn(deltaX1*deltaX2)/((self.NMesh)**3)
        # exK = fftn(deltaX1*deltaX3)/((self.NMesh)**3)
        # eyK = fftn(deltaX2*deltaX3)/((self.NMesh)**3)
        # ezK = fftn((2*deltaX3*deltaX3 - deltaX1*deltaX1 - deltaX2*deltaX2)/6)/((self.NMesh)**3)
        # del deltaX1, deltaX2, deltaX3

        # self.e1X = ifftn(e1K)*(self.NMesh**3)
        # self.e2X = ifftn(e2K)*(self.NMesh**3)
        # self.exX = ifftn(exK)*(self.NMesh**3)
        # self.eyX = ifftn(eyK)*(self.NMesh**3)
        # self.ezX = ifftn(ezK)*(self.NMesh**3)

        # Calculate coefficients
        # k1 = (self.fn[:, None, None]**2 - self.fn[None, :, None]**2 + np.zeros(self.NMesh)[None, None, :])/(2*self.k_ind**2)
        # k1[0,0,0] = 0
        # k2 = (2*self.fn[:, None, None]*self.fn[None, :, None]*np.ones(self.NMesh)[None, None, :])/(2*self.k_ind**2)
        # k2[0,0,0] = 0
        # kx = (2*self.fn[None, :, None]*self.fn[None, None, :]*np.ones(self.NMesh)[:, None, None])/(2*self.k_ind**2)
        # kx[0,0,0] = 0
        # ky = (2*self.fn[:, None, None]*self.fn[None, None, :]*np.ones(self.NMesh)[None, :, None])/(2*self.k_ind**2)
        # ky[0,0,0] = 0
        # kz = (2*self.fn[None, None, :]**2 - self.fn[:, None, None]**2 - self.fn[None, :, None]**2)/(2*self.k_ind**2)
        # kz[0,0,0] = 0
        ###############################################################################################################
        temp1 = 1/(2*self.k_ind**2)
        
#         self.e1X = (deltaX1*deltaX1 - deltaX2*deltaX2)*0.5
        
        
        e0K = ((self.fn[:, None, None]**2 - self.fn[None, :, None]**2 + np.zeros(self.NMesh)[None, None, :])*temp1*fftn((deltaX1*deltaX1 - deltaX2*deltaX2)*0.5)
                  + (2*self.fn[:, None, None]*self.fn[None, :, None]*np.ones(self.NMesh)[None, None, :])*temp1*fftn(deltaX1*deltaX2)
                  + (2*self.fn[:, None, None]*self.fn[None, None, :]*np.ones(self.NMesh)[None, :, None])*temp1*fftn(deltaX1*deltaX3)
                  + (2*self.fn[None, :, None]*self.fn[None, None, :]*np.ones(self.NMesh)[:, None, None])*temp1*fftn(deltaX2*deltaX3)
                  + (2*self.fn[None, None, :]**2 - self.fn[:, None, None]**2 - self.fn[None, :, None]**2)*temp1*fftn((2*deltaX3*deltaX3 - deltaX1*deltaX1 - deltaX2*deltaX2)/6))
        e0K[0, 0, 0] = 0
        e0X = ifftn(e0K).real
#         self.deltaK_re = fftn(e0X)
#         self.OneDimDeBias()
        self.deltaX_re = e0X
        end = time.time()
        print("Time Consumed in Reconstruction is ", end - start)
        
    def Reconstruct2D(self):
        print('Reconstruct2D')
        start = time.time()
        """
        Perform reconstruction
        calculate e1, e2, ex, ey, ez
        """
        k_perp = (self.fn[:, None, None]**2.+ self.fn[None, :, None]**2.+ np.zeros(self.NMesh)[None, None, :]**2)**(0.5)
        k_ind = self.k_ind
        k_perp[0,0,:] = 1
        del self.Data, self.deltaX
        deltaX1 = ifftn(self.deltaK1)
        deltaX2 = ifftn(self.deltaK2)
        deltaX3 = ifftn(self.deltaK3)
        del self.deltaK1, self.deltaK2, self.deltaK3
        e1K = fftn((deltaX1*deltaX1 - deltaX2*deltaX2)*0.5)
        e2K = fftn(deltaX1*deltaX2)

        # Calculate coefficients
        k1 = (self.fn[:, None, None]**2 - self.fn[None, :, None]**2 + np.zeros(self.NMesh)[None, None, :])*k_ind**2/k_perp**4
        k1[0,0,:] = 0
        k2 = (2*self.fn[:, None, None]*self.fn[None, :, None]*np.ones(self.NMesh)[None, None, :])*k_ind**2/k_perp**4
        k2[0,0,:] = 0
        e0K = e1K*k1 + e2K*k2

        ###############################################################################################################
        e0X = ifftn(e0K).real
#         self.deltaK_re = fftn(e0X)
        self.deltaX_re = self.DeBias(self.deltax, e0X)
#         self.OneDimDeBias()
        
#         self.deltaX_re = e0X
        end = time.time()
        print("Time Consumed in Reconstruction2D is ", end - start)
    
    def PhotoZ(self, SigmaKai):
        print('PhotoZ')
        """
        Add PhotoZ error to Reconstructed field
        """
        self.k_para = (np.zeros(self.NMesh)[:, None, None]**2.
				  + np.zeros(self.NMesh)[None, :, None]**2.
				  + self.fn[None, None, :]**2)**(0.5)
        
        ErrorWindow = np.exp(-0.5*self.Kf*self.k_para**2*SigmaKai**2)

        self.deltaK *= ErrorWindow
        self.deltaX = ifftn(self.deltaK).real*self.NMesh**3
        
        del ErrorWindow
        
    def CutOff(self, Data, threshold, BoxSize):
        deltaX = Data - 1
        deltaK = fftn(deltaX)
        Kf = 2*np.pi/BoxSize
        Num = int(np.floor(threshold/Kf))
        deltaK[Num:, Num:, Num:] = 0
        Data = ifftn(deltaK) + 1
        return Data
    

        