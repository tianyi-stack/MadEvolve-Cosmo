import math
import numpy as np
from numpy import fft
from numpy.fft import fftn,ifftn
from CosCal.Utils import Utils
from CosCal.FastLoop import Dimension

class CPower(Utils):
    """ Power Spectrum class """
    def __init__(self):
        super().__init__()
        self.deltaX = None # 3-d 1+delta in real domain
        self.deltaK = None # 3-d 1+delta in frequency domain
        self.k = [] # average k in each bin
        self.Pk = [] # 1-d Power Spectrum in frequency domain
        self.Px = None # 1-d Power Spectrum in real domain
        self.PK = None # 3-d Power Spectrum in frequency domain
        self.PX = None # 3-d Power Spectrum in real domain
        self.Kf = 2*math.pi # Fundamental freqency
        self.Knyq = math.pi # Nyquist frequecy
        self.modes = [] # modes

    def GetPower(self, mode, Nmu = 5, Compensate = False):
        """
        Get power spectrum
        """
        self.CompensateJudge = Compensate
        self.Power()
        
        if mode == '1D':
            self.Power1D()
            # Output 1-dimension power spectrum
            return(self.k, self.Pk)
        if mode == '2DBin':
            self.Power2DBin(Nmu)
            # Output 2-dimension power sepctrum
            return(self.k_bin, self.Pkmu)
        if mode =='2D':
            self.Power2D()
            return(self.k_perp, self.k_para, self.Pk)
        if mode == '3D':
            # Output 3-dimension power sepctrum
            return self.PK
        else:
            raise ValueError('Power spectrum mode is not available')

    def GetShotNoise(self):
        """
        Calculate shot noise, it is not abstracted from power spectrum
        SN = 1/nbar
        """
        print('L is ', self.BoxSize)
        print('NUMP is ', self.NUMP)
        return self.BoxSize**3/self.NUMP

    def Power(self, Compensate = False):
        """
        Calculate 3-D Power Spectrum and related parameters
        """           
        self.NUMP = self.Size
        self.PK = np.real((self.deltaK/self.NMesh**3)*np.conjugate((self.deltaK/self.NMesh**3)))*self.BoxSize**3
        
        if self.CompensateJudge == True:
            self.Compensate()

    def Power1D(self):
        """
        Calculate Power Spectrum
        """ 
        print('1D Power')            
        self.modes, self.k, self.Pk = Dimension.UniDimension(self.PK.astype(np.float32), self.fn.astype(np.float64), self.NMesh)        
        self.Pk[0] = 0
        self.k *= self.Kf

    def Power2DBin(self, Nmu):
        """
        Calculate 2-d Power Spectrum, mu is the cos of los
        """
        print('2DBin Power')
        self.modes, self.k_bin, self.Pkmu = Dimension.BiDimensionBin(self.PK.astype(np.float32), self.fn.astype(np.float64), self.NMesh, Nmu)
        self.k_bin *= self.Kf
        
    def Power2D(self):
        """
        Calculate 2-d Power Spectrum, mu is the cos of los
        """
        print('2D Power')
        modes, self.k_perp, self.k_para, self.Pk = Dimension.BiDimension(self.PK.astype(np.float32), self.fn, self.NMesh)

    def DeConvolution(self, p):
        """
        DeConvolution for Window Kernel in Fourier Space
        Here's the algorith for CIC Kernel (p = 2)
        """
        print('DeConvolution')
        self.dn = np.sinc(self.Kf*self.fn/(2*self.Knyq))**(-p)
        self.d_ind = self.dn[:, None, None]*self.dn[None, :, None]*self.dn[None, None, :]
        self.deltaK *= self.d_ind
        self.deltaX  = ifftn(self.deltaK)
        self.Data = self.deltaX + 1
        del self.dn, self.d_ind

        # Aliasing
        # self.wn = 1/(1 - (2/3)*np.sin(np.pi*self.fn*self.Kf/(2*self.Knyq))**2)
        # self.w_ind = self.wn[:, None, None]*self.wn[None, :, None]*self.wn[None, None, :]
        # self.PK *= self.w_ind
    
    def Compensate(self):
        """
        Compensate both aliasing and deconvolution
        """
        print('Compensate')
        self.wn = 1/(1 - (2/3)*np.sin(np.pi*self.fn*self.Kf/(2*self.Knyq))**2)
        self.w_ind = self.wn[:, None, None]*self.wn[None, :, None]*self.wn[None, None, :]
        self.PK *= self.w_ind

