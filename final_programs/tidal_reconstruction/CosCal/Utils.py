import numpy as np 
# from scipy import integrate 
from CosCal.CMesh import CMesh
# import matplotlib.pyplot as plt
# from pandas import Series, DataFrame
from numpy.fft import fftn,ifftn
from CosCal.FastLoop import Dimension
# plt.get_cmap('Blues') 
class Utils(CMesh):
    """
    Several tools performing on deltaX and deltaK
    """
    def __init__(self):
        super().__init__()

    	# Generate HeatMap of 1+delta on mesh
    def Preview(self, filename):
        print('Mesh Preview')
        # Calculate the average delta acrossing z axis
        Preview_Data = np.zeros([self.NMesh, self.NMesh])
        for I in range(self.NMesh):
            for J in range(self.NMesh):
                for K in range(self.NMesh):
                    Preview_Data[I,J] += (self.Data[I, J, K]-1)/self.NMesh

        df = DataFrame(Preview_Data)
        plt.imshow(df.values)
        plt.savefig(filename)

    def SlicePreview(self, Direction, SliceNumber, Path, Data):
        print('Slice Preview')
        """
        Preview a slice of data
        Direction: 1 x-axis, 2 y-axis, 3 z-axis
        SliceNumber: From 1 to NMesh
        Path: Location to save fig
        """

        if Direction == 1:
            # los = [1,0,0]
            View_Data = Data[SliceNumber - 1, :, :]
            df = DataFrame(View_Data)
            plt.imshow(df.values, vmin = -0.5, vmax = 2, cmap=plt.cm.Blues)
            plt.savefig(Path)
        
        if Direction == 2:
            # los = [0,1,0]
            View_Data = Data[:, SliceNumber - 1, :]
            df = DataFrame(View_Data)
            plt.imshow(df.values, vmin = -0.5, vmax = 2, cmap=plt.cm.Blues)
            plt.savefig(Path)

        if Direction == 3:
            # los = [0,0,1]
            View_Data = Data[:, :, SliceNumber - 1]
            df = DataFrame(View_Data)
            plt.imshow(df.values, vmin = -0.5, vmax = 2, cmap=plt.cm.Blues)
            plt.savefig(Path)
            
    def Slice(self, Direction, SliceNumber):
        print('Slice')
        """
        Preview a slice of data
        Direction: 1 x-axis, 2 y-axis, 3 z-axis
        SliceNumber: From 1 to NMesh
        Path: Location to save fig
        """
        
        if Direction == 1:
            # los = [1,0,0]
            View_Data = self.Data[SliceNumber - 1, :, :]
        
        if Direction == 2:
            # los = [0,1,0]
            View_Data = self.Data[:, SliceNumber - 1, :]

        if Direction == 3:
            # los = [0,0,1]
            View_Data = self.Data[:, :, SliceNumber - 1]
           
        return View_Data
            
    def Interlace(self):
        """
        Acting on deltaX mesh to reduce aliasing
        Move deltaX at 1/2 on each direction and average with original grid
        On Frequency domain this function is to return a coefficient matrix
        """
        self.deltaK = 0.5*self.deltaK *(1 +  (-1)**(self.fn[:, None, None] 
                    + self.fn[None, :, None] 
                    + self.fn[None, None, :]))
        self.deltaX = ifftn(self.deltaK)*(self.NMesh**3)

    def GaussianWindow(self, sigma):
        print('Smooth')
        """
        Utilize Gaussian Filter to smooth Power Spectrum
        sigma is R
        """
        window = np.exp(-0.5 * (self.k_ind * self.Kf) ** 2. * sigma**2.)
        self.deltaK *= window
        self.deltaX = (ifftn(self.deltaK)).real*self.NMesh**3
        del window
    
    def Bias(self, deltaXDM, deltaXHalo):
        print('Bias and Noise')
        """
        Calculate Bias and Noise
        """
        deltaKDM = fftn(deltaXDM)/self.NMesh**3
        deltaKHalo = fftn(deltaXHalo)/self.NMesh**3
        
        del deltaXDM, deltaXHalo
        
        Bias_1 = (np.conjugate(deltaKHalo)*deltaKDM).real
        Bias_2 = (np.conjugate(deltaKDM)*deltaKDM).real
        Noise_1 = (np.conjugate(deltaKHalo)*deltaKHalo).real
        
        del deltaKDM, deltaKHalo
        
        modes, k, bias1 = Dimension.UniDimension(Bias_1.astype(np.float32), self.fn, self.NMesh)
        modes, k, bias2 = Dimension.UniDimension(Bias_2.astype(np.float32), self.fn, self.NMesh)
        modes, k, noise1 = Dimension.UniDimension(Noise_1.astype(np.float32), self.fn, self.NMesh)
        
        bias2[0] = 1
        bias = bias1/bias2
        
#         Bias = Dimension.One2Three(bias, self.fn, self.NMesh)
        
#         Noise = (np.conjugate(deltaKHalo - Bias*deltaKDM)*(deltaKHalo - Bias*deltaKDM)).real*self.BoxSize**3
#         modes, k, noise = Dimension.UniDimension(Noise.astype(np.float64), self.fn, self.NMesh)

#         noise = (noise1 - bias1**2/bias2)*self.BoxSize**3
        noise = (noise1/bias**2 - bias2)*self.BoxSize**3
        noise[0] = 0
        k *= self.Kf
        bias[0] = 0

        return k, bias, noise
    
    def Bias2D(self, deltaXDM, deltaXHalo):
        print('Bias and Noise --2D')
        """
        Calculate 2D Bias and Noise
        """
        deltaKDM = fftn(deltaXDM)/self.NMesh**3
        deltaKHalo = fftn(deltaXHalo)/self.NMesh**3
        
        del deltaXDM, deltaXHalo
        
        Bias_1 = (np.conjugate(deltaKHalo)*deltaKDM).real
        Bias_2 = (np.conjugate(deltaKDM)*deltaKDM).real
        Noise_1 = (np.conjugate(deltaKHalo)*deltaKHalo).real
        
        del deltaKDM, deltaKHalo
        
        modes, k_perp, k_para, bias1 = Dimension.BiDimension(Bias_1.astype(np.float32), self.fn, self.NMesh)
        modes, k_perp, k_para, bias2 = Dimension.BiDimension(Bias_2.astype(np.float32), self.fn, self.NMesh)
        modes, k_perp, k_para, noise1 = Dimension.BiDimension(Noise_1.astype(np.float32), self.fn, self.NMesh)
        
        bias2[0,0] = 1
        bias = bias1/bias2
        bias[0,0] = 0
        
#         noise = (noise1 - bias1**2/bias2)*self.BoxSize**3
        noise = (noise1/bias**2 - bias2)*self.BoxSize**3
        noise[0, 0] = 0
        k_perp *= self.Kf
        k_para *= self.Kf
        return k_perp, k_para, bias, noise
    
    def Bias2DBin(self, deltaXDM, deltaXHalo, Nmu):
        print('Bias and Noise --2D-Bin')
        """
        Calculate 2D Bias and Noise to bins
        """
        deltaKDM = fftn(deltaXDM)/self.NMesh**3
        deltaKHalo = fftn(deltaXHalo)/self.NMesh**3
        
        del deltaXDM, deltaXHalo
        
        Bias_1 = ((np.conjugate(deltaKHalo)*deltaKDM).real).astype(np.float32)
        Bias_2 = ((np.conjugate(deltaKDM)*deltaKDM).real).astype(np.float32)
        Noise_1 = ((np.conjugate(deltaKHalo)*deltaKHalo).real).astype(np.float32)
        
        del deltaKDM, deltaKHalo
        
        modes, k, bias1 = Dimension.BiDimensionBin(Bias_1.astype(np.float32), self.fn, self.NMesh, Nmu)
        modes, k, bias2 = Dimension.BiDimensionBin(Bias_2.astype(np.float32), self.fn, self.NMesh, Nmu)
        modes, k, noise1 = Dimension.BiDimensionBin(Noise_1.astype(np.float32), self.fn, self.NMesh, Nmu)
        
        bias2[0,0] = 1
        bias = bias1/bias2
        bias[0,0] = 0
        
#         noise = (noise1 - bias1**2/bias2)*self.BoxSize**3
        noise = (noise1/bias**2 - bias2)*self.BoxSize**3
        noise[0, 0] = 0
        k *= self.Kf
        
        return k, bias, noise
        
    def SigmaKai(self, z, Omega_m, Omega_Lambda, H0, Error):
        """
        Calculate SigmaKai for PhotoZ error
        z : red shift
        Omega_m : 
        Omega_Lambda :
        H0 = Hubble Param (km/s/Mpc)
        Error = SigmaZ / (1 + z)
        """
        c = 299792.458
        Hz = H0*((1 + z)**3*Omega_m + Omega_Lambda)**0.5
        SigmaKai = c*(1 + z)/Hz *Error
        return SigmaKai
    
    def Distance(self, z0, Omega_m, Omega_Lambda, H0):
        """
        Calculate distance in red shift space
        """
        c = 299792.458
        f = lambda z : c/(H0*((1 + z)**3*Omega_m + Omega_Lambda)**0.5)
        distance, error = integrate.quad(f, 0, z0)
        return distance
    
    def DeBias(self, deltaXDM, deltaXHalo):
        print('OneDimDeBias')
        deltaKDM = fftn(deltaXDM)
        deltaKHalo = fftn(deltaXHalo)
        Pc = deltaKDM * np.conjugate(deltaKHalo)
        Pr = deltaKHalo * np.conjugate(deltaKHalo)
        tr = Pc/Pr
        tr[0,0,0] = 0
        tr_OneDim = Dimension.One2Three(tr, self.fn, self.NMesh)
        deltaKHalo *= tr_OneDim
        deltaXHalo = ifftn(deltaKHalo)
        
        return deltaXHalo + 1
    
    def Gauss(self, deltaX, R):
        """
        Gaussian filter exp(-k^2R^2/2)
        R is different according to different bin
        """
        print('Gauss Filter')
        # sigma of gaussian filter
        deltaK = fftn(deltaX)
        window = np.exp(-0.5 * (self.k_ind * self.Kf) ** 2. * R**2.).astype(np.float32)
        deltaK *= window
        deltaX = ifftn(deltaK)
        return deltaX
