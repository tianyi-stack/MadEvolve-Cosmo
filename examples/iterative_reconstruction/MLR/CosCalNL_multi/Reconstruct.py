import time
import numpy as np
from CosCalNL_multi.Fast import Fast
from CosCalNL_multi.FastLoop_multi import FastLoop_multi

class Reconstruct(Fast):
    def __init__(self, BoxSize = 1000, NMesh = 512, *args, **kwargs):
        super(Reconstruct, self).__init__()

    
    def Gaussian_Window(self, R):
        """
        Return a Gaussian window with scale R
        """
        self.deltaK = np.fft.rfftn(self.deltaX)
        window = np.exp(-0.5 * (self.k_ind * self.Kf) ** 2. * R**2.).astype(np.float32)
        self.deltaK *= window
        self.deltaK = self.Truncate(self.deltaK)
    
    def Zeldovich_Approx(self):
        """
        Return the displacement result by Zeldovich's approximation
        """
        self.k_ind[0,0,0] = 1
        temp = -1j*self.deltaK/self.k_ind**2/self.Kf
        self.k_ind[0,0,0] = 0
        self.Dis = np.empty([self.NMesh, self.NMesh, self.NMesh, 3], dtype = np.float32)
        self.Dis[:, :, :, 0]= np.fft.irfftn(temp*self.fnx).real.astype(np.float32)
        self.Dis[:, :, :, 1] = np.fft.irfftn(temp*self.fny).real.astype(np.float32)
        self.Dis[:, :, :, 2] = np.fft.irfftn(temp*self.fnz).real.astype(np.float32)
            
    def Shift(self):
        """
        Shift the particle on by interpolating displacement field to each particle.
        """
        self.Position = FastLoop_multi.Shift_multi(self.Position.astype(np.float32), self.Dis.astype(np.float32), self.NMesh, self.BoxSize, self.Size)

        del self.Dis
        
    def Displace_Reconstruct(self):
        """
        Reconstruct the density field according to the estimated displacement field.
        """
        self.Displace_Interpolate()
        self.delta0kx = (1j*self.Kf)*self.fnx*self.dxk
        self.delta0ky = (1j*self.Kf)*self.fny*self.dyk
        self.delta0kz = (1j*self.Kf)*self.fnz*self.dzk
        self.deltaX0 = np.fft.irfftn(self.delta0kx + self.delta0ky + self.delta0kz)
        
        
    def DeConvolution(self, p):
        """
        DeConvolution for Window Kernel in Fourier Space
        Here's the algorith for CIC Kernel (p = 2)
        """
        self.dn = np.sinc(self.Kf*self.fn/(2*self.Knyq))**(-p)
        self.rdn = np.sinc(self.Kf*self.rfn/(2*self.Knyq))**(-p)
        self.d_ind = self.dn[:, None, None]*self.dn[None, :, None]*self.rdn[None, None, :]
        self.deltaK *= self.d_ind
        del self.dn, self.d_ind
        
    def DeConvolutionxyz(self, p):
        """
        DeConvolution for Window Kernel in Fourier Space
        Here's the algorith for CIC Kernel (p = 2)
        """
        print('DeConvolutionxyz')
        self.dn = np.sinc(self.Kf*self.fn/(2*self.Knyq))**(-p)
        self.d_ind = self.dn[:, None, None]*self.dn[None, :, None]*self.dn[None, None, :]
        self.dxk = np.fft.fftn(self.dx)
        self.dyk = np.fft.fftn(self.dy)
        self.dzk = np.fft.fftn(self.dz)
        self.dxk *= self.d_ind
        self.dyk *= self.d_ind
        self.dzk *= self.d_ind
        del self.dn, self.d_ind
        
        
    
   
