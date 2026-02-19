import time
import numpy as np
from CosCalNL_multi.Base import Base
from CosCalNL_multi.FastLoop_multi import FastLoop_multi

class Fast(Base):
    def __init__(self, BoxSize = 1000, NMesh = 512, *args, **kwargs):
        super(Fast, self).__init__()

    def Paint(self, window = 'cic'):
        """
        Mapping the catalog onto the mesh. Both deconvolution and interlacing is operated.
        """
        if window == 'cic':
            self.Density = FastLoop_multi.CICPaint_multi(self.Position, self.NMesh, self.BoxSize, self.Size)
            self.deltaX = self.Density - 1
        self.deltaK = np.fft.rfftn(self.deltaX)
#         self.DeConvolution(2)
    
    def Displace_Interpolate(self):
        """
        Interpolate the displacement to the grid.
        """
        Dis = FastLoop_multi.DisInter(self.Position0.astype(np.float32), self.Position.astype(np.float32), self.Density, self.NMesh, self.BoxSize, self.Size)
        self.sx = Dis[:, :, :, 0]
        self.sy = Dis[:, :, :, 1]
        self.sz = Dis[:, :, :, 2]
        del Dis
        self.dxk = self.Truncate(np.fft.rfftn(self.sx))
        self.dyk = self.Truncate(np.fft.rfftn(self.sy))
        self.dzk = self.Truncate(np.fft.rfftn(self.sz))
        
    def Truncate(self, data):
        kmax = self.NMesh/2
        data[self.k_ind > kmax] = 0
        return data

    
