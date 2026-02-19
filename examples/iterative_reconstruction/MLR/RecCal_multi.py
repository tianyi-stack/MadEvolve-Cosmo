import sys
import time
import numpy as np
from CosCalNL_multi.FastLoop_multi import FastLoop_multi
from Quijote_lib import readgadget
from Quijote_lib import readfof

NMesh = 512
BoxSize = 1000
R_init = 10
R_min = 1
Para = sys.argv[1]
Num = sys.argv[2]

def main():
    start = time.time()
    
    # Simulation box configuration
    fn = np.fft.fftfreq(NMesh, 1. / NMesh).astype(np.float32)
    rfn = np.fft.rfftfreq(NMesh, 1. / NMesh).astype(np.float32)
    fnx = (fn[:, None, None]
              + np.zeros(NMesh, dtype = np.float32)[None, :, None]
              + np.zeros(int(NMesh/2 + 1), dtype = np.float32)[None, None, :])
    fny = (np.zeros(NMesh, dtype = np.float32)[:, None, None]
              + fn[None, :, None] 
              + np.zeros(int(NMesh/2 + 1), dtype = np.float32)[None, None, :])
    fnz = (np.zeros(NMesh, dtype = np.float32)[:, None, None]
              + np.zeros(NMesh, dtype = np.float32)[None, :, None]
              + rfn[None, None, :])
    k_ind = ((fn[:, None, None]**2.
              + fn[None, :, None]**2.
              + rfn[None, None, :]**2.)**(0.5)).astype(np.float32)
    # Fundamental Frequency and Nyquist Frequency
    Kf = 2*np.pi/BoxSize
    
    
    # Reading the snapshots
    snapPATH = '/scratch/zangsh/Quijote_Simulations/Snapshots/' + Para + '/' + Num + '/snapdir_004/snap_004'
    snapshot = snapPATH
    ptype   = [1] 
    Position = (readgadget.read_block(snapPATH, "POS ", ptype)/1e3).astype(np.float32)
    Size = Position.shape[0]
    Position0 = np.array(Position)
    
    # Loop to iterate
    for i in range(8):
        # Paint the catalog on the mesh using cic
        delta = FastLoop_multi.CICPaint_multi(Position, NMesh, BoxSize, Size) - 1
        
        # Calculate the smoothiing scale
        R = max(0.5**i*R_init, R_min)
        
        # Apply the Gaussian window function
        deltaK = np.fft.rfftn(delta)
        window = np.exp(-0.5 * (k_ind * Kf) ** 2. * R ** 2.).astype(np.float32)
        deltaK *= window
#         deltaK = Truncate(deltaK)
        
        # Calculate the Zeldovich displacement
        k_ind[0,0,0] = 1
        temp = -1j*deltaK/k_ind**2/Kf
        k_ind[0,0,0] = 0
        Dis = np.empty([NMesh, NMesh, NMesh, 3], dtype = np.float32)
        Dis[:, :, :, 0]= np.fft.irfftn(temp*fnx).real.astype(np.float32)
        Dis[:, :, :, 1] = np.fft.irfftn(temp*fny).real.astype(np.float32)
        Dis[:, :, :, 2] = np.fft.irfftn(temp*fnz).real.astype(np.float32)
        
        Position = FastLoop_multi.Shift_multi(Position.astype(np.float32), Dis.astype(np.float32), NMesh, BoxSize, Size)
    
    # Generate the displacement field
    Density = FastLoop_multi.CICPaint_multi(Position, NMesh, BoxSize, Size)
    Dis = FastLoop_multi.DisInter(Position0.astype(np.float32), 
                                  Position.astype(np.float32), 
                                  Density, 
                                  NMesh, 
                                  BoxSize, 
                                  Size)
    sx = Dis[:, :, :, 0]
    sy = Dis[:, :, :, 1]
    sz = Dis[:, :, :, 2]
    del Dis
#     dxk = Truncate(np.fft.rfftn(sx))
#     dyk = Truncate(np.fft.rfftn(sy))
#     dzk = Truncate(np.fft.rfftn(sz))
    dxk = np.fft.rfftn(sx)
    dyk = np.fft.rfftn(sy)
    dzk = np.fft.rfftn(sz)
    delta0kx = (1j*Kf)*fnx*dxk
    delta0ky = (1j*Kf)*fny*dyk
    delta0kz = (1j*Kf)*fnz*dzk
    deltaX0 = np.fft.irfftn(delta0kx + delta0ky + delta0kz)

    # Calculating power
    DenSavePATH = '/scratch/zangsh/Quijote_Simulations/Density/' + Para + '/' + str(Num) + '/Den_DMDisIter2_512.bin'

    deltaX0.astype(np.float32).tofile(DenSavePATH)
    
    end = time.time()
    print('%s %s finished, costs %f s.'%(Para, Num, end - start))
        
if __name__ == '__main__':
    main()
