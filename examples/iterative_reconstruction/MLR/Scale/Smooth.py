import numpy as np

DenPATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/fiducial/0/Den_IC_Scale.bin'
SavePATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/fiducial/0/Den_IC_Slice.bin'
R = 2
NMesh = 512
BoxSize = 1000

def main():
    delta = np.fromfile(DenPATH, dtype = np.float32).reshape([NMesh, NMesh, NMesh]) - 1
    deltak = np.fft.fftn(delta)
    
    fn = np.fft.fftfreq(NMesh, 1. / NMesh).astype(np.float32)
    k_ind = ((fn[:, None, None]**2.
              + fn[None, :, None]**2.
              + fn[None, None, :]**2.)**(0.5)).astype(np.float32)
    Kf = 2*np.pi/BoxSize
    window = np.exp(-0.5 * (k_ind * Kf) ** 2. * R**2.).astype(np.float32)
    deltak *= window
    data = np.fft.ifftn(deltak).real + 1
    data.astype(np.float32).tofile(SavePATH)

main()