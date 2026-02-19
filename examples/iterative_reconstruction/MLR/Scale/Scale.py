import numpy as np

DenPATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/fiducial/0/Den_IC_512.bin'
SavePATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/fiducial/0/Den_IC_Scale.bin'
R = 2
NMesh = 512
BoxSize = 1000
f = 100.3

def main():
    delta = np.fromfile(DenPATH, dtype = np.float32).reshape([NMesh, NMesh, NMesh]) - 1
    deltak = np.fft.fftn(delta)
    deltak *= f
    data = np.fft.ifftn(deltak).real + 1
    data.astype(np.float32).tofile(SavePATH)

main()