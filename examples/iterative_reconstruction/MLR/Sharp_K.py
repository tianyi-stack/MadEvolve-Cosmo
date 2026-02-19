import sys
import time
import numpy as np

NMesh_org = 512
NMesh_new = 128
fac = NMesh_new/NMesh_org
BoxSize = 1000
Para = sys.argv[1]
Num = sys.argv[2]

def main():
    InputPATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/' + str(Para) + '/' + str(Num) + '/Den_DMDisIter_test.bin'
    OutputPATH = '/scratch/p/pen/zangsh/Quijote_Simulations/Density/to_wzy/Den_DMDisIter_128_fiducial_' + str(Num) + '.bin'

    deltak = np.fft.fftn(np.fromfile(InputPATH, dtype = np.float32).reshape([NMesh_org,NMesh_org, NMesh_org]) - 1)
    deltak_shift = np.fft.fftshift(deltak)
    deltak_shift_down = deltak_shift[int(NMesh_org/2 - NMesh_new/2):int(NMesh_org/2 + NMesh_new/2), int(NMesh_org/2 - NMesh_new/2):int(NMesh_org/2 + NMesh_new/2), int(NMesh_org/2 - NMesh_new/2):int(NMesh_org/2 + NMesh_new/2)]
    dealtk_down = np.fft.ifftshift(deltak_shift_down*fac**3)
    delta_down = np.fft.ifftn(dealtk_down).real

    den = delta_down + 1
    den.astype(np.float32).tofile(OutputPATH)

main()