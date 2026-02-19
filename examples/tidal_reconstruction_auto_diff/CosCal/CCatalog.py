import time
import math
# import h5py
import numpy as np
from scipy import integrate
# from nbodykit.lab import *

class CCatalog():
    """ Catalog Element class """
    def __init__(self):
        super().__init__()
        self.Position = None
        self.Velocity = None
        self.Mass = None
        self.Size = 0
        self.H0 = 1 # Habour Constant

    def GetPosition(self):
        return self.Position

    def GetVelocity(self):
        return self.Velocity

    def GetMass(self):
        return self.Mass

    def GetSize(self):
        return self.Size

    def GetNUMP(self):
        return self.NUMP

    def GetH0(self):
        return self.H0
    
    def Getlog10Mvir(self):
        return self.log10Mvir

    def Read(self, Catalog):
        # Read data from catalog
        self.Position = Catalog.compute(Catalog['Position'])
        self.Size = Catalog.size

    def RSD(self, H, z, direction):
        """
        Add redshift distortion effect
        xs = x + dot(v, xhat)/H
        """
        print(self.Velocity.std()*0.6774)
        self.Position[:, direction] += (67.74/100)*self.Velocity[:, direction]/((1/(z + 1)) * H)
        self.Velocity = (67.74/100)*self.Velocity/((1/(1+z)) * H)
        self.Position = self.Position%self.BoxSize
        
    def PhotoZ(self, SigmaKai):
        """
        Add Photo-Z error
        SigmaKai = h*c*(1 + z)/Hz *Error
        0.003:15.20465894391183  10.299635968605875 
        0.01:50.6821964797061 34.33211989535291
        """
        np.random.seed(424)
        PhotoZ = np.random.randn(self.Size)*SigmaKai
        self.Position[:, 2] += PhotoZ
        
    def ReadCSV(self, filename, BoxSize, NMesh):
        """
        Read from csv
        """
        names =['x', 'y', 'z', 'vx', 'vy', 'vz','log10M']

        # read the catalog data
        f = CSVCatalog(filename, names)
        f['Position'] = f['x'][:, None] * [1, 0, 0] * BoxSize+ f['y'][:, None] * [0, 1, 0] * BoxSize + f['z'][:, None] * [0, 0, 1] * BoxSize
        f.attrs['BoxSize'] = BoxSize
        mesh = f.to_mesh(window='cic', Nmesh=NMesh, BoxSize = BoxSize)
        mesh.save('halo/data/fof_PVL10M_Ng256.bigfile', mode='real', dataset='Field')
        
    def ReadBigFile(self, filePATH, BoxSize, NMesh):
        """
        Read From Big File 
        """
        self.H = BoxSize/NMesh
        self.Size = int(np.fromfile(filePATH + '/Position/000000',dtype=np.float64).size/3)
        self.Position = np.fromfile(filePATH + '/Position/000000',dtype=np.float64).reshape(self.Size, 3)
        self.Velocity = np.fromfile(filePATH + '/Velocity/000000',dtype=np.float64).reshape(self.Size, 3)
        self.log10Mvir = np.fromfile(filePATH + '/log10Mvir/000000',dtype=np.float64)
        self.BoxSize = BoxSize
        
    def ReadTrial(self, Size, Position, Velocity, BoxSize, NMesh):
        """
        Read for debug
        """
        self.H = BoxSize/NMesh
        self.Position = Position
        self.Velocity = Velocity
        self.Size = Size
    
    def Readfof(self, filePATH, BoxSize, NMesh):
        """
        Read from fof data file
        """
        self.H = BoxSize/NMesh
        f = open(filePATH)
        content = f.readlines()
        self.Size = len(content)

        self.Position = np.zeros([self.Size, 3])
        self.Velocity = np.zeros([self.Size, 3])
        self.log10Mvir = np.zeros(self.Size)
        
        for i in range(self.Size):
            self.Position[i, :] = content[i].split()[0:3]
            self.Velocity[i, :] = content[i].split()[3:6]
            self.log10Mvir[i] = content[i].split()[6]
        
        self.Position *= BoxSize
    
    def Readhdf5(self, filePATH, BoxSize, NMesh):
        """
        Read from hdf5 file
        """
        self.H = BoxSize/NMesh
        self.Position = np.zeros([1,3])
        self.Velocity = np.zeros([1,3])
        
        for i in range(64):
            print(i)
            filename = filePATH + '/snap_0.6250.' + str(i) + '.hdf5'
            f = h5py.File(filename, "r")
            Position = f['/PartType1']['/PartType1/Coordinates']
            Velocity = f['/PartType1']['/PartType1/Velocities']
            self.Size += Position.shape[0]
            self.Position = np.vstack((self.Position,Position))
            self.Velocity = np.vstack((self.Velocity,Velocity))
        self.Position = self.Position[1:, :]
        self.Velocity = self.Velocity[1:, :]
       
    def Interlace(self):
        self.Position += self.H*0.5*np.ones([self.Size, 3])
        
    def Select(self, threshold):
        select = self.log10Mvir > threshold
        self.Position = self.Position[select, :]
        self.Velocity = self.Velocity[select, :]
        self.log10Mvir = self.log10Mvir[select]
        self.Size = self.log10Mvir.size
        
    def GetProb(self, log10Mmin, Sigmalog10M):
        """
        Generate Probability
        """
        Prob = np.zeros(self.Size)
        for i in range(self.Size):
            Prob[i] = 0.5*(1 + (2/(np.pi)**0.5)*integrate.quad(lambda t : math.exp(-t**2), 0, (self.log10Mvir[i] - log10Mmin)/Sigmalog10M)[0])
        return Prob
            
    
    def HOD(self, log10Mmin, Sigmalog10M):
        """
        Apply Halo Occupation Distribution to the Catalog
        log10Mmin: Typical Minimun Mass
        Sigmalog10M: Profile of the soft mass cutoff
        """
        print("HOD")
        start = time.time()
        
#         self.Select(log10Mmin)
        np.random.seed(424)
        Random = np.random.rand(self.Size)
        Prob = self.GetProb(log10Mmin, Sigmalog10M)
        select = Random < Prob
        self.Position = self.Position[select, :]
        self.Velocity = self.Velocity[select, :]
        self.log10Mvir = self.log10Mvir[select]
        self.Size = self.log10Mvir.size
        
        end = time.time()
        print("Time Consumed in HOD is ", end - start)
        
    def MassCut(self, log10Mmin):
        print("Mass Cut")
        select = self.log10Mvir > log10Mmin
        self.Position = self.Position[select, :]
        self.Velocity = self.Velocity[select, :]
        self.log10Mvir = self.log10Mvir[select]
        self.Size = self.log10Mvir.size
