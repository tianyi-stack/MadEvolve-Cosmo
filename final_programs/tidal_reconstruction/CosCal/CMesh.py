import sys
sys.path.append('../')
import time
import numpy as np
# import pandas as pd
# from nbodykit.lab import *
# from pandas import Series, DataFrame
from CosCal.CCatalog import CCatalog
# import matplotlib.pyplot as plt
from numpy import fft
from numpy.fft import fftn,ifftn
from CosCal.CIC import CIC
from CosCal.NGP import NGP

class CMesh():
	""" Mesh Element class """
	def __init__(self):
		super().__init__()
		self.NMesh = 16 # Default mesh number is 16
		self.Data = None # Array to save denstiy data
		self.BoxSize = 1.0 # Default Box size is 1
		self.Window = 'cic' # Default Window Kernel is CIC
		self.Size = 0
	
	def GetNMesh(self):
		return self.NMesh
	
	def GetData(self):
		return self.Data

	def GetdeltaK(self):
		return self.deltaK

	def GetdeltaX(self):
		return self.deltaX

	def GetBoxSize(self):
		return self.BoxSize

	def GetNUMP(self):
		return self.NUMP

	def GetSize(self):
		return self.Size
	
	def ParameterSet(self):
		"""
		Set Parameters after reading data
		"""
		# Calculate Mesh Size
		self.H = self.BoxSize/self.NMesh
		# Wave Number Parameters
		self.fn = np.fft.fftfreq(self.NMesh, 1. / self.NMesh).astype(np.float64)
		self.k_ind = ((self.fn[:, None, None]**2.
				  + self.fn[None, :, None]**2.
				  + self.fn[None, None, :]**2.)**(0.5)).astype(np.float32)
		# Fundamental Frequency and Nyquist Frequency
		self.Kf = 2*np.pi/self.BoxSize
		self.Knyq = np.pi*self.NMesh/self.BoxSize
		# Generate deltaX
		self.deltaX = (self.Data - 1).astype(np.float32)
		# fft
		self.deltaK = fftn(self.deltaX)

	def ReadBin(self, FilePath, NMesh, BoxSize):
		print('Mesh Read From Binary File')
		start = time.time()     
		"""
		Read Mesh Value from binary file
		"""
		# Update Mesh Number
		self.NMesh = NMesh
		# Read Box size
		self.BoxSize = BoxSize
		with open(FilePath, 'rb') as f:
			input_array = np.fromfile(f,dtype=np.float32)
		self.Data = input_array.reshape(NMesh, NMesh, NMesh).astype(np.float32)
		self.ParameterSet()
		end = time.time()
		print('Time consumed in Reading Binary File is ', end - start)
    
	def ReadArray(self, Array, NMesh, BoxSize):
		print('Read From Array')
		self.Data = Array
		self.BoxSize = BoxSize
		self.NMesh = NMesh
		self.ParameterSet()
        

	def ReadCatalog(self, Catalog, NMesh, BoxSize, Window, Interlace = False):
		print('Mesh Read')
		start = time.time()
		# Update Mesh Number
		self.NMesh = NMesh
		# Read Box size
		self.BoxSize = BoxSize
		# Read Window type
		self.Window = Window
		# Allocate space for mesh data
		self.Data = np.zeros([self.NMesh, self.NMesh, self.NMesh])
		# Calculate Mesh Size
		self.H = self.BoxSize/self.NMesh
		# Paint
		self.Size = Catalog.GetSize()
		Position = Catalog.GetPosition()
		
		# CIC Window Kernel
		if self.Window == 'cic':
			self.Data = CIC.CICPaint(Position.astype(np.float32), self.NMesh, self.BoxSize, self.Size)

		# NGP Window Kernel
		if self.Window == 'ngp':
			self.Data = NGP.NGPPaint(Position, self.NMesh, self.BoxSize, self.Size)

		# Interlace
		if Interlace == True:
			Data1 = self.Data
			self.Data = np.zeros([self.NMesh, self.NMesh, self.NMesh])
			for i in range(self.Size):
				self.CIC(Position[i,:] + 0.5*np.array([self.H, self.H, self.H]))
			self.Data = 0.5*(Data1 + self.Data)

		self.ParameterSet()
		end = time.time()
		print("Time Consumed in Painting Mesh is ", end - start)

	def ReadMeshValue(self, Mesh):
		# Read mesh data from mesh
		print('Mesh Read')
		self.Data = Mesh.paint(mode='real').value
		self.NMesh = Mesh.attrs['Nmesh'][0]
		self.BoxSize = Mesh.attrs['BoxSize'][0]
		self.Size = Mesh.attrs['N']

		self.ParameterSet()

	# NGP Function
	def NGP(self, Position):
		N = np.zeros(3)
		for i in range(3):
			N[i] = int(np.floor(Position[i]/self.H + 0.5))
		Np = [int(N[0]), int(N[1]), int(N[2])]
		for i in range(3):
			if Np[i] < 0:
				Np[i] = self.NMesh - 1
			if Np[i] >= self.NMesh:
				Np[i] = 0
		W = 1
		self.Data[Np[0], Np[1], Np[2]] += W

	# Paint 1+delta
	def Paint(self):
		print('Mesh Paint')
		databar = self.Data.mean()
		for I in range(self.NMesh):
			for J in range(self.NMesh):
				for K in range(self.NMesh):
					self.Data[I,J,K] = self.Data[I,J,K]/databar

	def deltaXUpdate(self, deltaX):
		"""
		Update deltaX mesh and deltaK mesh
		"""
		self.deltaX = deltaX
		self.deltaK = fftn(self.deltaX)/(self.NMesh**3)
		
		del deltaX

	def deltaKUpdate(self, deltaK):
		"""
		Update deltaX mesh and deltaK mesh
		"""
		self.deltaK = deltaK
		self.deltaX = ifftn(self.deltaK)*(self.NMesh**3)
		
		del deltaK
