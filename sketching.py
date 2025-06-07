from abc import ABC, abstractmethod
import numpy as np
from scipy.fft import dct

# Abstract base class (never instantiated)
class Sketch(ABC):
	def __init__(self, m, s, seed= None):
		self.m = m
		self.s = s
		self.rng= np.random.default_rng(seed= seed)

	@abstractmethod
	def apply_sketch(self, vecs):
		pass

	# Overload of the @ operator (allows to write S @ v)
	def __matmul__(self, vecs):
		return self.apply_sketch(vecs)
	
	@abstractmethod
	def get_matrix(self):
		pass
	
class GaussianSketch(Sketch):
	def __init__(self, m, s, seed= None):
		super().__init__(m, s, seed)
		self.S= self.rng.normal(size= (self.s, self.m))/np.sqrt(self.s)
	
	def apply_sketch(self, vecs):
		return self.S @ vecs
	
	def get_matrix(self):
		return self.S
	
class SRDCT(Sketch) :
	def __init__(self, m, s, seed= None):
		super().__init__(m, s, seed)
		self.rade_vec= np.sqrt(m/s)*self.rng.choice([-1.0, 1.0], size= self.m)
		self.indices= np.sort(self.rng.choice(self.m, size= self.s, replace=False))

	def apply_sketch(self, vecs):
		if vecs.ndim == 1 :
			s_vec= self.rade_vec * vecs
			s_vec= dct(s_vec, type= 2, norm= 'ortho', axis= 0)
			s_vec= s_vec[self.indices]
		else :
			s_vec= self.rade_vec[:, None] * vecs
			s_vec= dct(s_vec, type= 2, norm= 'ortho', axis= 0)
			s_vec= s_vec[self.indices, :]
		return s_vec
	
	def get_matrix(self):
		return (self.rade_vec[:, None]*dct(np.eye(self.m), type= 2, norm= 'ortho', axis= 0))[self.indices, :]