
import numpy as np
import wuml

class model_as_exponential: #Assuming that data is 1 dimension
	def __init__(self, X):
		X = np.squeeze(wuml.ensure_numpy(X))
		self.λ = 1/np.mean(X, axis=0)

	def __call__(self, x):
		try: x[x<0]=np.NAN
		except: pass

		λ = self.λ
		S = λ*np.exp(-λ*x)

		try: S[np.isnan(S)] = 0
		except: pass

		return S

	def cdf(self,x):
		λ = self.λ
		return (1 - np.exp(-λ*x))

		#[Area, error] = wuml.integrate(self.__call__, 0, x)
		#return Area

