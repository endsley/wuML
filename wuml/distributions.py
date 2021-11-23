
from scipy.stats import multivariate_normal
import numpy as np
import wuml

class multivariate_gaussian: #Assuming that data is 1 dimension
	def __init__(self, dat=None, mean=None, cov=None):
		self.X = X = wuml.ensure_numpy(dat)	

		if X is not None:
			μ = np.mean(X, axis=0)
			Σ = np.corrcoef(X.T)

			self.pₓ = multivariate_normal(mean=μ, cov=Σ)

		elif mean is not None and cov is not None:
			self.pₓ = multivariate_normal(mean=mean, cov=cov)

	def generate_samples(self, num_of_samples, return_data_type='wData'):
		dat = self.pₓ.rvs(size=num_of_samples, random_state=None)
		return ensure_data_type(dat, type_name=return_data_type)


	def __call__(self, data, return_log_likelihood=False, return_data_type='wData'):
		inp = wuml.ensure_numpy(data)	
		prob = self.pₓ.pdf(inp)

		if return_log_likelihood: output = np.log(prob)
		else: output = prob

		return ensure_data_type(output, type_name=return_data_type)


class exponential: #Assuming that data is 1 dimension
	def __init__(self, X=None, mean=None):
		if X is not None:
			X = np.squeeze(wuml.ensure_numpy(X))
			self.λ = 1/np.mean(X, axis=0)
		elif mean is not None:
			self.λ = 1/mean

	def __call__(self, x):
		try: x[x<0]=np.NAN
		except: pass

		λ = self.λ
		S = λ*np.exp(-λ*x)

		try: S[np.isnan(S)] = 0
		except: pass

		return S


	def generate_samples(size=1):
		# λ exp(-λx) , μ=1/λ
		return np.random.exponential(scale=self.λ, size=size)



	def cdf(self,x):
		λ = self.λ
		return (1 - np.exp(-λ*x))

		#[Area, error] = wuml.integrate(self.__call__, 0, x)
		#return Area

