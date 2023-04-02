
import wuml
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from wuml.type_check import *


class KDE:
	def __init__(self, data=None, load_model_path=None):
		if load_model_path is not None:
			#self.model = wuml.pickle_load(load_model_path)
			[self.model, self.columns] = wuml.pickle_load(load_model_path)
			return 

		self.columns = None
		if wtype(data) == 'wData':
			self.columns = data.columns

		data = ensure_numpy(data)

		# use grid search cross-validation to optimize the bandwidth
		params = {'bandwidth': np.logspace(-1, 1, 20)}
		self.grid = GridSearchCV(KernelDensity(), params)
	
		self.grid.fit(data)
		self.σ = self.grid.best_estimator_.bandwidth
		self.model = self.grid.best_estimator_

	def generate_samples(self, num_of_samples, return_data_type='wData'):
		rV = np.round(100*np.random.rand()).astype(int)
		
		samples = self.model.sample(num_of_samples, random_state=rV)
		return ensure_data_type(samples, type_name=return_data_type, column_names=self.columns)

	def integrate(self, x0, x1):	# This only works for 1D data
		[result, error] = wuml.integrate(self, x0, x1)
		return result

	def save(self, pth): 
		outDat = [self.model, self.columns]
		wuml.pickle_dump(outDat, pth)

	def __call__(self, data, return_log_likelihood=False):
		X = ensure_numpy(data)

		if return_log_likelihood: return self.model.score_samples(X)
		else: return np.exp(self.model.score_samples(X))

