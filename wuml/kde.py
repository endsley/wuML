
import wuml
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from wuml.type_check import *


class KDE:
	def __init__(self, data):
		data = ensure_numpy(data)

		# use grid search cross-validation to optimize the bandwidth
		params = {'bandwidth': np.logspace(-1, 1, 20)}
		self.grid = GridSearchCV(KernelDensity(), params)
	
		self.grid.fit(data)
		self.σ = self.grid.best_estimator_.bandwidth
		self.kde = self.grid.best_estimator_

	def generate_samples(self, num_of_samples):
		return self.kde.sample(num_of_samples, random_state=0)

	def integrate(self, x0, x1):	# This only works for 1D data
		[result, error] = wuml.integrate(self, x0, x1)
		return result

	def __call__(self, data, return_log_likelihood=False):
		X = ensure_numpy(data)

		if return_log_likelihood: return self.kde.score_samples(X)
		else: return np.exp(self.kde.score_samples(X))

