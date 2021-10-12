#!/usr/bin/env python

import numpy as np
import wuml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import wplotlib 


class dimension_reduction:
	def __init__(self, data, n_components=2, method='PCA', learning_rate=30, show_plot=False, kernel='rbf'):
		'''
			n_components: number of dimension to reduce down to
			method: 'PCA', 'TSNE', 'KPCA'
			learning_rate: used for TSNE, if too large, everything become equidistance, defulat=30
		'''

		X = wuml.ensure_numpy(data)
		if method == 'PCA':
			model = PCA(n_components=X.shape[1])
			use_all_dims = model.fit_transform(X)
			self.Ӽ = use_all_dims[:,0:n_components]

			self.normalized_eigs = model.explained_variance_ratio_
			self.eigen_values = model.singular_values_
			self.cumulative_eigs = np.cumsum(self.eigen_values)/np.sum(self.eigen_values)

		elif method == 'TSNE':
			model = TSNE(n_components=n_components, learning_rate=learning_rate)
			self.Ӽ = model.fit_transform(X)
		elif method == 'KPCA':
			model = KernelPCA(n_components=n_components, kernel=kernel)
			self.Ӽ = model.fit_transform(X)

		self.model = model
		self.method = method

		if show_plot:
			lp = wplotlib.scatter(figsize=(10,5))		# (width, height)
			lp.plot_scatter(self.Ӽ[:,0], self.Ӽ[:,1], 'Data After ' + method, 'X axis', 'Y axis')

	def __call__(self, X):
		if self.method == 'PCA' or self.method == 'KPCA':
			return self.model.transform(X)
		elif self.method == 'TSNE':
			raise ValueError('TSNE does not have a transform function.')
		

	def __getitem__(self, item):
		return self.Ӽ[item]

	def __str__(self):
		return str(self.Ӽ)

