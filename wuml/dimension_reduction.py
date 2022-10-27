#!/usr/bin/env python

import numpy as np
import wplotlib 
import wuml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import FactorAnalysis

from wuml.type_check import *

class dimension_reduction:
	def __init__(self, data, n_components=None, method='PCA', learning_rate=30, show_plot=False, kernel='rbf', n_neighbors=5, gamma=1 ):
		'''
			n_components: number of dimension to reduce down to
			method: 'PCA', 'TSNE', 'KPCA', 'isoMap', 'LLE', 'MDS', 'Spectral Embedding','Factor Analysis', 'LDA'
			learning_rate: used for TSNE, if too large, everything become equidistance, defulat=30
			n_neighbors: used for isoMap
			gamma: used for KPCA
		'''

		df = ensure_DataFrame(data)
		X = ensure_numpy(data)
		if method == 'PCA':
			if n_components is None: n_components = 2
			model = PCA(n_components=X.shape[1])
			
			use_all_dims = model.fit_transform(X)
			self.Ӽ = use_all_dims[:,0:n_components]

			self.normalized_eigs = model.explained_variance_ratio_
			self.eigen_values = model.singular_values_
			self.cumulative_eigs = np.cumsum(self.eigen_values)/np.sum(self.eigen_values)
			column_names = ['λ%d'%x for x in range(1, model.components_.shape[1]+1)]
			self.eig_vectors = ensure_DataFrame(model.components_, columns=column_names, index=df.columns)
		elif method == 'LDA':
			NP = wuml.ensure_numpy

			model = LinearDiscriminantAnalysis()
			model.fit(NP(data.X), NP(data.Y))
			nX = wuml.ensure_numpy(data)

			if n_components is not None: model.coef_ = model.coef_[0:n_components,:]
			self.Ӽ = nX.dot(model.coef_.T)
		elif method == 'TSNE':
			if n_components is None: n_components = 2
			model = TSNE(n_components=n_components, learning_rate=learning_rate)
			self.Ӽ = model.fit_transform(X)
		elif method == 'KPCA':
			if n_components is None: n_components = 2
			model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
			self.Ӽ = model.fit_transform(X)
		elif method == 'isoMap':
			if n_components is None: n_components = 2
			model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
			self.Ӽ = model.fit_transform(X)
		elif method == 'LLE':
			if n_components is None: n_components = 2
			model = LocallyLinearEmbedding(n_components=n_components)
			self.Ӽ = model.fit_transform(X)
		elif method == 'MDS':
			if n_components is None: n_components = 2
			model = MDS(n_components=n_components)
			self.Ӽ = model.fit_transform(X)
		elif method == 'Spectral Embedding':
			if n_components is None: n_components = 2
			model = SpectralEmbedding(n_components=n_components)
			self.Ӽ = model.fit_transform(X)
		elif method == 'Factor Analysis':
			if n_components is None: n_components = 2
			model = FactorAnalysis(n_components=n_components, random_state=0)
			self.Ӽ = model.fit_transform(X)
		else:
			raise ValueError('Error: Unrecognized Dimension Reduction Method: %s.'%method)

		self.model = model
		self.method = method
		self.shape = self.Ӽ.shape

		if show_plot:
			lp = wplotlib.scatter(self.Ӽ[:,0], self.Ӽ[:,1], 'Data After ' + method, 'X axis', 'Y axis', figsize=(10,5))		# (width, height)

	def __call__(self, X):
		methods = ['PCA', 'KPCA', 'isoMap','LLE', 'Factor Analysis']
		if self.method in methods:
			return self.model.transform(X)
		elif self.method == 'LDA':
			nX = wuml.ensure_numpy(X)
			return nX.dot(self.model.coef_.T)
		else:
			raise ValueError('This function does not have a transform function yet.')
		

	def __getitem__(self, item):
		return self.Ӽ[item]

	def __str__(self):
		return str(self.Ӽ)


def show_multiple_dimension_reduction_results(data, learning_rate=15, n_neighbors=5, gamma=1):
	methods = ['PCA', 'TSNE', 'KPCA', 'isoMap', 'LLE', 'MDS', 'Spectral Embedding','Factor Analysis']

	results = {}
	#lp = wplotlib.scatter(figsize=(7,8))		# (width, height)
	for i, m in enumerate(methods):
		results[m] = dimension_reduction(data, n_components=2, method=m, 
											learning_rate=learning_rate, show_plot=False, 
											n_neighbors=n_neighbors, gamma=gamma)

		if i == 0:
			lp = wplotlib.scatter(results[m].Ӽ[:,0], results[m].Ӽ[:,1], m, '', '', subplot=421 + i, show=False, figsize=(7,12))
		else:
			wplotlib.scatter(results[m].Ӽ[:,0], results[m].Ӽ[:,1], m, '', '', subplot=421 + i)

	lp.show()
	return results
