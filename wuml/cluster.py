import numpy as np
import pandas as pd
import wuml
import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')

import wplotlib
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

class clustering:
	'''
	Automatically run clustering on data

	data: can be any data type
	method='Spectral Clustering', 'KMeans', 'GMM', 'Agglomerative'
	gamma= only used for Spectral clustering, default to 1
	'''

	def __init__(self, data, n_clusters, y=None, y_column_name=None, method='Spectral Clustering', kernel=None, gamma=1):
		NP = wuml.ensure_numpy
		S = np.squeeze

		self.X = X = NP(data)
		if y is not None:
			y = S(NP(y))
		elif y_column_name is not None:
			y = data[y_column_name].values
		elif type(data).__name__ == 'wData':
			try: y = data.Y
			except: pass

		if method == 'Spectral Clustering':
			model = SpectralClustering(n_clusters=n_clusters,assign_labels='discretize', random_state=0, gamma=gamma)
		elif method == 'KMeans':
			model = KMeans(n_clusters=n_clusters, random_state=0)
		elif method == 'GMM':
			model = GaussianMixture(n_components=n_clusters, random_state=0)
			model.fit(X)
			self.labels = model.predict(X)
		elif method == 'Agglomerative':
			model= AgglomerativeClustering(n_clusters=n_clusters)
		else: raise ValueError('Unrecognized Clustering Method')


		try:
			model.fit(X)
			self.labels = model.labels_
		except: pass

		self.method = method
		self.y = y
		self.n_clusters = n_clusters
		self.data = data

	def __str__(self):
		return str(self.labels)

	def plot_scatter_result(self, dimension_reduction_method='all'):
		methods = ['PCA', 'TSNE', 'KPCA']
		if dimension_reduction_method == 'all':
			for i, m in enumerate(methods):
				title = 'Using %s'%(self.method)
				imgText = '%d clusters\n%s'%(self.n_clusters, m)
				ϰ = wuml.dimension_reduction(self.X, method=m)
				if i == 0:
					plts = wplotlib.plot_clusters(ϰ, self.labels, title=title, subplot=131+i, figsize=(10,3), imgText=imgText)
				else:
					plts = wplotlib.plot_clusters(ϰ, self.labels, title=title, subplot=131+i, imgText=imgText)
				
			plts.show()
		else:
			title = '%d Clusters via %s'%(self.n_clusters, dimension_reduction_method)
			ϰ = wuml.dimension_reduction(self.X, method=dimension_reduction_method)
			wplotlib.plot_clusters(ϰ, self.labels, title=title)


	def nmi_against_true_label(self, print_out=True):
		NPR = np.round
		if self.y is None: raise Warning("warning: y is not defined!!")

		nmi = wuml.NMI(self.y, self.labels)

		column_names = ['Method', 'NMI']
		data = np.array([[self.method, nmi]])

		df = pd.DataFrame(data, columns=column_names,index=[''])
		if print_out: print(df)
		self.results = df
		return df


	def save_clustering_result_to_csv(self, path, with_data=False):
		if with_data:
			X = wuml.ensure_wData(self.data)
			X.append_columns(self.labels, column_names='cluster_labels')
			X.to_csv(path, float_format='%d')
		else:
			wuml.csv_out(self.labels, path, float_format='%d')


def run_every_clustering_algorithm(data, n_clusters, gamma=1):
	X = wuml.ensure_numpy(data)
	c_methods= ['Spectral Clustering', 'KMeans', 'GMM', 'Agglomerative']

	results = {}
	for C in c_methods:
		results[C] = clustering(data, n_clusters, method=C, gamma=gamma)
		results[C].plot_scatter_result()


