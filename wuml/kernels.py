
import sklearn.metrics
import numpy as np
import wuml

def rbk_kernel(X, σ, zero_diagonal=False, use_random_features=False, random_feature_method='rff', random_feature_width=500):
	#random_feature_method = 'rff' or 'orthogonal'

	if use_random_features:
		sorf = wuml.random_feature(sigma=σ, random_feature_method=random_feature_method, sample_num=random_feature_width)
		Kx = sorf.get_kernel(X)			# kernel matrix from orthogonal
		if zero_diagonal: np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	else:
		γ = 1.0/(2*σ*σ)
		Kx = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
		if zero_diagonal: np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0

	return Kx

def get_rbf_γ(X):
	σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
	γ = 1.0/(2*σ*σ)
	return γ

