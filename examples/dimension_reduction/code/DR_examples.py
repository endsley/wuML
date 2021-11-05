#!/usr/bin/env python

import wuml 
import numpy as np


X1 = np.vstack((np.random.randn(10,1), 0.01*np.random.randn(10,1) + 10))
X2 = np.vstack((0.01*np.random.randn(10,1), np.random.randn(10,1) + 10))
X3 = 0.01*np.random.randn(20,2)
X = np.hstack((X1, X2, X3))
X = wuml.ensure_DataFrame(X, columns=['A','B','C','D'])
print('Dimension Before', X.shape)


#	Run a single method
Xᵈ = wuml.dimension_reduction(X, 2, method='PCA', show_plot=True)
print('Dimension After', Xᵈ.shape)
print('Eigen Vectors')
print(Xᵈ.eig_vectors)
print('\nNormalized Eigen Values')
print(Xᵈ.normalized_eigs)
print('\nAcccumulative Eigen Values')
print(Xᵈ.cumulative_eigs)

#	Run a all methods
wuml.show_multiple_dimension_reduction_results(X, learning_rate=14, n_neighbors=10, gamma=0.01)
