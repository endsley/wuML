#!/usr/bin/env python
import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
import numpy as np


X1 = np.vstack((np.random.randn(10,1), 0.01*np.random.randn(10,1) + 10))
X2 = np.vstack((0.01*np.random.randn(10,1), np.random.randn(10,1) + 10))
X3 = 0.01*np.random.randn(20,2)
X = np.hstack((X1, X2, X3))
X = wuml.ensure_DataFrame(X, columns=['A','B','C','D'])
wuml.jupyter_print('Dimension Before', X.shape)


#	Run a single method
Xᵈ = wuml.dimension_reduction(X, 2, method='PCA', show_plot=True)
wuml.jupyter_print('Dimension After', Xᵈ.shape)
wuml.jupyter_print('Eigen Vectors')
wuml.jupyter_print(Xᵈ.eig_vectors)
wuml.jupyter_print('\nNormalized Eigen Values')
wuml.jupyter_print(Xᵈ.normalized_eigs)
wuml.jupyter_print('\nAcccumulative Eigen Values')
wuml.jupyter_print(Xᵈ.cumulative_eigs)

#	Run a all methods
wuml.show_multiple_dimension_reduction_results(X, learning_rate=14, n_neighbors=10, gamma=0.01)
