#!/usr/bin/env python

import wuml 
import numpy as np


X = np.vstack((np.random.randn(10,4), np.random.randn(10,4) + 10))
print(X.shape)

#	Run a single method
DR = wuml.dimension_reduction(X, method='PCA', show_plot=True)
print(DR.shape)


#	Run a all methods
wuml.show_multiple_dimension_reduction_results(X, 2, learning_rate=14, n_neighbors=10, gamma=0.05)
