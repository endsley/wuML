#!/usr/bin/env python

import wuml 
import numpy as np

X = np.vstack((np.random.randn(10,4), np.random.randn(10,4) + 10))
X_test = np.random.randn(2,4)

DR = wuml.dimension_reduction(X, method='PCA', show_plot=True)

print(DR.normalized_eigs)
print(DR)
print(DR.eigen_vectors)

import pdb; pdb.set_trace()
