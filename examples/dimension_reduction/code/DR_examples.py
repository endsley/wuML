#!/usr/bin/env python

import wuml 
import numpy as np


X = np.vstack((np.random.randn(10,4), np.random.randn(10,4) + 10))
X_test = np.random.randn(2,4)
print(X)

DR = wuml.dimension_reduction(X, method='TSNE', show_plot=True, learning_rate=20)
print(DR)

DR = wuml.dimension_reduction(X, method='PCA', show_plot=True)
print(DR)
print(DR(X_test))

DR = wuml.dimension_reduction(X, method='KPCA', show_plot=True)
print(DR)
print(DR(X_test))


