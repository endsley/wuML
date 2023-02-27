#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import numpy as np
import wuml

def generate_data_from_same_distribution(n=30, with_regression=True, distribution_type='uniform'):
	#	Generate a data with 4 dimensions where
	#	x1 x2 has positive influence
	#	x3 has no influence
	#	x4 has negative influence
	if distribution_type=='uniform':
		X = np.random.rand(n, 4)
	elif distribution_type=='gaussian':
		X = np.random.randn(n, 4)

	Y = []
	for m in range(X.shape[0]):
		# 5x₁ + x₂ + x₁x₂ - 8x₄ - 2x₄x₄ + δ
		y = 5*X[m,0] + X[m,1] + X[m,0]*X[m,1] - 8*X[m,3] - 2*X[m,3]*X[m,3] + 0.01*np.random.randn()
		Y.append(y)
	
	if with_regression:
		dat = wuml.wData(X_npArray=X, Y_npArray=Y, column_names=['A','B','C','D'], label_type='continuous')
		dat.to_csv('shap_regress_example_gaussian.csv', add_row_indices=False, include_column_names=True)
	else:
		Y = np.array(Y)
		m = np.mean(Y)
		Y[Y > m] = 1
		Y[Y < m] = 0
	
		dat = wuml.wData(X_npArray=X, Y_npArray=Y, column_names=['A','B','C','D'], label_type='discrete')
		dat.to_csv('shap_classifier_example.csv', add_row_indices=False, include_column_names=True)

def generate_data_from_diff_distribution(n=30, with_regression=True):
	#	Generate a data with 4 dimensions where
	#	x1 x2 has positive influence
	#	x3 has no influence
	#	x4 has negative influence
	
	X1 = np.atleast_2d(np.random.randn(n)).T							# Gaussian Distribution
	X2 = np.atleast_2d(wuml.gen_exponential(λ=1, size=n)).T - 2			# Exponential Distribution
	X3 = np.atleast_2d(np.random.rand(n)).T								# Uniform distribution
	X4 = np.atleast_2d(wuml.gen_categorical(4, n, [0.2,0.2,0.1,0.5])).T	# Categorical distribution
	X = np.hstack((X1,X2,X3,X4))

	Y = []
	for m in range(X.shape[0]):
		y = 5*X[m,0] + X[m,1] + X[m,0]*X[m,1] - 8*X[m,3] - 2*X[m,3]*X[m,3] + 0.1*np.random.randn()
		Y.append(y)
	
	if with_regression:
		dat = wuml.wData(X_npArray=X, Y_npArray=Y, first_row_is_label=True, column_names=['A','B','C','D'])
		dat.to_csv('shap_regress_example_mix_distributions.csv', add_row_indices=False, include_column_names=True, label_type='continuous')
	else:
		Y = np.array(Y)
		m = np.mean(Y)
		Y[Y > m] = 1
		Y[Y < m] = 0
	
		dat = wuml.wData(X_npArray=X, Y_npArray=Y, first_row_is_label=True, column_names=['A','B','C','D'], label_type='discrete')
		#dat.to_csv('shap_classifier_example.csv', add_row_indices=False, include_column_names=True)


generate_data_from_same_distribution(n=30, with_regression=False)
