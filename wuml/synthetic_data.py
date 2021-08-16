#!/usr/bin/env python

import numpy as np
import wuml


def L2X_data_A(N, xpath=None, ypath=None, Block_path=None):
	'''
		This function generates 4 dimensions {X1,X2,X3,X4} and 100 samples.
		The the y from 1st 50 samples only came from X1,X2
		The the y from 2nd 50 samples only came from X3,X4
		Standard regression cannot zero out half of the samples
	'''
	def gen_sample():
		X1 = np.random.uniform(0,2) + 5
		X2 = np.random.uniform(0,2) + 5
		X3 = np.random.exponential(scale=1.0)
		X4 = np.random.randn()
	
		return np.array([[X1,X2,X3,X4]])

	b = np.array([[2],[1],[3],[3]])
	X = np.empty((0,4))
	for n in range(N):
		X = np.vstack((X,gen_sample()))

	O = np.ones((50,2))
	Z = np.zeros((50,2))
	B = np.block([[O, Z],[Z,O]])
	y = (X*B).dot(b)

	if xpath is not None: np.savetxt(xpath, X, delimiter=',', fmt='%.4f') 
	if ypath is not None: np.savetxt(ypath, y, delimiter=',', fmt='%.4f') 
	if Block_path is not None: np.savetxt(Block_path, B, delimiter=',', fmt='%d') 

	return [X,y,b]
	
X = L2X_data_A(100, xpath='l2x_synthetic.csv', ypath='l2x_synthetic_label.csv', Block_path='l2x_synthetic_blocks.csv')
