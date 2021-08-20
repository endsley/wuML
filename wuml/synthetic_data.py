#!/usr/bin/env python

import numpy as np
import wuml


def L2X_data_A(N, xpath=None, ypath=None, Block_path=None):
	'''
		Warning: This data is an example of how L2X won't work b/c
					the underlying distribution between data of different
					feature contribution is the same.

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

	
	O = np.ones((int(N/2),2))
	Z = np.zeros((int(N/2),2))
	B = np.block([[O, Z],[Z,O]])
	y = (X*B).dot(b)

	if xpath is not None: np.savetxt(xpath, X, delimiter=',', fmt='%.4f') 
	if ypath is not None: np.savetxt(ypath, y, delimiter=',', fmt='%.4f') 
	if Block_path is not None: np.savetxt(Block_path, B, delimiter=',', fmt='%d') 

	return [X,y,b]
	
def L2X_data_B(N, xpath=None, ypath=None, Block_path=None):
	'''
		Note: This data is an example of how L2X will work b/c
					the underlying distribution between data of different
					feature contribution are different.

		This function generates 4 dimensions {X1,X2,X3,X4} and 100 samples.
		The the y from 1st 50 samples only came from X1,X2
		The the y from 2nd 50 samples only came from X3,X4
		Standard regression cannot zero out half of the samples
	'''
	def gen_samples_A():
		X1 = np.random.beta(0.5,0.5) + 10
		X2 = np.random.beta(0.5,0.5) + 10
		X3 = np.random.beta(0.5,0.5) + 10
		X4 = np.random.rayleigh() + 10
	
		return np.array([[X1,X2,X3,X4]])


	def gen_samples_B():
		X1 = 3*np.random.uniform(0,2) 
		X2 = 3*np.random.uniform(0,2) 
		X3 = 3*np.random.exponential(scale=1.0)
		X4 = 3*np.random.randn()
	
		return np.array([[X1,X2,X3,X4]])

	Ns = int(N/2)

	b = np.array([[2],[1],[3],[3]])
	X = np.empty((0,4))
	for n in range(int(Ns)):
		X = np.vstack((X,gen_samples_A()))
	for n in range(int(Ns)):
		X = np.vstack((X,gen_samples_B()))

	O = np.ones((int(Ns),2))
	Z = np.zeros((int(Ns),2))
	B = np.block([[O, Z],[Z,O]])
	y = (X*B).dot(b)

	if xpath is not None: np.savetxt(xpath, X, delimiter=',', fmt='%.4f') 
	if ypath is not None: np.savetxt(ypath, y, delimiter=',', fmt='%.4f') 
	if Block_path is not None: np.savetxt(Block_path, B, delimiter=',', fmt='%d') 

	return [X,y,b]

X = L2X_data_B(5000, xpath='l2x_synthetic.csv', ypath='l2x_synthetic_label.csv', Block_path='l2x_synthetic_blocks.csv')
