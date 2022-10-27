#!/usr/bin/env python
import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import wplotlib
from wuml import jupyter_print
from ot.datasets import make_1D_gauss as gauss
import numpy as np
import ot



#	X and Y can be a matrix or a vector
#	if they are vectors, we assume they are counts of the histogram, returning a scalar value for the distance
#	if they are matrices, we assume each row is a histogram and output a distance matrix
#	if M is none, we assume the increments of distance is 1 between tickers
def wasserstein_distance(X, Y, M=None, XY_as_histogram=True, output_as='ndarray', method='set negative to small noise'):

	X = np.squeeze(wuml.ensure_numpy(X))
	Y = np.squeeze(wuml.ensure_numpy(Y))

	if XY_as_histogram:
		if len(X.shape) == 1:
			Nx = X.shape[0]
			if M is None:
				D = np.arange(Nx, dtype=np.float64)
				D = D.reshape((Nx, 1))
				M = ot.dist(D, D, 'euclidean')

			# normalize the distribution to 1
			X = wuml.map_vector_to_distribution_data(X, method=method)
			Y = wuml.map_vector_to_distribution_data(Y, method=method)
			
			return ot.emd2(X, Y, M)

		elif len(X.shape) == 2:
			n = X.shape[0]
			wasDis_matrix = np.zeros((n,n))
			for i in range(n):
				for j in range(n):
					Nx = X[i].shape[0]
					if M is None:
						D = np.arange(Nx, dtype=np.float64)
						D = D.reshape((Nx, 1))
						M = ot.dist(D, D, 'euclidean')

					if X[i].flags['C_CONTIGUOUS'] == False: Xc = X[i].copy(order = 'C')
					else: Xc = X[i]
					if Y[j].flags['C_CONTIGUOUS'] == False: Yc = Y[j].copy(order = 'C')
					else: Yc = Y[j]

					# normalize the distribution to 1
					Xc = wuml.map_vector_to_distribution_data(Xc, method=method)
					Yc = wuml.map_vector_to_distribution_data(Yc, method=method)
					wasDis_matrix[i,j] = ot.emd2(Xc, Yc, M)

					wuml.write_to_current_line('Completion %.3f, at (%d,%d) -> %.4f'%((i*n+j)/(n*n), i, j, wasDis_matrix[i,j]))
			print('\n')
			if output_as == 'ndarray': return wasDis_matrix
			elif output_as == 'wData': return wuml.wData(X_npArray=wasDis_matrix) # exact linear program
		else:
			raise ValueError('Wasserstein Distance does not handle 3 dimensional data.')
	else:
		raise ValueError('XY_as_histogram as False is not yet supported')



