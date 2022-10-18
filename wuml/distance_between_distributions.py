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
def wasserstein_distance(X, Y, M=None, XY_as_histogram=True, verbose=False):

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
			#import pdb; pdb.set_trace()
			return ot.emd2(X, Y, M) # exact linear program
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
					if Y[i].flags['C_CONTIGUOUS'] == False: Yc = Y[i].copy(order = 'C')
					else: Yc = Y[i]

					wasDis_matrix[i,j] = ot.emd2(Xc, Yc, M)

					wuml.write_to_current_line('Completion %.3f'%((i*n+j)/(n*n)))

			return wasDis_matrix
		else:
			raise ValueError('Wasserstein Distance does not handle 3 dimensional data.')
	else:
		raise ValueError('XY_as_histogram as False is not yet supported')



