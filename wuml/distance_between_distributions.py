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
def wasserstein_distance(X, Y, M=None, XY_as_histogram=True):

	X = np.squeeze(wuml.ensure_numpy(X))
	Y = np.squeeze(wuml.ensure_numpy(Y))

	if XY_as_histogram:
		if len(X.shape) == 1:
			Nx = len(X)
			Ny = len(Y)
			
			# normalize the distribution to 1
			#import pdb; pdb.set_trace()

			if M is None:
				D = np.arange(Nx, dtype=np.float64)
				D = D.reshape((Nx, 1))
				M = ot.dist(D, D, 'euclidean')

			return ot.emd2(X, Y, M) # exact linear program
		else:
			raise ValueError('XY_as_histogram as False is not yet supported')
	else:
		raise ValueError('XY_as_histogram as False is not yet supported')



