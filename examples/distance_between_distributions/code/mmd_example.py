#!/usr/bin/env python
import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

from wuml.IO import jupyter_print
from wuml.distance_between_distributions import mmd
import numpy as np



X = np.random.randn(50,2)			# X and Y have samples from the same distribution
Y = np.random.randn(50,2)
Z = np.random.rand(50,2) + 2		# X and Z are different

#	X, Y, and Z are just matrices of data
Dxy = mmd(X,Y)
Dxz = mmd(X,Z)

jupyter_print('D(X,Y) = %.4f'%Dxy)
jupyter_print('D(X,Z) = %.4f'%Dxz)

