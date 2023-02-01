#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import numpy as np
import wuml


#	example data of 2 classes
X1 = np.random.randn(20, 2) + 5
X2 = np.random.randn(7, 2) - 5
X = np.vstack((X1,X2))
y = np.append(np.ones(20), np.zeros(7))


#	There are currently 2 options
#	oversampling : will oversample data from fewer sample with replacement
#	'smote' : will take data from fewer sample class and find 5 nearest neighbor and get their interpolated sample as new

rebalancer = wuml.rebalance_data(X, y, method='oversampling')
print(X,'\n')
print(rebalancer.balanced_data)

