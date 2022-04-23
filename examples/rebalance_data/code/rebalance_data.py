#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import numpy as np
import wuml

X = np.random.randn(100, 10)
y = np.append(np.ones(20), np.zeros(80))

rebalancer = wuml.rebalance_data(X, y)
print(rebalancer.balanced_data.shape)
print(rebalancer.balanced_data)

