#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
import numpy as np


#data = wuml.make_moons()
n = 1000
x1 = np.random.randn(n,1)
x2 = x1 
x3 = np.random.randn(n,1)
x4 = x3 + 0.05*np.random.randn(n,1)
x5 = np.random.randn(n,1)
data = np.hstack((x1,x2,x3,x4,x5))
data = wuml.ensure_wData(data, column_names=['A','B','C','D','E'])

FS = wuml.feature_selection(data)
wuml.jupyter_print(FS.removal_history)
wuml.jupyter_print(FS.removal_table)
