#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import numpy as np
import wuml
from wuml import norm
import wplotlib


#	load required data
data = wuml.wData(xpath='wine.csv', ypath='wine_label.csv', label_type='discrete', path_prefix=['../../data/', '../../data/'] ) 
X1 = wuml.normalize(data, 'l1', return_new_copy=True)	
#	Every row is normalized to have l1 = 1
wuml.jupyter_print(norm(X1, norm='l1'))


X2 = wuml.normalize(data, 'l2', return_new_copy=True)
#	Every row is normalized to have l2 = 1
wuml.jupyter_print(norm(X2, norm='l2'))

