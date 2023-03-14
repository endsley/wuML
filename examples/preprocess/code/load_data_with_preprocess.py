#!/usr/bin/env python


import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
import numpy as np




data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='center and scale')

μ = np.mean(data.X, axis=0)
σ = np.std(data.X, axis=0)
wuml.print_two_matrices_side_by_side(μ, σ)



