#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 

#	We first remove column A because there are too many missing values
data = wuml.wData('../../data/missin_example.csv', first_row_is_label=True, columns_to_ignore=['A'])
wuml.jupyter_print('We start with this dataset')
wuml.jupyter_print(data,'\n')

X = wuml.impute(data, ignore_first_index_column=False)
wuml.jupyter_print('We first impute all the residual missing entries')
wuml.jupyter_print(X,'\n')


X = wuml.map_data_between_0_and_1(X, map_type='linear') # map_type: linear, or cdf
wuml.jupyter_print(X[0:20,:])


X = wuml.map_data_between_0_and_1(X, map_type='cdf') # map_type: linear, or cdf
wuml.jupyter_print(X[0:20,:])

