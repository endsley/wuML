#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml

## Show featurehistogram after data decimation
## 	header : the feature name you want to use to see the histogram. 
## 	index_col : if 0, it means the 1st row of dataFrame is the feature names. None would be nothing
data = wuml.wData('../../data/Chem_decimated_imputed.csv', first_row_is_label=True)



## You can pick a column from the data using its column name, here it is 'finalga_best'
wuml.get_feature_histograms(data['finalga_best'], title='Histogram of Data Labels', ylogScale=False)
wuml.get_feature_histograms(data['finalga_best'], title='Histogram of Data Labels', ylogScale=True)


