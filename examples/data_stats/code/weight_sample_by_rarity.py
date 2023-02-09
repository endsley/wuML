#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


	
	
#'''
#	Identifies a weight associated with each sample based on its likelihood. 
#	Given p(X1) > p(Xi) for all i
#	Using KDE if p(X1)/p(X2)=2  the weight for X1 = 1, and X2 = 2
#	This means that if X1 is the most likely samples and if X1 is 
#	2 times more likely than X2, then X1 would have a weight of 1
#	and X2 would have a weight of 2.
#	This weight can then be used to balance the sample importance for regression
#
#'''

data = wuml.wData('../../data/Chem_decimated_imputed.csv', first_row_is_label=True)
data.delete_column('id')	# the id should not be part of the likelihood 
sample_weights = wuml.get_likelihood_weight(data['finalga_best'])

print(wuml.output_two_columns_side_by_side(data['finalga_best'], sample_weights, labels=['age','weight'], rounding=3))



