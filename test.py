#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


	
	
'''
	Identifies a weight associated with each sample based on its likelihood. 
	Given p(X1) > p(Xi) for all i
	Using KDE if p(X1)/p(X2)=2  the weight for X1 = 1, and X2 = 2
	This means that if X1 is the most likely samples and if X1 is 
	2 times more likely than X2, then X1 would have a weight of 1
	and X2 would have a weight of 2.
	This weight can then be used to balance the sample importance for regression

'''

data = wuml.wData('examples/data/Chem_decimated_imputed.csv', row_id_with_label=0)
data.delete_column('id')	# the id should not be part of the likelihood 
sample_weights = wuml.get_likelihood_weight(data)
import pdb; pdb.set_trace()

H = histograms()
H.histogram(sample_weights.X, num_bins=20, title='Sample Weight Histogram', facecolor='blue', α=0.5, path=None)
sample_weights.to_csv('../data/Chem_sample_weights.csv', include_column_names=False)


