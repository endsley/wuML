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


data = wuml.wData('../../data/shap_regress_example_uniform.csv', first_row_is_label=True)

print(wuml.feature_wise_HSIC(data))
print('\n\n')
print(wuml.feature_wise_HSIC(data, get_top_dependent_pairs=True))
print('\n\n')
print(wuml.feature_wise_HSIC(data, label_name='label', get_top_dependent_pairs=True))


# This function handles missing data as well by removing missing entries during pairwise HSIC
data = wuml.wData('../../data/missin_example.csv', first_row_is_label=True)
print(data)
print('\n\n')
print(wuml.feature_wise_HSIC(data))



