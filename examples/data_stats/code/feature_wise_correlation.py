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


#cmds = wuml.get_commandLine_input()

data = wuml.wData('../../data/shap_regress_example_uniform.csv', first_row_is_label=True)
wuml.jupyter_print(data, endString='\n')

corrMatrix = wuml.feature_wise_correlation(data)
wuml.jupyter_print(corrMatrix)

top_correlated_pair = wuml.feature_wise_correlation(data, get_top_corr_pairs=True)
wuml.jupyter_print(top_correlated_pair, endString='\n')

most_correlated_to_label = wuml.feature_wise_correlation(data, label_name='label', get_top_corr_pairs=True)
wuml.jupyter_print(most_correlated_to_label, endString='\n')





