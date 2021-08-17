#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
	
l2x_synthetic_blocks.csv  l2x_synthetic.csv         l2x_synthetic_label.csv

wuml.set_terminal_print_options(precision=2)
Xdata = wuml.wData('../data/l2x_synthetic.csv', label_type='continuous')
Ydata = wuml.wData('../data/l2x_synthetic_label.csv', label_type='continuous')

import pdb; pdb.set_trace()


#P = wuml.l2x(data, max_epoch=4, learning_rate=0.001, data_imbalance_weights=imbW, pre_trained_file='weighted_L2X_5p.pk')
#ŷ = P(data)
#out = wuml.output_regression_result(data.Y, ŷ)

