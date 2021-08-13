#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
	
wuml.set_terminal_print_options(precision=2)
data = wuml.wData('../data/Chem_decimated_imputed.csv', 
					label_type='continuous', row_id_with_label=0, 
					label_column_name='finalga_best', columns_to_ignore=['id'])

imbW = wuml.wData('../data/Chem_sample_weights.csv')
P = wuml.l2x(data, max_epoch=4000, learning_rate=0.001, data_imbalance_weights=imbW)
P.train()
ŷ = P(data)

wuml.output_regression_result(data.Y, ŷ, write_path='./y_vs_ŷ.txt')

