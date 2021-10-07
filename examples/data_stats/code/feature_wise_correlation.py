#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('../../data/shap_regress_example_uniform.csv', row_id_with_label=0)

print(wuml.feature_wise_correlation(data), '\n\n')
print(wuml.feature_wise_correlation(data, get_top_corr_pairs=True), '\n\n')
print(wuml.feature_wise_correlation(data, label_name='label', get_top_corr_pairs=True), '\n\n')
import pdb; pdb.set_trace()

