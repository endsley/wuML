#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('../../data/shap_regress_example_uniform.csv', row_id_with_label=0)
data.df.style

#print(wuml.feature_wise_correlation(data).df.style)
corrMatrix = wuml.feature_wise_correlation(data)
corrMatrix.df.style


top_correlated_pair = wuml.feature_wise_correlation(data, get_top_corr_pairs=True)
top_correlated_pair.df.style


most_correlated_to_label = wuml.feature_wise_correlation(data, label_name='label', get_top_corr_pairs=True)
most_correlated_to_label.df.style





