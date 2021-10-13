#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('examples/data/shap_regress_example_uniform.csv', row_id_with_label=0)
FC = wuml.feature_wise_correlation(data, num_of_top_dependent_pairs_to_plot=2)

