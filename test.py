#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('examples/data/shap_regress_example_uniform.csv', row_id_with_label=0)
X = data[:,0:3]
Ys = data[:,3:5]
depList = wuml.HSIC_of_feature_groups_vs_label_list(X, Ys)
print(depList)

