#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('examples/data/missin_example.csv', row_id_with_label=0)
print(data)

print(wuml.feature_wise_HSIC(data))
print(wuml.feature_wise_HSIC(data, get_top_dependent_pairs=True))

