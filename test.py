#!/usr/bin/env python
import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
from wplotlib import scatter
from sklearn.gaussian_process import GaussianProcessRegressor

data = wuml.wData('examples/data/NN_with_labels.csv', first_row_is_label=True)

suckPattern = data[:,5:11]
wuml.center_and_scale(suckPattern)
babyAge = data.get_columns(['BabyAssessAge_B1W'])




#regressor = wuml.regression(suckPattern, y=babyAge, regressor='GP', alpha=0.5, gamma=0.5)
#regressor = wuml.regression(suckPattern, y=babyAge, regressor='Predef_NeuralNet', network_info_print=True, max_epoch=2000)
#summary = regressor.result_summary()

res = wuml.run_every_regressor(suckPattern, y=babyAge, order_by='Test mse', 
								alpha=1, gamma=1, l1_ratio=0.2, max_epoch=2000, network_info_print=True)
print(res['Train/Test Summary'])

import pdb; pdb.set_trace()



