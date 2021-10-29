#!/usr/bin/env python
import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
from wplotlib import scatter
from sklearn.gaussian_process import GaussianProcessRegressor

data = wuml.wData(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', label_type='discrete', first_row_is_label=False)
#ikdr = wuml.IKDR(data, q=3)
#ikdr.fit(data, data.Y)
#Ŷ = ikdr.predict(data)
#import pdb; pdb.set_trace()

results = wuml.run_every_classifier(data, q=12)
print(results['Train/Test Summary'])

#ikdr = wuml.classification(data, q=12, classifier='IKDR', split_train_test=True)
#ikdr.result_summary(print_out=True)



#suckPattern = data[:,5:11]
#wuml.center_and_scale(suckPattern)
#babyAge = data.get_columns(['BabyAssessAge_B1W'])
#
#regressor = wuml.regression(suckPattern, y=babyAge, regressor='kernel ridge', alpha=0.5, gamma=0.5)
##regressor = wuml.regression(suckPattern, y=babyAge, regressor='Predef_NeuralNet', network_info_print=True, max_epoch=2000)
#summary = regressor.result_summary()
#YY = regressor.show_true_v_predicted()
import pdb; pdb.set_trace()




