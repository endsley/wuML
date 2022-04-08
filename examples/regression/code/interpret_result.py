#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
import pandas as pd
import matplotlib.pyplot as plt
import wplotlib

#How to Interpret feature importance after certain models
wuml.jupyter_print('A non-linear model was used to generate the data. ')
wuml.jupyter_print('Note that x1 x4 has the largest impact')
wuml.jupyter_print('5x_1 + x_2 + x_1 x_2 - 8x_4 - 2x_4^2 + \delta', latex=True)
wuml.jupyter_print('\delta \leftarrow noise', latex=True)

data = wuml.wData('../../data/shap_regress_example_uniform.csv', label_column_name='label', label_type='continuous', first_row_is_label=True)
cnames = data.get_column_names_as_a_list()


reg = wuml.regression(data, regressor='linear', alpha=0.05, gamma=0.05, l1_ratio=0.05)
reg.plot_feature_importance('Feature Importance for Linear Regression', cnames)


reg2 = wuml.regression(data, regressor='Elastic net', alpha=0.05, gamma=0.05, l1_ratio=0.05)
reg2.plot_feature_importance('Feature Importance for Elastic Net', cnames)


reg3 = wuml.regression(data, regressor='Lasso', alpha=0.05)
reg3.plot_feature_importance('Feature Importance for Lasso', cnames)


reg4 = wuml.regression(data, regressor='Lasso', alpha=0.05)
reg4.plot_feature_importance('Feature Importance for Lasso', cnames)


reg5 = wuml.regression(data, regressor='RandomForest')
reg5.plot_feature_importance('Feature Importance for Random Forest', cnames)

