#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 


data = wuml.wData('../../data/shap_regress_example_uniform.csv', label_column_name='label', label_type='continuous', first_row_is_label=True)

reg = wuml.regression(data, regressor='AdaBoost', alpha=0.05, gamma=0.05, l1_ratio=0.05)
wuml.jupyter_print('Running a single regressor')
wuml.jupyter_print(reg.result_summary(print_out=False))
reg.output_sorted_feature_importance_table(data.columns)



wuml.jupyter_print('\n\nRun all regressors sorted by least test error')
result = wuml.run_every_regressor(data, alpha=0.1, gamma=0.05, l1_ratio=0.05)
wuml.jupyter_print(result['Train/Test Summary'])
result['linear'].plot_feature_importance('linear Feature Importance', data.columns)

