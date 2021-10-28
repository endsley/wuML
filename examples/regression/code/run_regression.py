#!/usr/bin/env python

import wuml 


data = wuml.wData('../../data/shap_regress_example_uniform.csv', label_column_name='label', label_type='continuous', first_row_is_label=True)

reg = wuml.regression(data, regressor='Elastic net', alpha=0.05, gamma=0.05, l1_ratio=0.05)
print('Running a single regressor')
print(reg)

print('\n\nRun all regressors sorted by least test error')
result = wuml.run_every_regressor(data, alpha=0.1, gamma=0.05, l1_ratio=0.05)
print(result)

