#!/usr/bin/env python

import wuml 


data = wuml.wData('../../data/shap_classifier_example_uniform.csv', label_column_name='label', label_type='continuous', first_row_is_label=True)

cf = wuml.classification(data, classifier='NeuralNet')
print('Running a single classifier')
print(cf)

print('\n\nRun all regressors sorted by Accuracy')
result = wuml.run_every_classifier(data, y=data.Y, order_by='Test')
print(result)

