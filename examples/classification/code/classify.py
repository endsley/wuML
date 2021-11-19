#!/usr/bin/env python

import wuml 

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', label_type='discrete', first_row_is_label=False)

cf = wuml.classification(data, classifier='NaiveBayes')
print('Running a single classifier')
print(cf)

print('\n\nRun all regressors sorted by Accuracy')
results = wuml.run_every_classifier(data, y=data.Y, order_by='Test')
Y = results['GP'](data)
print(Y)
print(results['Train/Test Summary'])
import pdb; pdb.set_trace()

