#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 


data = wuml.make_classification_data( n_samples=200, n_features=5, n_informative=3)
cf = wuml.classification(data, classifier='GP')
wuml.jupyter_print('Running a single classifier')
wuml.jupyter_print(cf)

wuml.jupyter_print('\n\nSorted Feature Importance')
cf.output_sorted_feature_importance_table(data.columns)


wuml.jupyter_print('\n\nRun all classifiers sorted by Accuracy')
models = wuml.run_every_classifier(data, y=data.Y, order_by='Test')
wuml.jupyter_print(models['Train/Test Summary'])


wuml.jupyter_print('\n\nPick out SVM and plot Feature Importance')
models['SVM'].plot_feature_importance('SVM feature importance', data.columns)

