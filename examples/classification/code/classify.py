#!/usr/bin/env python

import os
import sys
#if os.path.exists('/home/chieh/code/wPlotLib'):
#	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
from sklearn.inspection import permutation_importance


data = wuml.make_classification_data( n_samples=200, n_features=5, n_informative=3)
cf = wuml.classification(data, classifier='NaiveBayes')
wuml.jupyter_print('Running a single classifier')
wuml.jupyter_print(cf)


wuml.jupyter_print('\n\nRun all classifiers sorted by Accuracy')
models = wuml.run_every_classifier(data, y=data.Y, order_by='Test')
Y = models['GP'](data)
wuml.jupyter_print(models['Train/Test Summary'])

import pdb; pdb.set_trace()
models['GP'].plot_feature_importance('Test', data.df.columns, title_fontsize=12, axis_fontsize=9, xticker_rotate=0, ticker_fontsize=9)


#importance_GP = permutation_importance(models['GP'].model, data.X, data.Y, scoring='accuracy')
#importance = importance_GP.importances_mean
### summarize feature importance
#
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.5f' % (i,v))
	

import pdb; pdb.set_trace()
## plot feature importance
#pyplot.bar([x for x in range(len(importance))], importance)
#pyplot.show()

