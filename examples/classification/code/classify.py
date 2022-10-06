#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 


data = wuml.make_classification_data( n_samples=200, n_features=5, n_informative=3)

wuml.jupyter_print('Running a single classifier')
cf = wuml.classification(data, classifier='IKDR', q=3)
wuml.jupyter_print(cf.result_summary(print_out=False))
wuml.jupyter_print('\nThe linear weights that designates feature importance')
wuml.jupyter_print(cf.model.get_feature_importance())

wuml.jupyter_print('\nSorted Feature Importance')
cf.output_sorted_feature_importance_table(data.columns)
wuml.jupyter_print('\nNotice how the feature importances from IKDR is similar to using permutation_importance from sklearn')

wuml.jupyter_print('\n\nRun all classifiers sorted by Accuracy')
#default regressor=['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR','LogisticRegression']
models = wuml.run_every_classifier(data, y=data.Y, order_by='Test',
		regressors=['GP', 'SVM', 'KNN', 'NeuralNet', 'LDA', 'IKDR','LogisticRegression'])
wuml.jupyter_print(models['Train/Test Summary'])


wuml.jupyter_print('\n\nPick out SVM and plot Feature Importance')
models['SVM'].plot_feature_importance('SVM feature importance', data.columns)

