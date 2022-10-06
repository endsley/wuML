#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
from wuml import jupyter_print


#list of possible classifiers = 
#	['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR','LogisticRegression']
#	['LDA+GP', 'LDA+SVM', 'LDA+RandomForest', 'LDA+KNN', 'LDA+NeuralNet', 'LDA', 'LDA+NaiveBayes', 'LDA+IKDR','LDA+LogisticRegression']

data = wuml.make_classification_data( n_samples=300, n_features=6, n_informative=3, n_classes=3)

#jupyter_print('Running a single classifier')
#cf = wuml.classification(data, classifier='IKDR', q=3)
#jupyter_print(cf.result_summary(print_out=False))
#jupyter_print('\nThe linear weights that designates feature importance')
#jupyter_print(cf.model.get_feature_importance())
#
#jupyter_print('We can save the classifer to file for later use')
#cf.save_classifier_to_pickle_file('IKDR_classifier.pk')

jupyter_print('Now we load the classifier and use it')
ikdr = wuml.pickle_load('IKDR_classifier.pk')
jupyter_print('You can get the labels by passing the data into the class')
ӯ = ikdr(data)
summary = wuml.summarize_classification_result(data.Y, ӯ)
jupyter_print('Avg Error: %.3f'%summary.avg_error())


#
#jupyter_print('\nSorted Feature Importance')
#cf.output_sorted_feature_importance_table(data.columns)
#jupyter_print('\nNotice how the feature importances from IKDR is similar to using permutation_importance from sklearn')
#
#
#
#jupyter_print('\nWe can also reduce the dimension first via LDA')
#cf = wuml.classification(data, classifier='LDA+IKDR')#, reduce_dimension_first_method='LDA')
#jupyter_print(cf.result_summary(print_out=False))



#jupyter_print('\n\nRun all classifiers sorted by Accuracy')
##default regressor=['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR','LogisticRegression']
#models = wuml.run_every_classifier(data, y=data.Y, order_by='Test',
#		regressors=['GP', 'SVM', 'KNN', 'NeuralNet', 'LDA', 'IKDR','LogisticRegression','LDA+GP', 'LDA+SVM', 'LDA+RandomForest', 'LDA+KNN', 'LDA+NeuralNet', 'LDA+NaiveBayes', 'LDA+IKDR','LDA+LogisticRegression'])
#jupyter_print(models['Train/Test Summary'])
#
#
#jupyter_print('\n\nPick out SVM and plot Feature Importance')
#models['SVM'].plot_feature_importance('SVM feature importance', data.columns)

