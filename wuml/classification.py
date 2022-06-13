import numpy as np
import pandas as pd

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt



class classification:
	'''
	Automatically run classification on data

	data: can be any data type
	classifier='GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR'
	split_train_test: automatically splits the data, default is true
	q: for IKDR, dimension to reduce to
	'''

	def __init__(self, data, y=None, y_column_name=None, split_train_test=False, classifier='GP', kernel=None, 
				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001, q=2):
		NP = wuml.ensure_numpy
		S = np.squeeze

		X = NP(data)
		if y is not None:
			y = S(NP(y))
		elif y_column_name is not None:
			y = data[y_column_name].values
		elif type(data).__name__ == 'wData':
			y = data.Y
		else: raise ValueError('Undefined label Y')

		if split_train_test:
			self.X_train, X_test, self.y_train, y_test = wuml.split_training_test(data, label=y, xdata_type="%.4f", ydata_type="%.4f")
		else:
			self.X_train = NP(X)
			self.y_train = NP(y)

		if classifier == 'GP':
			kernel = 1.0 * RBF(1.0) 
			model = GaussianProcessClassifier(kernel=kernel, random_state=0)
		elif classifier == 'SVM':
			model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
		elif classifier == 'RandomForest':
			model = RandomForestClassifier(max_depth=3, random_state=0)
		elif classifier == 'KNN':
			model = KNeighborsClassifier(n_neighbors=4)
		elif classifier == 'NeuralNet':
			model = MLPClassifier(random_state=1, max_iter=400)
		elif classifier == 'LDA':
			model = LinearDiscriminantAnalysis()
		elif classifier == 'NaiveBayes':
			model = GaussianNB()
		elif classifier == 'GradientBoosting':
			model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
		elif classifier == 'IKDR':
			model = wuml.IKDR(data, q=q, y=y)
		else: raise ValueError('Unrecognized Classifier')

		model.fit(NP(self.X_train), S(NP(self.y_train)))
		self.ŷ_train = model.predict(NP(self.X_train))
		self.Train_acc = wuml.accuracy(S(NP(self.y_train)), self.ŷ_train)

		if split_train_test:
			self.ŷ_test = model.predict(NP(X_test))
			self.Test_acc = wuml.accuracy(S(NP(y_test)), self.ŷ_test)

		self.split_train_test = split_train_test
		self.model = model
		self.classifier = classifier
		self.kernel = kernel

		#self.results = self.result_summary(print_out=False)

	def output_sorted_feature_importance_table(self, Column_names, show_top_few=5): 	# feature importance computed via permutation_importance
		NP = wuml.ensure_numpy

		all_classifiers =['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR']
		Cnames = wuml.ensure_list(Column_names)

		importance_GP = permutation_importance(self.model, NP(self.X_train), self.y_train, scoring='accuracy')
		importance = importance_GP.importances_mean

		if self.classifier in all_classifiers:

			coefs = pd.DataFrame( importance, columns=['Coefficients'], index=Cnames)
			sorted_coefs = coefs.sort_values(by='Coefficients', ascending=False)
			first_few = sorted_coefs.head(show_top_few)
			wuml.jupyter_print(first_few)

	def plot_feature_importance(self, title, Column_names, title_fontsize=12, axis_fontsize=9, xticker_rotate=0, ticker_fontsize=9,
								yticker_rotate=0, ytick_locations=None, ytick_labels=None): # feature importance computed via permutation_importance

		NP = wuml.ensure_numpy
		all_classifiers =['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR']
		Cnames = wuml.ensure_list(Column_names)

		importance_GP = permutation_importance(self.model, NP(self.X_train), self.y_train, scoring='accuracy')
		importance = importance_GP.importances_mean

		if self.classifier in all_classifiers:

			coefs = pd.DataFrame( importance, columns=['Coefficients'], index=Cnames)
				
			#coefs.plot(kind='barh', figsize=(9, 7))
			coefs.plot(kind='barh')
			plt.title(title, fontsize=title_fontsize)
			plt.axvline(x=0, color='.5')
			plt.subplots_adjust(left=.3)
			plt.tight_layout()
			plt.ylabel('Features', fontsize=axis_fontsize)
			plt.xlabel('Features Influence on Results', fontsize=axis_fontsize)
			plt.yticks(fontsize=ticker_fontsize, rotation=yticker_rotate, ticks=ytick_locations, labels=ytick_labels )

			plt.show()

	def result_summary(self, print_out=True):
		NPR = np.round

		if self.split_train_test:
			column_names = ['classifier', 'Train', 'Test']
			data = np.array([[self.classifier, NPR(self.Train_acc, 4) , NPR(self.Test_acc, 4)]])
		else:
			column_names = ['classifier', 'Train']
			data = np.array([[self.classifier, NPR(self.Train_acc, 4)]])

		df = pd.DataFrame(data, columns=column_names,index=[''])
		if print_out: print(df)
		self.results = df
		return df



	def __call__(self, data):

		X = wuml.ensure_numpy(data)	
		#[self.ŷ, self.σ] = model.predict(X, return_std=True)

		try: [self.ŷ, self.σ] = self.model.predict(X, return_std=True, return_cov=False)
		except: self.ŷ = self.model.predict(X)

		self.ŷ = wuml.ensure_data_type(self.ŷ, type_name=type(data).__name__)
		try: return [self.ŷ, self.σ]
		except: return self.ŷ

		

	def __str__(self):
		#return self.results
		return str(self.result_summary(print_out=False))


def run_every_classifier(data, y=None, y_column_name=None, order_by='Test', q=None):
	'''
	order_by: 'Test', 'Train'
	q: for ikdr
	'''
	regressors=['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR']
	results = {}
	
	if q is None: q = int(data.shape[1]/2)
	df = pd.DataFrame()
	for reg in regressors:
		results[reg] = classification(data, y=y, classifier=reg, split_train_test=True, q=q)
		df = df.append(results[reg].result_summary(print_out=False))

	if order_by == 'Test': results['Train/Test Summary'] = df.sort_values(['Test','Train'], ascending=False)
	else: results['Train/Test Summary'] = df.sort_values(['Train','Test'], ascending=False)

	return results

