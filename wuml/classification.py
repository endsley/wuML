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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy import stats


class classification:
	'''
	Automatically run classification on data

	data: can be any data type
	classifier='GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR', 'LogisticRegression'
	split_train_test: automatically splits the data, default is true
	q: for IKDR, dimension to reduce to
	'''

	def __init__(self, data, y=None, test_data=None, testY=None, y_column_name=None, split_train_test=True, 	
				classifier='GP', kernel='rbf', 	# kernels = 'rbf', 'linear'
				reduce_dimension_first_method=None, # 'LDA'
				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001, q=2,
				accuracy_rounding=3, regularization_weight=1):
		NP = wuml.ensure_numpy
		S = np.squeeze

		self.X_train = None
		self.data = data
		X = NP(data)
		if y is not None:
			y = S(NP(y))
		elif y_column_name is not None:
			y = data[y_column_name].values
		elif type(data).__name__ == 'wData':
			y = data.Y
		else: raise ValueError('Undefined label Y')

		if test_data is not None:
			if wuml.wtype(data) == 'wData': self.X_train = data.X
			elif wuml.wtype(data) == 'ndarray': self.X_train = data

			if y is None: self.y_train = data.Y
			else: self.y_train = y

			if wuml.wtype(test_data) == 'wData': self.X_test = test_data.X
			elif wuml.wtype(test_data) == 'ndarray': self.X_test = test_data
			
			if testY is None: y_test = test_data.Y
			else:y_test = testY
			split_train_test = True
		else:
			if split_train_test:
				self.X_train, self.X_test, self.y_train, y_test = wuml.split_training_test(data, label=y, xdata_type="%.4f", ydata_type="%.4f")
			else:
				self.X_train = NP(X)
				self.y_train = NP(y)

		#Check for dimension reduction, example LDA+SVM
		cfname = classifier.split('+')
		self.original_classifier_name = classifier
		if len(cfname) == 2:
			reduce_dimension_first_method = cfname[0]
			classifier = cfname[1]

		# Reduce the dimension of data first if LDA is required
		if reduce_dimension_first_method == 'LDA':
			self.dim_reduct = LinearDiscriminantAnalysis()
			self.dim_reduct.fit(NP(self.X_train), S(NP(self.y_train)))
			self.X_train = NP(self.X_train).dot(self.dim_reduct.coef_.T)
			if split_train_test: self.X_test = NP(self.X_test).dot(self.dim_reduct.coef_.T)


		if classifier == 'GP':
			if kernel == 'rbf': kernel = 1.0 * RBF(1.0) 
			model = GaussianProcessClassifier(kernel=kernel, random_state=0)
		elif classifier == 'SVM':
			model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernel, C=regularization_weight))
		elif classifier == 'RandomForest':
			model = RandomForestClassifier(max_depth=3, random_state=0)
		elif classifier == 'KNN':
			model = KNeighborsClassifier(n_neighbors=4)
		elif classifier == 'NeuralNet':
			model = MLPClassifier(random_state=1, max_iter=max_epoch)
		elif classifier == 'LDA':
			model = LinearDiscriminantAnalysis()
		elif classifier == 'NaiveBayes':
			model = GaussianNB()
		elif classifier == 'GradientBoosting':
			model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
		elif classifier == 'LogisticRegression':
			model = LogisticRegression(random_state=0)
		elif classifier == 'IKDR':
			model = wuml.IKDR(self.X_train, q=q, y=self.y_train)
		else: raise ValueError('Unrecognized Classifier')

		NPR = np.round

		model.fit(NP(self.X_train), S(NP(self.y_train)))
		self.ŷ_train = model.predict(NP(self.X_train))
		self.Train_acc = NPR(wuml.accuracy(S(NP(self.y_train)), self.ŷ_train), accuracy_rounding)

		if split_train_test:
			self.ŷ_test = model.predict(NP(self.X_test))
			self.Test_acc = NPR(wuml.accuracy(S(NP(y_test)), self.ŷ_test), accuracy_rounding)

		#import pdb; pdb.set_trace()
		self.split_train_test = split_train_test
		self.model = model
		self.classifier = classifier
		self.kernel = kernel
		self.reduce_dimension_first_method = reduce_dimension_first_method
		#self.results = self.result_summary(print_out=False)

	def project_data_onto_linear_weights(self, X=None):
		# if X is none, it will use the original data
		if wuml.wtype(self.model) != 'LinearDiscriminantAnalysis':
			print('Warning: the function project_data_onto_linear_weights only works for LDA')
			return X

		if X is None:
			nX = wuml.ensure_numpy(self.data)
			pX = nX.dot(self.model.coef_.T)

		return pX
			
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
		if self.split_train_test:
			column_names = ['classifier', 'Train', 'Test']
			data = np.array([[self.original_classifier_name, self.Train_acc , self.Test_acc]])
		else:
			column_names = ['classifier', 'Train']
			data = np.array([[self.original_classifier_name, self.Train_acc]])

		df = pd.DataFrame(data, columns=column_names,index=[''])
		if print_out: print(df)
		self.results = df
		return df


	def save_classifier_to_pickle_file(self, path):
		wuml.pickle_dump(self, path)

	def __call__(self, data):

		X = wuml.ensure_numpy(data)	
		if self.reduce_dimension_first_method == 'LDA':
			X = X.dot(self.dim_reduct.coef_.T)

		try: [self.ŷ, self.σ] = self.model.predict(X, return_std=True, return_cov=False)
		except: self.ŷ = self.model.predict(X)

		self.ŷ = wuml.ensure_data_type(self.ŷ, type_name=type(data).__name__)
		try: return [self.ŷ, self.σ]
		except: return self.ŷ

		

	def __str__(self):
		#return self.results
		return str(self.result_summary(print_out=False))


def run_every_classifier(data, y=None, y_column_name=None, order_by='Test', q=None, kernel='rbf',
						regressors=['GP', 'SVM', 'RandomForest', 'KNN', 'NeuralNet', 'LDA', 'NaiveBayes', 'IKDR','LogisticRegression']	):
	'''
	data is type wData
	order_by: 'Test', 'Train'
	q: for ikdr
	'''
	
	results = {}
	
	if q is None: q = int(data.shape[1]/2)
	df = pd.DataFrame()
	for reg in regressors:
		wuml.write_to_current_line('Running %s'%reg)
		results[reg] = classification(data, y=y, classifier=reg, split_train_test=True, q=q, kernel='rbf')
		df = df.append(results[reg].result_summary(print_out=False))

	if order_by == 'Test': results['Train/Test Summary'] = df.sort_values(['Test','Train'], ascending=False)
	else: results['Train/Test Summary'] = df.sort_values(['Train','Test'], ascending=False)

	return results


#	Generate 10 classifiers and use bagging concensus to reach a label
class ten_folder_classifier:
	def __init__(self, data, y=None, y_column_name=None, q=2, kernel='linear', classifier='SVM'):

		tenFoldData = wuml.gen_10_fold_data(data=data)
		self.tb = tb = wuml.result_table(column_names=['Fold', 'Train Acc', 'Test Acc'])
	
		self.classifier_list = []
		for i, fold in enumerate(tenFoldData):
			wuml.write_to_current_line('Running fold :%d'%(i+1))
			[X_train, Y_train, X_test, Y_test] = fold
			cf = wuml.classification(X_train, y=Y_train, test_data=X_test, testY=Y_test, classifier=classifier, kernel=kernel, q=q)
			self.classifier_list.append(cf)
			tb.add_row([i+1, cf.Train_acc , cf.Test_acc])
	
		TrainAcc = np.mean(tb.get_column('Train Acc').X)
		TestAcc = np.mean(tb.get_column('Test Acc').X)
		tb.add_row(['Avg Acc', TrainAcc , TestAcc])

	def show_results(self):
		wuml.jupyter_print(self.tb)

	def predict(self, data):
		return self.__call__(data)

	def __call__(self, data):
		all_labels = None
		for model in self.classifier_list:
			if all_labels is None:
				all_labels = wuml.ensure_wData(model(data))
			else:
				label = wuml.ensure_wData(model(data))
				all_labels.append_columns(model(data))

		m = stats.mode(all_labels.X, axis=1)
		return m[0]

	def fit(self, data, y=None):
		tenFoldData = wuml.gen_10_fold_data(data=data)
	
		self.classifier_list = []
		for i, fold in enumerate(tenFoldData):
			wuml.write_to_current_line('Running fold :%d'%(i+1))
			[X_train, Y_train, X_test, Y_test] = fold
			cf = wuml.classification(X_train, y=Y_train, test_data=X_test, testY=Y_test, classifier=classifier, kernel=kernel, q=q)
			self.classifier_list.append(cf)

	def save_classifier_to_pickle_file(self, path):
		wuml.pickle_dump(self, path)

