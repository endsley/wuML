import numpy as np
import pandas as pd
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
			X_train, X_test, y_train, y_test = wuml.split_training_test(data, label=y, xdata_type="%.4f", ydata_type="%.4f")
		else:
			X_train = X
			y_train = y

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

		model.fit(NP(X_train), S(NP(y_train)))
		self.ŷ_train = model.predict(NP(X_train))
		self.Train_acc = wuml.accuracy(S(NP(y_train)), self.ŷ_train)

		if split_train_test:
			self.ŷ_test = model.predict(NP(X_test))
			self.Test_acc = wuml.accuracy(S(NP(y_test)), self.ŷ_test)

		self.split_train_test = split_train_test
		self.model = model
		self.classifier = classifier
		self.kernel = kernel

		#self.results = self.result_summary(print_out=False)

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

