
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import numpy as np
import pandas as pd
import wuml

class regression:
	'''
	Automatically run regression on data

	data: can be any data type
	regressor='GP', 'linear', 'NeuralNet', 'kernel ridge', 'AdaBoost', 'Elastic net', 'RandomForest', 'Predef_NeuralNet'
				if 'Predef_NeuralNet' is used, you can additionally set arguments : networkStructure, max_epoch, learning_rate
	split_train_test: automatically splits the data, default is true
	'''

	def __init__(self, data, y=None, y_column_name=None, split_train_test=True, regressor='GP', kernel=None, 
				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001	):
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
			self.X_train, self.X_test, self.y_train, self.y_test = wuml.split_training_test(data, label=y, xdata_type="%.4f", ydata_type="%.4f")
		else:
			self.X_train = X
			self.y_train = y

		if regressor == 'GP':
			model = GaussianProcessRegressor(kernel=kernel, random_state=0)
		elif regressor == 'linear':
			model = LinearRegression()
		elif regressor == 'kernel ridge':
			model = KernelRidge(alpha=1.0)
		elif regressor == 'AdaBoost':
			model = AdaBoostRegressor(random_state=0, n_estimators=100)
		elif regressor == 'Elastic net':
			model = ElasticNet(random_state=0)
		elif regressor == 'NeuralNet':
			model = MLPRegressor(random_state=1, max_iter=1000)
		elif regressor == 'RandomForest':
			model = RandomForestRegressor(max_depth=3, random_state=0)
		elif regressor == 'Predef_NeuralNet':
			Xt = wuml.ensure_wData(self.X_train)
			model = wuml.basicNetwork('mse', Xt, Y=S(NP(self.y_train)), networkStructure=networkStructure, 
										max_epoch=max_epoch, learning_rate=learning_rate, network_info_print=False)

		model.fit(NP(self.X_train), S(NP(self.y_train)))
		try: [self.ŷ_train, self.σ] = model.predict(NP(self.X_train), return_std=True)
		except: 
			self.ŷ_train = model.predict(NP(self.X_train))

		self.mse_train = mean_squared_error(S(NP(self.y_train)), self.ŷ_train)

		if split_train_test:
			try: [self.ŷ_test, self.σ_test] = model.predict(NP(self.X_test), return_std=True)
			except: self.ŷ_test = model.predict(NP(self.X_test))
			self.mse_test = mean_squared_error(S(NP(self.y_test)), self.ŷ_test)

		self.split_train_test = split_train_test
		self.model = model
		self.regressor = regressor
		self.kernel = kernel

	def score(self, score_type='r2'):

		if score_type == 'r2' or score_type == 'all scores':
			self.train_r2_score = r2_score(self.y_train, self.ŷ_train)
			if self.split_train_test: 
				self.test_r2_score = r2_score(self.y_test, self.ŷ_test)

		if score_type == 'mean absolute error' or score_type == 'all scores':
			self.train_avg_abs_Δ = mean_absolute_error(self.y_train, self.ŷ_train)
			if self.split_train_test: 
				self.test_avg_abs_Δ = mean_absolute_error(self.y_test, self.ŷ_test)

		if score_type == 'max error' or score_type == 'all scores':
			self.train_max_abs_Δ = max_error(self.y_train, self.ŷ_train)
			if self.split_train_test: 
				self.test_max_abs_Δ = max_error(self.y_test, self.ŷ_test)



	def result_summary(self, print_out=True):
		NPR = np.round
		self.score(score_type='all scores')

		if self.split_train_test:
			column_names = ['regressor', 	'Train mse', 'Test mse', 
											'Train r2 Score', 'Test r2 Score', 
											'Train avg abs error', 'Test avg abs error', 
											'Train max error', 'Test max error' ]
			data = np.array([[	self.regressor, NPR(self.mse_train,4), 
												NPR(self.mse_test,4), 
												NPR(self.train_r2_score,4), 
												NPR(self.test_r2_score,4), 
												NPR(self.train_avg_abs_Δ,4), 
												NPR(self.test_avg_abs_Δ,4), 
												NPR(self.train_max_abs_Δ,4), 
												NPR(self.test_max_abs_Δ, 4) ]])
		else:
			column_names = ['regressor', 'Train mse', 'Train r2 Score',											
											'Train avg abs error', 'Train max error']

			data = np.array([[self.regressor, 	NPR(self.mse_train,4), 
												NPR(self.train_r2_score,4), 
												NPR(self.train_avg_abs_Δ,4), 
												NPR(self.train_max_abs_Δ,4)]])

		df = pd.DataFrame(data, columns=column_names,index=[''])
		if print_out: print(df)
		return df



	def __call__(self, data):
		X = wuml.ensure_numpy(data)	

		try:
			[self.ŷ, self.σ] = self.model.predict(X, return_std=True, return_cov=False)
		except:
			self.ŷ = self.model.predict(X)

		return wuml.ensure_data_type(self.ŷ, type_name=type(data).__name__)

		#else: raise ValueError('Regressor not recognized, must use regressor="GP"')
		

	def __str__(self):
		return str(self.result_summary(print_out=False))


def run_every_regressor(data, y=None, y_column_name=None, order_by='Test mse'):
	'''
	order_by: 'Test mse', 'Train mse'
	'''
	regressors=['Elastic net', 'linear', 'kernel ridge', 'AdaBoost', 'GP', 'NeuralNet', 'RandomForest', 'Predef_NeuralNet']

	df = pd.DataFrame()
	for reg in regressors:
		reg = regression(data, y=y, regressor=reg)
		df = df.append(reg.result_summary(print_out=False))

	df = df.sort_values(order_by, ascending=True)
	return df

