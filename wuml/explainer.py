

import shap		#pip install shap
import xgboost
import sys
import wuml
import numpy as np

class explainer():
	def __init__(self, data, explainer_algorithm='shap', link='identity',
						model=None, 										# Use this if you have your own model, input/output should be nparray
						load_network_path=None,								# You can load your own pytorch network
						loss='mse',  										#use these options if you use a default regression
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=1000, learning_rate=0.001, print_network_training_status=True):

		'''
			data : must be wData type with data.X and data.Y defined
			loss: Can be str : 'mse', 'L1', 'CE', 'hindge' or you can define your own loss function
			explainer_algorithm: 'shap', 'XGBRegressor'
			use link='identity' for regression and link='logit' for classification
		'''

		if type(data).__name__ != 'wData': 
			wuml.error_msg_exit('Error: data input into explainer must be wData type with data.X, data.Y defined.')

		if explainer_algorithm == 'shap':
			X = wuml.ensure_numpy(data)
			if load_network_path is not None:	
				if type(load_network_path) != type(''): 
					print('\nError : load_network_path is a string that you load the network, it must be a basicNetwork type!!!')
					print('\tIf you want to input a function, use model=yourFunction argument instead!!!\n')
					sys.exit()
					
				self.model = wuml.pickle_load(load_network_path)
				if data.label_type == 'continuous': 
					self.model.eval(output_type='ndarray')
					self.Explr = shap.KernelExplainer(self.model, np.zeros((1, X.shape[1])), link=link)		
				else: 
					self.model.eval(output_type='ndarray', out_structural='softmax')
					self.Explr = shap.KernelExplainer(self.model, X, link=link)		
			elif model is not None:
				self.model = model
				self.Explr = shap.KernelExplainer(model, np.zeros((1, X.shape[1])), link=link)
			else:
				self.model = wuml.basicNetwork(loss, data, networkStructure=networkStructure, max_epoch=max_epoch, learning_rate=learning_rate)
				self.model.train(print_status=print_network_training_status)
				if data.label_type == 'continuous': 
					self.model.eval(output_type='ndarray')		# If classification, output softmax
					self.Explr = shap.KernelExplainer(self.model, np.zeros((1, X.shape[1])), link=link)		#, l1_reg=False
				else: 
					self.model.eval(output_type='ndarray', out_structural='softmax')
					self.Explr = shap.KernelExplainer(self.model, X, link=link)		#, l1_reg=False


		elif explainer_algorithm == 'XGBRegressor':
			self.model = xgboost.XGBRegressor().fit(data.df, data.Y)
			self.Explr = shap.Explainer(self.model)


		self.explainer_algorithm = explainer_algorithm


#	def force_plot(self, single_shape_row, X):
#		shap.force_plot(self.Explr.expected_value[0], single_shape_row, X, link="identify")
#
#	def waterfall(self, single_shape_row):
#		shap.plots.waterfall(single_shape_row)

	def __call__(self, data, nsamples=20, output_all_results=False):
		A = self.explainer_algorithm
		if A == 'shap':
			X = wuml.ensure_numpy(data)
			return self.Explr.shap_values(X, nsamples=nsamples)[0]
		elif A == 'XGBRegressor':
			X = wuml.ensure_DataFrame(data)
			results = self.Explr(X)
			if output_all_results:
				return results
			else:
				return results.values


