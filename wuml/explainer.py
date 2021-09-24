

import shap		#pip install shap
import wuml
import numpy as np

class explainer():
	def __init__(self, data, explainer_algorithm='shap', link='identity',
						model=None, 										# Use this if you have your own model, input/output should be nparray
						load_network_path=None,								# You can load your own pytorch network
						loss='mse',  										#use these options if you use a default regression
						networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], 
						max_epoch=1000, learning_rate=0.001):

		'''
			loss: Can be str : 'mse', 'L1', 'CE', 'hindge' or you can define your own loss function
			use link='identity' for regression and link='logit' for classification
		'''

		X = wuml.ensure_numpy(data)

		if explainer_algorithm == 'shap':
			if load_network_path is not None:	
				self.net = wuml.pickle_load(load_network_path)
				self.net.eval(output_type='ndarray')
				self.Explr = shap.KernelExplainer(self.net, np.zeros((1, X.shape[1])), link=link)		
			elif model is not None:
				self.Explr = shap.KernelExplainer(model, np.zeros((1, X.shape[1])), link=link)
			else:
				self.net = wuml.basicNetwork(loss, data, networkStructure=networkStructure, max_epoch=max_epoch, learning_rate=learning_rate)
				self.net.train()
				self.net.eval(output_type='ndarray')
				self.Explr = shap.KernelExplainer(self.net, np.zeros((1, X.shape[1])), link=link)		#, l1_reg=False

	
#	def force_plot(self, single_shape_row, X):
#		shap.force_plot(self.Explr.expected_value[0], single_shape_row, X, link="identify")
#
#	def waterfall(self, single_shape_row):
#		shap.plots.waterfall(single_shape_row)

	def __call__(self, data, nsamples=20):
		X = wuml.ensure_numpy(data)
		return self.Explr.shap_values(X, nsamples=nsamples)[0]



