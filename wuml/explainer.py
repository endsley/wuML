import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from interpret import show
from interpret.blackbox import LimeTabular


class explainer():
	def __init__(self, data, model, explainer_algorithm='lime'):
		'''
			data : need to be in wData
		'''
		model.output_y_as = 'not in column'

		if explainer_algorithm='shap':
			pass
		elif explainer_algorithm='lime':
			lime = LimeTabular(predict_fn=model, data=data.X)
			lime_local = lime.explain_local(model.X_test.X[0:2], model.y_test[0:2])
			show(lime_local)
		else:
			raise ValueError('\n\tError: %s is an unrecognized explainer_algorithm, it must be "shap" or "lime"'%explainer_algorithm)

	def __call__(self, data, nsamples=20, output_all_results=False):
		A = self.explainer_algorithm
		if A == 'shap':
			X = wuml.ensure_numpy(data)
			shapOut = self.Explr.shap_values(X, nsamples=nsamples)[0]
			return wuml.ensure_DataFrame(shapOut)

		elif A == 'XGBRegressor':
			X = wuml.ensure_DataFrame(data)
			results = self.Explr(X)
			if output_all_results:
				return results
			else:
				return wuml.ensure_DataFrame(results.values)


