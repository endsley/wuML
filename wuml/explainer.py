import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
if os.path.exists('/home/chieh/code/interpret'):
	sys.path.insert(0,'/home/chieh/code/interpret')

import wuml
from wuml.type_check import *
from wuml.IO import *
import numpy as np
import pandas as pd

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from interpret import show
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel

class explainer():
	def __init__(self, data, model, explainer_algorithm='shap'):
		'''
			data : need to be in wData
		'''
		data = wuml.ensure_wData(data)
		model.explainer_mode = True
		self.joint_table = None

		if wtype(model) == 'autoencoder': self.model_prediction = model.objective_network
		else: self.model_prediction = model
		

		explain_types = {'lime':LimeTabular, 'shap':ShapKernel}
		self.explainer = explain_types[explainer_algorithm](predict_fn=self.model_predict_wrapper, data=data.df)

	def model_predict_wrapper(self, X_input):
		y = self.model_prediction(X_input)
		y = ensure_numpy(y, ensure_column_format=False)
		y = np.squeeze(y)

		if len(y.shape) == 0: y = np.array([y])
		return y



	def __call__(self, data, nsamples=20, display=True):
		data = wuml.ensure_wData(data)
		self.column_names = data.column_names
		
		local_explain = self.explainer.explain_local(data.df, data.Y)
		if display: show(local_explain)

		self.joint_table = self.feature_importance_table(local_explain, print_out=display)
		self.global_summary = self.most_important_features(local_explain, print_out=display)
		return [self.joint_table, self.global_summary]

	def feature_importance_table(self, local_explain, print_out=True):
		if self.joint_table is not None: return self.joint_table

		E = local_explain._internal_obj['specific']
		explain_list = []
		for local_explain in E:
			α = local_explain['perf']['actual_score']
			β = local_explain['perf']['predicted_score']
			γ = np.absolute(α - β)
			σ = np.append(local_explain['scores'], [α, β, γ])
			n = np.append(local_explain['names'], ['y', 'ŷ', 'Δy'])

			item = pd.DataFrame(np.array([σ]), columns=n)
			explain_list.append(item)

		cnames = np.append(self.column_names, ['y', 'ŷ', 'Δy'])
		result = pd.concat(explain_list)
		r2 = result.reindex(columns=cnames)
		self.joint_table = ensure_wData(r2.reset_index(drop=True))

		if print_out: jupyter_print(self.joint_table, endString='\n')
		return self.joint_table

	def most_important_features(self, local_explain, print_out=True):
		E = local_explain._internal_obj['specific']
		feature_names = E[0]['names']
		D = {}
		Dm = {}
		for j in feature_names: 
			D[j] = 0
			Dm[j] = 0

		for i in E:
			most_important_feature = i['names'][np.argmax(np.absolute(i['scores']))]
			D[most_important_feature] += 1		#increase count

			for j, k in enumerate(i['names']): 
				Dm[k] += np.absolute(i['scores'][j])

		D = wuml.ensure_wData(pd.DataFrame(D.values(),index=D.keys(), columns=['Most chosen']))
		Dm = wuml.ensure_wData(pd.DataFrame(Dm.values(),index=Dm.keys(), columns=['Most weighted']))
		D.sort_by('Most chosen', ascending=False)
		Dm.sort_by('Most weighted', ascending=False)
		if print_out: wuml.print_two_matrices_side_by_side(D.df, Dm.df, title1='', title2='')
	
		return [D, Dm]
		
