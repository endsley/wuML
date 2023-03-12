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
import marshal, types

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from interpret import show
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel

class explainer():
	def __init__(self, data, model, explainer_algorithm='shap', which_model_output_to_use=None):
		'''
			data : need to be in wData
			which_model_output_to_use: sometimes the model output multiple outputs in a list instead of just the label, 
					this integer tells us which item within the last should be used
		'''
		self.which_model_output_to_use = which_model_output_to_use
		self.data = wuml.ensure_wData(data)
		self.explainer_algorithm = explainer_algorithm
		self.column_names = self.data.column_names

		reduce_down_to = 70
		if self.data.shape[0] > reduce_down_to:
			km = wuml.clustering(self.data, n_clusters=reduce_down_to, method='KMeans')
			Cs = km.model.cluster_centers_
			self.data = wuml.ensure_wData(Cs, column_names=data.columns)
			jupyter_print('\nNote: Since there are too many input samples, kmeans was employed to reduce the sample down to %d\n'%reduce_down_to)

		self.joint_table = None		# initialize a table of all feature importances
		if wtype(model) == 'autoencoder': self.model_prediction = model.objective_network
		else: self.model_prediction = model
		

		explain_types = {'lime':LimeTabular, 'shap':ShapKernel}
		self.explainer = explain_types[explainer_algorithm](predict_fn=self.model_predict_wrapper, data=self.data.df)
		model.explainer = self


	def output_network_data_for_storage(self, existing_db=None):
		if existing_db is None: existing_db = {}

		existing_db['explainer'] = 'exists'
		existing_db['reference_data'] = self.data.X
		existing_db['explainer_algorithm'] = self.explainer_algorithm
		existing_db['which_model_output_to_use'] = self.which_model_output_to_use
		existing_db['column_names'] = self.column_names

		return existing_db

	def model_predict_wrapper(self, X_input):
		y = self.model_prediction(X_input)
		if self.which_model_output_to_use is not None:
			y = y[self.which_model_output_to_use]

		y = ensure_numpy(y, ensure_column_format=False)
		y = np.squeeze(y)

		if len(y.shape) == 0: y = np.array([y])
		return y



	def __call__(self, data, y=None, display=True, outpath=None, figsize=None):
		data = wuml.ensure_wData(data)
		if y is not None:
			local_explain = self.explainer.explain_local(data.df, y)
		else:
			local_explain = self.explainer.explain_local(data.df, data.Y)

		if display: show(local_explain)

		self.joint_table = self.feature_importance_table(local_explain, print_out=display)
		self.global_summary = self.most_important_features(local_explain, print_out=display)

		#if figsize is not None or outpath is not None:
		#	# plot out and save importance
		#	[y, ŷ, Δy] = ensure_list(np.round(self.joint_table.X, 4)[0])[-3:]
		#	pred_str = 'y = %.3f\nŷ = %.3f'%(y, ŷ)
		#	for i in range(self.joint_table.shape[0]): 
		#		feature_importance = self.joint_table.X[i]
		#		feature_importance = feature_importance[0:len(feature_importance)-3]
		#		B = wplotlib.bar(self.column_names, feature_importance, 'Sample %s: '%i, 'Features', 'Importance', horizontal=True, imgText=pred_str, xTextShift=1.05, yTextShift=0, figsize=figsize, outpath=output)

		return [self.joint_table, self.global_summary]

	def feature_importance_table(self, local_explain, print_out=True):
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

		if print_out: jupyter_print(self.joint_table, endString='\n', display_all_rows=True, display_all_columns=True)
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
		

	def plot_individual_sample_importance(self, one_sample, y=None, sample_id='', figsize=None):
		X = ensure_proper_model_input_format(one_sample)
		[self.joint_table, self.global_summary] = self.__call__(X, y=y, display=False)

		cnames = ensure_list(self.column_names)
		[y, ŷ, Δy] = ensure_list(np.round(self.joint_table.X, 4)[0])[-3:]
		pred_str = 'y = %.3f\nŷ = %.3f'%(y, ŷ)
		feature_importance = ensure_list(ensure_numpy(self.joint_table).round(4))[0]
		feature_importance = feature_importance[0:len(feature_importance)-3]		# remove the last 3 elements cus they are ['y', 'ŷ', 'Δy']
		B = wplotlib.bar(cnames, feature_importance, 'Sample %s: '%sample_id, 'Features', 'Importance', horizontal=True, imgText=pred_str, xTextShift=1.05, yTextShift=0, figsize=figsize)	
