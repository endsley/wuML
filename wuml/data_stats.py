
import numpy as np; np.random.seed(0)
from wplotlib import lines		#pip install wplotlib
from wplotlib import heatMap
from wplotlib import histograms
from wplotlib import scatter
import wuml 
from wuml.data_loading import wData

import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import itertools
import random


def get_feature_histograms(X, path=None, title='', ylogScale=False):
	X = wuml.ensure_numpy(X)

	if path is not None:
		header = './results/DatStats/'
		wuml.ensure_path_exists('./results')
		wuml.ensure_path_exists(header)

	H = histograms()
	H.histogram(X, num_bins=10, title=title, fontsize=12, facecolor='blue', α=0.5, path=path, subplot=None, ylogScale=ylogScale)

def identify_missing_data_per_feature(data):
	df = wuml.ensure_DataFrame(data)

	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	mdp = missing_data_counter_per_feature = []
	for column in df:
		x = df[column].values
		missing_percentage = np.sum(np.isnan(x))/n
		mdp.append(missing_percentage)

	return mdp

def missing_data_stats(data, save_plots=False): 
	df = wuml.ensure_DataFrame(data)

	if save_plots:
		header = './results/DatStats/'
		wuml.ensure_path_exists('./results')
		wuml.ensure_path_exists(header)
	
		mdp = identify_missing_data_per_feature(df)
		mdp = np.array(mdp)
		textstr = ''
		x = np.arange(1, len(mdp)+1)
	
		lp = lines()
		lp.plot_line(x, mdp, 'Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=textstr, outpath= header + 'feature_missing_percentage.png', 
					xtick_locations=x, xtick_labels=df.columns.to_numpy(), xticker_rotate=90)

		X2 = np.isnan(df.values).astype(int)
		hMap = heatMap()
		hMap.draw_HeatMap(X2, title='Missing Data Heat Map', 
								xlabel='Feature ID', ylabel='Sample ID',
								path= header + 'missing_data_heatMap.png')
	
		buffer = io.StringIO()
		df.info(buf=buffer, verbose=True)
		s = buffer.getvalue()
		wuml.write_to(s, header + 'feature_stats.txt')
	else:
		mdp = identify_missing_data_per_feature(df)
		mdp = np.array(mdp)
		textstr = ''
		x = np.arange(1, len(mdp)+1)
	
		lp = lines()
		lp.plot_line(x, mdp, 'Missing Percentage', 'Feature ID', 'Percentage Missing', imgText=textstr,
					xtick_locations=x, xtick_labels=df.columns.to_numpy(), xticker_rotate=90)
	
		X2 = np.isnan(df.values).astype(int)
		hMap = heatMap()
		hMap.draw_HeatMap(X2, title='Missing Data Heat Map', 
								xlabel='Feature ID', ylabel='Sample ID')
	
#		buffer = io.StringIO()
#		df.info(buf=buffer, verbose=True)
#		s = buffer.getvalue()


def get_redundant_pairs(df):
	'''Get diagonal and lower triangular pairs of correlation matrix'''
	pairs_to_drop = set()
	cols = df.columns
	for i in range(0, df.shape[1]):
		for j in range(0, i+1):
			pairs_to_drop.add((cols[i], cols[j]))
	return pairs_to_drop


def get_top_abs_correlations(df, n=5):
	#au_corr = df.corr().abs().unstack()
	au_corr = df.unstack()
	labels_to_drop = get_redundant_pairs(df)
	au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False, key=abs)
	return au_corr[0:n]

def feature_wise_correlation(data, n=10, label_name=None, get_top_corr_pairs=False, num_of_top_dependent_pairs_to_plot=0):
	'''
		if label_name is label string, then it only compares the features against the label
		num_of_top_dependent_pairs_to_plot: if > 0, it will plot out the most correlated pairs
	'''

	df = wuml.ensure_DataFrame(data)
	if n > df.shape[1]: n = df.shape[1]

	if label_name is not None and label_name not in df.columns:
		raise ValueError('Error : %s is an unrecognized column name. \nThe list of names are %s'%(label_name, str(df.columns)))

	corrMatrix = df.corr()
	topCorr = get_top_abs_correlations(corrMatrix, n=n).to_frame()
	if not get_top_corr_pairs and label_name not in df.columns: 
		outDF = corrMatrix
	elif label_name is None:
		outDF = topCorr
	else:
		corrVector = corrMatrix[label_name].to_frame()
		topCorr = corrVector.sort_values(label_name, key=abs, ascending=False)
		topCorr = topCorr[1:n]
		outDF = topCorr

	if num_of_top_dependent_pairs_to_plot>0:
		subTopCorr = topCorr.head(num_of_top_dependent_pairs_to_plot)
		idx = subTopCorr.index
		idx = idx.to_frame().values
		
		for i, item in enumerate(idx):
			if len(item) == 1:
				α = item
				β = label_name
				A = df[α].to_numpy()
				B = df[label_name].to_numpy()

			elif len(item) == 2:
				α, β = item
			
				A = df[α].to_numpy()
				B = df[β].to_numpy()
	
			corV = subTopCorr.values[i]
			textstr = r'Order ID: %d, Correlation : %.3f' % (i+1, corV)
			lp = scatter(figsize=(10,5))		# (width, height)
			lp.plot_scatter(A, B, α + ' vs ' + β, α, β, imgText=textstr)


	return wuml.ensure_data_type(outDF, type_name=type(data).__name__)


def feature_wise_HSIC(data, n=10, label_name=None, get_top_dependent_pairs=False, num_of_top_dependent_pairs_to_plot=0):
	'''
		if label_name is label string, then it only compares the features against the label
		num_of_top_dependent_pairs_to_plot: if > 0, it will plot out the most correlated pairs
	'''
	X = wuml.ensure_numpy(data)
	df = wuml.ensure_DataFrame(data)
	d = X.shape[1]

	if n > df.shape[1]: n = df.shape[1]
	if label_name is not None and label_name not in df.columns:
		raise ValueError('Error : %s is an unrecognized column name. \nThe list of names are %s'%(label_name, str(df.columns)))


	lst = list(range(d))
	pair_order_list = itertools.combinations(lst,2)
	depMatrix = np.eye(d)
	for α, β in list(pair_order_list):
		x1 = np.atleast_2d(X[:,α]).T
		x2 = np.atleast_2d(X[:,β]).T

		joinX = wuml.ensure_DataFrame(np.hstack((x1,x2)))
		joinX = joinX.dropna()
		withoutNan = joinX.values
		depMatrix[α,β] = depMatrix[β,α] = wuml.HSIC(withoutNan[:,0], withoutNan[:,1], sigma_type='mpd')

	depM_DF = wuml.ensure_DataFrame(depMatrix, columns=df.columns, index=df.columns)
	topCorr = get_top_abs_correlations(depM_DF, n=n).to_frame()
	if not get_top_dependent_pairs and label_name not in df.columns: 
		outDF = depM_DF
	elif label_name is None:
		outDF = topCorr
	else:
		corrVector = depM_DF[label_name].to_frame()
		topCorr = corrVector.sort_values(label_name, key=abs, ascending=False)
		topCorr = topCorr[1:n]
		outDF = topCorr

	if num_of_top_dependent_pairs_to_plot>0:
		subTopCorr = topCorr.head(num_of_top_dependent_pairs_to_plot)
		idx = subTopCorr.index
		idx = idx.to_frame().values
		
		for i, item in enumerate(idx):
			if len(item) == 1:
				α = item
				β = label_name
				A = df[α].to_numpy()
				B = df[label_name].to_numpy()

			elif len(item) == 2:
				α, β = item
			
				A = df[α].to_numpy()
				B = df[β].to_numpy()
	
			corV = subTopCorr.values[i]
			textstr = r'Order ID: %d, Correlation : %.3f' % (i+1, corV)
			lp = scatter(figsize=(10,5))		# (width, height)
			lp.plot_scatter(A, B, α + ' vs ' + β, α, β, imgText=textstr)


	return wuml.ensure_data_type(outDF, type_name=type(data).__name__)



def HSIC_of_feature_groups_vs_label_list(data, data_compared_to):
	'''
		Compare the entire "data" to each column of "data_compared_to"
	'''
	
	X = wuml.ensure_numpy(data)
	Ys = wuml.ensure_numpy(data_compared_to)
	Ys_df = wuml.ensure_DataFrame(data_compared_to)
	num_of_Ys = Ys.shape[1]

	hsic_list = []
	for i in range(num_of_Ys):
		hsic_list.append(wuml.HSIC(X, Ys[:,i])) #, sigma_type='mpd'

	df = wuml.ensure_DataFrame(np.array(hsic_list))
	df.index = Ys_df.columns
	df.columns = ['feature_group']
	df = df.sort_values('feature_group', axis=0, ascending=False)
	
	return wuml.ensure_wData(df)
