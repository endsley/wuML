
import numpy as np; np.random.seed(0)
from wplotlib import lines		#pip install wplotlib
from wplotlib import heatMap
from wplotlib import histograms
import wuml 
from wuml.data_loading import wData

import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import random


def get_feature_histograms(X, path=None, title='', ylogScale=False):
	X = wuml.ensure_numpy(X)

	if path is not None:
		header = './results/DatStats/'
		wuml.ensure_path_exists('./results')
		wuml.ensure_path_exists(header)

	H = histograms()
	H.histogram(X, num_bins=10, title=title, fontsize=12, facecolor='blue', α=0.5, path=path, subplot=None, ylogScale=ylogScale)

def identify_missing_data_per_feature(df):
	if type(df).__name__ != 'DataFrame': 
		df = df.get_data_as('DataFrame')

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
					imgText=textstr, outpath= header + 'feature_missing_percentage.png')
	
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
		lp.plot_line(x, mdp, 'Missing Percentage', 'Feature ID', 'Percentage Missing', 
					imgText=textstr)
	
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

def feature_wise_correlation(data, n=10, label_name=None, get_top_corr_pairs=False):
	'''
		if label_name is label string, then it only compares the features against the label
	'''

	df = wuml.ensure_DataFrame(data)
	corrMatrix = df.corr()
	if not get_top_corr_pairs: return corrMatrix
	if n > corrMatrix.shape[0]: n = corrMatrix.shape[0]

	if label_name is None:
		outDF = get_top_abs_correlations(corrMatrix, n=n).to_frame()
	else:
		corrVector = corrMatrix[label_name].to_frame()
		SortedcorrVector = corrVector.sort_values(label_name, key=abs, ascending=False)
		outDF = SortedcorrVector[0:n]

	return wuml.ensure_data_type(outDF, type_name=type(data).__name__)
