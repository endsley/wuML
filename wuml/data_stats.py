
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


def get_feature_histograms(X, path=None, title=''):
	header = './results/DatStats/'
	wuml.ensure_path_exists('./results')
	wuml.ensure_path_exists(header)

	H = histograms()
	H.histogram(X, num_bins=10, title=title, fontsize=12, facecolor='blue', α=0.5, path=path, subplot=None)

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

def missing_data_stats(df): 
	if type(df).__name__ != 'DataFrame': 
		df = df.get_data_as('DataFrame')

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



