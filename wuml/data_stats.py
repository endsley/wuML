
import numpy as np; np.random.seed(0)
from wplotlib import lines		#pip install wplotlib
from wplotlib import heatMap
import wuml 

import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import random


def missing_data_stats(df): 
	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	wuml.ensure_path_exists('./DatStats')

	mdp = missing_data_counter_per_feature = []
	for column in df:
		x = df[column].values
		missing_percentage = np.sum(np.isnan(x))/n
		mdp.append(missing_percentage)


	mdp = np.array(mdp)
	textstr = ''
	x = np.arange(1, len(mdp)+1)

	lp = lines()
	lp.plot_line(x, mdp, 'Missing Percentage', 'Feature ID', 'Percentage Missing', 
				imgText=textstr, outpath='./DatStats/feature_missing_percentage.png')


	X2 = np.isnan(X).astype(int)
	hMap = heatMap()
	hMap.draw_HeatMap(X2, title='Missing Data Heat Map', 
							xlabel='Feature ID', ylabel='Sample ID',
							path='./DatStats/missing_data_heatMap.png')

	#wuml.write_to(str(df.info()), './DatStats/feature_stats.txt')
	#import pdb; pdb.set_trace()

	buffer = io.StringIO()
	df.info(buf=buffer, verbose=True)
	s = buffer.getvalue()
	wuml.write_to(s, './DatStats/feature_stats.txt')


