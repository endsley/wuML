
import numpy as np
from wplotlib import lines		#pip install wplotlib
import wpreprocess as wPr


def missing_data_stats(df): 
	X = df.values
	n = X.shape[0]
	d = X.shape[1]

	wPr.ensure_path_exists('./DatStats')

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

	mdp[mdp < 0.3]
	import pdb; pdb.set_trace()

