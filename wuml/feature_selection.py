
import numpy as np
import pandas as pd
import wuml

class feature_selection: 
	def __init__(self, data, Y=None, method='HSIC filter', exit_hsic=0.8):
		X = wuml.ensure_wData(data)

		if method == 'HSIC filter':
			self.hsic_filter(X,Y, exit_hsic)

	def hsic_filter(self, X, Y, exit_hsic=0.5):
		d = X.shape[1]
		x = X.X
		removal_table = pd.DataFrame(columns=['column ID', 'Residual HSIC'])
		removal_history = pd.DataFrame(columns=X.columns)
		name_list = X.columns.to_list()

		if Y is None:
			col_ids = np.arange(d).tolist()
			highest_hsic = 1
			while highest_hsic > exit_hsic and len(col_ids) > 1:
				highest_hsic = 0
				highest_hsic_column = None
				history_row = {}		# row of each hsic history
				for i, j in enumerate(col_ids):
					n = col_ids[:i] + col_ids[i+1:]
					z = x[:, n]

					H = wuml.HSIC(z,x, sigma_type='mpd', normalize_hsic=True)
					history_row[name_list[col_ids[i]]] = H
					if highest_hsic < H:
						highest_hsic = H
						highest_hsic_column = col_ids[i]

					#import pdb; pdb.set_trace()
				removal_table = removal_table.append({'column ID': name_list[highest_hsic_column], 'Residual HSIC': highest_hsic}, ignore_index=True)	
				removal_history = removal_history.append( history_row, ignore_index=True)
				col_ids.remove(highest_hsic_column)

		self.removal_table = removal_table
		self.removal_history = removal_history

	def __str__(self): 
		return str(self.removal_table)

	def __repr__(self): 
		return str(self.removal_table)

