#!/usr/bin/env python

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
import wuml
import numpy as np


class rebalance_data:
	def __init__(self, data, y=None, method='smote'):

		label_column_name=None
		if y is None:
			if wuml.wtype(data) == 'wData':
				y = X.Y
				X = data.X
			else:
				raise ValueError('data must be wData if the label y is None')
			label_column_name = data.get_column_names_as_a_list()
		else:
			X = wuml.ensure_numpy(data)
			y = wuml.ensure_numpy(y)
			
	
		sm = SMOTE(random_state=42)
		X_res, y_res = sm.fit_resample(X, y)
	
		self.balanced_data = wuml.wData(X_npArray=X_res, Y_npArray=y_res, first_row_is_label=False, label_column_name=label_column_name, label_type='discrete')

if __name__ == "__main__":
	X = np.random.randn(100, 10)
	y = np.append(np.ones(20), np.zeros(80))

	rebalancer = rebalance_data(X, y)
	print(rebalancer.balanced_data.shape)
	print(rebalancer.balanced_data)
	import pdb; pdb.set_trace()
