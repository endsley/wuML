#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, RandomOverSampler
import wuml
import numpy as np


class rebalance_data:
	def __init__(self, data, y=None, method='smote'):
		'''method : 'smote', 'oversampling' '''

		self.label_column_name=None
		self.columns = None
		self.label_column_id = None

		if y is None:
			if wuml.wtype(data) == 'wData':
				y = data.Y
				X = data.X
				self.columns = data.columns
				self.label_column_name = data.label_column_name
				self.label_column_id = data.label_column_id
			else:
				raise ValueError('data must be wData if the label y is None')
		else:
			X = wuml.ensure_numpy(data)
			y = wuml.ensure_numpy(y)
			
		if method == 'smote':
			yStats = wuml.get_label_stats(y, print_stat=False)
			sn = yStats.get_columns('num sample')
			smallest_class_sample_num =  np.min(sn.X)
			if smallest_class_sample_num < 6: raise ValueError('Error: rebalance data with smote must have at least 6 samples in the smallest class.')
	
			sm = SMOTE(random_state=42)
			X_res, y_res = sm.fit_resample(X, y)
		elif method == 'oversampling':
			ROS = RandomOverSampler()
			X_res, y_res = ROS.fit_resample(X, y)


		self.balanced_data = wuml.wData(X_npArray=X_res, Y_npArray=y_res, first_row_is_label=False, label_type='discrete',
										column_names=self.columns, label_column_name=self.label_column_name, label_column_id=self.label_column_id)


if __name__ == "__main__":
	X1 = np.random.randn(20, 2) + 5
	X2 = np.random.randn(7, 2) - 5
	X = np.vstack((X1,X2))
	y = np.append(np.ones(20), np.zeros(7))

	rebalancer = rebalance_data(X, y, method='oversampling')
	print(X,'\n')

	print(rebalancer.balanced_data.shape)
	print(rebalancer.balanced_data)
	import pdb; pdb.set_trace()
