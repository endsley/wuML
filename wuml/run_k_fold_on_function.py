#!/usr/bin/env python

import numpy as np
import wuml
from sklearn.model_selection import KFold

# data_list is a list of data, example: [X, Y] , or just [X]
def run_K_fold_on_function(K, data_list, func, func_args): 
	X = data_list[0]
	Xnp = wuml.ensure_numpy(X)
	all_results = []

	kf = KFold(n_splits=K, shuffle=True)
	kf.get_n_splits(Xnp)

	fold_id = 1
	data_one_fold_list = []
	for train_index, test_index in kf.split(Xnp):
		for dat in data_list:
			datNP = wuml.ensure_numpy(dat)
			data_one_fold_list.append([datNP[train_index], datNP[test_index]])

		func_args['one_fold_data_list'] = data_one_fold_list
		func_args['fold_id'] = fold_id
		fold_id += 1
		single_result = func(**func_args)
		all_results.append(single_result)

	return all_results
