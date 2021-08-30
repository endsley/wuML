#!/usr/bin/env python

import wuml
import numpy as np
import torch
import torch.nn as nn
import wplotlib
from torch.autograd import Variable




data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
data = wuml.center_and_scale(data)

weights = wuml.wData(xpath='examples/data/Chem_sample_weights.csv')
#weights = weights.get_data_as('Tensor')



#You must have the first 2 variables included (fold_id, one_fold_data_list)
def run_each_fold(fold_id, one_fold_data_list):
	[X_train, X_test] = one_fold_data_list[0]
	[Y_train, Y_test] = one_fold_data_list[1]
	[W_train, W_test] = one_fold_data_list[2]

	single_fold_train_data = wuml.wData( X_npArray=X_train, Y_npArray=Y_train, label_type='continuous')
	single_fold_test_data = wuml.wData( X_npArray=X_test, Y_npArray=Y_test, label_type='continuous')
	W_train = wuml.ensure_tensor(W_train)

	def costFunction(x, y, ŷ, ind):
		relu = nn.ReLU()
	
		W = torch.squeeze(W_train[ind])
		n = len(ind)
		ŷ = torch.squeeze(ŷ)

		penalty = torch.sum(relu(W*(ŷ - y)))/n
		return torch.sum(W*((y - ŷ)**2))/n + 0.8*penalty

	
	bNet = wuml.basicNetwork(costFunction, single_fold_train_data, networkStructure=[(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')], max_epoch=6000, learning_rate=0.001)
	bNet.train()

	Ŷ_train = bNet(single_fold_train_data, output_type='ndarray')
	Ŷ = bNet(single_fold_test_data, output_type='ndarray')

	train_result = wuml.summarize_regression_result(single_fold_train_data.Y, Ŷ_train)
	train_avg_error = train_result.avg_error()

	test_result = wuml.summarize_regression_result(single_fold_test_data.Y, Ŷ)
	test_avg_error = test_result.avg_error()
	print('\n\nFold %d, train error: %.4f, test error: %.4f'%(fold_id, train_avg_error, test_avg_error))

	return [train_avg_error, test_avg_error]

all_results = wuml.run_K_fold_on_function(10, [data.X, data.Y, weights.X], run_each_fold, {}) 
