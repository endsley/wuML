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
weights = weights.get_data_as('Tensor')



#You must have the first 2 variables included (fold_id, one_fold_data_list)
def run_each_fold(fold_id, one_fold_data_list):
	[X_train, X_test] = one_fold_data_list[0]
	[Y_train, Y_test] = one_fold_data_list[1]
	[W_train, W_test] = one_fold_data_list[2]

	def costFunction(x, y, ŷ, ind):
		relu = nn.ReLU()
	
		W = torch.squeeze(W_train[ind])
		n = len(ind)
		ŷ = torch.squeeze(ŷ)
	
		penalty = torch.sum(relu(W*(ŷ - y)))/n
		return torch.sum(W*((y - ŷ)**2))/n + 0.8*penalty

	
	bNet = wuml.basicNetwork(costFunction, X_train, networkStructure=[(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')], max_epoch=6000, learning_rate=0.001)
	bNet.train()
	
	Ŷ = bNet(data, output_type='ndarray')
	output = wuml.output_regression_result(data.Y, Ŷ)
	print(output)


wuml.run_K_fold_on_function(10, [data.X, data.Y], run_each_fold, {}) 
