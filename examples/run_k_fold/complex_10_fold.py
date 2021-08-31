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

#Define how to run each fold
#You must have the first 2 variables included (fold_id, one_fold_data_list)
def run_each_fold(fold_id, one_fold_data_list):
	[X_train, X_test] = one_fold_data_list[0]
	[Y_train, Y_test] = one_fold_data_list[1]
	[W_train, W_test] = one_fold_data_list[2]

	single_fold_train_data = wuml.wData( X_npArray=X_train, Y_npArray=Y_train, label_type='continuous')
	single_fold_test_data = wuml.wData( X_npArray=X_test, Y_npArray=Y_test, label_type='continuous')
	W_train = wuml.ensure_tensor(W_train)
	
	#	Since I am running a neural network, I need to define a cost function
	def costFunction(x, y, ŷ, ind):
		relu = nn.ReLU()
	
		W = torch.squeeze(W_train[ind])
		n = len(ind)
		ŷ = torch.squeeze(ŷ)
		y = torch.squeeze(y)

		penalty = torch.sum(relu(W*(ŷ - y)))/n
		return torch.sum(W*((y - ŷ)**2))/n + 0.8*penalty

	
	bNet = wuml.basicNetwork(costFunction, single_fold_train_data, networkStructure=[(200,'relu'),(200,'relu'),(1,'none')], max_epoch=5000, learning_rate=0.001)
	bNet.train()

	Ŷ_train = bNet(single_fold_train_data, output_type='ndarray')
	Ŷ = bNet(single_fold_test_data, output_type='ndarray')

	train_result = wuml.summarize_regression_result(single_fold_train_data.Y, Ŷ_train)
	train_avg_error = train_result.avg_error()

	test_result = wuml.summarize_regression_result(single_fold_test_data.Y, Ŷ)
	test_avg_error = test_result.avg_error()
	print('\n\nFold %d, train error: %.4f, test error: %.4f'%(fold_id, train_avg_error, test_avg_error))

	return [train_result, test_result, bNet] # <- all_results is a list of these lists


if __name__=="__main__":
	all_results = wuml.run_K_fold_on_function(10, [data.X, data.Y, weights.X], run_each_fold, {}) 
	
	lowest_error = 100
	outStr = ''
	lowest = ''
	avg_train_error = []
	avg_test_error = []
	for idx, single_result in enumerate(all_results):
		[train_result, test_result, bNet] = single_result
		ε1 = train_result.avg_error()
		ε2 = test_result.avg_error()
		avg_train_error.append(ε1)
		avg_test_error.append(ε2)
	
		outStr += 'Fold %d, train error: %.4f, test error: %.4f\n'%(idx, ε1, ε2)
	
		if lowest_error > test_result.avg_error():
			lowest_error = test_result.avg_error()
			lowest = 'Lowest Error : Fold %d, train error: %.4f, test error: %.4f\n\n'%(idx, ε1, ε2)
			bN = wuml.basicNetwork(None, None, simplify_network_for_storage=bNet)
			bN.train_result = train_result
			bN.test_result = test_result
			wuml.pickle_dump(bN, 'best_network.pk')

	Avg_train = np.mean(np.array(avg_train_error))
	Avg_test = np.mean(np.array(avg_test_error))
	Topline = 'Average Train Error: %.4f, Average Test Error: %.4f\n'%(Avg_train, Avg_test)
	wuml.write_to(Topline + lowest + outStr, './10_fold_summary.txt')


	#	Using the following code can reload the network

	#bN = wuml.pickle_load('best_network.pk')
	#Ŷ = bN(data)
	#output = wuml.output_regression_result(data.Y, Ŷ, write_path='YvŶ.txt')
	#import pdb; pdb.set_trace()

