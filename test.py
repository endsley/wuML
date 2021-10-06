#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib

data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, data_name='Chem_decimated_imputed', 
										data_path='examples/data/', save_as='no saving',
										xdata_type="%.4f", ydata_type="%.4f", test_percentage=0.1)


X_train = wuml.center_and_scale(X_train)
weights = wuml.get_likelihood_weight(y_train)
weights = weights.get_data_as('Tensor')

def costFunction(x, y, ŷ, ind):
	relu = torch.nn.ReLU()

	W = torch.squeeze(weights[ind])
	n = len(ind)
	ŷ = torch.squeeze(ŷ)
	y = torch.squeeze(y)

	penalty = torch.sum(relu(W*(ŷ - y)))/n	# This will penalize predictions higher than true labels
	loss = torch.sum(W*((y - ŷ)**2))/n + 0.8*penalty
	return loss


bNet = wuml.basicNetwork(costFunction, X_train, networkStructure=[(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')], max_epoch=6000, learning_rate=0.001)
bNet.train()

Ŷ_train = bNet(X_train, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
Ŷ_test = bNet(X_test, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

SR = wuml.summarize_regression_result(X_train.Y, Ŷ_train)
print(SR.true_vs_predict(sort_based_on_label=True))
import pdb; pdb.set_trace()




