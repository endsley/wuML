#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib
from torch.autograd import Variable

''' Weighted Regression
	Each sample is weighted based on its likelihood to balance the data
	
'''
 
data = wuml.wData(xpath='../data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])

data = wuml.center_and_scale(data)

weights = wuml.wData(xpath='../data/Chem_sample_weights.csv')
weights = weights.get_data_as('Tensor')

def costFunction(x, y, ŷ, ind):
	relu = nn.ReLU()

	W = torch.squeeze(weights[ind])
	n = len(ind)
	ŷ = torch.squeeze(ŷ)

	penalty = torch.sum(relu(W*(ŷ - y)))/n	# This will penalize predictions higher than true labels
	return torch.sum(W*((y - ŷ)**2))/n + 0.8*penalty

bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')], max_epoch=6000, learning_rate=0.001)
bNet.train()

Ŷ = bNet(data, output_type='ndarray')
output = wuml.output_regression_result(data.Y, Ŷ)
print(output)


