#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml
import sklearn
import shap
import wplotlib
import numpy as np
from wplotlib import histograms
import torch
import torch.nn as nn
from torch.autograd import Variable


dataTrain = wuml.wData(xpath='../../data/gestAge_train.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					first_row_is_label=True)
dataTrain = wuml.center_and_scale(dataTrain)
dataTest = wuml.wData(xpath='../../data/gestAge_test.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					first_row_is_label=True)

dataTest = wuml.center_and_scale(dataTest)
weights = wuml.wData(xpath='../../data/Chem_sample_weights.csv')
weights = weights.get_data_as('Tensor')

def costFunction(x, y, ŷ, ind):
	relu = nn.ReLU()
#
	W = torch.squeeze(weights[ind])
	n = len(ind)
	ŷ = torch.squeeze(ŷ)
	y = torch.squeeze(y)
#
	penalty = torch.sum(relu(W*(ŷ - y)))/n	# This will penalize predictions higher than true labels
	return torch.sum(W*((y - ŷ)**2))/n + 1*penalty

structures = []
structures.append([(400,'relu'),(400,'relu'),(400,'relu'),(1,'none')])
structures.append([(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')])
structures.append([(200,'relu'),(200,'relu'),(1,'none')])
structures.append([(100,'relu'),(100,'relu'),(1,'none')])
structures.append([(100,'relu'),(1,'none')])
structures.append([(60,'relu'),(1,'none')])
#epochs = [100, 200, 400, 800, 1600, 3200, 6000]
epochs = [1, 2, 4, 8]
variousSettings = list(zip(structures, epochs))

for S in structures:
	for E in epochs:
		
		Fn = 'RN_' + str(len(S)) + '_' + wuml.gen_random_string(str_len=3) + '_' + str(E)
		wuml.ensure_path_exists('./tmp/')
		wuml.ensure_path_exists('./tmp/' + Fn)
	
		bNet = wuml.basicNetwork(costFunction, dataTrain, networkStructure=S, max_epoch=E, learning_rate=0.001)
		bNet.train()
		
		Ŷ1 = bNet(dataTrain, output_type='ndarray')
		SR_train = wuml.summarize_regression_result(dataTrain.Y, Ŷ1)
		SR_train.true_vs_predict(write_path='./tmp/' + Fn + '/Yvŷ_train.txt', sort_based_on_label=True)
		E1 = SR_train.avg_error()
	
	
		Ŷ = bNet(dataTest, output_type='ndarray')
		SR_test = wuml.summarize_regression_result(dataTest.Y, Ŷ)
		SR_test.true_vs_predict(write_path='./tmp/' + Fn + '/Yvŷ_test.txt', sort_based_on_label=True)
		wuml.jupyter_print('\nRun:', Fn, 'Train Error:', E1, 'Test Error:', SR_test.avg_error(), '\n')
		
		wuml.pickle_dump(bNet, './tmp/' + Fn + '/best_network.pk')
		wuml.write_to(bNet.info(printOut=False), './tmp/' + Fn + '/best_network_info.txt')		
		sortedY = wuml.sort_matrix_rows_by_a_column(SR_test.side_by_side_Y, 0)
	
		del bNet


