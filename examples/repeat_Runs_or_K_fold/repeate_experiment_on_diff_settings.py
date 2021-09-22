#!/usr/bin/env python

import wuml
import sklearn
import shap
import wplotlib
import numpy as np
from wplotlib import histograms
import torch
import torch.nn as nn
from torch.autograd import Variable


dataTrain = wuml.wData(xpath='examples/data/gestAge_train.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)
dataTrain = wuml.center_and_scale(dataTrain)
dataTest = wuml.wData(xpath='examples/data/gestAge_test.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', 
					row_id_with_label=0)

dataTest = wuml.center_and_scale(dataTest)
weights = wuml.wData(xpath='examples/data/Chem_sample_weights.csv')
weights = weights.get_data_as('Tensor')

def costFunction(x, y, ŷ, ind):
	relu = nn.ReLU()

	W = torch.squeeze(weights[ind])
	n = len(ind)
	ŷ = torch.squeeze(ŷ)
	y = torch.squeeze(y)

	penalty = torch.sum(relu(W*(ŷ - y)))/n	# This will penalize predictions higher than true labels
	return torch.sum(W*((y - ŷ)**2))/n + 1*penalty

structures = []
structures.append([(400,'relu'),(400,'relu'),(400,'relu'),(1,'none')])
structures.append([(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')])
structures.append([(200,'relu'),(200,'relu'),(1,'none')])
structures.append([(100,'relu'),(100,'relu'),(1,'none')])
structures.append([(100,'relu'),(1,'none')])
structures.append([(60,'relu'),(1,'none')])
epochs = [100, 200, 400, 800, 1600, 3200, 6000]
#epochs = [1, 2, 4, 8]
variousSettings = list(zip(structures, epochs))

for S in structures:
	for E in epochs:
		
		Fn = 'RN_' + str(len(S)) + '_' + wuml.gen_random_string(str_len=3) + '_' + str(E)
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
		print('\nRun:', Fn, 'Train Error:', E1, 'Test Error:', SR_test.avg_error(), '\n')
		
		wuml.pickle_dump(bNet, './tmp/' + Fn + '/best_network.pk')
		wuml.write_to(bNet.info(printOut=False), './tmp/' + Fn + '/best_network_info.txt')		
		sortedY = wuml.sort_matrix_rows_by_a_column(SR_test.side_by_side_Y, 0)
	
		del bNet



##	Creation
#data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
#					label_type='continuous', label_column_name='finalga_best', 
#					row_id_with_label=0, columns_to_ignore=['id'])
#
#X_train, X_test, y_train, y_test = wuml.split_training_test(data, 'gestAge', data_path='./examples/data/', 
#															save_as='DataFrame', test_percentage=0.2, xdata_type="%.4f", 
#															ydata_type="%d")
#
#
#H = histograms()
#H.histogram(y_train, num_bins=10, title='Training Label Distribution', 
#			subplot=121, facecolor='green', α=0.5, showImg=False, normalize=False)
#H.histogram(y_test, num_bins=10, title='Test Label Distribution', 
#			subplot=122, facecolor='green', α=0.5, showImg=False, normalize=False)
#
#H.show(save_path='img/Train_Test_split.png')


