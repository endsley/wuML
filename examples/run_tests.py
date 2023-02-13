#!/usr/bin/env python

import os
import sys

folder_list = []
folder_list.append('classification')
folder_list.append('data_stats')
folder_list.append('dimension_reduction')
folder_list.append('distance_between_distributions')
folder_list.append('distribution_modeling')
folder_list.append('feature_selection')
folder_list.append('IO')
folder_list.append('random_features')
folder_list.append('rebalance_data')
folder_list.append('wData')
folder_list.append('clustering')
folder_list.append('dependencies')
folder_list.append('measure')
folder_list.append('preprocess')
folder_list.append('NeuralNet')


for f in folder_list:
	print('Going into ' + f)
	os.chdir('./' + f + '/code')
	os.system('./test_code.py')
	os.chdir('../../')

