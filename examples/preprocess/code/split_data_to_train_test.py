#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml
import sklearn
import wplotlib
from wplotlib import histograms


data = wuml.wData(xpath='../../data/small.csv', batch_size=2, 
					label_type='continuous', label_column_name='Y', 
					mv_columns_to_extra_data='B',
					first_row_is_label=True)

# Original data, label, extra labels
wuml.jupyter_print(wuml.block_matrix_concatenate([data, data.Y, data.xDat]))

# data_path : will save the the files gestAge_train.csv and gestAge_test.csv to ../../data folder
# test_percentage : 0.1 implies that 90% will be training and 10% will be test
X_train, X_test, y_train, y_test = wuml.split_training_test(data, test_percentage=0.2)

# after splitting data to training and test, notice the split
wuml.jupyter_print(wuml.block_matrix_concatenate([X_train, X_train.Y, X_train.xDat]))
wuml.jupyter_print(wuml.block_matrix_concatenate([X_test, X_test.Y, X_test.xDat]))

