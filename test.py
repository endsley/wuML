#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib
from torch.autograd import Variable


import wuml 
import torch
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
	
#l2x_synthetic_blocks.csv  l2x_synthetic.csv         l2x_synthetic_label.csv

wuml.set_terminal_print_options(precision=2)
data = wuml.wData(xpath='examples/data/l2x_synthetic.csv', ypath='examples/data/l2x_synthetic_label.csv', label_type='continuous', batch_size=20)
data = wuml.center_and_scale(data)
d = data.X.shape[1]

θˢ = [(2000,'relu'),(2000,'relu'),(2000,'relu'),(d*2,'none')]
#θˢ = [(2000,'relu'),(2000,'relu'),(2000,'relu'),(d*2,'none')]
θᴾ = [(200,'relu'),(200,'relu'),(200,'relu'),(1,'none')]
#θᴾ = [(1,'none')]

P = wuml.l2x(data, max_epoch=100, learning_rate=0.001, use_binary=True,
				selector_network_structure=θˢ,predictor_network_structure=θᴾ)
P.train()
ŷ = P(data.X)


wuml.output_regression_result(data.Y, ŷ)

S = P(data.X, test_phase=True)
import pdb; pdb.set_trace()
#out = wuml.output_regression_result(data.Y, ŷ)






#import wuml
#import numpy as np
#import torch
#import wplotlib
#
#
##	Splits the data into training and test
#
#data = wuml.wData(xpath='examples/data/regress.csv', ypath='examples/data/regress_label.csv', batch_size=20, label_type='continuous')
#wuml.split_training_test(data, data_name='regress', data_path='./examples/data/', xdata_type="%.4f", ydata_type="%.4f", test_percentage=0.2)
#
#
#def costFunction(x, y, ŷ, ind):
#	ŷ = torch.squeeze(ŷ)
#	return torch.sum((y- ŷ) ** 2)	
#
#data_train = wuml.wData(xpath='examples/data/regress_train.csv', ypath='examples/data/regress_train_label.csv', batch_size=20, label_type='continuous')
#data_test = wuml.wData(xpath='examples/data/regress_test.csv', ypath='examples/data/regress_test_label.csv', batch_size=20, label_type='continuous')
#
## costFunction can be a function or string: mse, L1
#bNet = wuml.basicNetwork('mse', data_train, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001)
#bNet.train()
#
#
#Ŷ_train = bNet(data_train, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
#Ŷ_test = bNet(data_test, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
#
#newX = np.expand_dims(np.arange(0,5,0.1),1)
#Ŷ_line = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
#
#
##	plot the results out
#splot = wplotlib.scatter()
#splot.add_plot(data_train.X, data_train.Y, marker='o', color='blue')
#splot.add_plot(data_test.X, data_test.Y, marker='o', color='red')
#
#lp = wplotlib.lines()	
#lp.add_plot(newX, Ŷ_line)
#
#splot.show(title='Train/Test Network Regression', xlabel='x-axis', ylabel='y-axis',
#			imgText='Blue dot:Training Data\nRed dot: Test Data')
#
#import pdb; pdb.set_trace()
