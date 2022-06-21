#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np
import torch
import wplotlib


#	Examples of both "with" and "without" batch normalization
#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')]
#				Potential Activation Functions are: softmax, relu, tanh, sigmoid, none
#		2. Define a cost function
#		3. Call train()

data = wuml.wData(xpath='../../data/regress.csv', ypath='../../data/regress_label.csv', batch_size=20, label_type='continuous')
def costFunction(x, y, ŷ, ind):
	return torch.sum((y- ŷ) ** 2)	


#-----------------------------------------------
#	Create basic network and train
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(30,'relu'),(50,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001)
bNet.train(print_status=True)
Ŷ = bNet(data, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

#	Check out our predictions
SR = wuml.summarize_regression_result(data.Y, Ŷ)
print(SR.true_vs_predict(print_result=False))


#	Draw the regression line
newX = np.expand_dims(np.arange(0,5,0.1),1)
Ŷline = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor




#-----------------------------------------------
#	Create network with batch normalization and train		
#	For bn layers, ('bn', False), the 2nd boolean term sets if we use γ,β terms or not
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(30,'relu'),('bn', True), (50,'relu'),('bn', True),(1,'none')], max_epoch=500, learning_rate=0.001)
bNet.train(print_status=True)
Ŷ = bNet(data, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

#	Check out our predictions
SR = wuml.summarize_regression_result(data.Y, Ŷ)
print(SR.true_vs_predict(print_result=False))


#	Draw the regression line
newX_bn = np.expand_dims(np.arange(0,5,0.1),1)
bNet.eval()		# if using BN you should set setting to eval before evaluating test samples
Ŷline_bn = bNet(newX_bn, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor



#	plot the results out
splot = wplotlib.scatter(data.X, data.Y, marker='o', show=False, subplot=121)
lp = wplotlib.lines(newX, Ŷline, title_font=11, title='Without Batch Normalization', xlim=[0,5], ylim=[0,5], show=False)	# show must be false if it is a subplot

splot = wplotlib.scatter(data.X, data.Y, marker='o', show=False, subplot=122)
lp = wplotlib.lines(newX, Ŷline, title_font=11, title='With Batch normalization', xlim=[0,5], ylim=[0,5], )	
