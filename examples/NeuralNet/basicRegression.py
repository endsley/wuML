#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib


#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'none')]
#				Potential Activation Functions are: softmax, relu, tanh, sigmoid, none
#		2. Define a cost function
#		3. Call train()

data = wuml.wData(xpath='../data/regress.csv', ypath='../data/regress_label.csv', batch_size=20)

def costFunction(x, y, ŷ, ind):
	ŷ = torch.squeeze(ŷ)
	return torch.sum((y- ŷ) ** 2)	

bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(1,'none')], max_epoch=500, learning_rate=0.001)
bNet.train()

#	Test out on test data
newX = np.expand_dims(np.arange(0,5,0.1),1)
Ŷ = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

#	plot the results out
splot = wplotlib.scatter()
splot.add_plot(data.X, data.Y, marker='o')

lp = wplotlib.lines()	
lp.add_plot(newX, Ŷ)

splot.show(title='Basic Network Regression', xlabel='x-axis', ylabel='y-axis')


