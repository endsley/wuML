#!/usr/bin/env python

import wuml
import numpy as np
import torch
import torch.nn as nn
import wplotlib


#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'softmax')]
#		2. Define a cost function
#		3. Call train()

data = wuml.load_csv(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', batch_size=20)

def costFunction(x, y, ŷ, ind):
	lossFun = nn.CrossEntropyLoss() 
	loss = lossFun(ŷ, y) #weird from pytorch, dim of y is 1, and ŷ is 20x3	
	return loss


#It is important for pytorch that with classification, you need to define Y_dataType=torch.int64
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(3,'none')], 
						Y_dataType=torch.int64, max_epoch=500, learning_rate=0.001)
bNet.train()


#_, predicted = torch.max(outputs, 1)
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


##	Test out on test data
#newX = np.expand_dims(np.arange(0,5,0.1),1)
#Ŷ = bNet(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
#
##	plot the results out
#splot = wplotlib.scatter()
#splot.add_plot(data.X, data.Y, marker='o')
#
#lp = wplotlib.lines()	
#lp.add_plot(newX, Ŷ)
#
#splot.show(title='Basic Network Regression', xlabel='x-axis', ylabel='y-axis')
#
#import pdb; pdb.set_trace()
