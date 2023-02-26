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
import torch.nn as nn
import wplotlib
from sklearn.metrics import accuracy_score


#	The idea of training a neural network boils down to 3 steps
#		1. Define a network structure
#			Example: This is a 3 layer network with 100 node width
#				networkStructure=[(100,'relu'),(100,'relu'),(1,'softmax')]
#		2. Define a cost function
#		3. Call train()
#		4. Display results

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', batch_size=20, label_type='discrete')

def costFunction(x, y, ŷ, ind):
	lossFun = nn.CrossEntropyLoss() 
	loss = lossFun(ŷ, y) #weird from pytorch, dim of y is 1, and ŷ is 20x3	
	return loss


#It is important for pytorch that with classification, you need to define Y_dataType=torch.int64
#You can define a costFunction, but for classification it can be directly set to 'CE'
#bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(3,'none')], 
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(3,'none')], 
						Y_dataType=torch.LongTensor, max_epoch=400, learning_rate=0.001)
bNet.train(print_status=False)

#	Report Results
Ŷ = bNet(data.X, output_type='ndarray', out_structural='1d_labels')
CR = wuml.summarize_classification_result(data.Y, Ŷ)




