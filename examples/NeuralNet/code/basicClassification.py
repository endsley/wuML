#!/usr/bin/env python

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

data = wuml.wData(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', batch_size=20)

def costFunction(x, y, ŷ, ind):
	lossFun = nn.CrossEntropyLoss() 
	loss = lossFun(ŷ, y) #weird from pytorch, dim of y is 1, and ŷ is 20x3	
	return loss


#It is important for pytorch that with classification, you need to define Y_dataType=torch.int64
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(100,'relu'),(100,'relu'),(3,'none')], 
						Y_dataType=torch.int64, max_epoch=3000, learning_rate=0.001)
bNet.train()
netOutput = bNet(data.X)

#	Output Accuracy
_, Ŷ = torch.max(netOutput, 1)
Acc= accuracy_score(data.Y, Ŷ.cpu().numpy())
print('Accuracy: %.3f'%Acc)

