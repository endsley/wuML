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
import torch
import torch.nn as nn

def costFunction(x, x̂, ẙ, y, ŷ, ind):	
#	x -> encoder -> x̂
#	x̂ -> encoder_linear_output -> ẙ	
#	x̂ -> decoder -> ŷ	
#	possible autoencoder objective λ could be 0
#	loss = (x - ŷ)ᒾ + λ * objective(ẙ, y)
#
#	This function can return 1 value or 3 values in a list
#	if return 1 value, just the loss
#	if return 3 values, [total_loss, reconstruction_loss, extra network from ẙ loss]
#
#	In this example, we perform both reconstruction and CE loss
#
	CE_loss = nn.CrossEntropyLoss() #weird pytorch, dim of y is 1, and ŷ is 20x3
	R = torch.sum((x - ŷ) ** 2)/(128*13)	#scaled by batch size times data dimension
	CE = CE_loss(ẙ, y)
	loss = R + CE
	return [loss, R, CE]


def costFunction2(x, x̂, ẙ, y, ŷ, ind):	
#	Example of just return 1 value, this is just a regular autoencoder
	return torch.sum((x - ŷ) ** 2)



data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					preprocess_data='center and scale', batch_size=128, label_type='discrete')

AE = wuml.autoencoder(12, data, default_depth=2, costFunction=costFunction, # costFunction and costFunction2 both works
						max_epoch=2000, encoder_output_weight_structure=[(3,'none')] ) 
AE.fit()

#	Notice that the reconstruction did a reasonable job
ŷ = AE(data)
wuml.jupyter_print(ŷ[0:5,0:10])
wuml.jupyter_print(data.X[0:5,0:10])


#	This is the bottleneck output which is d=12
x̂ = AE.reduce_dimension(data, output_type='wData')
wuml.jupyter_print(x̂.shape)


#	This is the objective network output 
ẙ = AE.objective_network(data)

#	Here we use the bottleneck to perform SVM classification
cf = wuml.classification(x̂, classifier='SVM')
wuml.jupyter_print(cf.result_summary(print_out=False))


#	Here we use the objective network output to perform LogisticRegression classification
cf = wuml.classification(ẙ, classifier='LogisticRegression')
wuml.jupyter_print(cf.result_summary(print_out=False))

