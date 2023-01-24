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

#	x -> encoder -> x̂
#	x̂ -> encoder_linear_output -> ẙ	
#	x̂ -> decoder -> ŷ	
#	possible autoencoder objective λ could be 0
#	loss = (x - ŷ)ᒾ + λ * objective(ẙ, y)
def costFunction(x, x̂, ẙ, y, ŷ, ind):
	#import pdb; pdb.set_trace()
	#return torch.sum((x - ŷ) ** 2)

	CE_loss = nn.CrossEntropyLoss() #weird from pytorch, dim of y is 1, and ŷ is 20x3
	R = torch.sum((x - ŷ) ** 2)	

	import pdb; pdb.set_trace()
	loss = R + CE_loss(ẙ, y) 				
	return loss

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					preprocess_data='center and scale', batch_size=128, label_type='discrete')
#data = wuml.make_classification_data( n_samples=200, n_features=10, n_informative=4, n_classes=2)
#self.yout = self.y1 = F.relu(self.l0(self.y0))



AE = wuml.autoencoder(12, data, default_depth=2, costFunction=costFunction, max_epoch=2000, encoder_output_weight_structure=[(3,'none')] ) 
AE.fit()
y = AE(data)
x̂ = AE.reduce_dimension(data, output_type='wData')

cf = wuml.classification(x̂, classifier='SVM')
wuml.jupyter_print(cf.result_summary(print_out=False))

import pdb; pdb.set_trace()
