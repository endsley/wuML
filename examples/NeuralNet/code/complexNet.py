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


def costFunction(all_data, all_networks):	
	net1 = all_networks[0]
	net2 = all_networks[1]

	#	the 1st 3 items of all_data will always be X, y and index, 
	#	the rest will be what you include
	X = all_data[0]
	y = all_data[1]
	indx = all_data[2]
	y2 = all_data[3]
	b = net1(X)
	c = net2(b)
	return 1


#	This data has both regression and classification labels (3 classes)
#	the network will train on both labels by
#		using the 1st network to get 3 softmax outputs, 
#		from the 1st network, it will connect to the 2nd network, 
#			expand to width of 5 and compress down to 1 for regression
data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					extra_data='../../data/wine_regress_label.csv', 
					preprocess_data='center and scale', 
					 batch_size=16, label_type='discrete')

netStructureList = []
netStructureList.append([(10,'relu'),(3,'none')])
netStructureList.append([(5,'relu'),(1,'none')])
netInputDimList = [13, 3]

cNet = wuml.combinedNetwork(data, netStructureList, netInputDimList, costFunction, max_epoch=2000,
							Y_dataType=torch.LongTensor, extra_dataType=[torch.FloatTensor]) 
cNet.fit()

