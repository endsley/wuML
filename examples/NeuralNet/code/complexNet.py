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


def costFunction(all_data):	
	return 1


data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					preprocess_data='center and scale', batch_size=128, label_type='discrete')

netStructureList = []
netStructureList.append([(10,'relu'),(3,'none')])
netStructureList.append([(10,'relu'),(1,'none')])
netInputDimList = [13, 3]

cNet = wuml.combinedNetwork(data, netStructureList, netInputDimList, costFunction, max_epoch=2000 ) 
cNet.fit()

