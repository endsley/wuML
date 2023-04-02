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


data = wuml.wData(xpath='./wine.csv', ypath='./wine_label.csv', 
					preprocess_data='center and scale', batch_size=128, label_type='discrete')

W = np.array([[2],[10],[12],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
regressX = data.X.dot(W)
wuml.label_csv_out(regressX, './wine_regress_label.csv', float_format='%.4f')

