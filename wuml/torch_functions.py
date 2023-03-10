#!/usr/bin/env python
import os 
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import torch
import torch.nn as nn

#	Set this variable in the very beginning to cpu to ignore the gpu 
pytorch_device = None

#	Pytorch helpers
#	----------------------------------------------
def get_current_device():
	if wuml.pytorch_device is not None: return wuml.pytorch_device

	if torch.cuda.is_available(): wuml.pytorch_device = 'cuda'
	else: wuml.pytorch_device = 'cpu'
	return wuml.pytorch_device

#	Loss function
#	----------------------------------------------
def CrossEntropyLoss(y, ŷ):
	lossFun = nn.CrossEntropyLoss() 

	#weird from pytorch, dim of y is 20, and ŷ is 20x3
	# it seems that the loss function performs
	#	1. softmax
	return lossFun(ŷ, y) 	

def MSELoss(y, ŷ):
	mseFun = nn.MSELoss()
	if ŷ.shape != y.shape:
		raise ValueError('Error: \n\tIn wuml.MSELoss y.shape=%s and ŷ.shape=%s have dimension mismatch [maybe ŷ.squeeze() is needed].'%(str(ŷ.shape), str(y.shape)))

	return mseFun(ŷ, y)

def softmax(x, turn_into_label=False):
	m = nn.Softmax()
	xout = m(x)

	if turn_into_label:
		_, yout = torch.max(xout, 1)
		return yout
	else:
		return xout

