
import torch
import torch.nn as nn


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

