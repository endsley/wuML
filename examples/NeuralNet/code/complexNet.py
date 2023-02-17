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

#	This code shows how you can mix and match different networks
#	This network simultaneously minimize CE and MSE loss


def status_printing(all_losses, epoch, lr):
	[total_loss, CE_loss, Regress_loss] = all_losses
	txt = '\tepoch: %d, Tloss: %.4f, CEloss: %.4f, MSELoss: %.4f, Learning Rate: %.8f'%((epoch+1), total_loss, CE_loss, Regress_loss, lr)
	wuml.write_to_current_line(txt)


#	You can also control the behavior of the network on call
def network_behavior_on_call(all_data, all_networks):
	net1 = all_networks[0]
	net2 = all_networks[1]
#
	#	the 1st 2 items of all_data will always be X, and y
	#	the rest will be what you include
	X = all_data[0]
	y = all_data[1]
	y2= all_data[2]

	ŷₐ = net1(X)
	ŷᵦ = net2(ŷₐ)

	labels = wuml.softmax(ŷₐ)
	return [labels, ŷᵦ]



def costFunction(all_data, all_networks):	
	net1 = all_networks[0]
	net2 = all_networks[1]
#
	#	the 1st 3 items of all_data will always be X, y, index
	#	the rest will be what you include
	X = all_data[0]
	y = all_data[1]
	indx = all_data[2]
	y2= all_data[3]

	# run data through the networks
	ŷₐ = net1(X)
	ŷᵦ = net2(ŷₐ)
#
	CE_loss = wuml.CrossEntropyLoss(y, ŷₐ)
	Regress_loss = wuml.MSELoss(y2, ŷᵦ)
	total_loss = 0.1*CE_loss + Regress_loss
#
	return [total_loss, CE_loss, Regress_loss]






#	This data has both regression and classification labels (3 classes)
#	the network will train on both labels by
#		using the 1st network to get 3 softmax outputs, 
#		from the 1st network, it will connect to the 2nd network, 
#			expand to width of 5 and compress down to 1 for regression
data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					extra_data='../../data/wine_regress_label.csv', 
					preprocess_data='center and scale', 
					 batch_size=16, label_type='discrete')
Y2 = data.extra_data_dictionary['numpy'][0]

netStructureList = []
netStructureList.append([(10,'relu'),(3,'none')])
netStructureList.append([(5,'relu'),(1,'none')])
netInputDimList = [13, 3]

cNet = wuml.combinedNetwork(data, netStructureList, netInputDimList, costFunction, max_epoch=2000,
							on_new_epoch_call_back=status_printing,
							network_behavior_on_call=network_behavior_on_call,
							Y_dataType=torch.LongTensor, extra_dataType=[torch.FloatTensor]) 
cNet.fit()
[labels, ŷᵦ] = cNet(data)


CR = wuml.summarize_classification_result(data.Y, labels)
wuml.jupyter_print('\nAccuracy : %.3f\n\n'%CR.avg_error())

SR = wuml.summarize_regression_result(Y2, ŷᵦ)
Reg_result = SR.true_vs_predict(print_result=False)
wuml.jupyter_print(Reg_result, display_all_rows=True)


