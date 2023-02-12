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
import wplotlib
from wuml import HSIC


data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', randomly_shuffle_batch=True,
					batch_size=20, label_type='discrete', encode_discrete_label_to_one_hot=True )

#	Remember that you are trying to "minimize" this objective
def costFunction(x, y, ŷ, ind):
	H = HSIC(ŷ, y, X_kernel='linear', Y_kernel='linear', sigma_type='mpd' )
	# compare the HSIC value against numpy library
	#y = y.detach().cpu().numpy()
	#ŷ = ŷ.detach().cpu().numpy()
	#H2 = HSIC(ŷ, y, X_kernel='linear', Y_kernel='linear', sigma_type='mpd' )
	#import pdb; pdb.set_trace()
	return -H  


#-----------------------------------------------
#	Create basic network and train
bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(30,'relu'),('bn', True), (50,'relu'),('bn', True),(3,'none')], max_epoch=500, learning_rate=0.001)
bNet.train(print_status=True)
Ŷ = bNet(data, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor

Network_out_v_Label = np.hstack((Ŷ, data.Y))
wuml.jupyter_print(Network_out_v_Label)


