#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np
#import torch
import wplotlib

#	The original network was trained by basicRegression.py
#	here we load the trained network and use it
#    ---------------------------------------------------------
net = wuml.load_torch_network('./basicRegressionNet.pk')
data = wuml.wData(xpath='../../data/regress.csv', ypath='../../data/regress_label.csv', batch_size=20, label_type='continuous')
Ŷ = net(data)


#	Check out our predictions
SR = wuml.summarize_regression_result(data.Y, Ŷ)
SR.true_vs_predict(print_out=True)


#	Draw the regression line
newX = np.expand_dims(np.arange(0,5,0.1),1)
Ŷline = net(newX, output_type='ndarray')		#Takes Numpy array or Tensor as input and outputs a Tensor
splot = wplotlib.scatter(data.X, data.Y, marker='o', show=False)
lp = wplotlib.lines(newX, Ŷline, title_font=11, title='Fitting from stored network', xlim=[0,5], ylim=[0,5], show=True)	

#    ---------------------------------------------------------




#	The original network was trained by complexNet.py
#	here we load the trained network and use it
#    ---------------------------------------------------------
data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					extra_data='../../data/wine_regress_label.csv', 
					preprocess_data='center and scale', 
					 batch_size=16, label_type='discrete')
Y2 = data.extra_data_dictionary['numpy'][0]

net = wuml.load_torch_network('./ComplexNet.pk')
[labels, ŷᵦ] = net(data)

#	output results
CR = wuml.summarize_classification_result(data.Y, labels)
SR = wuml.summarize_regression_result(Y2, ŷᵦ)
#    ---------------------------------------------------------




#	The original network was trained by autoencoder.py
#	here we load the trained network and use it
#    ---------------------------------------------------------
data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					preprocess_data='center and scale', batch_size=128, label_type='discrete')

AE = wuml.load_torch_network('./autoencoder.pk')
#	This is the objective network output 
ẙ = AE.objective_network(data)

#	Here we use the objective network output to perform LogisticRegression classification
cf = wuml.classification(ẙ, classifier='LogisticRegression')
wuml.jupyter_print(cf.result_summary(print_out=False))


#    ---------------------------------------------------------
