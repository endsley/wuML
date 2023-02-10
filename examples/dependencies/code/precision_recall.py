#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import numpy as np
import wuml

#	Binary even, 1 is positive
y = np.array([1,1,1,1,0,0,0,0,0,0])	# true labels
ŷ = np.array([1,1,1,1,1,0,0,0,0,0]) # predicted labels

#	Precision is the probably of you being correct if you predicted a positive event
#		here we have 5 predicted positive events, but only 4 were correct 80%
#	Recall is the probably of predicting a positive even if it did occure
#		here we have 4 true positive events and we got all of them so 100%
P = wuml.precision(y, ŷ)
R = wuml.recall(y, ŷ)

wuml.jupyter_print('Precision: %.3f'%P)
wuml.jupyter_print('Recall : %.3f'%R)

