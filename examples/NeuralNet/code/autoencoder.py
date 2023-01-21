#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np


data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', batch_size=20, label_type='discrete')
wuml.autoencoder(3, data) 

