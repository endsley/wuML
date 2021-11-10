#!/usr/bin/env python

import wuml 
import numpy as np


#data = wuml.make_moons()

x1 = np.random.randn(20,1)
x2 = x1 #+ 0.05*np.random.randn(20,1)
x3 = np.random.randn(20,1)

data = np.hstack((x1,x2,x3))
wuml.feature_selection(data)
