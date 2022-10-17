#!/usr/bin/env python
import os 
import sys

if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import wplotlib 
import numpy as np
from wuml import jupyter_print




n = 20
tick = np.array([1, 2, 3, 4, 5, 6])
a = np.array([0.4, 0.2, 0.4, 0, 0, 0])
b = np.array([0.4, 0.4, 0.2, 0, 0, 0])
c = np.array([0, 0, 0, 0.4, 0.2, 0.4])


A = wplotlib.bar(tick,a, 'Normalized Histogram Distribution 1', 'values', 'Count', subplot=311, figsize=(7,8))
wplotlib.bar(tick,b, 'Normalized Histogram Distribution 2', 'values', 'Count', subplot=312)
wplotlib.bar(tick,c, 'Normalized Histogram Distribution 3', 'values', 'Count', subplot=313)
A.show()

jupyter_print('The distance between a and b should be 0.2 = 0.2*1')
jupyter_print('The distance between a and c should be 3 = 0.4*3 + 0.2*3 + 0.4*3')
jupyter_print('Notice that the distance (a,b) is much shorter than distance (a,c)')

wd_1 = wuml.wasserstein_distance(a, b)
wd_2 = wuml.wasserstein_distance(a, c)
jupyter_print('W(a,b) = %.3f , W(a,c) = %.3f'%(wd_1, wd_2))
