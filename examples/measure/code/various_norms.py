#!/usr/bin/env python


import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np
from wuml import norm
from wuml import jupyter_print

# default norm is 'l2'

X = np.array([[1,2],[3,1]])
jupyter_print(wuml.pretty_np_array(X, front_tab='\t', title='Original Matrix'))
jupyter_print(wuml.pretty_np_array(norm(X), front_tab='\t', title='L2 Norm for each row'))
jupyter_print(wuml.pretty_np_array(norm(X, 'l1'), front_tab='\t', title='L1 Norm for each row'))
jupyter_print(wuml.pretty_np_array(norm(X, 'fro'), front_tab='\t', title='Frobenius Norm'))
jupyter_print(wuml.pretty_np_array(norm(X, 'nuclear'), front_tab='\t', title='nuclear Norm'))
print('\n')


X = np.array([1,2])
jupyter_print(wuml.pretty_np_array(X, front_tab='\t', title='Original Vector'))
jupyter_print(wuml.pretty_np_array(norm(X, 'l1'), front_tab='\t', title='l1 Norm'))


