#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml
import numpy as np
import wplotlib




X = np.array([[1,2,4],[3,4,5]])
x = np.array([[3,3,3]])

data = wuml.wData(X_npArray=X, column_names=['A','B', 'C'])
wuml.jupyter_print('Original Data')
wuml.jupyter_print(data, endString='\n')

C = data.get_columns('C')	
data.delete_column('C')
wuml.jupyter_print('With Column deleted')
wuml.jupyter_print(data, endString='\n')

data.append_columns(C)
wuml.jupyter_print('With Column added back')
wuml.jupyter_print(data, endString='\n')

data.append_rows(x)
wuml.jupyter_print('With row added')
wuml.jupyter_print(data, endString='\n')

data.rename_columns(['c','e','f'])
wuml.jupyter_print('With column labels renamed')
wuml.jupyter_print(data, endString='\n')

data.sort_by('f', ascending=True)
wuml.jupyter_print('Sort data by column f')
wuml.jupyter_print(data, endString='\n')

data.reset_index()
wuml.jupyter_print('With index of the rows reset')
wuml.jupyter_print(data, endString='\n')



import pdb; pdb.set_trace()

