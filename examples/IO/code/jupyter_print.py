#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import pandas as pd
import numpy as np
import matplotlib as mpl
from IPython.display import display, Math, Latex, HTML
#from IPython.display import clear_output, HTML


#	print string
wuml.jupyter_print('Hello String')


#	print data frame
df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, np.nan],[19, 439, 6, 452, 226,232]],
                  index=pd.Index(['Tumour (Positive)', 'Non-Tumour (Negative)'], name='Actual Label:'),
                  columns=pd.MultiIndex.from_product([['Decision Tree', 'Regression', 'Random'],['Tumour', 'Non-Tumour']], names=['Model:', 'Predicted:']))

wuml.jupyter_print(df)



#	print latex
wuml.jupyter_print('\int x\, dx', latex=True)
wuml.jupyter_print('\int x\, dx')
