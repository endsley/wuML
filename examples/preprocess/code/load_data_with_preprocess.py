#!/usr/bin/env python

import wuml 
import numpy as np

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='center and scale')

wuml.jupyter_print('mean : ', np.mean(data.X, axis=0))
wuml.jupyter_print('std : ', np.std(data.X, axis=0))

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='between 0 and 1')

wuml.jupyter_print('max : ', np.max(data.X))
wuml.jupyter_print('min : ', np.std(data.X))

