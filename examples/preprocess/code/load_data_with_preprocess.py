#!/usr/bin/env python

import wuml 
import numpy as np

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='center and scale')

print('mean : ', np.mean(data.X, axis=0))
print('std : ', np.std(data.X, axis=0))

data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='between 0 and 1')

print('max : ', np.max(data.X))
print('min : ', np.std(data.X))

