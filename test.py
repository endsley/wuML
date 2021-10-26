#!/usr/bin/env python

import wuml 
import numpy as np

data = wuml.wData(xpath='examples/data/wine.csv', ypath='examples/data/wine_label.csv', 
					label_type='discrete', first_row_is_label=False, 
					preprocess_data='center and scale')


SC = wuml.clustering(data, n_clusters=3, method='GMM', gamma=0.05)
SC.plot_result(dimension_reduction_method='PCA')
SC.nmi_against_true_label()
import pdb; pdb.set_trace()
