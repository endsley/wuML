#!/usr/bin/env python

import wuml
import sklearn
import shap
import wplotlib
import numpy as np
from wplotlib import histograms


data = wuml.wData(xpath='../data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
X_train, X_test, y_train, y_test = wuml.split_training_test(data, 'gestAge', data_path='../data/', 
															test_percentage=0.2, xdata_type="%.4f", ydata_type="%d")


H = histograms()
H.histogram(y_train, num_bins=10, title='Training Label Distribution', 
			subplot=121, facecolor='green', α=0.5, showImg=False, normalize=False)
H.histogram(y_test, num_bins=10, title='Test Label Distribution', 
			subplot=122, facecolor='green', α=0.5, showImg=False, normalize=False)

H.show()


