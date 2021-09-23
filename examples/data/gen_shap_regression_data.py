#!/usr/bin/env python

import numpy as np
import wuml

#	Generate a data for regression with 4 dimensions where
#	x1 x2 has positive influence
#	x3 has no influence
#	x4 has negative influence

X = np.random.randn(30, 4)
Y = []
for m in range(X.shape[0]):
	y = X[m,0] + X[m,1] + X[m,0]*X[m,1] - 4*X[m,3] - X[m,3]*X[m,3] + 0.2*np.random.randn()
	Y.append(y)

dat = wuml.wData(X_npArray=X, Y_npArray=Y, row_id_with_label=0, column_names=['A','B','C','D'])
dat.to_csv('shap_regress_example.csv', add_row_indices=False, include_column_names=True)

