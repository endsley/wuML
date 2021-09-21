#!/usr/bin/env python

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import wuml
import shap
import time

#X,y = shap.datasets.diabetes()
#X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
data = wuml.center_and_scale(data)
bNet= wuml.pickle_load('./tmp/4496/best_network_updated.pk')



# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_train_summary = shap.kmeans(data.X, 10)

#def print_accuracy(f):
#	print("Root mean squared test error = {0}".format(np.sqrt(np.mean((f(X_test) - y_test)**2))))
#	time.sleep(0.5) # to let the print get out before any progress bars

shap.initjs()
lin_regr = linear_model.LinearRegression()
lin_regr.fit(data.X, data.Y)

#print_accuracy(lin_regr.predict)

ex = shap.KernelExplainer(lin_regr.predict, X_train_summary)
shap_values = ex.shap_values(np.atleast_2d(data.X[0,:]))
shap.force_plot(ex.expected_value, shap_values, data.X[0,:])

shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

