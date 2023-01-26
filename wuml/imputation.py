
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from wuml.type_check import *
import numpy as np


def impute(data, imputerType='iterative', ignore_first_index_column=False):
	X = ensure_numpy(data)

	if ignore_first_index_column:
		c1 = np.atleast_2d(X[:,0]).T
		X = X[:,1:]

	imp_mean = IterativeImputer(random_state=0)
	imp_mean.fit(X)
	Xi = imp_mean.transform(X)

	if ignore_first_index_column:
		Xi = np.hstack((c1,Xi))

	Xout = ensure_data_type(Xi, type_name=type(data).__name__)	# returns the same data type as input type
	if wtype(Xout) == 'wData' and wtype(data) == 'wData':
		Xout.rename_columns(data.get_column_names_as_a_list())
		Xout.label_column_name = data.label_column_name

		Xout.Y = data.Y

	return Xout
	

