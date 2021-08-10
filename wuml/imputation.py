
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np


def impute(data, imputerType='iterative', ignore_first_index_column=False):

	if type(data).__name__ == 'ndarray': 
		X = data
	elif type(data).__name__ == 'wData': 
		X = data.df.values
	elif type(data).__name__ == 'DataFrame': 
		X = data.values
	else:
		raise

	if ignore_first_index_column:
		c1 = np.atleast_2d(X[:,0]).T
		X = X[:,1:]

	imp_mean = IterativeImputer(random_state=0)
	imp_mean.fit(X)
	Xi = imp_mean.transform(X)

	if ignore_first_index_column:
		Xi = np.hstack((c1,Xi))

	if type(data).__name__ == 'ndarray': 
		return Xi
	elif type(data).__name__ == 'wData': 
		data.df[:] = Xi
		return data
	elif type(data).__name__ == 'DataFrame': 
		data[:] = Xi
		return data

