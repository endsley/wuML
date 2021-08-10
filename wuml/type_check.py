
import numpy as np

def ensure_numpy(data):
	if type(data).__name__ == 'ndarray': 
		if len(data.shape) == 1:
			X = np.atleast_2d(data).T
		else:
			X = data
	elif type(data).__name__ == 'wData': 
		X = data.df.values
	elif type(data).__name__ == 'DataFrame': 
		X = data.values
	elif np.isscalar(data):
		X = np.array([[data]])

	return X


#def correct_variable_type(variable, intended_type):
#	if type(variable).__name__ == intended_type: return True
#	return False
