import numpy as np
from wuml.type_check import *


def norm(data, norm='l2'):
	# If norm='l1':
	#	If data is a vector: compute l1 norm
	#	If data is a matrix: compute l1 norm for each row
	# If norm='l2':
	#	If data is a vector: compute l2 norm
	#	If data is a matrix: compute l2 norm for each row
	# If norm='fro':
	#	If data is a vector: compute frobenius norm
	#	If data is a matrix: compute frobenius norm 
	# If norm='nuc':
	#	If data is a vector: compute nuclear norm
	#	If data is a matrix: compute nuclear norm 

	X = np.squeeze(ensure_numpy(data))

	if norm=='l1':
		if len(X.shape) == 1:
			return np.linalg.norm(X, ord=1)
		else:
			return np.linalg.norm(X, ord=1, axis=1)
	elif norm=='l2':
		if len(X.shape) == 1:
			return np.linalg.norm(X, ord=2)
		else:
			return np.linalg.norm(X, ord=2, axis=1)
	elif norm=='fro':
		return np.linalg.norm(X, ord='fro')
	elif norm=='nuc' or norm=='nuclear':
		return np.linalg.norm(X, ord='nuc')

