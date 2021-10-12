
import numpy as np
import random
import string


def gen_squared_symmetric_matrix(length, distribution='Gaussian'):
	if distribution == 'Gaussian':
		X = np.random.randn(length, length)
	elif distribution == 'Uniform':
		X = np.random.rand(length, length)

	X = X.dot(X.T)
	return X

def gen_random_string(str_len=5, letters=string.ascii_letters):
	#letters = string.ascii_lowercase
	#letters = string.ascii_letters
	#letters = string.ascii_uppercase
	#letters = string.digits
	#letters = string.punctuation
	return ''.join(random.choice(letters) for i in range(str_len)) 

def gen_exponential(λ=1, size=1):
	# λ exp(-λx) , μ=1/λ
	return np.random.exponential(scale=1/λ, size=size)


#category_range	: if 5, it will select from 0,1,2,3,4
#n				: number of samples returned	
#probabilities 	: the probability of each instance
#example		: np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
def gen_categorical(category_range, n, probabilities):
	return np.random.choice(category_range, n, p=probabilities)
	
