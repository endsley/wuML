
import numpy as np
import random
import string

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
