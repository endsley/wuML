#!/usr/bin/env python

import itertools
from itertools import permutations
 

def zip_with_joint_permutation(*arg):
	return list(itertools.product(*arg))





if __name__ == "__main__":
	a = [1,2]
	b = ['a','b']
	e = ['s','t']

	O = zip_with_joint_permutation(a,b,e)
	for i, j, k in O: print(i,j, k) 

	import pdb; pdb.set_trace()
