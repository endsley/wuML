#!/usr/bin/env python

import numpy as np
import wuml
import torch

'''
	Notes : https://github.com/endsley/math_notebook/blob/master/neural_network/Gumbel_Softmax.pdf

	The gumbel softmax generates samples based on a categorial distribution
	that can be incorporated into a neural network. 
	Given a categorical distribution of {pᑊ pᒾ pᶾ ...}, it will generate one-hot vectors given these probabilities.

	Implement: Make sure that the rows add up to 1
'''


X = np.array([[0.4405, 0.4045, 0.0754, 0.0796],			# The rows should add up to 1
				[0.2287, 0.2234, 0.2676, 0.2802],
				[0.2518, 0.2524, 0.1696, 0.3262],
				[0.2495, 0.1744, 0.2126, 0.3635],
				[0.1979, 0.31  , 0.2165, 0.2755],
				[0.2003, 0.2329, 0.2982, 0.2686]])

# numpy implementation
C = np.zeros((6,4))
for n in range(10000):
	C += np.round(wuml.gumbel(X))

C = C/10000

# torch implementation
Ct = torch.zeros(6,4)
for n in range(10000):
	Ct += torch.round(wuml.gumbel(torch.from_numpy(X)))

Ct = Ct.numpy()/10000

print('True Probability')	
print(X,'\n')

print('Gumbel generated Probability from sampling')
print(C,'\n')

print('Gumbel generated Probability from sampling via pytorch')
print(Ct,'\n')

