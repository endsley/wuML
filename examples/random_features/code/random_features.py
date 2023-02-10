#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
import wplotlib
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



#	On lower dimension data, it seems that RFF is good enough
X = np.random.randn(7,4)
σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
γ = 1.0/(2*σ*σ)
#
K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
#
sorf = wuml.random_feature(sigma=σ, random_feature_method='orthogonal', sample_num=12)
Kₒ = sorf.get_kernel(X)			# kernel matrix from orthogonal

wuml.jupyter_print('You can obtain the feature map directly via the function get_feature_map')
Φₒ = sorf.get_feature_map(X)			# kernel matrix from orthogonal
wuml.jupyter_print(Φₒ); print('\n')

#
rff = wuml.random_feature(sigma=σ, random_feature_method='rff')
Kᵣ = rff.get_kernel(X)			# kernel matrix from orthogonal
#
#
wuml.print_two_matrices_side_by_side(K[0:8,0:8], Kₒ[0:8,0:8], title1='Real', title2='Approx by SORF', auto_print=True)
wuml.print_two_matrices_side_by_side(K[0:8,0:8], Kᵣ[0:8,0:8], title1='Real', title2='Approx by RFF', auto_print=True)

wuml.jupyter_print("Don't worry that RFF performs better on lower dimension datasets with few samples (only 12)")
ε = mean_absolute_error(K, Kₒ)
wuml.jupyter_print('Mean absolute Error with SORF %.3f'%ε)
ε = mean_absolute_error(K, Kᵣ)
wuml.jupyter_print('Mean absolute Error with RFF %.3f\n'%ε)

ε = mean_squared_error(K, Kₒ)
wuml.jupyter_print('Mean Squared Error with SORF %.3f'%ε)
ε = mean_squared_error(K, Kᵣ)
wuml.jupyter_print('Mean Squared Error with SORF %.3f'%ε)

