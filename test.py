#!/usr/bin/env python

import wuml 
import numpy as np
from wplotlib import histograms
	



#wuml.set_numpy_print_options(precision=5)

data = wuml.wData(X_npArray=np.random.randn(100))
Pₓ = wuml.KDE(data)

X = Pₓ.generate_samples(300)
H = histograms()
H.histogram(X, num_bins=10, title='Basic Histogram', facecolor='blue', α=0.5, path=None)


#probs = Pₓ(data)
probs = Pₓ(0)
print(probs)

import pdb; pdb.set_trace()


