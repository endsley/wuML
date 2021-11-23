#!/usr/bin/env python
import wuml
import numpy as np
from scipy.stats import multivariate_normal

var = multivariate_normal(mean=[0,1], cov=[[1,0.5],[0.5,1]])
dat = var.rvs(size=2000, random_state=None)

P1 = wuml.multivariate_gaussian(dat)
P2 = wuml.flow(dat, max_epochs=200)

p1 = P1(dat, return_data_type='ndarray')
p2 = P2(dat, return_data_type='ndarray')

print(p1[0:10])
print(p2[0:10])


