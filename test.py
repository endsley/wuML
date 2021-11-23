#!/usr/bin/env python
import wuml
import numpy as np
from scipy.stats import multivariate_normal

var = multivariate_normal(mean=[0,1], cov=[[1,0.5],[0.5,1]])
dat = var.rvs(size=5000, random_state=None)

P1 = wuml.multivariate_gaussian(dat)
P2 = wuml.flow(dat)

p1 = P1(dat)
p2 = P2(dat)

import pdb; pdb.set_trace()

