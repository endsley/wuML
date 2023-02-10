#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


from wuml.distributions import *
import wplotlib
import numpy as np


#we first generate samples from an exponential distribution
Ex = exponential(mean=1)
ε = Ex.generate_samples(size=400)


#	now that we have samples from an exponential distribution
#	we can recreate the exponential model via MLE
E = wuml.exponential(X=ε)

Xp = np.arange(0.1,5,0.05)
probs = E(Xp)

e1 = 1 - E.cdf(1)
e2 = 1 - E.cdf(2)

msg = ('P(X > 1) = %.3f\n'%e1)
msg += ('P(X > 2) =  %.3f'%(e2))


H = wplotlib.histograms(ε, num_bins=40, title='Basic Histogram', xlabel='value', ylabel='count', facecolor='blue', α=0.5, path=None, normalize=True, show=False )
l = wplotlib.lines(Xp, probs, 'Histogram and Distribution Modeled via MLE', 'Value away from 0', 'Probability Distribution', show=False)
l.add_text(Xp, probs, msg, α=0.35, β=0.95)
l.show()


