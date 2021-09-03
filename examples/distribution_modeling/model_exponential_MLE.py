#!/usr/bin/env python

import wuml
import wplotlib
import numpy as np


X = wuml.gen_exponential(λ=2, size=3000)	# generates 2 exp(-2x)
E = wuml.model_as_exponential(X)

Xp = np.arange(0.05,8,0.05)
Q = 2*np.exp(-2*Xp)							# theoretical true distribution
P = E(Xp)									# estimated distribution via MSE

H = wplotlib.histograms()
l = wplotlib.lines()
H.histogram(X, num_bins=20, title='Basic Histogram', xlabel='value', ylabel='count', facecolor='blue', α=0.5, path=None, normalize=True, showImg=False )
l.add_plot(Xp,Q, color='red', marker=',')
l.add_text(Xp, Q, 'Red: true distribution\nBlue: Estimated distribution', α=0.35, β=0.95)
l.plot_line(Xp, P, 'Histogram and Modeled Distribution', 'X', 'Probability')

A = E.cdf(1)
print('True cdf to 1: %.3f, Approximated cdf to 1: %.3f'%(0.865, A))

