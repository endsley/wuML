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


X = wuml.gen_exponential(λ=2, size=3000)	# generates 2 exp(-2x)
E = wuml.exponential(X)

Xp = np.arange(0.05,8,0.05)
Q = 2*np.exp(-2*Xp)							# theoretical true distribution
P = E(Xp)									# estimated distribution via MSE


H = wplotlib.histograms(X, num_bins=20, facecolor='blue', α=0.5, path=None, normalize=True, show=False )
l = wplotlib.lines(Xp,Q, color='red', marker=',', imgText='Red: true distribution\nBlue: Estimated distribution', xTextShift=0.35, yTextShift=0.95, show=False )
wplotlib.lines(Xp,P, color='blue', title='Basic Histogram', xlabel='value', ylabel='count')


A = E.cdf(1)
wuml.jupyter_print('True λ: %.3f, Approximated λ: %.3f'%(2, E.λ))
wuml.jupyter_print('True cdf to 1: %.3f, Approximated cdf to 1: %.3f'%(0.865, A))

