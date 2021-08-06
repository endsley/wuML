#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines
	

data = wuml.wData(X_npArray=np.random.randn(1000))
Pₓ = wuml.KDE(data)

X = np.arange(-3,3,0.05)
realProb = scipy.stats.norm(0, 1).pdf(X)
estimatedProb = Pₓ(X)
newX = Pₓ.generate_samples(300)

textstr = 'Blue: True Density\nRed: KDE estimated density\nGreen: Histogram sampled from KDE'
lp = lines()
H = histograms()
lp.add_plot(X,realProb, color='blue', marker=',')
lp.add_plot(X,estimatedProb, color='red', marker=',')
lp.add_text(X,estimatedProb, textstr, β=0.8)
H.histogram(newX, num_bins=10, title='Using KDE to Estimate Distributions', facecolor='green', α=0.5, showImg=False, normalize=True)

H.show()


