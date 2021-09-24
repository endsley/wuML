#!/usr/bin/env python

import wuml
import wplotlib
import numpy as np


data = wuml.wData(xpath='../../data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
data = wuml.center_and_scale(data)

bN= wuml.pickle_load('./network_saved/best_network.pk')
Ŷ = np.squeeze(bN(data, output_type='ndarray'))
ε = np.absolute(data.Y - Ŷ)
E = wuml.model_as_exponential(ε)

Xp = np.arange(0.1,5,0.05)
probs = E(Xp)

e1 = 1 - E.cdf(1)
e2 = 1 - E.cdf(2)

msg = ('P(X > 1) = %.3f\n'%e1)
msg += ('P(X > 2) =  %.3f'%(e2))


H = wplotlib.histograms()
l = wplotlib.lines()
H.histogram(ε, num_bins=40, title='Basic Histogram', xlabel='value', ylabel='count', facecolor='blue', α=0.5, path=None, normalize=True, showImg=False )
l.add_text(Xp, probs, msg, α=0.35, β=0.95)
l.plot_line(Xp, probs, 'Histogram and Distribution Modeled via MLE', 'Gestational Age Error', 'Probability Distribution')


