#!/usr/bin/env python

import wuml 
from wplotlib import histograms

data = wuml.wData('examples/data/NN_dat_only.csv', first_row_is_label=True, )
data_U = wuml.use_cdf_to_map_data_between_0_and_1(data)

sample_weights = wuml.get_likelihood_weight(data, 'Weight Raw Dat')
sample_weights_U = wuml.get_likelihood_weight(data_U, 'Weight Uniform Dat')
sample_weights.append_columns(sample_weights_U)
print(sample_weights)

import pdb; pdb.set_trace()

data.df['rarity_score'] = sample_weights.X
print(data)

H = histograms()
H.histogram(sample_weights.X, num_bins=10, title='Rarity Score Histogram', facecolor='green', α=0.5, normalize=False)
H.histogram(sample_weights.X, num_bins=10, title='Rarity Score Histogram', facecolor='green', α=0.5, normalize=True)
