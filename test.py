#!/usr/bin/env python

import wuml 
from wplotlib import scatter

wuml.set_terminal_print_options(precision=3)
data = wuml.make_moons(n_samples=1500)


Pᵳ = wuml.flow(data, max_epochs=2000, num_flows=10, network_width=1024)
probᵳ = Pᵳ(data)
samplesᵳ = Pᵳ.generate_samples(2000)
samplesᵳ.plot_2_columns_as_scatter(0, 1)


Pᴋ = wuml.KDE(data)
probᴋ = Pᴋ(data)
samplesᴋ = Pᴋ.generate_samples(2000)
samplesᴋ.plot_2_columns_as_scatter(0, 1)

import pdb; pdb.set_trace()
S = scatter(data.X[:,0], data.X[:,1], title='Original data', subplot=131, figsize=(10,7))
scatter(samplesᴋ.X[:,0], samplesᴋ.X[:,1], title='KDE Generated', subplot=132)
scatter(samplesᵳ.X[:,0], samplesᵳ.X[:,1], title='Flow Generated', subplot=133)
S.show()
