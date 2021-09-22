#!/usr/bin/env python

import wuml
import numpy as np


data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
data = wuml.center_and_scale(data)

bN= wuml.pickle_load('./examples/run_k_fold/best_network.pk')
Ŷ = np.squeeze(bN(data, output_type='ndarray'))
Y = data.Y

type1_error = np.sum((Y > 37).astype(int)*(Ŷ < 37.5).astype(int))/Y.shape[0]
type2_error = np.sum((Y < 37).astype(int)*(Ŷ > 37.5).astype(int))/Y.shape[0]
print('Type 1: %.3f, Type 2: %.3f'%(type1_error, type2_error))
#Type 1: 0.053, Type 2: 0.006

import pdb; pdb.set_trace()


