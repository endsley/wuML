#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines

#ci = ['FamilyID', 'Mom_DOB', 'Baby_DOB','GA_Birth']
#data = wuml.wData('examples/data/NN_with_labels.csv', row_id_with_label=0, columns_to_ignore=ci, replace_this_entry_with_nan=991)
#mdp = wuml.missing_data_stats(data)
#data = wuml.decimate_data_with_missing_entries(data, column_threshold=0.9, row_threshold=0.9)
#data = wuml.impute(data, imputerType='iterative')
#data.to_csv('examples/data/NN_with_labels_imputed.csv', include_column_names=True)
#import pdb; pdb.set_trace()

data = wuml.wData('examples/data/NN_with_labels_imputed.csv', row_id_with_label=0, replace_this_entry_with_nan=991)
X = data[:,5:11]
Ys = data[:,0:5]
depList = wuml.HSIC_of_feature_groups_vs_label_list(X, Ys)
print(depList)

