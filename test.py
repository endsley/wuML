#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('examples/data/Chem_decimated_imputed.csv', label_column_name='finalga_best', 
					label_type='continuous', row_id_with_label=0)

H1 = histograms()
H1.histogram(data.Y, num_bins=20, title='Gestational Age Histogram', facecolor='blue', α=0.5, path=None)


sample_weights = wuml.get_likelihood_weight(data.Y)
H = histograms()
H.histogram(sample_weights.X, num_bins=20, title='Sample Weight Histogram', facecolor='blue', α=0.5, path=None, ylogScale=True )

print(wuml.output_two_columns_side_by_side(data.Y, sample_weights.X, labels=['Y','W'], rounding=3))
sample_weights.to_csv('../data/Chem_sample_weights.csv', include_column_names=False)







##!/usr/bin/env python
#import wuml
#
#data = wuml.wData(xpath='examples/data/shap_classifier_example_uniform.csv',  label_type='discrete', 
#					label_column_name='label', row_id_with_label=0)
#
#EXP = wuml.explainer(data, loss='CE', explainer_algorithm='shap', link='logit', max_epoch=20, 
#					networkStructure=[(100,'relu'),(100,'relu'),(2,'none')]	)
#
#Ŷ = EXP.model(data, out_structural='1d_labels')
#SC = wuml.summarize_classification_result(data.Y, Ŷ)
#res = SC.true_vs_predict(sort_based_on_label=True, print_result=False)
#print(res)
#
#shapV = EXP(data.X)
#print(shapV)

