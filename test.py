#!/usr/bin/env python

import wuml
import sklearn
import shap
import wplotlib
import numpy as np


data = wuml.wData(xpath='examples/data/Chem_decimated_imputed.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])
data = wuml.center_and_scale(data)

bNet= wuml.pickle_load('./tmp/4496/best_network_updated.pk')

##	update the network to new code
#bN = wuml.basicNetwork(None, None, simplify_network_for_storage=bNet, flatten_network_output_during_usage=True)
#bN.train_result = bNet.train_result
#bN.test_result = bNet.test_result
#wuml.pickle_dump(bN, './tmp/4496/best_network_updated.pk')

#Y2 = bNet(data)
#import pdb; pdb.set_trace()


## use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(bNet, data.df, link="logit")
shap_values = explainer.shap_values(data.df, nsamples='auto')
import pdb; pdb.set_trace()


# plot the SHAP values for the Setosa output of the first instance
#shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")
