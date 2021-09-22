#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib
from torch.autograd import Variable

wuml.set_terminal_print_options(precision=2)
data = wuml.wData(xpath='../data/wine.csv', ypath='../data/wine_label.csv', label_type='discrete', batch_size=20)
data = wuml.center_and_scale(data)


#You must have the first 2 variables included (fold_id, one_fold_data_list)
def run_classifier(fold_id, one_fold_data_list):
	[X_train, X_test] = one_fold_data_list[0]
	[Y_train, Y_test] = one_fold_data_list[1]

	svm = wuml.svm(X_train, Y_train)	# train 
	Ŷ = svm(X_test)						# predict
	acc = wuml.accuracy(Y_test, Ŷ)
	print(fold_id, ' : ', acc)

# dictionary {} is used as additional arguments into function beyond (fold_id, one_fold_data_list)
wuml.run_K_fold_on_function(10, [data.X, data.Y], run_classifier, {}) 
