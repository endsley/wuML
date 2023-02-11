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


data = wuml.wData(xpath='../../data/wine.csv', ypath='../../data/wine_label.csv', label_type='discrete')
tenFoldData = wuml.gen_10_fold_data(data=data)
tb = wuml.result_table(column_names=['Fold', 'Train Acc', 'Test Acc'])

for i, fold in enumerate(tenFoldData):
	[X_train, Y_train, X_test, Y_test] = fold
	svm = wuml.classification(X_train, y=Y_train, test_data=X_test, testY=Y_test, classifier='SVM')
	tb.add_row([i+1, svm.Train_acc , svm.Test_acc])
	wuml.write_to_current_line('run %d: Train Acc: %.3f, Test Acc:%.3f'%(i, svm.Train_acc , svm.Test_acc))


wuml.jupyter_print('\n')
TrainAcc = np.mean(tb.get_column('Train Acc').X)
TestAcc = np.mean(tb.get_column('Test Acc').X)
tb.add_row(['Avg Acc', TrainAcc , TestAcc])
wuml.jupyter_print(tb, display_all_rows=True)
