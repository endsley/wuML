#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
from wuml import jupyter_print



data = wuml.make_classification_data( n_samples=300, n_features=6, n_informative=3, n_classes=3)

tenF_LDA_LR = wuml.ten_folder_classifier(data, classifier='SVM')
tenF_LDA_LR.show_results()

jupyter_print('\nIf we use the entire 10 models and obtain labels by voting, we get')
ӯ = tenF_LDA_LR(data)
summary = wuml.summarize_classification_result(data.Y, ӯ)

