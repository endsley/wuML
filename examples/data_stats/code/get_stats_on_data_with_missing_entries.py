#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml

data = wuml.wData('../../data/missin_example.csv', first_row_is_label=True)
wuml.missing_data_stats(data)

