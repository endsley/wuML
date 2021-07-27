#!/usr/bin/env python

import wuml 

data = wuml.load_csv('../data/chem.exposures.csv', row_id_with_label=0)
dfSmall = wuml.decimate_data_with_missing_entries(data, column_threshold=0.7, row_threshold=0.7,newDataFramePath='')

