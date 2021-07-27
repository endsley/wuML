#!/usr/bin/env python
import wuml


data = wuml.load_csv('../data/chem.exposures.csv', row_id_with_label=0)
wuml.missing_data_stats(data)
print('You can view the results in ./results/DaStats folder')

