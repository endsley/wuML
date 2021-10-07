#!/usr/bin/env python
import wuml

data = wuml.wData('../../data/missin_example.csv', row_id_with_label=0)
wuml.missing_data_stats(data)

