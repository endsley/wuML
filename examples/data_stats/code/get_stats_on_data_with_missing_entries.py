#!/usr/bin/env python
import wuml

data = wuml.wData('../../data/missin_example.csv', first_row_is_label=True)
wuml.missing_data_stats(data)

