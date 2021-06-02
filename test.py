#!/usr/bin/env python

import wpreprocess as wPr
import numpy as np
import pandas as pd
import sys



df = pd.read_csv ('./data/chem.exposures.csv', header='infer',index_col=0)
#df = pd.read_csv ('./data/wine.csv', header=None)

wPr.missing_data_stats(df)

#print(np.mean(X,axis=0))


#foo = [wPr.center_and_scale]
#X = wPr.read_csv('./data/chem.exposures.csv', preprocess_list=foo)
#X = wPr.read_csv('./data/wine.csv', preprocess_list=foo)


