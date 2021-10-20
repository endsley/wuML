#!/usr/bin/env python

import wuml 

ci = ['FamilyID', 'Mom_DOB', 'Baby_DOB','GA_Birth']
data = wuml.wData('examples/data/NN_with_labels.csv', first_row_is_label=True, columns_to_ignore=ci, replace_this_entry_with_nan=991)
suckPattern = data.df[['NNS_Duration' ,'NNS_Frequency' ,'NNS_Amplitude' ,'NNS_Bursts' ,'NNS_Cycles' ,'NNS_Cycles_Per_Bursts']]

sample_weights = wuml.get_likelihood_weight(suckPattern, weight_names='Raw Rarity Weights')
compare_columns = data.get_columns(['sex' ,'m_education' ,'m_age' ,'Bweight_g' ,'BabyAssessAge_B1W'])
compare_columns.append_columns(sample_weights)
corrMatrix = wuml.feature_wise_correlation(compare_columns)
print(corrMatrix)


# What if we preprocess the data by mapping it into [0,1] Domain?
suckPattern_U = wuml.use_cdf_to_map_data_between_0_and_1(suckPattern)
sample_weights_U = wuml.get_likelihood_weight(suckPattern_U, weight_names='Uniform Rarity Weights')

otherDats = data.df[['sex' ,'m_education' ,'m_age' ,'Bweight_g' ,'BabyAssessAge_B1W']]
otherDats_U = wuml.use_cdf_to_map_data_between_0_and_1(otherDats)
otherDats_U.append_columns(sample_weights_U) 
corrMatrix = wuml.feature_wise_correlation(otherDats_U)
print(corrMatrix)
#corrMatrix.df.style
