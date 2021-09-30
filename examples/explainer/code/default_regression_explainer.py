#!/usr/bin/env python
import wuml

data = wuml.wData(xpath='../../data/shap_regress_example_mix_distributions.csv', batch_size=20, 
					label_type='continuous', label_column_name='label', row_id_with_label=0)

Udata = wuml.use_cdf_to_map_data_between_0_and_1(data, output_type_name='wData')

EXP = wuml.explainer(Udata, explainer_algorithm='XGBRegressor')
shap_values = EXP(Udata, output_all_results=False)
print(shap_values)


