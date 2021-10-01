#!/usr/bin/env python
import wuml

data = wuml.wData(xpath='../../data/shap_classifier_example_uniform.csv',  label_type='discrete', 
					label_column_name='label', row_id_with_label=0)

EXP = wuml.explainer(data, loss='CE', explainer_algorithm='shap', link='logit', max_epoch=200, 
					networkStructure=[(100,'relu'),(100,'relu'),(2,'none')]	)
Ŷ = EXP.model(data)
SC = summarize_classification_result(data.Y. Ŷ)
res = SC.true_vs_predict(sort_based_on_label=True, print_result=False)
print(res)

shapV = EXP(data.X)
print(shapV)

