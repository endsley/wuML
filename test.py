#!/usr/bin/env python
import wuml



data = wuml.wData(xpath='examples/data/shap_classifier_example_uniform.csv',  label_type='discrete', 
					label_column_name='label', row_id_with_label=0)

EXP = wuml.explainer(data, loss='CE', explainer_algorithm='shap', link='logit', max_epoch=200, 
					networkStructure=[(100,'relu'),(100,'relu'),(2,'none')]	)
Ŷ = EXP.model(data)
shapV = EXP(data.X)

