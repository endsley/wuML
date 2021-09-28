# wuml
## Chieh's quick ML library
## Pip Installation
```sh
pip install wuml
```


## Examples Usages

### Data Preprocessing
[Load data + Decimate rows and column with too much missing + auto-imputation](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/deal_with_missing_data.ipynb) \
[Split data into Training Test + Look at the histogram of their labels](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/train_test_histogram.ipynb) \
[Split data into Training Test + Run a basic Neural Network](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/train_test_on_basic_network.ipynb)\
\
[Use Reverse CDF to map data into between 0 and 1](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/use_reverse_cdf_to_map_data_to_between_0_and_1.ipynb) 


### Build Neural Networks via Pytorch
[Simple Regression](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/basicRegression.ipynb) \
[Weighted Regression](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/weighted_regression.ipynb) \
\
[Simple Classification](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/basicClassification.ipynb) 


### Distribution Modeling
[KDE Example](https://github.com/endsley/wuML/blob/main/examples/distribution_modeling/ipynb/basicKDE_estimate.ipynb) \
[Maximum Likelihood on Exponential Distribution Example](https://github.com/endsley/wuML/blob/main/examples/distribution_modeling/ipynb/basic_exponential_MLE_modeling.ipynb)


### Explaining Models
[Run Shap on data between 0 and 1](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/regression_pytorch_explainer_uniform.ipynb) 
	(Examples of a good result since data is between 0 and 1) \
[Run Shap on data of Normal Distribution](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/regression_pytorch_explainer_gaussian.ipynb) 
	(Examples of a bad result since data can be negative or positive)


