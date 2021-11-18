# wuml
## Chieh's quick ML library
## Pip Installation
```sh
pip install wuml
```


## Examples Usages

### Data Statistics
[Learn about missing data stats](https://github.com/endsley/wuML/blob/main/examples/data_stats/ipynb/get_stats_on_data_with_missing_entries.ipynb) \
[Feature wise Correlation Matrices](https://github.com/endsley/wuML/blob/main/examples/data_stats/ipynb/feature_wise_correlation.ipynb) \
[Feature wise HSIC Matrices](https://github.com/endsley/wuML/blob/main/examples/data_stats/ipynb/feature_wise_HSIC.ipynb) 

### Dependency Measures
[Comparing HSIC to Correlation](https://github.com/endsley/wuML/blob/main/examples/dependencies/ipynb/comparing_HSIC_to_correlation.ipynb)


### Data Preprocessing
[Obtain sample weight based on label likelihood](https://github.com/endsley/wuML/blob/main/examples/data_stats/ipynb/weight_sample_by_rarity.ipynb) \
[Show histogram of a feature](https://github.com/endsley/wuML/blob/main/examples/data_stats/ipynb/show_feature_histogram.ipynb) \
[Split data into Training Test + Look at the histogram of their labels](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/train_test_histogram.ipynb) \
[Split data into Training Test + Run a basic Neural Network](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/train_test_on_basic_network.ipynb)\
\
[Map data into between 0 and 1](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/map_data_to_between_0_and_1.ipynb) 
\
[Load data + Decimate rows and column with too much missing + auto-imputation](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/deal_with_missing_data.ipynb) \
[Load data + center/scaled or between 0 and 1](https://github.com/endsley/wuML/blob/main/examples/preprocess/ipynb/load_data_with_preprocess.ipynb) 


### Build Neural Networks via Pytorch
[Simple Regression](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/basicRegression.ipynb) \
[Weighted Regression](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/weighted_regression.ipynb) \
\
[Simple Classification](https://github.com/endsley/wuML/blob/main/examples/NeuralNet/ipynb/basicClassification.ipynb) 


### Distribution Modeling
[KDE Example](https://github.com/endsley/wuML/blob/main/examples/distribution_modeling/ipynb/basicKDE_estimate.ipynb) \
[Maximum Likelihood on Exponential Distribution Example](https://github.com/endsley/wuML/blob/main/examples/distribution_modeling/ipynb/basic_exponential_MLE_modeling.ipynb)\
[Using Flow-based Deep Generative Model](https://github.com/endsley/wuML/blob/main/examples/distribution_modeling/ipynb/flow_example.ipynb)


### Feature Selection
[Unsupervised Filtering via HSIC](https://github.com/endsley/wuML/blob/main/examples/feature_selection/ipynb/selection_by_hsic.ipynb) 

### Explaining Models
[Run Default Shap explainer (Regression)](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/default_regression_explainer.ipynb) \
[Run Shap on data between 0 and 1 (Regression)](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/regression_pytorch_explainer_uniform.ipynb) 
	(Examples of a good result since data is between 0 and 1) \
[Run Shap on data of Normal Distribution (Regression)](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/regression_pytorch_explainer_gaussian.ipynb) 
	(Examples of a bad result since data can be negative or positive)\
[Run Shap on data of mixed Distributions that are mapped into distribution between 0 and 1(Regression)](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/comparing_shap_results_on_different_data_distributions.ipynb) \
	(Examples of dealing with data of mixed distribution all map into uniform)\
\
[Run Shap Classification](https://github.com/endsley/wuML/blob/main/examples/explainer/ipynb/classification_pytorch_explainer_uniform.ipynb) 


### Regression / Classification
[Run Several Basic Regressors](https://github.com/endsley/wuML/blob/main/examples/regression/ipynb/run_regression.ipynb) \
[Run Several Basic Classifiers](https://github.com/endsley/wuML/blob/main/examples/classification/ipynb/classify.ipynb) 

### Dimension Reduction
[Run Several Dimension Reduction Examples](https://github.com/endsley/wuML/blob/main/examples/dimension_reduction/ipynb/DR_examples.ipynb) 

### Clustering
[Run Several Clustering Examples](https://github.com/endsley/wuML/blob/main/examples/clustering/ipynb/cluster.ipynb) 

### Math Operations
[EigenDecomposition](https://github.com/endsley/wuML/blob/main/examples/operations/ipynb/eigDecomp.ipynb) \
[Integrate a univariate function](https://github.com/endsley/wuML/blob/main/examples/operations/ipynb/integrate_function.ipynb) 

