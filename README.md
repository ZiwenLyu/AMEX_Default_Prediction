# Credit_Default_Prediction

## Overview
Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. This is a kaggle competition held by American Express, a globally integrated payments companyone and one of the card issuers in the world. 
This project will leverage an industrial scale data set to build a machine learning model that challenges the current model in production. Training, validation, and testing datasets include time-series behavioral data and anonymized customer profile information. 

## Data
The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.

The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:

D_* = Delinquency variables

S_* = Spend variables

P_* = Payment variables

B_* = Balance variables

R_* = Risk variables

with the following features being categorical:
['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

American Express provides three datasets to work on: `train_data.csv`, `test_data.csv`, `train_labels.csv`. In total, therea are 189 features, 5,531,451 statements in the train dataset, and 11,363,762 statements in the test dataset. Since the original dataset is too big (over 50 GIB), use the feather-format data from @Munum https://www.kaggle.com/datasets/munumbutt/amexfeather

## Explotary Data Analysis (EDA)
### The date range for two datasets
![The date range for two datasets](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/date%20range%20for%20two%20datasets.png)
As the graph illustrates, the train dataset covers the date from 2017-04 to 2018-04, and the test dataset ranges from 2018-04 to 2019-10. Two datasets don't have period overlapped.

### The number of statements per customer
![The number of statements per customer](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/Count%20of%20statements%20per%20customer.png)
We could see that over 80% of customers have 13 statements in both datasets. We will need to aggregate the data by each customer to predict their default probabilities.

### The default status distribution in categorical features
![The default status distribution in categorical features](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20categorical%20features.png)

### The defaul status distribution in numeric features
As the graphs show, for some variables (such as P_2, D_55) the default and paid distributions vary a lot.
![The defaul status distribution in numeric features](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features.png)
![The defaul status distribution in numeric features2](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features2.png)
![The defaul status distribution in numeric features3](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features3.png)
![The defaul status distribution in numeric features4](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features4.png)
![The defaul status distribution in numeric features5](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features5.png)
![The defaul status distribution in numeric features6](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features6.png)
![The defaul status distribution in numeric features7](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features7.png)
![The defaul status distribution in numeric features8](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features8.png)
![The defaul status distribution in numeric features9](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features9.png)
![The defaul status distribution in numeric features10](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features10.png)
![The defaul status distribution in numeric features11](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/default%20status%20distribution%20in%20num%20features11.png)

## Study Missing Values
### Null distribution per row
Both train and test datasets contain variables with high portions of missing values. Let's check the distribution of nulls and see if values are missed randomly or highly-correlated with the default status.
![The defaul status distribution in numeric features11](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/number%20of%20null%20values%20by%20default%20status%20(train).png)
In the train dataset: We can learn from the result that the default curve tends to skew left, compared to the non-default curve, which means rows with less null values are more likely to have defaults. And we also can see that rows with 15-22 null values contain more targets "default." Adding "number of null values per row" as a feature to our dataset will help the model to better predict. It seems null values are informative to the classifier.
### Correlations of default status and columns with null values
![Correlations of default status and columns with null values](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/Correlation%20of%20target%20and%20null%20columns.png)
From the chart we can see that, the correlation of the target and columns which contain null values ranges from -0.61 to 0.55. Variable D_87 seems not to be correlated with the target, so we can drop D_87. Though other features have a huge amount of nulls, they matter to the target status to some extent and we need them to predict default status.

## Data Preprocess
Since we need to predict default probability of each customer and we have their multiple statements, statements are aggregated into one piece per customer. Numeric variables of several statements will be aggregated into one by means, standard deviation, min, max, and last. Categorical variables will be aggregated by their counts, last number, countss of non-null unique values.
Though each categorical feature has less than 10 unique values, the train dataset already has more than 100 columns and we adopt labelencoder rather than onehotencoding.

## Machine Learning: LightGBM Classifier
Considered the huge amount of data the model will process, the proportion of missing values we need to preserve, as well as some of the outliers seen in the distributions, here I will choose LightGBM.

The evaluation metric, `M`, for this competition is the mean of two measures of rank ordering: Normalized Gini Coefficient, `G`, and default rate captured at 4%, `D`.

`M = 0.5⋅(G+D)`

The default rate captured at 4% is the percentage of the positive labels (defaults) captured within the highest-ranked 4% of the predictions, and represents a Sensitivity/Recall statistic.
For both of the sub-metrics `G` and `D`, the negative labels are given a weight of 20 to adjust for downsampling.

This metric has a maximum value of 1.0.

At first, I set a range of parameters and applied `optuna` to automatically choose parameters and run trials for the model. The model scored 0.7889. Then based on the suggested parameters, I manually adjust parameters and train the model. The validation scored 0.7922 by AMEX metric, reached an accuracy of 90.34%.

![validation_score](https://github.com/ZiwenLyu/AMEX_Default_Prediction/blob/main/graphs/validation_score.png)
