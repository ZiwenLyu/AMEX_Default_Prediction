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

As the graphs show, for some variables (such as P_2, D_55) the default and paid distributions vary a lot.
