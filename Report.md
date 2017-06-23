

#Introduction:

Enron Corporation was an American energy, commodities, and services company based in Houston, Texas. It was founded in 1985 as the result of a merger between Houston Natural Gas and InterNorth, both relatively small regional companies in the U.S. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years. 

### About the dataset: 
Dataset contains 146 persons in total, however, 18 persons of them are identified as POI while the rest of them are non POI. There are 19 features for each person, person's salary and emails receiver for instance. This dataset contains missing values, for example there is no restricted stock deferred features for 128 persons and no director fees feature for 129 persons.  

## Feature Engineering:

Two features are created recently:
1- **to_poi_message_ratio** :
Measuring how many e-mails are frequently sent by a person to POIs.
2- **from_poi_message_ratio**:
Measuring how many e-mails are frequently received by a person from POIs.

## Feature Selection:

Using cross validation this feature is performed as a way to find the best number of features that should be remained. Also, f_classif function is performed to get the best k features, where f_classif function is a sorting function for SelectKBest in scikit-learn.

In addition, Random Forest and Adaboost classifiers are created and tested by using same methodology (cross validation). 

[![Capture1.png](https://s15.postimg.org/um8rr867f/Capture1.png)](https://postimg.org/image/lr7xgphev/)

As a result, Random Forest has a significant high precision score than  Adaboost. On the other hand, Both classifiers have a high accuracy score with different dumber of features and the opposite is true, Both classifiers have a low recall score, the highest three scores are obtained when using the best 3 features from SelectKBest which there are bonus, total_stock_value and exercised_stock_options. The importance score of them are 0.33,0.36 and 0.30 respectively, that is resulted from Random forest The features scores are 21.06, 24.47 and 25.10 for bonus, total_stock_value and exercised_stock_options respectively by applying SelectKBest. No feature scaling is used as it is not required in my case.

## Tune an Algorithm:

GridSearchCV is applied to obtain a higher recall score by finding the best hyper parameters for Random Forest and Adaboost classifiers. the main idea of doing that is to find the correct values and performing data better than know later on. 

In terms of Random Forest, 3 valued is received:
max_features and min_sample_splits = 2 and n_estimators = 100
which the higher values of max_features and n_estimator will be the higher complexity of the classifier and the higher value of the remain one will be the smaller complexity of the classifier.

In terms of adaboost, n_estimators = 200 and learning_rate = 0.6.

##Validate and Evaluate:

Cross validation method is applied in the pervious 2 parts to measure how effective the algorithm would work om sample cases. StratifiedShuffleSplit method is used in order to applying cross validation. Summary of performance measurements:

[![Capture.png](https://s4.postimg.org/5zixhc7vh/Capture.png)](https://postimg.org/image/mnafju2mx/)

Based on the pervious table scores, Random Forest model is the final classifier and it produced the highest scores for precision and recall as well. 

When preparing and running python scripts I thought sorting this data and simplifying the complexity is impossible with no features, also, the final features selected using cross validation did not consist of these types features. Then, I confidently agree that the classifier should by applied in reality. 

Written with [StackEdit](https://stackedit.io/).
