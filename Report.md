 

***Question 1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?***

#Introduction:

The goal of this project is to use financial and email data from Enron Corporation in order to build z classifier that can distinguish persons of interest POIs who possibly involved in the fraud and corruption. it was an American energy, commodities, and services company based in Houston, Texas. It was founded in 1985 as the result of a merger between Houston Natural Gas and InterNorth, both relatively small regional companies in the U.S. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years. ). The corpus is widely used for various machine learning problem and although it has been labeled already, the value is the potential application for similar cases in other companies or spam filtering application. 

### About the dataset: 
Dataset contains 146 persons in total, however, 18 persons of them are identified as POI while the rest of them are non POI. There are 19 features for each person, person's salary and emails receiver for instance. This dataset contains missing values, for example there is no restricted_stock_deferred features for 128 persons and no director_fees feature for 129 persons.  

[![Capture111.png](https://s9.postimg.org/j5zbnwqb3/Capture111.png)](https://postimg.org/image/ou5mesunf/)

TOTAL is the outlier name as a summary of all persons. it can be verified by calculating the sum of all persons salaries then the result is exactly as same as the salary for TOTAL. TOTAL in this case is outlier and will affect the performance of the classifier, thus, TOTAL is removed. Eventually, 145 persons ( records ) is remaining after removing TOTAL from the dataset.

---
***Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.***

scikit-learn SelectKBest is used to select best 10 influential features and used those features for all the upcoming algorithm. As a result, 9 out of 10 features related to financial data and only 1 features called shared_receipt_with_poi (messages from/to the POI divided by to/from messages from the person) were attempted to engineere by us. Main purpose of composing ratio of POI message is the expectation of that POI contact each other more often than non-POI and the relationship could be non-linear. The initial assumption behind these features is: the relationship between POI is much more stronger than between POI and non-POIs.


## Feature Engineering:
Two features are created recently:
1- **to_poi_message_ratio** :
Measuring how many e-mails are frequently sent by a person to POIs.
2- **from_poi_message_ratio**:
Measuring how many e-mails are frequently received by a person from POIs.

After feature engineering & using SelectKBest, all features are scaled using min-max scales. For a comprehensive look on the chosen features, we can find their respective score after using SelectKBest:

Selected Features      		 Score
exercised_stock_options 		22.510
total_stock_value			22.349
bonus					20.792
salary					18.289
deferred_income			11.425
long_term_incentive			9.922
restricted_stock			9.284
total_payments			8.772
shared_receipt_with_poi		8.589
loan_advances				7.184


***Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?***
***Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm?***

GridSearchCV is applied to obtain a higher recall score by finding the best hyper parameters for Random Forest and Adaboost classifiers. the main idea of doing that is to find the correct values and performing data better than know later on. 

max_features, min_sample_splits and n_estimators are tuned for randomforest classifier. the higher values of max_features and n_estimator will be the higher complexity of the classifier and the higher value of the remain one will be the smaller complexity of the classifier.
The optimal values with respect to (RECALL SCORE):
*In terms of Random Forest: max_features and min_sample_splits = 2 and n_estimators = 100
*In terms of adaboost, n_estimators = 200 and learning_rate = 0.6.

Parameters tuning refers to the adjustment of the algorithm when training, in order to improve the fit on the test set. Parameter can influence the outcome of the learning process, the more tuned the parameters, the more biased the algorithm will be to the training data & test harness. The strategy can be effective but it can also lead to more fragile models & overfit the test harness but don't perform well in practice

***Question 5:What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?***

Validation comprises set of techniques to make sure our model generalizes with the remaining part of the dataset. A classic mistakes, which was briefly mistaken by me, is over-fitting where the model performed well on training set but have substantial lower result on test set.

Since, the number of observation is small and the proportion on POIs in low, StratifiedShuffleSplit method is used in order to applying cross validation. Summary of performance measurements of RF and AB tuned classifiers generated by CV:

[![Capture.png](https://s4.postimg.org/5zixhc7vh/Capture.png)](https://postimg.org/image/mnafju2mx/)
Where,
Precision measures the proportion of persons who are classified as POIs by the classifier and are indeed a POIs. *Precision = number of persons classified as POIs and are indeed a POIs / number of persons classified as POIs*
Recall  measures the proportion of persons who are indeed POIs correctly identified by classifier. *Recall = number of persons classified as POIs and are indeed a POIs / number of persons who are indeed POIs*

Based on the pervious table scores, **Random Forest model is the final classifier** as it produced the highest scores for precision and recall as well. 

***Question 6:Give at least 2 evaluation metrics, and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm's performance***

For this assignment, I used precision & recall as 2 main evaluation metrics. The best performance belongs to Random Forest (precision: 0.562 & recall: 0.392) which is also the final model of choice. Precision refer to the ratio of true positive (predicted as POI) to the records that are actually POI while recall described ratio of true positives to people flagged as POI.

Essentially speaking, with a precision score of 0.562, it tells us if this model predicts 100 POIs, there would be 56 people are actually POIs and the rest are innocent. With recall score of 0.392, this model finds 39% of all real POIs in prediction. This model is perfect for finding bad people without any missing, but with 39% probability of wrong. Due to the nature of the dataset, accuracy is not a perfect measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success. I confidently agree that the classifier should by applied in reality. 

Written with [StackEdit](https://stackedit.io/).
