     
***Question 1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?***

#Introduction:

The goal of this project is to use financial and email data from Enron Corporation in order to build z classifier that can distinguish persons of interest POIs who possibly involved in the fraud and corruption. it was an American energy, commodities, and services company based in Houston, Texas. It was founded in 1985 as the result of a merger between Houston Natural Gas and InterNorth, both relatively small regional companies in the U.S. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years. ). The corpus is widely used for various machine learning problem and although it has been labeled already, the value is the potential application for similar cases in other companies or spam filtering application. 

### About the dataset: 
Dataset contains 146 persons in total, however, 18 persons of them are identified as POI while the rest of them are non POI. There are 19 features for each person, person's salary and emails receiver for instance. This dataset contains missing values, for example there is no restricted_stock_deferred features for 128 persons and no director_fees feature for 129 persons.  

[![Capture111.png](https://s9.postimg.org/j5zbnwqb3/Capture111.png)](https://postimg.org/image/ou5mesunf/)

TOTAL is the outlier name as a summary of all persons. it can be verified by calculating the sum of all persons salaries then the result is exactly as same as the salary for TOTAL. TOTAL in this case is outlier and will affect the performance of the classifier, thus, TOTAL is removed. Eventually, 145 persons ( records ) is remaining after removing TOTAL from the dataset.

---
***Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.***

I used *f_classif* function as the scoring function for SelectKBest method in scikit_learn then scikit-learn *SelectKBest* is used to select best more influential features and used those features for all the upcoming algorithm. As a result, I've got 10 features, 9 out of 10 features related to financial data and only 1 features called shared_receipt_with_poi (messages from/to the POI divided by to/from messages from the person) were attempted to engineere by us. Main purpose of composing ratio of POI message is the expectation of that POI contact each other more often than non-POI and the relationship could be non-linear. The initial assumption behind these features is: the relationship between POI is much more stronger than between POI and non-POIs. The fact that shared_receipt_with_poi is included after using SelectKBest proved that this is a crucial features, as they also slightly increased the precision and recall of most of the machine learning algorithms used in later part of the analysis (e.g precision & recall for Support Vector Classifer before adding new feature are 0.503 & 0.223 respectively, while after adding new feature, the results are 0.504 & 0.225)


In more details for the Two features that are created recently:
1- **to_poi_message_ratio** :
Measuring how many e-mails are frequently sent by a person to POIs.
2- **from_poi_message_ratio**:
Measuring how many e-mails are frequently received by a person from POIs.

After feature engineering & using SelectKBest, all features are scaled using min-max scales. For a comprehensive look on the chosen features, we can find their respective score after using SelectKBest:

Selected Features      		 Score

exercised_stock_options 		22.510

total_stock_value			      22.349

bonus					          20.792

salary			              		18.289

deferred_income			       11.425

long_term_incentive			   9.922

restricted_stock			      9.284

total_payments		        	8.772

shared_receipt_with_poi		8.589

loan_advances			        	7.184


***Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?***
***Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm?***

I improved this section by applying Support Vector Machine & Logistic Regression to  Random Forest Classifer "from the pervious submitiom".

Post-tuning results of them as following:
Algorithm:   Logistic Regression     Percision:     0.382       Recall:  0.415

Algorithm:   Support Vector Classifier     Percision:    0.518        Recall:  0.219

Algorithm:   Random Forest Classifier     Percision:       0.321     Recall:   0.158

Parameters tuning refers to the adjustment of the algorithm when training, in order to improve the fit on the test set. Parameter can influence the outcome of the learning process, the more tuned the parameters, the more biased the algorithm will be to the training data & test harness. The strategy can be effective but it can also lead to more fragile models & overfit the test harness but don't perform well in practice


After tuning all algorithems, it found that Logistic Regression is the good one.See the following paremeters values:

C=1e-08, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
max_iter=100, multi_class='ovr', penalty='l2', random_state=42, solver='liblinear', tol=0.001, verbose=0))

where:
C means inverse regularization, solver parameter Im using 'liblinear' as the dataset is very small).

***Question 5:What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?***

Validation comprises set of techniques to make sure our model generalizes with the remaining part of the dataset. A classic mistakes, which was briefly mistaken by me, is over-fitting where the model performed well on training set but have substantial lower result on test set.

I used cross-validation ( evaluate_clf function in poi_id.py where I start 1000 trials and divided the dataset into 3:1 training-to-test ratio. Main reason why we would use StratifiedSuffleSplit rather than other splitting techniques avaible is due to the nature of our dataset, which is extremely small with only 14 Persons of Interest. A single split into a training & test set would not give a better estimate of error accuracy. Therefore, we need to randomly split the data into multiple trials while keeping the fraction of POIs in each trials relatively constant.


***Question 6:Give at least 2 evaluation metrics, and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm's performance***

For this assignment, I used precision & recall as 2 main evaluation metrics. The best performance belongs to logistic regression (precision: 0.382 & recall: 0.415). Precision refer to the ratio of true positive (predicted as POI) to the records that are actually POI while recall described ratio of true positives to people flagged as POI.

Essentially speaking, with a precision score of 0.386, it tells us if this model predicts 100 POIs, there would be 38 people are actually POIs and the rest are innocent. With recall score of 0.4252, this model finds 42% of all real POIs in prediction. This model is perfect for finding bad people without any missing, but with 42% probability of wrong. Due to the nature of the dataset, accuracy is not a perfect measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success. I confidently agree that the classifier should by applied in reality. 

Written with [StackEdit](https://stackedit.io/).
