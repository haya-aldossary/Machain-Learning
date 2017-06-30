                                                                                                                                                                                 
***Question 1: Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?***

#Introduction:

The goal of this project is to use financial and email data from Enron Corporation in order to build z classifier that can distinguish persons of interest POIs who possibly involved in the fraud and corruption. it was an American energy, commodities, and services company based in Houston, Texas. It was founded in 1985 as the result of a merger between Houston Natural Gas and InterNorth, both relatively small regional companies in the U.S. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years. ). The corpus is widely used for various machine learning problem and although it has been labeled already, the value is the potential application for similar cases in other companies or spam filtering application. 

### About the dataset: Dataset contains 146 persons in total, however, 18 persons of them are identified as POI while the rest of them are non POI. There are 19 features for each person, person's salary and emails receiver for instance. This dataset contains missing values, for example there is no restricted_stock_deferred features for 128 persons and no director_fees feature for 129 persons.  

  [![Capture111.png](https://s9.postimg.org/j5zbnwqb3/Capture111.png)](https://postimg.org/image/ou5mesunf/)

TOTAL is the outlier name as a summary of all persons. it can be verified by calculating the sum of all persons salaries then the result is exactly as same as the salary for TOTAL. TOTAL in this case is outlier and will affect the performance of the classifier, thus, TOTAL is removed. Eventually, 145 persons ( records ) is remaining after removing TOTAL from the dataset.

---
***Question 2: What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it.***

I used scikit-learn SelectKBest to select best 10 influential features and used those featuers for all the upcoming algorithm. Unsurprisingly, 9 out of 10 features related to financial data and only 1 features called shared_receipt_with_poi (messages from/to the POI divided by to/from messages from the person) were attempted to engineere. 


Main purpose of composing ratio of POI message is we expect POI contact each other more often than non-POI and the relationship could be non-linear. The initial assumption behind these features is: the relationship between POI is much more stronger than between POI and non-POIs, and if we quickly did back-of-the-envelope Excel scatter plot, there might be truth to that hypothesis. The fact that shared_receipt_with_poi is included after using SelectKBest proved that this is a crucial features, as they also slightly increased the precision and recall of most of the machine learning algorithms used in later part of the analysis (e.g precision & recall for Support Vector Classifer before adding new feature are 0.503 & 0.223 respectively, while after adding new feature, the results are 0.504 & 0.225). Two features are created recently:

1- **to_poi_message_ratio** :
Measuring how many e-mails are frequently sent by a person to POIs.
2- **from_poi_message_ratio**:
Measuring how many e-mails are frequently received by a person from POIs.


[![1.png](https://s12.postimg.org/qwhok8gql/image.png)](https://postimg.org/image/6p48rxj95/)[![2.png](https://s7.postimg.org/5dn2i8kxn/image.png)](https://postimg.org/image/3lu3nc1kn/)

It appears that using random forest model tends to have higher precision and very low recall compared with its precision score.
For Logistic Regression model tends to have higher precision points than in it is recall points. And the accuracy for both classifiers are approximately the same.

Based on the plots, the recall scores are relatively low for all classifiers no matter how many features were selected.
The highret recall I obtaind for classifiers when using 10 features as listed in the next table with their importance scores in the second column that happend by leveraged the use of the scikit-learn's SelectKBest module to select the 10 most influential features as they slightly increased the precision and accuracy of most of the machine learning algorithms tested.

After feature engineering & using SelectKBest, I then scaled all features using min-max scalers. As briefly investigated through exporting CSV, we can see all email and financial data are varied by several order of magnitudes. Therefore, it is vital that we feature-scaling for the features to be considered evenly. For a comprehensive look on the chosen features, we can look at their respective score after using SelectKBest by the table below:

[![11.png](https://s16.postimg.org/4ug59mcit/image.png)](https://postimg.org/image/5wqbs5vc1/)

loan_advances had a considerably high score with only 3 non-NaN values. The K-best approach is an 
automated univariate feature selection algorithm.

LR:
precision: 0.170865993728
recall:    0.185535425685


***Question 3: What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?***
***Question 4: What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm?***

I improved this section by applying Support Vector Machine & Logistic Regression to  Random Forest Classifer "from the pervious submitiom".

Post-tuning results of them as following:

Algorithm:   Logistic Regression     Percision:     0.39       Recall:  0.42   Accuracy: 0.83

Algorithm:   Support Vector Classifier     Percision:    0.51        Recall:  0.22    Accuracy: 0.87

Algorithm:   Random Forest Classifier     Percision:       0.33     Recall:   0.16      Accuracy: 0.86

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

For this assignment, I used precision & recall as 2 main evaluation metrics. The best performance belongs to logistic regression (precision: 0.39 & recall: 0.42). Precision refer to the ratio of true positive (predicted as POI) to the records that are actually POI while recall described ratio of true positives to people flagged as POI.

Essentially speaking, with recall score of 0.42, it tells us if this model predicts 100 POIs, there would be 42 people are actually POIs and the rest are innocent. With a precision score of 0.39, this model finds 38% of all real POIs in prediction. This model is perfect for finding bad people without any missing, but with 39% probability of wrong. Due to the nature of the dataset, accuracy is not a perfect measurement as even if non-POI are all flagged, the accuracy score will yield that the model is a success. I confidently agree that the classifier should by applied in reality. 

Written with [StackEdit](https://stackedit.io/).
