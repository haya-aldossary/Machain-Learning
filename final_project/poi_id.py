#!/usr/bin/python


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import cross_validation
from numpy import mean
import pandas as pd
import numpy as np
from time import time
from matplotlib import pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
				 
print "total number of features = ",len(features_list) - 1

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print "number of data points in the dataset: ", len(data_dict)

print "print out the person names in the dataset: "
s = []
for person in data_dict.keys():
    s.append(person)
    if len(s) == 4:
        print '{:<30}{:<30}{:<30}{:<30}'.format(s[0],s[1],s[2],s[3])
        s = []
print '{:<30}{:<30}'.format(s[0],s[1])

print "'Total' = outlier, need to e removed from the dataset"

npoi = 0
for p in data_dict.values():
    if p['poi']:
        npoi += 1
print "number of person of interest = ", npoi



print "Number of missing values in all features: "
NaNInFeatures = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            NaNInFeatures[j] += 1

for i, feature in enumerate(features_list):
    print feature, NaNInFeatures[i]

print "print out some values of the observation 'TOTAL'"
for name, person in data_dict.iteritems():
	if name == 'TOTAL':
		print person

salary  = []
for name, person in data_dict.iteritems():
    if float(person['salary']) > 0:
        salary.append(float(person['salary']))
print "Sum of salary amount of other persons =",np.sum(salary)/2 

## Remove the outlier
data_dict.pop('TOTAL')


print "number of data points in the dataset after remove 'TOTAl' =", len(data_dict)

### Task 3: Create new feature(s):
### Store to my_dataset for easy export below.

my_dataset = data_dict

def compute_fraction(poi_messages, all_messages):
    """ return fraction of messages from/to that person to/from POI"""    
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.
    fraction = poi_messages / all_messages
    return fraction

for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

my_feature_list = features_list+['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                                 'shared_receipt_with_poi', 'fraction_to_poi']

num_features = 10 

def get_k_best(data_dict, features_list, k):

#runs scikit-learn's SelectKBest feature selection returns dict where keys=features, values=scores

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

best_features = get_k_best(my_dataset, my_feature_list, num_features)
target_label = 'poi'
my_feature_list = [target_label] + best_features.keys()

print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
## Try to use stratifieldshufflesplit to find the best subset of features to use

#  Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

### Support Vector Machine Classifier

from sklearn.svm import SVC

s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'auto')


### Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)


###TAsk 5: evaluate function

def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):

    print clf

    accuracy = []
    precision = []
    recall = []
    first = True

    for trial in range(num_iters):

        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "accuracy: {}".format(mean(accuracy))	
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall), mean(accuracy)


## Evaluate all functions

evaluate_clf(l_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(rf_clf, features, labels)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

Clf = l_clf
dump_classifier_and_data(Clf, my_dataset, my_feature_list)
