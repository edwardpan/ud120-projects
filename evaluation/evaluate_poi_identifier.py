#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeRegressor()
clf.fit(features_train, labels_train)

predict_res = clf.predict(features_test)

print "poi: ", np.sum(predict_res)
print "total: ", len(features_test)
print "accuracy: ", accuracy_score(labels_test, predict_res)

print "precision: ", precision_score(labels_test, predict_res)
print "recall: ", recall_score(labels_test, predict_res)


from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import matplotlib.pyplot as plt

# Set the learning curve parameters; you'll need this for learning_curves
size = 10
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

# Defining our regression algorithm
reg = DecisionTreeRegressor()
# Fit our model using X and y
reg.fit(features_train, labels_train)
print "Regressor score: {:.4f}".format(reg.score(features_train, labels_train))

# TODO: Use learning_curve imported above to create learning curves for both the
#       training data and testing data. You'll need reg, X, y, cv and score from above.

train_sizes, train_scores, test_scores = learning_curve(reg, features, labels, cv=cv, scoring=score)

# Taking the mean of the test and training scores
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)

# Plotting the training curves and the testing curves using train_scores_mean and test_scores_mean
plt.plot(train_sizes ,train_scores_mean,'-o',color='b',label="train_scores_mean")
plt.plot(train_sizes,test_scores_mean ,'-o',color='r',label="test_scores_mean")

# Plot aesthetics
plt.ylim(-0.1, 1.1)
plt.ylabel("Curve Score")
plt.xlabel("Training Points")
plt.legend(bbox_to_anchor=(1.1, 1.1))
plt.show()