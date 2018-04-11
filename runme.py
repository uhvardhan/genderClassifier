from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


clf_tree = tree.DecisionTreeClassifier()

# CHALLENGE - creating 3 more classifiers...
clf_knn = KNeighborsClassifier()
clf_gauss = GaussianNB()
clf_svm = SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

#Labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_knn = clf_knn.fit(X,Y)
clf_gauss = clf_gauss.fit(X,Y)
clf_svm = clf_svm.fit(X,Y)

prediction_tree = clf_tree.predict(X)
accuracy_tree = accuracy_score(Y,prediction_tree) * 100
print('Accuracy for DecisionTree: {}'.format(accuracy_tree))

prediction_knn = clf_knn.predict(X)
accuracy_knn = accuracy_score(Y, prediction_knn) * 100
print('Accuracy for K Nearest Neighbors: {}'.format(accuracy_knn))

prediction_gauss = clf_gauss.predict(X)
accuracy_gauss = accuracy_score(Y, prediction_gauss) * 100
print('Accuracy for Naive Bayes: {}'.format(accuracy_gauss))

prediction_svm = clf_svm.predict(X)
accuracy_svm = accuracy_score(Y,prediction_svm) * 100
print('Accuracy for Support Vector Machines: {}'.format(accuracy_svm))

# CHALLENGE compare their reusults and print the best one!
# The best classifier from svm, per, KNN
index = np.argmax([accuracy_knn, accuracy_gauss, accuracy_svm])
classifiers = {0: 'KNN', 1: 'Naive Bayes', 2: 'SVM'}
print('Best gender classifier is {}'.format(classifiers[index]))