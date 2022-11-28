import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

essays = pd.read_csv('../..data/processed/essays_with_topic_scores.csv', index_col=0)

essays.set_index('essay_id', inplace=True)

scores = essays['domain1_score']
length = essays['length']

X = essays.drop(['domain1_score'], axis=1)
y = essays['domain1_score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)

dt_entropy.fit(X_train, y_train)

y_pred= dt_entropy.predict(X_test)


accuracy_entropy = accuracy_score(y_pred, y_test)
print(accuracy_entropy)

dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

dt_gini.fit(X_train, y_train)

y_pred= dt_gini.predict(X_test)

accuracy_gini = accuracy_score(y_pred, y_test)
print(accuracy_gini)

dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

Accuracy_CV_scores = cross_val_score(dt_gini, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1) 

print(Accuracy_CV_scores)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

SEED=3


lr = LogisticRegression(random_state=SEED)


knn = KNN(n_neighbors=5)

dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=SEED)

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)

for clf_name, clf in classifiers:    

    clf.fit(X_train, y_train)    

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test) 

    print('{:s} : {:.3f}'.format(clf_name, accuracy))

vc = VotingClassifier(estimators=classifiers)     

vc.fit(X_train, y_train)   

y_pred = vc.predict(X_test)


accuracy = accuracy_score(y_pred, y_test)
print('Voting Classifier: {:.3f}'.format(accuracy))

from sklearn.ensemble import BaggingClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

dt = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=0.016, random_state=4)

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=12)

bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

acc_test = accuracy_score(y_pred, y_test)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

dt = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=0.016, random_state=4)

bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True, n_jobs=-1, random_state=13)

bc.fit(X_train, y_train)


y_pred = bc.predict(X_test)

acc_test = accuracy_score(y_pred, y_test)

oob_accuracy = bc.oob_score_

print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 

print('OOB accuracy of bc: {:.2f}'.format(oob_accuracy))
