# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 02:07:27 2017

@author: KMusial
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from IPython.display import display
from sklearn.cross_validation import train_test_split

in_file = 'numerai_training_data.csv'
full_data = pd.read_csv(in_file)
outcomes = full_data['target']
data = full_data.drop('target', axis = 1)

tournamentdata = pd.read_csv('numerai_tournament_data.csv')
t_id = tournamentdata.columns[0]
data1 = list(tournamentdata.columns[1:22])
X_tour = tournamentdata[data1]
y_tour = tournamentdata[t_id]

X_train = None
X_test = None
y_train = None
y_test = None
X_train, X_test, y_train, y_test = train_test_split(data, outcomes, test_size=0.24, random_state=40)

print 'GNB'
clf_A = GaussianNB()
clf_A.fit(X_train, y_train)
y_pred = clf_A.predict(X_test)
print y_pred
accuracy = f1_score(y_test, y_pred)
print accuracy
y_predicttournament = clf_A.predict(X_tour)
print y_predicttournament
#print type(y_predicttournament)

print 'logisticregression'
clf_B = LogisticRegression(random_state=42)
clf_B.fit(X_train, y_train)
y_pred = clf_B.predict(X_test)
print y_pred
accuracy = f1_score(y_test, y_pred)
print accuracy
y_predicttournament = clf_B.predict(X_tour)
print y_predicttournament

print 'SVC'
clf_C = SVC(random_state=42)
clf_C.fit(X_train, y_train)
y_pred = clf_C.predict(X_test)
print y_pred
accuracy = f1_score(y_test, y_pred)
print accuracy
y_predicttournament = clf_C.predict(X_tour)
print y_predicttournament