# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:31:30 2018

@author: arpan
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import operator

from sklearn.cross_validation import KFold

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn import tree
import seaborn as sns

from IPython.display import Image

%matplotlib inline
#import dataset
# add the rows names
header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',\
               'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']

# read csv file with Cleveland heart diseases data
heart = pd.read_csv('heart.csv', names=header_row)
heart[:5]
#dataset preprocessing
# we have unknown values '?'
# change unrecognized value '?' into mean value through the column
for c in heart.columns[:-1]:
    heart[c] = heart[c].apply(lambda x: heart[heart[c]!='?'][c].astype(float).mean() if x == "?" else x)
    heart[c] = heart[c].astype(float)
#coverting class into binary value
# if "diagnosis" == 0, member does not have disease A - we put 0
# if "diagnosis" >= 1, member possess the disease A - we put 1
heart.loc[:, "diag_int"] = heart.loc[:, "diagnosis"].apply(lambda x: 1 if x >= 1 else 0)


#normalize the dataset

# create the normalizer and fit it # create  
preprocessing.Normalizer().fit_transform(heart)


# split dataset into train and test
heart_train, heart_test, goal_train, goal_test = cross_validation.train_test_split(heart.loc[:,'age':'thal'], \
                                                 heart.loc[:,'diag_int'], test_size=0.33, random_state=0)
# calculate the correlation between features

corr = heart.corr()
heart.corr()
#training the classifier using Naive Bayes
cf=GaussianNB()
cf.fit(heart_train,goal_train)
#predicting the test result
heart_pred=cf.predict(heart_test)
#confusion matrix
cm=confusion_matrix(goal_test,heart_pred)
#get the performance metrics
scores = ['accuracy', 'f1', 'precision', 'recall']

metrics = {score: cross_validation.cross_val_score(cf,heart_test, goal_test, scoring=score).mean() for score in scores}

metrics
#since accuracy is only 76.9% we use next classifier Desicion Tree
#finding best parameter for decison tree

# build Decision tree model
best_score_dt = 0

criterion = ['gini', 'entropy']

for c in criterion:             

            clf = tree.DecisionTreeClassifier(criterion=c)

            clf.fit(heart_train, goal_train)
            print("Decision tree Cross-Validation scores:")
            scores = cross_validation.cross_val_score(clf, heart.loc[:,'age':'thal'], heart.loc[:,'diag_int'], cv=10)
            print (scores)
            print("Mean Decision tree Cross-Validation score = ", np.mean(scores))

            if np.mean(scores) > best_score_dt:
                best_score_dt = np.mean(scores)
                best_param_dt = (c)
                    
    
print("The best parameters for model are ", best_param_dt)
print("The Cross-Validation score = ", best_score_dt)
# develop the model with the best parameters

lss_best_dt = tree.DecisionTreeClassifier(criterion = 'entropy')
lss_best_dt.fit(heart_train, goal_train)
print("Decision tree Test score:")
print(lss_best_dt.score(heart_test, goal_test))
# develop the model with the best parameters

lss_best_dt = tree.DecisionTreeClassifier(criterion = 'entropy')
lss_best_dt.fit(heart_train, goal_train)
print("Decision tree Test score:")
print(lss_best_dt.score(heart_test, goal_test))
# we can see the accuracy is also low about 73% 

#creating our own ANN

#ANN
#importing libraries keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier=Sequential()
#adding input layer and first hidden layer 
classifier.add(Dense(input_dim=13, kernel_initializer="uniform", units=7, activation="relu"))
#addinh second hidden layer
classifier.add(Dense( kernel_initializer="uniform", units=7, activation="relu"))
#adding output layer
classifier.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))
#compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting ANN to training set
classifier.fit(heart_train,goal_train,batch_size=10,epochs=100)
# Predicting the Test set results
y_pred = classifier.predict(heart_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, y_pred)
#since the accuracy for training is 85% we can use this model
#but we have to check whether overfitting occurred or not
#accuracy for training phase
accuracy=(cm[0][0]+cm[1][1])/np.sum(cm) *100
# we got the accuracy of about 80%,so overfitting was not occured we can use this model

#save model for user input task
import h5py
classifier.save("ANNmodel.h5")

