# -*- coding: utf-8 -*-
"""Bail.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QaviOyEqDgMnhoq3tcMMduz7KM7_nhM5
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Balanced_Data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y.reshape(-1, 1)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), list(range(6, 18)))], remainder='passthrough')
X = np.array(ct.fit_transform(X))
y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred, y_test)))

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
cm = multilabel_confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

pred = classifier.predict([[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 17, 1, 19, 73, 100000,
 36623]])

if pred[0][1] == 1.0:
  print("Yes")
else:
  print("No")
  
