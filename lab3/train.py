import numpy as np
import pickle

with open("dataset.pkl","rb") as file: dataset = pickle.load(file)

X = dataset[:,:-1]
y = dataset[:,-1].reshape((-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

Classifier = AdaBoostClassifier(DecisionTreeClassifier,10)
Classifier.fit(X_train, y_train)

y_pre = Classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pre, target_names=["Face", "Not Face"], digits = 4))# print("Hello AdaBoost!")