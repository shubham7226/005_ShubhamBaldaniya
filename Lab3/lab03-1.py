
from google.colab import drive
drive.mount("/content/drive")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB

dataset = pd.read_csv("/content/drive/My Drive/ML/L3/L3/PracticeDataSets/Dataset1.csv")

# labelEncoder Object
label_encoder = preprocessing.LabelEncoder()
Y_rows=None
for data_heading in dataset:
  if data_heading!="Play":
    print(f"\n\nHeading :- {data_heading}")
    #print(list(dataset[data_heading]))
    dummy = pd.get_dummies(dataset[data_heading])
    #print("\n\nDummy :\n",dummy)
    dataset = dataset.drop([data_heading],axis=1)
    dataset = pd.concat([dataset,dummy],axis=1)
    #print("\n\nFinal Data :\n",dataset)
  else:
    Y_rows = label_encoder.fit_transform(dataset[data_heading])
    dataset = dataset.drop([data_heading],axis=1)

print(dataset,Y_rows)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset, Y_rows, test_size = 0.30, random_state = 54)

# create model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Predict Y from X_text
Y_predicted = model.predict(X_test)
print(X_test)
print(Y_predicted)

from sklearn import metrics

print(f"Accuracy is :- {metrics.accuracy_score(Y_test, Y_predicted)}")

# print precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


precision = precision_score(Y_test, Y_predicted)
recall = recall_score(Y_test, Y_predicted)


print(f"precision :- {precision}")
print(f"recall :- {recall}")

# Excersice
# Task1

# Temp = "Hot" and Weather = "overcast"
#              1  1  0
#Outlook(O,R,S)=0  1  0,	Temp(C,H,M)=0  1  0,	Humidity(High,Low,Normal)=0  0 1,	Wind(F,T)=1,0,	Play=0
output = model.predict([[0,1,0, 0,1,0  ,0,0,1 ,1,0]])
print(f"final prediction :- {output}")

#Overcast , High, Normal, False
output = model.predict([[1,0,0  ,0,1,0    ,0,0,1  ,1,0]])
print(f"final prediction :- {output}")

# Excersice
# Task1

# Overcast, Mild, Normal, True

output = model.predict([[1,0,0, 0,0,1 ,0,0,1  ,0,1]])
print(f"final prediction :- {output}")

