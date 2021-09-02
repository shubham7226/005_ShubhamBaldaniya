#!/usr/bin/env python
# coding: utf-8


# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree


# Prepare dataset

# In[2]:


#Predictor variables
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
            'Rainy', 'Rainy', 'Sunny', 'Rainy','Overcast', 'Overcast', 'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
            'False', 'False', 'False', 'True', 'True', 'False', 'True']

#Class Label:
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']


# 
# Encoding the features using one-hot encoding

# In[3]:


from sklearn.preprocessing import OneHotEncoder
X = tuple(zip(Outlook, Temperature, Humidity, Wind))
print(X)
X_enc = OneHotEncoder().fit_transform(X).toarray()
print(X_enc)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_enc = le.fit_transform(Play)


# # step 4 :- merge different features to prepare dataset

# In[4]:


# create a tuples of different values
print("\nFeatures\n")
for pair in X_enc:
  print(pair)


# In[5]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_enc, Y_enc, test_size = 0.3, random_state = 5)
print(x_train)


# # step 5 :- train "Create and train DecisionTreeClassifier"

# In[6]:


# Create a decisionTreeCalssifier object

from sklearn import metrics

dtc = DecisionTreeClassifier(criterion = "entropy")
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

# Accuracy
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))


# # step 6 :- Predict output for new data

# In[7]:


# Prediction
print(x_test)

# y_predicted = dtc.predict(x_test)
print("y predicted : ", y_pred)

# y_predicted = dtc.predict(x_test)
print(f"Actual y_test {y_test}")


# In[8]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[9]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print(metrics.classification_report(y_test, y_pred,zero_division=0))


# In[10]:


disp = metrics.plot_confusion_matrix(dtc, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# In[11]:


from subprocess import call
export_graphviz(dtc,out_file='tree_entropy.dot',
               feature_names=['outlook2','outlook1','outlook0','temperature2','temperature1','temperature0','humidity1','humidity0','wind1','wind0'],
               class_names=['do_not_play','play'], 
               filled=True)

# Convert to png
call(['dot', '-Tpng', 'tree_entropy.dot', '-o', 'tree_entropy.png', '-Gdpi=600'])

# Display in python
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree_entropy.png'))
plt.axis('off');
plt.show();


# In[11]:




