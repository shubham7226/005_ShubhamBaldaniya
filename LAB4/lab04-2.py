#!/usr/bin/env python
# coding: utf-8


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from subprocess import call


# In[11]:


wine_data = datasets.load_wine()
ds = pd.DataFrame(wine_data.data, columns = wine_data.feature_names)
print(f"#examples :{ds.shape[0]} and #features: {ds.shape[1]}")


# In[13]:


print(ds.head())
print("\n\nFeatures:", wine_data.feature_names)
print("\nLabels:", np.unique(wine_data.target_names))


# **Splitting the dataset for training(80%) and testing(20%).
# Random state = 12 (Roll No. 12)**
# 

# In[14]:


x_train, x_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size = 0.20, random_state = 5)


# In[15]:


#creating instance of classifier and performing training
dtclassifier = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 10)
dtclassifier.fit(x_train,y_train)


# **Testing**

# In[17]:


# Testing
y_prediction = dtclassifier.predict(x_test)

#  Accuracy
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy Score:\n", accuracy)

#  Confusion Matrix
c_matrix = confusion_matrix(y_test, y_prediction)
print("\nConfusion Matrix:\n",c_matrix)

#  Precision
precision = precision_score(y_test, y_prediction, average=None)
print("\nPrecision Score:\n", precision)

#  Recall
recall = recall_score(y_test, y_prediction, average=None)
print("\nRecall Score:\n", recall)


# In[23]:


export_graphviz(dtclassifier, out_file='wine_tree.dot',
                feature_names=list(wine_data.feature_names),
               class_names=list(wine_data.target_names),
                filled=True)

# Convert to png
call(['dot', '-Tpng', 'wine_tree.dot', '-o', 'wine_tree.png', '-Gdpi=600'])
plt.figure(figsize = (15, 20))
plt.imshow(plt.imread('wine_tree.png'))
plt.axis('off')
plt.show()

