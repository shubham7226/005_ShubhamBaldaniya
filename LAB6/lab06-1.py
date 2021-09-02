#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Task 1 : Try logistic regression on BuyComputer dataset and set Random state=Your_RollNumber. 


# In[2]:


import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("BuyComputer.csv")
data.drop(columns=['User ID',],axis=1,inplace=True)
data.head()


# In[4]:


from sklearn.model_selection import train_test_split

y = data.iloc[:,-1].values
X = data.iloc[:,:-1].values
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[5]:


from sklearn.preprocessing import StandardScaler
import torch

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[6]:


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# In[7]:


import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)


# In[8]:


num_epochs = 140
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[9]:


for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'\n Accuracy: {acc.item()*100:.2f}')

