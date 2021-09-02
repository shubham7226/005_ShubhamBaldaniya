#!/usr/bin/env python
# coding: utf-8



# Import Numpy & PyTorch
import numpy as np


# A tensor is a number, vector, matrix or any n-dimensional array.

# ## Problem Statement

# We'll create a model that predicts crop yeilds for apples (*target variable*) by looking at the average temperature, rainfall and humidity (*input variables or features*) in different regions. 
# 
# Here's the training data:
# 
# >Temp | Rain | Humidity | Prediction
# >--- | --- | --- | ---
# > 73 | 67 | 43 | 56
# > 91 | 88 | 64 | 81
# > 87 | 134 | 58 | 119
# > 102 | 43 | 37 | 22
# > 69 | 96 | 70 | 103
# 
# In a **linear regression** model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias :
# 
# ```
# yeild_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# ```
# 
# It means that the yield of apples is a linear or planar function of the temperature, rainfall & humidity.
# 
# 
# 
# **Our objective**: Find a suitable set of *weights* and *biases* using the training data, to make accurate predictions.

# ## Training Data
# The training data can be represented using 2 matrices (inputs and targets), each with one row per observation and one column for variable.

# In[197]:


# Input (temp, rainfall, humidity)
X = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')


# In[198]:


# Target (apples)
Y = np.array([[56], 
                    [81], 
                    [119], 
                    [22], 
                    [103]], dtype='float32')


# Before we build a model, we need to convert inputs and targets to PyTorch tensors.

# ## Linear Regression Model (from scratch)
# 
# The *weights* and *biases* can also be represented as matrices, initialized with random values. The first row of `w` and the first element of `b` are use to predict the first target variable i.e. yield for apples, and similarly the second for oranges.

# The *model* is simply a function that performs a matrix multiplication of the input `x` and the weights `w` (transposed) and adds the bias `w0` (replicated for each observation).
# 
# $$
# \hspace{2.5cm} X \hspace{1.1cm} \times \hspace{1.2cm} W^T 
# $$
# 
# $$
# \left[ \begin{array}{cc}
# 1 & 73 & 67 & 43 \\
# 1 &91 & 88 & 64 \\
# \vdots & \vdots & \vdots & \vdots \\
# 1 &69 & 96 & 70
# \end{array} \right]
# %
# \times
# %
# \left[ \begin{array}{cc}
# w_{0}  \\
# w_{1}  \\
# w_{2}  \\
# w_{3} 
# \end{array} \right]
# %
# $$

# In[199]:


mu = np.mean(X, 0)
sigma = np.std(X, 0)
#normalizing the input
X = (X-mu) / sigma
X = np.hstack((np.ones((Y.size,1)),X))
print(X.shape)


# In[200]:



# Weights and biases
rg = np.random.default_rng(12)
w = rg.random((1, 4))
print(w)


# Because we've started with random weights and biases, the model does not perform a good job of predicting the target varaibles.

# ## Loss Function
# 
# We can compare the predictions with the actual targets, using the following method: 
# * Calculate the difference between the two matrices (`preds` and `targets`).
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# 
# The result is a single number, known as the **mean squared error** (MSE).

# In[201]:


# MSE loss function
def mse(t1, t2):
    diff = t1 - t2
    return np.sum(diff * diff) / diff.size


# In[202]:


# Compute error
preds = model(X,w)
cost_initial = mse(preds, Y)
print("Cost before regression: ",cost_initial)


# ## Compute Gradients
# 
# 

# In[203]:


# Define the model
def model(x,w):
    return x @ w.T


# In[204]:


def gradient_descent(X, y, w, learning_rate, n_iters):
    J_history = np.zeros((n_iters,1))
    for i in range(n_iters):
        h = model(X,w)
        diff = h - y
        delta = (learning_rate/Y.size)*(X.T@diff)
        new_w = w - delta.T
        w=new_w
        J_history[i] = mse(h, y)
    return (J_history, w)


# ## Train for multiple iteration
# 
# To reduce the loss further, we repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an epoch.

# In[205]:


import matplotlib.pyplot as plt
n_iters = 500
learning_rate = 0.01

initial_cost = mse(model(X,w),Y)

print("Initial cost is: ", initial_cost, "\n")

(J_history, optimal_params) = gradient_descent(X, Y, w, learning_rate, n_iters)

print("Optimal parameters are: \n", optimal_params, "\n")

print("Final cost is: ", J_history[-1])


# In[206]:


plt.plot(range(len(J_history)), J_history, 'r')

plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()


# In[209]:


# Calculate error
preds = model(X,optimal_params)
cost_final = mse(preds, Y)
# Print predictions
print("Prediction:\n",preds)
# Comparing predicted with targets
print("Targets:\n",Y)


# In[210]:


print("Cost after linear regression: ",cost_final)
print("Cost reduction percentage : {} %".format(((cost_initial- cost_final)/cost_initial)*100))

