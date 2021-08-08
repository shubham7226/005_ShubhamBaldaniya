import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/drive/My Drive/ML/L1/mtcars.csv')
d=pd.crosstab(index=data['cyl'],columns="count",dropna=True)
print(d)

print(data.info())

#Count Total Null values in each column
print("Total Null Data:",data.isnull().sum())

# Finding the Histogram
# From the given dataset ‘mtcars.csv’, plot a histogram to check the frequency distributi on of the variable ‘mpg’ (Miles per gallon).
plt.hist(data['mpg'],bins=5)
plt.show()

#scatter plot of ‘mpg’ (Miles per gallon) vs ‘wt’ (Weight of car)
plt.scatter(data['mpg'],data['wt'])
plt.show()

#In the dataframe, under the variable gear count total records in each value
df=pd.DataFrame(data,columns=['gear'])
print("Count How many values:\n",df['gear'].value_counts())

