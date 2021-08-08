import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#Load Data
from google.colab import drive
drive.mount('/content/drive')

datasets = pd.read_csv('/content/drive/My Drive/ML/L2/L2/Datasets/Exercise-CarData.csv', index_col=[0])
print("\nData :\n",datasets)

print("\nData statistics\n",datasets.describe())

datasets.dropna(how='all',inplace=True)
#print("\nNew Data :",datasets)
print(datasets.dtypes)
# All rows, all columns except last 
new_X = datasets.iloc[:, :-1].values
# Only last column  
new_Y = datasets.iloc[:, -1].values 

#FuelType
new_X[:,3]=new_X[:,3].astype('str')
le = LabelEncoder()
new_X[ : ,3] = le.fit_transform(new_X[ : ,3])

print("\n\nInput before imputation : \n\n", new_X[6])

str_to_num_dictionary={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nune":9,"ten":10}

# 3b. Imputation (Replacing null values with mean value of that attribute)
#for col-3
for i in range(new_X[:,3].size):
  #KM
  if new_X[i,2]=="??":
    new_X[i,2]=np.nan
  #HP
  if new_X[i,4]=="????":
    new_X[i,4]=np.nan
  #Doors
  temp_str = str(new_X[i,8])
  if temp_str.isnumeric():
    new_X[i,8]=int(temp_str)
  else:
    new_X[i,8]=str_to_num_dictionary[temp_str]
# Using Imputer function to replace NaN values with mean of that parameter value 
imputer = SimpleImputer(missing_values = np.nan,strategy = "mean")
mode_imputer = SimpleImputer(missing_values = np.nan,strategy = "most_frequent")

# Fitting the data, function learns the stats 
the_imputer = imputer.fit(new_X[:, 0:3])
# fit_transform() will execute those stats on the input ie. X[:, 1:3] 
new_X[:, 0:3] = the_imputer.transform(new_X[:, 0:3])

# Fitting the data, function learns the stats 
the_mode_imputer = mode_imputer.fit(new_X[:, 3:4])   
new_X[:, 3:4] = the_mode_imputer.transform(new_X[:, 3:4])

# Fitting the data, function learns the stats 
the_imputer = imputer.fit(new_X[:, 4:5])
new_X[:, 4:5] = the_imputer.transform(new_X[:, 4:5])

# Fitting the data, function learns the stats 
the_mode_imputer = mode_imputer.fit(new_X[:, 5:6])   
new_X[:, 5:6] = the_mode_imputer.transform(new_X[:, 5:6])

# filling the missing value with mean 
print("\n\nNew Input with Mean Value for NaN : \n\n", new_X[6])

new_datasets = pd.DataFrame(new_X,columns=datasets.columns[:-1])
new_datasets = new_datasets.astype(float)
new_datasets.dtypes

#feature selection
corr = new_datasets.corr()
print(corr.head())
sns.heatmap(corr)

columns = np.full((len(new_datasets.columns),), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = new_datasets.columns[columns]
print(selected_columns)

new_datasets = new_datasets[selected_columns]

# Step 5a : Perform scaling and standardization
new_X = new_datasets.iloc[:, :-1].values
scaler = MinMaxScaler()
std = StandardScaler()
new_X[:,0:3] = std.fit_transform(scaler.fit_transform(new_X[:,0:3]))
new_X[:,4:5] = std.fit_transform(scaler.fit_transform(new_X[:,4:5]))
new_X[:,7:9] = std.fit_transform(scaler.fit_transform(new_X[:,7:9]))

print("Dataset after preprocessing\n\n",new_datasets)

