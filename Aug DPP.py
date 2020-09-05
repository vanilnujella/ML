# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 12:27:52 2020

@author: vnujella
"""

# change working directory under Spider

# imort popular libraries
import numpy              as np   # for complex mathematical computations - speedy, less memory
import matplotlib.pyplot  as plt  # for graphs and charts
import pandas             as pd   # for data structures (Matrix of IV and DV) 

# import sklearn                  # for handling missing data, categorical data 

# import the data set from csv file

dataset = pd.read_csv("C:\Work\Learn\ML\ML-DataFiles\data.csv")

# Split data into Matrix of IV and DV
# iloc - integer location based mehtod, of using the index positions to pull out data in those locations
# ":" represents all
"""
X = dataset.iloc[:,[0,1,2]].values 
X = dataset.iloc[:,:-1].values 
X = dataset.iloc[:3,:-2].values 
X = dataset.iloc[:,[0,3]].values 
X = dataset.iloc[5,[1,2]].values 
X = dataset.iloc[5:8,[0,1,2]].values 
"""

X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,[3]].values

"""
# handle missing values  
# one option is remove those observations

dataset1 = dataset.dropna(axis=0,how="any")
dataset1 = dataset.dropna(axis=0,how="all")
dataset1 = dataset.dropna(axis=1,how="all") # remvoe columns which one or more null values
"""

# In case of missing data - fill it using MEAN, MEDIAN or MODE 
# ... using Imputer class from SKLearn library. But it no longer exists and need to use Simple Imputer
# imputer = Imputer(missing_values ='NaN', strategy = 'mean',axis=0)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # np.nan - means numbers which are null
# in the above statement set the logic to be applied 

imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


# Handling categorical data - text type of data - as models only understand numbers, not text
# To convert text to numerical using label encoder 
# onehotencoder is no longer in use directly. To be used with in ColumnTransformer Class.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# one label encoding completed - we have to do OneHotEncoding,
# It is required when no of distinct categorical values are more than 2

# [("Countries", OneHotEncoder(),[0])]

ct = ColumnTransformer([("Country",OneHotEncoder(),[0])], remainder='passthrough')
X = ct.fit_transform(X)

y[:,0] = labelencoder_X.fit_transform(y[:,0])

# Split data into training and test set 70 - 30 or 80 - 20 etc - in case of Supervised Learning AL
# Need test set to validate the model output for the confidence, on UNSEEN Data, to avoid overfitted model
# (like learning of students on few questions + exam on different questions, to check the ability)




















