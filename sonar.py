#!/usr/bin/env python
# coding: utf-8

# In[217]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[218]:


sonar_data = pd.read_csv('Copy of sonar data.csv', header = None)
sonar_data.head()



# In[219]:


sonar_data.shape


# In[220]:


sonar_data.describe()


# In[221]:


sonar_data[60].value_counts()


# In[222]:


sonar_data.groupby(60).mean()



# In[223]:


X  = sonar_data.drop(columns = 60 , axis =1)
Y = sonar_data[60]


# In[224]:


print(X)
print(Y)


# In[225]:


X_train, X_test, Y_train,Y_test =  train_test_split(X,Y,test_size = 0.1 , stratify = Y , random_state = 1)


# In[226]:


print(X.shape, X_train.shape , X_test.shape)


# In[227]:


model =  LogisticRegression()
print(X_train)
print(Y_train)
model.fit(X_train, Y_train)


# In[228]:


print(X_train)
print(Y_train)


# In[229]:


model.fit(X_train, Y_train)


# In[230]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[231]:


print("accuracy is", training_data_accuracy)


# In[232]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[233]:


print("accuracy is", test_data_accuracy)


# In[234]:


input_data = (0.0762,0.0666,0.0481,0.0394,0.0590,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.5730,0.5399,0.3161,0.2285,0.6995,1.0000,0.7262,0.4724,0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.2430,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.0230,0.0046,0.0156,0.0031,0.0054,0.0105,0.0110,0.0015,0.0072,0.0048,0.0107,0.0094)
input_data_as_numpy_array = np.asarray(input_data)

inputdatareshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(inputdatareshaped)
print(prediction)
if prediction[0] == 'R':
    print("object is a rock")

else:
     print("object is a mine")

