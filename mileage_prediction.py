#!/usr/bin/env python
# coding: utf-8

# # Objective: Build a predictive modelling algorithm to predict the mileage of cars based on input variables.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#for creating train test samples
from sklearn.model_selection import train_test_split

#for feature selection
from sklearn.feature_selection import SelectKBest, f_regression

#for building linear regression model
from sklearn.linear_model import LinearRegression


# In[2]:


#import automobile data
df=pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Introtallent\Python\Data Files used in Projects\automobile data.csv")


# In[3]:


df


# Target variable (y): MPG Miles per Gallon
# Indepemdamt variables(x):
# 
#     * Cylinders
#     * Displacement
#     * Horsepower
#     * Weight
#     * Acceleration
#     * Model Year
#     * Origin
#     * Car Name

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df['Horsepower']=pd.to_numeric(df['Horsepower'], errors='coerce');


# In[7]:


df.dtypes


# In[8]:


df.describe


# In[9]:


df.isnull().sum()


# In[10]:


#there are 6 missing values in Horsepower variable


# In[11]:


#Missing value imputation


# In[12]:


df['Horsepower']=df['Horsepower'].fillna(df['Horsepower'].median())


# In[13]:


df.isnull().sum()


# # check outliers

# In[14]:


plt.boxplot(df['MPG'])
plt.show()
#no outlier


# In[15]:


plt.boxplot(df['Displacement'])
plt.show()
#no outliers


# In[16]:


plt.boxplot(df['Horsepower'])
plt.show()
#has outliers


# In[17]:


plt.boxplot(df['Weight'])
plt.show()
#no outliers


# In[18]:


plt.boxplot(df['Acceleration'])
plt.show()
#has outliers


# In[19]:


# There are outliers in acceleration and horsepower variables


# In[20]:


def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    
    iqr=q3-q1
    
    ub=q3+1.5*iqr
    lb=q1-1.5*iqr
    
    good_data=d[(d[c]<ub) & (d[c]>lb)]
    return good_data


# In[23]:


df=remove_outlier(df,'Horsepower')
plt.boxplot(df['Horsepower'])
plt.show()


# In[26]:


df=remove_outlier(df,'Acceleration')
plt.boxplot(df['Acceleration'])
plt.show()


# # Exploratory Data Analysis
# * Distribution
# * Data Mix
# * Corelation

# In[27]:


df.columns


# In[28]:


sns.distplot(df['MPG'])


# In[29]:


sns.distplot(df['Displacement'])


# In[30]:


sns.distplot(df['Horsepower'])


# In[31]:


sns.distplot(df['Weight'])


# In[32]:


sns.distplot(df['Acceleration'])


# In[33]:


#Check datamix
#Cylinders, model year, origin, car name


# In[34]:


df.groupby('Cylinders')['Cylinders'].count().plot(kind='bar')


# In[35]:


df.groupby('Model_year')['Model_year'].count().plot(kind='bar')


# In[36]:


df.groupby('Origin')['Origin'].count().plot(kind='bar')


# In[37]:


df.groupby('Car_Name')['Car_Name'].count().plot(kind='bar')


# In[38]:


#create a set of numeric columns
df_numeric=df.select_dtypes(include=['int64','float64'])
df_numeric.head()


# In[39]:


#df_numeric has categorical values that we need to drop
#they are cylinders, model year, origin


# In[40]:


df_numeric=df_numeric.drop(['Cylinders', 'Model_year', 'Origin'] , axis=1)


# In[41]:


df_numeric.head()


# In[42]:


#Heatmap
sns.heatmap(df_numeric.corr(), cmap='YlGnBu', annot=True)


# In[43]:


#Key drivers areweight, horsepower and displacement


# # Dummy conversion (One-hot encoding)

# In[44]:


df=df.drop('Model_year', axis=1)


# In[45]:


df.dtypes


# In[46]:


df['Cylinders']=df['Cylinders'].astype('object')
df['Origin']=df['Origin'].astype('object')
df.dtypes


# In[47]:


df_categorical=df.select_dtypes(include='object')
df_categorical.head()


# In[48]:


#Dummy Conversion

df_dummies=pd.get_dummies(df_categorical, drop_first=True)
df_dummies.head()


# In[49]:


df_final=pd.concat([df_numeric,df_dummies], axis=1)
df_final.head()


# In[50]:


#creating x and y
x=df_final.drop('MPG', axis=1)
y=df_final['MPG']


# In[51]:


#Training and Testing Samples
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.3)


# In[52]:


xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[53]:


# Feature selection
#selecting 4 significant features

key_features = SelectKBest(score_func=f_regression, k=5)

xtrain_selected=key_features.fit_transform(xtrain,ytrain)

#get the indices of the selected feature
selected_indices=key_features.get_support(indices=True)

#get the names of the selected feature
selected_features=xtrain.columns[selected_indices]


# In[54]:


selected_features


# In[55]:


# Building a linear regression model

linreg=LinearRegression()
linreg.fit(xtrain_selected, ytrain)

linreg.score(xtrain_selected, ytrain)


# In[56]:


xtest_selected= xtest.iloc[:, selected_indices]


# In[57]:


#predicted mileage based on xtest
predicted_mpg=linreg.predict(xtest_selected)

#check aaccuracy of tested model
linreg.score(xtest_selected, ytest)


# In[ ]:




