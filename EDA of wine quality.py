#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis on wine quality and its contains

# ### Importing required libraries for the EDA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as pyplot
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ### Importing the data set  which consist of Wine quality data sets

# In[2]:



data=pd.read_csv('D:Kaggle/winequalityN.csv')
data=pd.DataFrame(data)
data.head()


# In[3]:


#Getting the information of Datasets
data.info()


# In[4]:


data['type'].isnull().count()
data.isna().sum()/100


# ## There are total type of wine distributions are
# ### white    4898
# ### red      1599

# ### Null value consisting in the dataset 

# In[5]:


sns.countplot(x =data.isna().sum(), data = data)


# In[6]:


#filling null value by default mean value after the complete case analysis
df=data.fillna(value='data')
df.head(4)


# ## It show the that the quality of the both the type of wine have max approx similar quality

# In[7]:


sns.distplot(df[df['type']=='white']['quality'],hist=False)
sns.distplot(df[df['type']=='red']['quality'],hist=False)
#it show the that the quality of the both the type of wine have max approx similar quality


# ## Checking the correlation of the given columns 
# 
# ###  It only include numerical numerical dataset not the cateroical dataset it show that the free sulfur dioxide and total sulfur dioxide have strong correlation , where as it show negative relation with the alcohol so that it suggest that the free sulfur di oxide is nearly at very small quantity is found in the alcohol of same type of analysize with the different column can be also be made based on reseaerch

# In[8]:


df.corr()


# In[9]:


#visualizing the correlation of the dataset through heat map
plt.figure(figsize=(14,12))

sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)

plt.yticks(rotation=0);


# In[13]:


#a higher density of alcohol establishments means more availability of alcohol is present  there


# ## A pairs plot allows us to see both distribution of single variables and relationships between two variables.

# In[14]:


sns.pairplot(df,hue="type", diag_kind="hist")


# In[ ]:




