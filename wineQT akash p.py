#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/lenovo/Downloads/WineQT.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


print(pd.isnull(df).sum())


# In[7]:


sns.barplot(x="pH",y="quality", data=df)


# In[8]:


sns.barplot(x="chlorides",y="quality", data=df)


# In[9]:


corr = df.corr()
fig = plt.figure(figsize =(15,12))
r = sns.heatmap(corr, cmap='Purples')
r.set_title("Correlation ")


# In[10]:


df.value_counts()


# In[11]:


df['quality'].value_counts()


# In[12]:


df.hist(bins=100, figsize=(10,15))
plt.show()


# In[ ]:





# In[13]:


df.corr()['quality'].sort_values()


# In[14]:


sns.barplot(df['quality'],df['alcohol'], data=df)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(df.drop([ 'alcohol'],axis=1), 
                                                    df['quality'], test_size=0.20, 
                                                    random_state=8)


# In[20]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[21]:


accuracy = model.score(X_test, y_test)
print(accuracy*100,'%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




