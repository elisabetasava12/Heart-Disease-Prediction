#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


df=pd.read_csv('C:/Users/sava_/OneDrive/Desktop/ALTELE/pt_python/heart (1).csv')


# In[26]:


df


# In[27]:


#verificam daca avem valori nule

df.isnull().values.any()


# In[28]:


#cautam corelatii daca exista

import seaborn as sns

corr_df=df.corr()
f,ax=plt.subplots(figsize=(12,9))


sns.heatmap(corr_df,vmax=.8,square=True,fmt='.2f',annot=True,cmap='winter')


# In[29]:



df.info()


# In[30]:


df=pd.get_dummies(df,columns=['sex','fbs','restecg','exang','slope','ca','cp','thal'])


# In[31]:


df

#inainte aveam 14 coloane


# In[32]:


#scalam celelate coloane carenu sunt categorical features

from sklearn.preprocessing import StandardScaler
StandardScaler=StandardScaler()
col_scalat=['age','trestbps','chol','thalach','oldpeak']
df[col_scalat]=StandardScaler.fit_transform(df[col_scalat])


# In[33]:


from sklearn.model_selection import train_test_split

x=df.drop('target',axis=1)
y=df['target']

#split 80-20

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[34]:


#o sa folosim SVM Classifier si RandomSearch pt cel mai bun varainta de parametrii

from sklearn.svm import LinearSVC

svc=LinearSVC()

from sklearn.model_selection import RandomizedSearchCV

penalties=['l1','l2']
tolerance=[1e-3,1e-4,1e-5]
Cul=[1,0.8,0.6,0.4,0.2]

param_random=[{'penalty':penalties,'tol':tolerance,'C':Cul}]

random=RandomizedSearchCV(LinearSVC(dual=False),param_random,cv=3,n_iter=100)

random.fit(x_train,y_train)

random.best_params_


# In[39]:


svc=LinearSVC(tol=0.001,penalty='l2',C=0.2)
svc.fit(x_train,y_train)

print('Train score:',svc.score(x_train,y_train))


# In[42]:


y_pred=svc.predict(x_test)

from sklearn.metrics import accuracy_score

print('Acuratetea:',accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:




