#!/usr/bin/env python
# coding: utf-8

# In[360]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[361]:


df = pd.read_csv("C:\\Users\\sudhi\\Downloads\\IRIS.csv")
df


# In[362]:


df.info()


# In[363]:


df.isnull().sum()


# In[364]:


df.describe


# In[365]:


df.shape


# In[366]:


df.dtypes


# In[367]:


df.corr()


# In[368]:


sns.heatmap(df.corr())


# In[369]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[370]:


df.iloc[:,-1] = le.fit_transform(df.iloc[:,-1])
df.iloc[:,-1]


# In[371]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[372]:


x = df.drop(["species"],axis=1)
y = df["species"]
y.unique()


# In[373]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[374]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.75,test_size=0.25)


# In[375]:


x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


# In[376]:


from sklearn.linear_model import LogisticRegression


# In[377]:


model = LogisticRegression()


# In[378]:


model.fit(x_train,y_train)


# In[379]:


y_pred = model.predict(x_test)
y_pred


# In[380]:


np.array(y_test)


# In[381]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[382]:


accuracy_score(y_test,y_pred)


# In[383]:


confusion_matrix(y_test,y_pred)


# In[384]:


sns.regplot(x="Actual",y="Predicted",data={"Actual":y_test,"Predicted":y_pred},logistic=True)


# In[385]:


df1 = pd.read_csv("C:\\Users\\sudhi\\Downloads\\IRIS.csv")
df1


# In[386]:


x = df1.drop(["species"],axis=1)
y = df1["species"]


# In[387]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.25,train_size=0.75)


# In[388]:


from sklearn.neighbors import KNeighborsClassifier


# In[389]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[390]:


knn.fit(x_train,y_train)


# In[391]:


y_pred = knn.predict(x_test)
y_pred


# In[392]:


y_test


# In[393]:


accuracy_score(y_test,y_pred)


# In[394]:


confusion_matrix(y_test,y_pred)


# In[395]:


df2 = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df2


# In[396]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[397]:


dt = DecisionTreeClassifier(criterion="entropy")


# In[398]:


dt.fit(x_train,y_train)


# In[399]:


y_pred = dt.predict(x_test)
y_pred


# In[400]:


accuracy_score(y_test,y_pred)


# In[401]:


confusion_matrix(y_test,y_pred)


# In[402]:


# How Decision Tree is plotted
plt.figure(figsize=(15,8))
tree.plot_tree(dt,filled=True)


# In[ ]:





# In[ ]:




