# -*- coding: utf-8 -*-
#House_Price_Prediction.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('homeprices (1).csv')

df.head()

plt.scatter(df['area'],df['price'])
plt.xlabel("area")
plt.ylabel("price (in K)")
plt.title("area Vs price")

X=df.iloc[:,0:1]
Y=df.iloc[:,-1]

X

Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,Y_train)

X_test

Y_test

lr.predict(X_test.iloc[0].values.reshape(1,1))

plt.scatter(df['area'],df['price'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel("area")
plt.ylabel("price (in K)")
plt.title("area Vs price")

m = lr.coef_

c=  lr.intercept_

m

c

# y = mx + c
m*5320 + c

