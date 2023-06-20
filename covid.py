import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model                                            
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



import pandas as pd
df=pd.read_csv("country_wise_latest.csv")

print(df)
 

#HEAD OF THE DATASET:
print("Head of the dataset")
print(df.head())

 

#TAIL OF THE DATASET:

print("Tail of the dataset")
print(df.tail())
 
#SHAPE OF THE DATASET:

print("Shape of the dataset")
print(df.shape)
 

#INFORMATION OF THE DATASET:

print("Information of the dataset")
print(df.info())
 
#Finding the Count of the dataset
print("Count the values:",df.count())

 

#Finding the Missing values in the dataset
print("missing values in the dataset:")
print(df.isnull().sum())
 
#Finding the Duplicated Items of the dataset
d=df[df.duplicated()]
print("Duplicate entries:")
print(d)

 
#descriptive statistics
print("Mean=\n",df.mean())
print("Median=\n",df.median())
print("Variance=\n",df.var())
print("Standard deviation=\n",df.std())
print("Maximum value=\n",df.max())
print("Minimum value=\n",df.min())

 


#Finding the Interquartile range of the dataset
print("Interquartile=",df.quantile())

 


#Aggregate functions
x=df.aggregate(["sum"])
print(x)
y=df.aggregate(["max"])
print(y)
z=df.aggregate(["mean"])
print(z)
s=df.aggregate(["sem"])
print(s)
p=df.aggregate(["var"])
print(p)
q=df.aggregate(["prod"])
print(q)

 
 

#Skewness
print(df.skew())
 

#THREE DIMENSIONAL PLOTTING:

import seaborn as sns
fig=plt.figure()
ax=plt.axes(projection='3d')
x=df['Deaths']
y=df['Recovered']
z=df['Confirmed']
ax.plot3D(x,y,z,'purple')
ax.set_title('covid-19 dataset')
plt.show()

 


#LINE PLOT:

plt.plot(df.Deaths,df.Confirmed)
plt.title("Death vs Confirmed ")
plt.xlabel("death")
plt.ylabel("Confirmed ")
plt.show()

 

#PAIRPLOT:
sns.pairplot(data=df)
plt.show()
 

#HISTOGRAM:
plt.hist(df.Deaths,bins=30)
plt.title("death")
plt.xlabel("death")
plt.show()
 
#SUBPLOT:
df.plot(kind='box',subplots=True,layout=(5,3),figsize=(12,12))
plt.show()
 
#DENSITYPLOT:
f,ax=plt.subplots(figsize=(10,6))
x=df['Deaths']
ax=sns.kdeplot(x,shade=True,color='r')
plt.show()

 
#HEATMAP:
sns.heatmap(df.corr())
plt.show()
 
#Linear Regression
x=df['Confirmed']
y =df['Deaths']


#HEAD :
x.head()
 
y.head()
 
#SHAPE OF X AND Y:
x.shape

y.shape

#RESHAPING:
x=x.values.reshape(-1,1)
x.shape


#SPLITTING X AND Y VALUES:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.99,random_state=100)
print(x.shape)
print(y.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



#FIND LINEAR REGRESSION:

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
LinearRegression()

regr.coef_


regr.intercept_


plt.scatter(x_train, y_train)
plt.plot(x_train,608.701 + 0.034*x_train, 'r')
plt.show()
 

y_pred = regr.predict(x_test)
res = (y_test - y_pred)

r_squared = r2_score(y_test, y_pred)
r_squared


print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
print('Mean Absolute Error: %.2f'% mean_absolute_error(y_test, y_pred))




#MULTIPLE REGRESSION:

X=df[['Confirmed','Deaths','Recovered']]
Y=df['Active']

#SPLITTING  X AND Y VALUES:

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)

reg=linear_model.LinearRegression()

#FITTING THE REGRESSION:
reg.fit(X_train,Y_train)
Y_predict=reg.predict(X_test)

#COEFFICIENT:
print('Coefficients:',reg.coef_)


#VARIANCE SCORE:
print('Variance score:{}'.format(reg.score(X_test,Y_test)))


#R^2 VALUE:
from sklearn.metrics import r2_score
print('r^2:',r2_score(Y_test,Y_predict))


#ROOT MEAN SQUARED ERROR:

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,Y_predict)
rmse=np.sqrt(mse)
print('RMSE:',rmse)



