import pandas as pd
import numpy as np

datasets= pd.read_csv('2.0 Salary_Data.csv')

x = datasets.iloc[:, :-1].values #makes a 2d array
y = datasets.iloc[:, 1].values  #makes simple array

##retrieving rows by iloc method 
#values to get only values to an array

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
clsfr=LinearRegression()
clsfr.fit(x_train,y_train)

result=clsfr.predict(x_test)

print('test values:',y_test)
print('predicted values:',result)



import matplotlib.pyplot as plt

plt.scatter(x_train,y_train,color='red')
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_train, clsfr.predict(x_train), color = 'blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
