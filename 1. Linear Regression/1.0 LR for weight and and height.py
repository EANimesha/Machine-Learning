train_data=[72,70,75,62,55,58,65,45,55,60,55,76,77,55,105,74,82,90,66,63,73,61,59,50,60,80,70,60,47]
train_target=[177,171,168,162,160,162,168,148,160,170,152,171,179,151,162,153,165,180,165,180,165,167,176,182,160,160,180,155,176]

print(len(train_data))
print(len(train_target))

import numpy as np 

train_data=np.array(train_data)
train_target=np.array(train_target)

train_data=train_data.reshape(len(train_data),1)  # make a 2d array with set of one element array list
#requirement in sklearn. train_data should be a 2D array

#print(train_data)

from sklearn.linear_model import LinearRegression

Algo=LinearRegression()  #load linear regression to the black box named Algo

Algo.fit(train_data,train_target) # training the ML-algorithm

test_data=[[84]]

result=Algo.predict(test_data)
 
print(result)

from matplotlib import pyplot as plt 

plt.scatter(train_data,train_target)
plt.show()