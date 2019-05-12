from sklearn import datasets  #importing datasets
iris=datasets.load_iris() #loading iris flower data set into iris

data=iris.data  #iris flower data, 150*4 array will be loaded to data
target=iris.target  #iris flower targets, 150*1 array will be loaded to data


from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)


from sklearn.neighbors import KNeighborsClassifier  #load KNN classifier
clsfr=KNeighborsClassifier(n_neighbors=3) #KNN classifier with k value=3
clsfr.fit(train_data,train_target)

result=clsfr.predict(test_data)

print('Predicted',result)
print('Actual',test_target)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(test_target,result)
print('accuracy:',accuracy)

from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # Axes 3D used for 3 axis graphs

fig=plt.figure()  #initialize 3D graph
ax=fig.add_subplot(111,projection='3d') #adding 3 axes to fig graph, 111-xyz true

for i in range(0,len(train_target)):
    if(train_target[i]==0):     # if target is setosa
        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='g')

    elif(train_target[i]==1):   # if target is verginica
        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='r')
    
    elif(train_target[i]==2):   # if target is versicolor
        ax.scatter(train_data[i][0],train_data[i][1],train_data[i][2],c='b')

print(result[0],test_target[0])
ax.scatter(test_data[0][0],test_data[0][1],test_data[0][2],c='c',marker='x')

plt.show()