from sklearn import datasets

wine= datasets.load_wine()

##print(wine.feature_names)
##print(wine.target_names)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

clsfr=KNeighborsClassifier(n_neighbors=10)
clsfr.fit(x_train,y_train)

result=clsfr.predict(x_test)

from sklearn import metrics
print('Accuracy:',metrics.accuracy_score(y_test,result))


