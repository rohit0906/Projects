import pandas as pd
import numpy as np
data=pd.read_csv('BrainTumorData.csv',delimiter=",")

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data['diagnosis']=le.fit_transform(data['diagnosis'])

data1=data.drop(['id','diagnosis','Unnamed: 32'],axis=1)
print(data1.describe())
X=np.array(data1)

'''from sklearn.decomposition import PCA

pca=PCA(n_components=2)
X_reduced=pca.fit_transform(X)
X_reduced=pd.DataFrame(data=X_reduced, columns=['x1','x2'])

print(X_reduced.describe())

import matplotlib.pyplot as plt

plt.scatter(X_reduced['x1'],X_reduced['x2'],c=data['diagnosis'])
plt.title("PCA figure n_components=2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

data1.hist()
plt.show()'''

from sklearn.model_selection import train_test_split
y=np.array(data['diagnosis'])

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

from sklearn. import
model=SVC()
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))