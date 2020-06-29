import pandas as pd
import numpy as np

hnames = ['preg', 'plas', 'pres', 'skin',
          'test', 'mass', 'pedi',
          'age', 'class' ]

dataframe =pd.read_csv('indians-diabetes.data.csv', names=hnames, delimiter=",", index_col=False)
print(dataframe.head())



print(dataframe.describe())

print(dataframe.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

#print(dataframe.corr())
fig, ax=plt.subplots()
sns.heatmap(dataframe.corr(), annot=True, ax=ax)
'''sns.pairplot(dataframe,vars=['plas','mass','age'],hue='class')
plt.show()

fig, ax=plt.subplots()
for column in dataframe:
    ax.plot(dataframe[column],label=column)


plt.legend()
'''


from sklearn.model_selection import train_test_split

X=np.array(dataframe.drop(['class'],axis=1))
y=np.array(dataframe['class'])

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=12)

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train,y_train)
print("Accuracy train set",model.score(X_train,y_train))
print("Accuracy test set",model.score(X_test,y_test))

'''


Accuracy train set 0.760586319218241
Accuracy test set 0.8181818181818182
'''
plt.show()