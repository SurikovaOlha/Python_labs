import numpy as np
import pandas as pd
import mglearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[1, 2, 3],[4, 5, 6]])
print("x:\n{}".format(x))

eye = np.eye(4)
print("Масив NumPy:n\{}".format(eye))

iris_dataset = load_iris()
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назви ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("Перші п'ять рядків масиву data:\n{}".format(iris_dataset['data'][:5]))

print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Форма масиву target: {}".format(iris_dataset['target'].shape))
print("Відповіді:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("Форма масиву X_train: {}".format(X_train.shape))
print("Форма масиву y_train: {}".format(y_train.shape))
print("Форма масиву X_test: {}".format(X_test.shape))
print("Форма масиву y_test: {}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массиву X_new: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Прогнози для тестового набору:\n{}".format(y_pred))
print("Правильність тестового набору: {:.2f}".format(np.mean(y_pred == y_test)))
