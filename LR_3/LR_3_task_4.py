import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict((Xtest))

# regr.coef_
print("Coefficients:\n", regr.coef_)
# regr.intercept_
print("Intercept:\n", regr.intercept_)
# r2_score
print('Variance score: %.2f' % r2_score(ytest, ypred))
# mean_squared_error
print("Mean squared errorr: %.2f" % mean_squared_error(ytest, ypred))
# mean_absolute_error
print("Mean absolute errorr: %.2f" % mean_absolute_error(ytest, ypred))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()