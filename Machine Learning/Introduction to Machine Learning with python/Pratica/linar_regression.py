import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np


"""X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(f'w = {lr.coef_}')
print(f'b = {lr.intercept_}')"""

"""X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)


print(f"Training score: {lr.score(X_train, y_train)}")
print(f"Test score: {lr.score(X_test, y_test)}")"""

"""X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train,y_train)
ridge10 = Ridge(alpha=0.1).fit(X_train, y_train)
print(f"Training set score: {ridge.score(X_train, y_train)}")
print(f"Test sset score: {ridge.score(X_test, y_test)}")
print()
print(f"Training set score: {ridge10.score(X_train, y_train)}")
print(f"Test sset score: {ridge10.score(X_test, y_test)}")"""


X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print(f"Training set score: {lasso.score(X_train, y_train)}")
print(f"Test sset score: {lasso.score(X_test, y_test)}")
print(f"Number of features used: {np.sum(lasso.coef_ != 0)}")