# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(np.array(X,dtype=float).reshape((10,1)))
y = sc_y.fit_transform(np.array(y,dtype=float).reshape((10,1)))

# Fitting SVR to the dataset
# L'algorithme de SVR ne prends pas en compte le dernier elt
# La classe svr ne fais pas de scaling par defaut donc nous somme obligé de le faire
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#np.ravel pour convertir le tableau en 1d
regressor.fit(X, np.ravel(y))

# Predicting a new result
#il faut faire une iversion etant donné que svr ne fais pas de scaling
#par defaut il faut remettrz tout dans l'ordre
y_pred = regressor.predict(sc_X.transform(np.array([6.5]).reshape((1,1))))
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
