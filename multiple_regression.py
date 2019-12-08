# Il existe 5 methodes pour optimiser un modele de prediction multi lineaire
#All-in
#backawrd elimination       ==>
#forward elimination        ==> step wise elimination 
#bidirection elimination    ==>
#score comparison

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorial data
#encoding the independant variable
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoid dummy trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""
#Fitting multiple Linear regression to training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set result

y_pred = regressor.predict(X_test)
############################################################################""
#backward elimination
############################################################################""
#build the optimal model using Backward Elimination
#p-value est une mesure statistique qui aide les scientifiques à déterminer si 
#leurs hypothèses sont correctes ou non. Les valeurs de P sont utilisées pour 
#déterminer si les résultats de leur expérience se situent dans la plage 
#normale de valeurs pour les événements observés. Habituellement, si la valeur 
#P d'un ensemble de données est inférieure à un certain montant prédéterminé 
#(comme, par exemple, 0,05), les scientifiques rejetteront "l'hypothèse nulle" 
#de leur expérience - en d'autres termes, ils excluront l'hypothèse que les 
#variables de leur expérience n'ont eu aucun effet significatif sur les 
#résultats. Aujourd'hui, les valeurs de p sont généralement trouvées sur une 
#table de référence en calculant d'abord une valeur de chi carré .
#https://www.wikihow.com/Calculate-P-Value pour plus de détail sur p-value

#build the optimal model possible

#add remove parameter with np.ones vector qui va servir de constant pour notre model
#sans ca nous ne pouvons pas calculer la valeur du p
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape=(50,1)).astype(int), values =X, axis=1)

#backward elimintation
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

#On supprime la variable ayant le p-value le plus grand while(en boucle jusqu'a condition)
#si la p-value <0.05 stop
#fin model (ready)
#sinon recommencer l'operation avec la valeur la plus petites
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

"""
Backward Elimination with p-values only:

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""

##############################################################################""
#forward selection
##############################################################################""

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

#selectionner l'élément avec le p-value le plus petit
#ici c'est x3
#puis ajouter un predictor a celui-ci
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()
#if p > 0.05 reprends en ajouter une nouvelle variable et aussi en gardant la variable
#avec le p-value le plus petit
#sinon end model est prêt

##############################################################################""
#Bidirection elimination
##############################################################################""
#Faire toutes les etapes de forward elimination
#faire toutes les etapes de backward elimination (puis retourné forward elimination)
#ainsi de suite jusqu'à ce qu'aucune variable ne puisse être ajouté ou enléve
#dans notre model

##############################################################################""
#all-in
##############################################################################""
#Ici j'utilise la librairie seaborn pour faire mon graphe
#j'essaie de voir laquel de mes variables independent me permet d'avoir un bon model
#et a une bone correlation avec le profit
import seaborn as sns

plt.figure(figsize=(7,4)) 
sns.heatmap(dataset.corr(), annot=True,cmap='cubehelix_r')
plt.show()
#selon le contact qu'on fait on realise que la premiere colonne a une correlation
#tres forte avec notre profit
#la seconde est negligable
#la troisieme peut etre prise en compte car elle a aussi une forte corrélation avec notre profit
#etait donné que la corrélation de la premiere colonne et la premier colonne est plus forte
#nous pouvons l'utiliser seul pour notre model

############################################################################""
#score comparison
############################################################################"" 
#https://www.researchgate.net/publication/318467372_Multiple_Linear_Regression_using_Python_Machine_Learning
