# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
"""
#import des librairies
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#import du dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3]

#Gerer les données manquantes Mais de la vielle ecole
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#LA NOUVELLE ECOLE UTILISE
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,1:3] = imp_mean.fit_transform(X[:,1:3])

#encoder les données texte en donnée numérique MAIS C4EST DE LA VIELLE ECOLE MAINTENANT
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:,0])

#Ajouter les dummy données pour ne pas que le programme pense que certaines données sont plus
#important que d'autres données

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#LA NOUVELLE ECOLE UTILISE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype = np.float)

#Separation des données en données d'entraitenement et de test
#sklearn.cross_validation est maintenant sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#METTRE LES DONNEES A L'ECHELLE
#c'est a dire mettre les différentes variables utilisées a la même echelle
#pour eviter des problèmes d'incompréhension par notre models de data_science
#les valeurs a l'echelle sont entre -1 et +1
#pour la mise a l'echelle on utilise la methode ecludienne
#la formule de standardisation
#x_stand = (x- mean(x))/standard_deviation(x) ou variation
#la formule de normalisation
#x_norm = (x - min(x))/(max(x) - min(x))
#
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#pour les données de test nous n'avons plus besoin de la fonction fit
#parce que la fonction fit lui a deja ete appliqué depuis les données entrainer
X_test = sc_X.transform(X_test)

#Pour plus de connaisssance
#EST IL IMPORTANT DE METTRE A L4ECHELLE LES DONN2ES DUMMY ?
#lA REPONSE EST CELA DEPEND DU CONTEXTE
#CELA DEPEND DE COMBIEN DE FOIS VOUS VOULEZ GARDEZ LES INTERPRETATIONS DE VOTRE MODELS
#pARCE QUE SI VOUS METTEZ TOUT CELA A L4ECHELLE CELA VA ETRE TRES INTERESSANT ET BON POUR VOS PREDICTIONS
#CAR TOUT SERA A LA MEME ECHELLE MAIS VOUS PERDREZ L4INTERPRETATION DE SAVOIR QUELLE OBSERVATION APPARTIENT A QUELLE DONNEES CATEGORIQUE

"""
PS: QUELQUES CHOSE D'IMPORTANT A SOULIGNER
SI LE MODEL N4EST PAS BAS2 SUR LES DISTANCES EUCLUDIENNNE ET QUE L4ALGORITHME UTILISE
EST L4ALGORITHME DE CONVERGENCE PLUS RAPIDE VOUS POURRIEZ NEANMOINS AVOIR BESOIR DE
DE METTRE A L4ECHELLE LES DONN2ES CAR CELA RISQUE PLUS OU MOINS DE VOUS PRENDRE PLUS DE TEMPS
CAS D4UN ARBRE DE DECISION QUI N4UTILISE PAS LES DISTANCES EUCLIDIENNES
"""
