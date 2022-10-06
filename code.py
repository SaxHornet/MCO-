###Question 3 et 4 
#!/usr/env python 3
import numpy as np
import pandas as pd
import statsmodels.api as sm

#Chargement des datas
dataset = pd.read_csv('winequality-white.csv')
#print (dataset.head())

#Création des variables
#Z = matrice Z
#y = vecteur y
Z = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #the relevant column name
y = dataset['quality'] #Variable indépendante est QUALITY dans le projet 2

print ("Dataset in running....\n\n\n")
print ("Dataset is .....\n\n\n")
print (Z,y) #affichage du dataset final

#Question 3 -calcul des résidus
#normalement calcul à la main, cf sujet !!
print ("Calulating in progress.......\n\n\n")
print ("=============================\n\n\n")
x = sm.OLS(y,sm.add_constant(Z)).fit()
print ("== The Rsquared -r²- for the question 3 is : \n\n\n" + str(x.rsquared))
print ("== All results of the regression including the r² are : \n\n\n", x.summary())

#Question 4 - Donner la valeur de R²
#Implémentation de la matrice Z
#Matrice X est son résulat, soit X = Z

#ajout de de la variable X
print ("Calulating in progress.......\n\n\n")
print ("=============================\n\n\n")
X = sm.OLS(y,sm.add_constant(Z)).fit()
print ("=== The Rsquared --r²- for the question 4 is : \n\n\n" + str(X.rsquared))
print ("== All results of the regression including the r² are : \n\n\n", X.summary())

####Vérification avec une autre librairie,sckit-learn
#!/usr/env python 3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Chargement des datas
dataset = pd.read_csv('winequality-white.csv')
#print (dataset.head())

#Création des variables
#Z = matrice Z
#y = vecteur y
Z = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] #the relevant column name
y = dataset['quality'] #Variable indépendante est QUALITY dans le projet 2

# Implémentation avec sckit-learn
# initialisation du modèle
regression_model = LinearRegression()
# Adapter les données (entraînement du modèle)
regression_model.fit(Z, y)
# Prédiction
y_predicted = regression_model.predict(Z)
# Évaluation du modèle
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
# Affichage des valeurs
print("Pente : " ,regression_model.coef_)
print("Ordonnée à l'origine : ", regression_model.intercept_)
print("Racine carrée de l'erreur quadratique moyenne - RMSE : ", rmse)
print('Sccore R2 : ', r2)
