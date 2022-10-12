#!/usr/env python 3
import numpy as np
import pandas as pd
from numpy import linalg as NO
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

print ("Datas chargées ...\n\n")
#Chargement des datas
Fichier = 'winequality-white.csv'
#df = pd.read_csv(Fichier, sep = ';', header = None, index_col = None, nrows = None).to_numpy()
df = pd.read_csv(Fichier, header = None, index_col = None, nrows = None).to_numpy()
Z = df[:, :-1]
Y = df[:, -1]

z = np.asarray(Z[1:, :]).astype(float)
df = pd.DataFrame(z)
scaler = StandardScaler()
df = scaler.fit_transform(df)
#print ("Matrice X après standardisation des variables : \n\n",df)

X = df
y = Y[1:].astype(float)
XtX = np.transpose(X)@ X
Xinv = np.linalg.inv(XtX)
betaX = Xinv @ np.transpose(X) @ y
print ("Le betaX : \n\n",betaX)

#calcul des résidus avec la norme 2 euclidienne
#(XbetaX - y)²

XbetaX = X @ betaX

residus = np.square(LA.norm(np.subtract(XbetaX,y)))

print ("Résidus question 4 : \n\n",residus)

#ipython question4.py 
#Datas chargées ...

#Le betaX : 

 #[ 0.05528457 -0.18777892  0.00267308  0.41324329 -0.00540194  0.06347717
 #-0.01214247 -0.44944011  0.10362774  0.07206042  0.23807087]
#Résidus question 4 : 

 #171983.33880876756
