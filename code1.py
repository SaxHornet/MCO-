#!/usr/env python 3
import numpy as np
import pandas as pd
from numpy import linalg as NO
from numpy import linalg as LA

print ("Datas chargées ...\n\n")
#Chargement des datas
Fichier = 'winequality-white.csv'
#df = pd.read_csv(Fichier, sep = ';', header = None, index_col = None, nrows = None).to_numpy()
df = pd.read_csv(Fichier, header = None, index_col = None, nrows = None).to_numpy()
Z = df[:, :-1]
Y = df[:, -1]

z = np.asarray(Z[1:, :]).astype(float)
y = Y[1:].astype(float)
ZtZ = np.transpose(z)@ z
Zinv = np.linalg.inv(ZtZ)
betaZ = Zinv@np.transpose(z)@ y
print ("Le betaZ : \n\n",betaZ)

#calcul des résidus avec la norme 2 euclidienne
#(ZbetaZ - y)²

ZbetaZ = z @ betaZ

residus = np.square(LA.norm(np.subtract(ZbetaZ,y)))

print ("Résidus question 3 : \n\n",residus)
