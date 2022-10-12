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


#ipython question3.py 
#Datas chargées ...

#Le betaZ : 

# [-5.05906229e-02 -1.95851023e+00 -2.93492412e-02  2.49883984e-02
# -9.42582369e-01  4.79078658e-03 -8.77630817e-04  2.04204607e+00
 # 1.68395142e-01  4.16453560e-01  3.65633380e-01]
#Résidus question 3 : 

 #2794.3435114292647
