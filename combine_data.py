import time
import pandas as pd
import numpy as np



ac = pd.read_csv("Actives_descriptors.csv")
inac = pd.read_csv("Inactives_descriptors.csv")

#Y column labels of 1 and zeros
acc = np.ones((ac.shape[0],1))
inacc = np.zeros((inac.shape[0],1))




#concatenate active and inactive data
la = np.concatenate((acc,inacc),axis=0) #axis can be 0 or 1 where 0 means row and 1 means column

comb = pd.concat([ac,inac],axis=0)

comb["Label"] = la

comb.to_csv("combined_raw_data_whole.csv", index=False)
#print('\nCOMBINED DATASETS')
#print(comb)
#comb=comb.fillna(0)
#comb=comb.drop('name',axis=1)