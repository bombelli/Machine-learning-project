import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Test_data_rus.csv")

comb = data.drop('Label', axis =1)

lab= data.Label

comb=comb.fillna(0)
comb=comb.drop('name',axis=1)


var=float(input('ENTER A CUTOFF VARIANCE: '))
Sel =  VarianceThreshold(threshold=var)
Sel.fit(comb)
feat = comb[comb.columns[Sel.get_support(indices=True)]]

print(feat)

feat["Label"] = lab

feat.to_csv("Test_data_rus_cleaned.csv", index = False)
'''
ss=StandardScaler()
dat=ss.fit_transform(X_res)
dat=pd.DataFrame(dat,columns=df)
print(dat)

comb = pd.concat([ac,inac],axis=0)
'''       


