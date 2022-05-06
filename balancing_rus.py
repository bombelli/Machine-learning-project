import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter           
#from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("combined_raw_data_whole.csv")

feat = data.drop('Label', axis =1)
#print(feat)

lab= data.Label 

print(Counter(lab))

rus = RandomUnderSampler(sampling_strategy="not minority")

#ros = RandomOverSampler(sampling_strategy="minority")

X_res, y_res = rus.fit_resample(feat, lab)
               
#print('\nOVERSAMPLED DATA')
#print(y_res)
print(Counter(y_res))

print(y_res)
comb = pd.concat([X_res,y_res],axis=1)

comb.to_csv('whole_data_rus.csv')


#la = np.concatenate((acc,inacc),axis=0)
