import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
print('\n STEP 3: DEALING WITH IMBALANCED DATA IF ANY...')
#time.sleep(4)
print('\nTHIS PIE CHART SHOWS THE DISTRIBUTION OF YOUR DATASETS')


import matplotlib.pyplot as plt
ac = pd.read_csv("Actives_descriptors.csv")
inac = pd.read_csv("Inactives_descriptors.csv")

#Y column labels of 1 and zeros
acc = np.ones((ac.shape[0],1))
inacc = np.zeros((inac.shape[0],1))


y = [len(acc),len(inacc)]
mylabels = ["Actives", "Inactives"]

plt.pie(y, labels = mylabels)
plt.show()
print('ACTIVES:',len(acc),'INACTIVES:',len(inacc))
#Actives_size = 17620 Inactives_size = 178,522
#time.sleep(4)
'''


from collections import Counter           
from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("combined_raw_data_whole.csv")

feat = data.drop('Label', axis =1)
#print(feat)

lab= data.Label 

print(Counter(lab))

ros = RandomOverSampler(sampling_strategy="minority")
X_res, y_res = ros.fit_resample(feat, lab)
               
#print('\nOVERSAMPLED DATA')
#print(y_res)
print(Counter(y_res))
#X_res.to_csv('X_res_results.csv')
comb = pd.concat([X_res,y_res],axis=1)

comb.to_csv('whole_data_ROS.csv')



'''
xx=np.array(y_res)
ab=[]
inab=[]

for i in xx:
    if i == 'Active':
        ab.append(1)
    else:
        inab.append(0)
                        
y = [len(ab),len(inab)]
                
print('PIE-CHART OF DATA AFTER OVERSAMPLING')
mylabels = ["Actives", "Inactives"]
plt.pie(y, labels = mylabels)
plt.show()
print('ACTIVES:',len(ab),'INACTIVES:',len(inab))
'''
                
'''
            
            elif i3 == 2:
                rus = RandomUnderSampler(sampling_strategy="not minority")
                X_res, y_res = rus.fit_resample(feat, lab)
                time.sleep(3)
                print('\nDOWNSAMPLED DATA')
                print(X_res)
                
                xx=np.array(y_res)
                ab=[]
                inab=[]

                for i in xx:
                    if i == 'Active':
                        ab.append(1)
                    else:
                        inab.append(0)
                        
                y = [len(ab),len(inab)]
                time.sleep(4)
                print('PIE-CHART OF DATA AFTER DOWNSAMPLING')
                mylabels = ["Actives", "Inactives"]
                plt.pie(y, labels = mylabels)
                plt.show()
                print('ACTIVES:',len(ab),'INACTIVES:',len(inab))

                c = 1;
                
        elif i2 =='n' or i2 == 'no':
            print('DATASET IS STILL IMBALANCED.....LETS PROCEED\n')
            lab.value_counts().plot.pie(autopct='%.2f')
            c = 1
            
            X_res=feat
        else:
            print('INVALID INPUT')
            c=4
time.sleep(2)
X_res=X_res.drop('Label',axis=1)   
#FEATURE SCALING
df = X_res.columns
time.sleep(4)
print('\n STEP 4: FEATURE SCALING')
time.sleep(3)
'''