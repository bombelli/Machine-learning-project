import os
import glob
import time
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow
from tkinter import *
import sklearn as sk
from numpy import loadtxt
from sklearn import metrics
from pandastable import Table
from keras.layers import Dense
from tkinter import filedialog
from keras.models import Sequential
#from tkintertable import TableCanvas
from tkinter import messagebox as msg
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt



print('\n\n SELECT THE TRAINING DATA')
time.sleep(2)
t= filedialog.askdirectory()
os.chdir(t)
Tk().withdraw()
m=filedialog.askopenfilename(initialdir = '/Desktop',      title = '    SELECT TRAINING DATASETS     ')
Train = loadtxt(m,delimiter=',')
n=filedialog.askopenfilename(initialdir = '/Desktop',   title = '       SELECT VALIDATION DATASETS   ')
val = loadtxt(n,delimiter=',')
feat = Train.shape[1]-1
		
x=Train[:,0:feat]
y=Train[:,feat]
x_val=val[:,0:feat]
y_val=val[:,feat]

KUDIDNN = Sequential()
KUDIDNN.add(Dense(180, input_dim=feat,activation='relu'))
KUDIDNN.add(Dense(16, activation='relu'))
KUDIDNN.add(Dense(1, activation='sigmoid'))

time.sleep(2)
KUDIDNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
#Weights implementation
weights ={0:1, 1:50} 

class_weight=weights
KUDIDNN.fit(x,y,epochs=35,batch_size=100,class_weight = class_weight)

KUDIDNN.save("KUDIDNNmod7")
KUDIDNN.save_weights("weights7.h5")

		
_, accuracy = KUDIDNN.evaluate(x_val,y_val)
g = accuracy*100
print('Accuracy: %.2f' %(accuracy*100))
msg.showinfo('MODEL ACCURACY', '%.2f' %g)
		

m=filedialog.askopenfilename(initialdir = '/Desktop',      
					  title = '                         SELECT DATASETS TO TEST     ')
TEST= loadtxt(m,delimiter=',')
print('    RUNNING PROGRAM\n\n')

feat = TEST.shape[1]-1
z = TEST[:,0:feat]
y= TEST[:,feat]
		
TP,FP,TN,FN=0,0,0,0
a,ina=0,0
		
predictions = KUDIDNN.predict_classes(z)
#k=[]
#for i in range(len(z)):
		
for i in range(len(z)):
    if int(y[i])==1:
        a+=1
        if int(predictions[i])==1:
            TP+=1
        if int(predictions[i])==0:
            FN+=1
	
    if int(y[i])==0:
        ina+=1
        if int(predictions[i])==1:
            FP+=1
        if int(predictions[i])==0:
            TN+=1
			
		#print('PREDICTION: ', predictions[i],'ACTUAL: ',y[i])
print('Confusion Matrix')
print(metrics.confusion_matrix(y, KUDIDNN.predict_classes(z)))
print('Classsification Report')
print(metrics.classification_report(y, KUDIDNN.predict_classes(z)))
print('AUC SCORE: ',metrics.roc_auc_score(y, KUDIDNN.predict_proba(z)))
        
fpr, tpr, _ = metrics.roc_curve(y, KUDIDNN.predict_proba(z))
        # Calculate the AUC
roc_auc = metrics.auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)
		# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

plt.subplot(211)
plt.title('Loss')
plt.plot(['loss'], label='train')
plt.plot(['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(['accuracy'], label='train')
plt.plot(['val_accuracy'], label='test')
plt.legend()
plt.show()
        
		
ACC = ((TP+TN)/len(z))*100
SEN = (TP/(TP+FN))*100
SPE = (TN/(TN+FP))*100
		
pre = (TP/TP+FP)
F1_score = (2*((pre*SEN)/(pre+SEN)))
		
num = (TP*TN)-(FP*FN)
		
val = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
den = math.sqrt(val)

		
MCC = (num/den)
		
		

		
		
msg.showinfo('ACCURACY', '%.2f' %ACC)
		
		
msg.showinfo('SENSITIVITY', '%.2f' %SEN)
		
msg.showinfo('SPECIFICITY', '%.2f' %SPE) 
msg.showinfo('F1 SCORE', '%.2f' %F1_score)
		
msg.showinfo('MCC', '%.2f' %MCC) 
print('ACTIVES =',a,' ','INACTIVES =',ina)
print('\n')
		
print('TOTAL SAMPLE =',len(z), 'TP =',TP,'FP =',FP,'TN =',TN,'FN =',FN)
print('ACCURACY:',int(ACC),' ', 'SENSITIVITY:',int(SEN),' ', 'SPECIFICITY:',int(SPE))
    