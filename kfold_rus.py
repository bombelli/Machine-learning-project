import pandas as pd
import numpy as np # correct data preparation for model evaluation with k-fold cross-validation
from sklearn import model_selection
from numpy import mean
from numpy import std
#from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline



data = pd.read_csv("whole_data_rus_cleaned.csv")

X = data.drop('Label', axis =1)

y= data.Label

# define the pipeline
steps = list()
steps.append(('scaler', StandardScaler()))
#steps.append(('model', LogisticRegression()))
#steps.append(('model', XGBClassifier()))
#steps.append(('model', RandomForestClassifier()))
#steps.append(('model', KNeighborsClassifier()))
#steps.append(('model', GaussianNB()))
#steps.append(('model', DecisionTreeClassifier()))
#steps.append(('model', SVC()))
steps.append(('model', AdaBoostClassifier()))
pipeline = Pipeline(steps=steps)

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'MCC': make_scorer(matthews_corrcoef)}
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model using cross-validation

scores = model_selection.cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
# report performance

#print(scores.mean())

res_df = pd.DataFrame(scores)
print(res_df)
print("Mean times and scores:\n", res_df.mean())


#print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))



'''
print('Accuracy: %.3f' % (accuracy*100))
print('Sensitivity: %.3f' % (recall*100))
print('precision: %.3f' % (precison*100))
print('F1_score: %.3f' % (f1_score*100))
'''