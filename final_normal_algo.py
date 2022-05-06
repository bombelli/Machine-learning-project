from numpy import mean
from numpy import std
import numpy
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

data = pd.read_csv("combined_raw_data_whole_cleaned.csv")
X = data.drop('Label', axis =1)
y= data.Label

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

def train_val_data(X, y):
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X)

	ros = RandomOverSampler(sampling_strategy="minority")
	#rus = RandomUnderSampler(sampling_strategy="not minority")
	X_ros,y_ros = ros.fit_resample(X, y)

	return X_ros, y_ros
'''
def test_data(X_test, y_test):
	scaler_test = StandardScaler()
	X_test = scaler_test.fit_transform(X_test)

	return X_test, y_test

X_train_tt, y_train_tt = train_val_data(X_train,y_train)
'''

X_ros , y_ros = train_val_data(X,y)

#model = RandomForestClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier()
#model = SVC()
#model = AdaBoostClassifier()
#model = XGBClassifier()
model = LogisticRegression()

scoring = {'accuracy' : make_scorer(accuracy_score), 
		   'precision' : make_scorer(precision_score),
		   'recall' : make_scorer(recall_score), 
		   'f1_score' : make_scorer(f1_score),
		   'AUC': make_scorer(roc_auc_score)}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=1)

scores = cross_validate(model, X_ros, y_ros, scoring=scoring, cv=cv, n_jobs=-1)

# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
res_df = pd.DataFrame(scores)
print(res_df)
print("Mean times and scores:\n", res_df.mean())


'''
model.fit(X_train_tt, y_train_tt)

X_test, y_test = test_data(X_test,y_test)

# predict probabilities for test set
yhat_probs = model.predict(X_test)

# predict crisp classes for test set
#yhat_classes = model.predict_classes(X_test, verbose=0)

#yhat_classes = (model.predict(X_test) > 0.5).astype("int32")




# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_probs)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_probs)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_probs)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_probs)
print('F1 score: %f' % f1)

# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
'''