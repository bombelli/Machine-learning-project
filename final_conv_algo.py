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
from sklearn.linear_model import SGDClassifier

data = pd.read_csv("combined_raw_data_whole_cleaned.csv")
X = data.drop('Label', axis =1)
y= data.Label

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

def train_val_data(X_train, y_train):
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)

	ros = RandomOverSampler(sampling_strategy="minority")
	#rus = RandomUnderSampler(sampling_strategy="not minority")
	#X_rus, y_rus = rus.fit_resample(X_train, y_train)
	X_ros, y_ros = rus.fit_resample(X_train, y_train)

	#X_train_tt, X_val, y_train_tt, y_val = train_test_split(X_res,y_res, test_size=0.1)

	return X_rus, y_rus

def test_data(X_test, y_test):
	scaler_test = StandardScaler()
	X_test = scaler_test.fit_transform(X_test)

	rus = RandomUnderSampler(sampling_strategy="not minority")
	X_rus_test, y_rus_test = rus.fit_resample(X_test, y_test)

	return X_rus_test, y_rus_test

X_rus, y_rus = train_val_data(X_train,y_train)

#model = RandomForestClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier()
#model = SVC(random_state= 1)
#model = AdaBoostClassifier()
#model = XGBClassifier()
#model = LogisticRegression()
model = SGDClassifier(random_state = 1, n_jobs = -1)

model.fit(X_rus, y_rus)

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

