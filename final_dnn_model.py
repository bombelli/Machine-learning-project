from numpy import mean
from numpy import std
import numpy
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def create_model(X_train_tt, X_val, y_train_tt, y_val):
	# create model
	model = Sequential()
	model.add(Dense(256, input_dim=622,kernel_initializer='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(128,kernel_initializer='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
		# Compile model
	optimizer = SGD(learning_rate=0.001, momentum=0.8)	
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.fit(X_train_tt, y_train_tt, epochs=75, batch_size=256, validation_data=(X_val, y_val))
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load dataset
data = pd.read_csv("combined_raw_data_whole_cleaned.csv")
X = data.drop('Label', axis =1)
y= data.Label

#Splitting data into various sections
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

def train_val_data(X_train, y_train):
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)

	#ros = RandomOverSampler(sampling_strategy="minority")
	rus = RandomUnderSampler(sampling_strategy="not minority")
	X_res, y_res = rus.fit_resample(X_train, y_train)

	X_train_tt, X_val, y_train_tt, y_val = train_test_split(X_res,y_res, test_size=0.1)

	return X_train_tt,X_val,y_train_tt,y_val

def test_data(X_test, y_test):
	scaler_test = StandardScaler()
	X_test = scaler_test.fit_transform(X_test)

	return X_test, y_test

#model = KerasClassifier(build_fn=create_model)

X_train_tt, X_val, y_train_tt, y_val = train_val_data(X_train,y_train)

model = create_model(X_train_tt, X_val, y_train_tt, y_val)


X_test, y_test = test_data(X_test,y_test)

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)

# predict crisp classes for test set
#yhat_classes = model.predict_classes(X_test, verbose=0)

yhat_classes = (model.predict(X_test) > 0.5).astype("int32")


# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)



