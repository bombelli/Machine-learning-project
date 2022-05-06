# Use scikit-learn to grid search the batch size and epochs
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
# Function to create model, required for KerasClassifier

def create_model(hidden_layers=1):
  #with tf.device('/cpu:0'):
  # create model
  model = Sequential()
  model.add(Dense(256, input_dim=622,kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0.5))
  #model.add(Dense(neurons,kernel_initializer='normal', activation='relu'))

  for i in range(hidden_layers):
      # Add one hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

  model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
    # Compile model
  #optimizer = SGD(learning_rate=learn_rate, momentum=momentum) 
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load dataset
data = pd.read_csv("whole_data_rus_cleaned.csv")
X = data.drop('Label', axis =1)
y= data.Label
#Preprocess the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

Xm_train, Xm_test, Ym_train, Ym_test = train_test_split(X, y, test_size=0.1, random_state=1)
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=256, verbose=0)
# define the grid search parameters
#batch_size = [16,32]
#epochs = [50,75,100,125,150,175,200]
#param_grid = dict(batch_size=batch_size, epochs=epochs)
#optimizer = ['Adam','SGD','Adagrad']
#param_grid = dict(optimizer=optimizer)
#learn_rate = [0.0001, 0.001, 0.01]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

#param_grid = dict(learn_rate=learn_rate, momentum=momentum)
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#init_mode = [ 'uniform', 'zero', 'normal']
#activation = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid', 'linear']
#dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
#neurons = [32, 64, 128, 256, 512, 1024]
hidden_layers = [3,4,5,6,7]
param_grid = dict(hidden_layers=hidden_layers)

grid = GridSearchCV(estimator=model,param_grid=param_grid, cv=5)
grid_result = grid.fit(Xm_train, Ym_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))
