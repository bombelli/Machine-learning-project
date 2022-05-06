import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("whole_data_rus.csv")
X = data.drop('Label', axis =1)

Y= data.Label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

X_train["Label"] = Y_train

X_train.to_csv("Train_data_rus.csv")

X_test["Label"] = Y_test

X_test.to_csv("Test_data_rus.csv")