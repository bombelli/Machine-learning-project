from sklearn.model_selection import StratifiedKFold,KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,cross_val_score, StratifiedKFold,RandomizedSearchCV

def create_baseline():
 

  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  return model

# Nested k-fold cross-validation (Subject_dependent)


#train/validation/test=0.8/0.2/0.2
inner_cv = StratifiedKFold(n_splits = 4,shuffle=True,random_state=42)
outer_cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=42)

accuracy=[]
p_grid=[]
estimators=[]
#p_grid={'batch_size':[400,800]}

from sklearn.preprocessing import LabelEncoder

#def get_new_labels(y):
    #y = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    #return y
#y = get_new_labels(y)

for train_index, test_index in outer_cv.split(x,y):
    print('Train Index:',train_index,'\n')
    print('Test Index:',test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes) 

    grid = RandomizedSearchCV(estimator=estimator,
                                param_distributions=p_grid,
                                cv=inner_cv,                            
                                refit='roc_auc_scorer',
                                return_train_score=True,
                                verbose=1,n_jobs=-1,n_iter=20)
    grid.fit(x_train, y_train)
    estimators.append(grid.best_estimator_)
    prediction = grid.predict(x_test)
    accuracy.append(grid.score(x_test,y_test))
    print('Accuracy:{}'.format(accuracy))

