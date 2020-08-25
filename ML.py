import pandas as pd
import numpy as np

#Machine learning validation
def fit_predict(clf, x_train, y_train, x_test, y_true, use_second_model=False, clf_sec=None, x_train_sec=pd.DataFrame(), y_train_sec=pd.DataFrame(),
                mode='normal', use_weight=False, early_stop_num=None):
    #Change the objective variable to integer type
    y_train = y_train.astype('int')
    y_true = y_true.astype('int')
    if use_second_model: y_train_sec = y_train_sec.astype('int') #For second model       
    
    #Calculate weight
    if use_weight:
        w = max(np.bincount(y_train))/np.bincount(y_train)
        w_train = []
        for i in list(y_train):
            w_train.append(w[i])
    
        #For second model
        if use_second_model:
            w = max(np.bincount(y_train_sec))/np.bincount(y_train_sec)
            w_train_sec = []
            for i in list(y_train_sec):
                w_train_sec.append(w[i])
    #Not use weight
    else:
        w_train = None
        if use_second_model: w_train_sec = None
    
    #Training model
    mode = 'normal' if mode!='xgb' else mode
    #"Normal" mode
    if mode=='normal':
        clf.fit(x_train, y_train)
        if use_second_model: clf_sec.fit(x_train_sec, y_train_sec)
        
    #"XGBoost" mode
    if mode=='xgb':
        eval_set = None if early_stop_num==None else [(x_test, y_true)]
        clf.fit(x_train, y_train, sample_weight=w_train, early_stopping_rounds=early_stop_num, eval_set=eval_set, 
                verbose=0)
        if use_second_model: clf_sec.fit(x_train_sec, y_train_sec, sample_weight=w_train_sec,
                                         early_stopping_rounds=early_stop_num, eval_set=eval_set, verbose=0)
    
    #Predict
    proba_both = clf.predict_proba(x_test)
    if use_second_model: proba_both_sec = clf_sec.predict_proba(x_test)
    
    
    return proba_both if not use_second_model else np.stack([proba_both, proba_both_sec])
    
