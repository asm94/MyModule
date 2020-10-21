import pandas as pd
import numpy as np
import shap
import pydotplus
from sklearn import tree
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

#Machine learning validation
def fit_predict(clf, x_train, y_train, x_test, y_true, use_second_model=False, clf_sec=None, x_train_sec=pd.DataFrame(), y_train_sec=pd.DataFrame(),
                mode='normal', use_weight=False, early_stop_num=None, cat_feature=None, normal_class=None):
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
    
    #one class mode
    if mode=='lof' or mode=='ocsvm':
        if normal_class == None:
            count = y_train.value_counts()    
            normal_class = count[count==count.max()].iloc[0].index
            
        tgt_train = x_train[y_train==normal_class]
        clf.fit(tgt_train)
        proba = clf.decision_function(x_test)
        
        if use_second_model:
            tgt_train_sec = x_train_sec[y_train_sec==normal_class]
            clf_sec.fit(tgt_train_sec) 
            proba_sec = clf_sec.decision_function(x_test)
            
        return proba if not use_second_model else np.stack([proba, proba_sec])
        
    #"XGBoost" mode
    elif mode=='xgb':
        eval_set = None if early_stop_num==None else [(x_test, y_true)]
        clf.fit(x_train, y_train, sample_weight=w_train, early_stopping_rounds=early_stop_num, eval_set=eval_set, 
                verbose=0)
        if use_second_model: clf_sec.fit(x_train_sec, y_train_sec, sample_weight=w_train_sec,
                                         early_stopping_rounds=early_stop_num, eval_set=eval_set, verbose=0)
    #"catboost" mode
    elif mode=='catb':
        eval_set = None if early_stop_num==None else (x_test, y_true)
        clf.fit(x_train, y_train, sample_weight=w_train, early_stopping_rounds=early_stop_num,
                eval_set=eval_set, cat_features=cat_feature, use_best_model=True, verbose=0)
        if use_second_model: clf_sec.fit(x_train, y_train, sample_weight=w_train, early_stopping_rounds=early_stop_num,
                                         eval_set=eval_set, cat_features=cat_feature, use_best_model=True, verbose=0)
    
    #other mode
    else:
        clf.fit(x_train, y_train)
        if use_second_model: clf_sec.fit(x_train_sec, y_train_sec)
    
    #Predict
    proba_both = clf.predict_proba(x_test)    
    if use_second_model: proba_both_sec = clf_sec.predict_proba(x_test)
    
    
    return proba_both if not use_second_model else np.stack([proba_both, proba_both_sec])


#Get shap-value of binary classification for decision tree
def get_shap_values(clf, x_train, y_train, x_test, pred_positive_only=False, pred_negative_only=False,
                    is_nega=False, do_display=False, display_idx=0):
    
    #Training model
    clf.fit(x_train, y_train)
    
    #Choice predicted class
    if pred_positive_only or pred_negative_only:
        y_pred = clf.predict(x_test)
        if pred_positive_only and not pred_negative_only:
            x_test = x_test.iloc[np.where(y_pred==1)[0],:]
        if not pred_positive_only and pred_negative_only:  
            x_test = x_test.iloc[np.where(y_pred==0)[0],:]
    
    #Make shap explaner
    explainer = shap.TreeExplainer(clf)
    
    #Get shap-value
    shap_values = explainer.shap_values(x_test)
            
    #Degree that true label is negative if 'is_nega' is TRUE
    if is_nega:
        odds_to_proba = np.frompyfunc(lambda x: -1*x, 1, 1) 
        shap_values = odds_to_proba(shap_values)
        explainer.expected_value *= -1
    
    #Display shap-value of target index
    if do_display:
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0,:], x_test.iloc[display_idx,:],matplotlib=True,link='logit')
    
    #Calculate sum of shap-value by columns                
    se=pd.Series()
    for i, clm in enumerate(x_test.columns):
        se[clm] = sum(list(map(lambda x: x/(1+x) if x>=0 else x/(1-x), shap_values[:,i])))            
    se['count'] = len(x_test)
    
    return se
    

##Display partial dependance
def plot_PDP(fitted_clf, data, tgt_clm, ax=None):
    data_temp = data.copy()
    grid = np.linspace(np.percentile(data_temp.loc[:,tgt_clm], 0.1),
                       np.percentile(data_temp.loc[:,tgt_clm], 99.5),
                       50)
    
    y_pred = np.zeros(len(grid))
    for i, val in enumerate(grid):
        data_temp.loc[:,tgt_clm] = val
        y_pred[i] = np.average(fitted_clf.predict_proba(data_temp)[:,1])
            
    y_pred_adjust=[x-0.5 for x in y_pred]
    
    ax.plot(grid, y_pred_adjust, '-', color = 'blue', linewidth = 2.5)
    ax.set_xlim(min(grid), max(grid))
    ax.set_ylim(min(y_pred_adjust)-abs(0.05*min(y_pred_adjust)), max(y_pred_adjust)+abs(0.05*max(y_pred_adjust)))

    ax.set_xlabel(tgt_clm)
    ax.set_ylabel('Partial Dependence')
    