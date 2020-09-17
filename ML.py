import pandas as pd
import numpy as np
import shap
import pydotplus
from sklearn import tree
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

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
def display_PDP(clf, df_train, path_dot=None):
    #Get partial dependance
    dot_data = tree.export_graphviz(clf.estimators_[0][0],
                                    feature_names=df_train.columns,
                                    class_names=['1','0'],
                                    filled=True,
                                    rounded=True,
                                    proportion=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.progs = {'dot': path_dot} #Like 'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe'
    
    #For save image
    #graph.write_png(r'G:\生データ\対象データ\出力\test_graph.png')
    #Image(graph.create_png())
    
    #Plot and display
    fig, axs = plot_partial_dependence(clf, df_train, list(df_train.columns), feature_names=list(df_train.columns))
    fig.set_size_inches(12, 18)
    plt.show()