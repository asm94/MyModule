import pandas as pd
import numpy as np
import shap
from sklearn.metrics import confusion_matrix

import sys

#Get shap-value of binary classification for decision tree
def get_shap_values(fitted_model, x_train, y_train, x_test, pred_positive_only=False, pred_negative_only=False,
                    is_nega=False, do_display=False, display_idx=0):
    
    #Choice predicted class
    if pred_positive_only or pred_negative_only:
        y_pred = fitted_model.predict(x_test)
        if pred_positive_only and not pred_negative_only:
            x_test = x_test.iloc[np.where(y_pred==1)[0],:]
        if not pred_positive_only and pred_negative_only:  
            x_test = x_test.iloc[np.where(y_pred==0)[0],:]
    
    #Make shap explaner
    explainer = shap.TreeExplainer(fitted_model)
    
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
        
    return shap_values
    

##Display partial dependance
def plot_PDP(fitted_clf, data, tgt_clm, cat_feature=False, ax=None):
    data_temp = data.copy()
    grid = np.linspace(np.percentile(data_temp.loc[:,tgt_clm], 0.1),
                       np.percentile(data_temp.loc[:,tgt_clm], 99.5),
                       50)
    
    y_pred = np.zeros(len(grid))
    for i, val in enumerate(grid):
        data_temp.loc[:,tgt_clm] = val if not cat_feature else int(val) if (val-int(val))<0.5 else int(val)+1
        y_pred[i] = np.average(fitted_clf.predict_proba(data_temp)[:,1])
            
    y_pred_adjust=[x-0.5 for x in y_pred]
    
    ax.plot(grid, y_pred_adjust, '-', color = 'blue', linewidth = 2.5)
    ax.set_xlim(min(grid), max(grid))
    ax.set_ylim(min(y_pred_adjust)-abs(0.05*min(y_pred_adjust)), max(y_pred_adjust)+abs(0.05*max(y_pred_adjust)))

    ax.set_xlabel(tgt_clm)
    ax.set_ylabel('Partial Dependence')
    
    
##Calculate model performance value corresponding to a single threshold
def calculate_performance(true_label, pred_label, num_class=2):
    
    #2 classes(positive=1, negative=0) 
    if num_class == 2:
        cm = confusion_matrix(true_label, pred_label, labels=[1, 0])
        tp, fn, fp, tn = cm.flatten()
        
        out = {'accuracy'       : ((tp+tn)/(tp+fn+fp+tn)) if (tp+fn+fp+tn)!=0 else 0,
               'precision'      : (tp/(tp+fp)) if (tp+fp)!=0 else 1,
               'nega_precision' : (tn/(tn+fn)) if (tn+fn)!=0 else 1,
               'recall'         : (tp/(fn+tp)) if (fn+tp)!=0 else 0,
               'specificity'    : (tn/(fp+tn)) if (fp+tn)!=0 else 0,
              }
        
        return out
    
    #3 classes
    elif num_class >= 3:
        cm = confusion_matrix(true_label, pred_label, labels=list(np.unique(true_label)))
        
        out = {'accuracy':0}
        for i in np.unique(true_label):
            out['accuracy'] += cm[i,i]
            out[f'precision_class{i}'] = cm[i,i] / sum(cm[:,i]) if sum(cm[:,i])!=0 else 1
            out[f'recall_class{i}'] = cm[i,i] / sum(cm[i]) if sum(cm[i])!=0 else 0
            
        out['accuracy'] = out['accuracy'] / cm.sum() if cm.sum()!=0 else 0
        
        return out
    
    #The other number of classes is undefined.
    else:
        sys.exit('ERROR:Unexpected number of classes.')
        
        
#Optimize border
def optimize_border(positive_proba, true_label, positive_proba_sec=[], step=0.1, maximize_metrics='f_score'):
        
    #Set range of border
    lower = 0.0
    upper = 1.0
    data = None
    if len(positive_proba) == len(positive_proba_sec) == len(true_label):       
        lower = 0.0
        upper = 1.1
        data = pd.DataFrame(columns = ['precision', 'recall', 'fpr', 'border', 'boder_sec'])
    else:
        lower = 0.00
        upper = 1.01
        data = pd.DataFrame(columns = ['precision', 'recall', 'fpr', 'border'])  
        step *= 0.1
       
    #Unify to numpy-type
    positive_proba = np.array(positive_proba)
    positive_proba_sec = np.array(positive_proba_sec)
       
    #Explore border optimized
    max_idx = 0
    best_pred = None
    best_border = 0
    for border in np.arange(lower, upper, step)[::-1]:
        #Dual model
        if len(positive_proba) == len(positive_proba_sec) == len(true_label):
            for border_sec in np.arange(lower, upper, step)[::-1]:
                pred_label = [0]*len(positive_proba)
                for i in range(0,len(positive_proba)):
                    pred_label[i] = 1 if positive_proba[i]>=border and positive_proba_sec[i]>=border_sec else 0
                
                res = calculate_performance(true_label, pred_label)
                precision = res['precision']
                recall = res['recall']
                fpr = 1-res['specificity']
                                
                temp_idx = None
                if maximize_metrics == 'f_score': temp_idx = (2*res['precision']*res['recall'])/(res['precision']+res['recall']) #F-score
                elif maximize_metrics == 'youden_index': temp_idx = recall+(1-fpr)-1 #Youden Index
                else:
                    print('Select from ["f_score", "youden_index"] and choose it as the "maximize_metrics" argument.')
                    return None
                
                if max_idx < temp_idx or best_pred == None:
                    best_pred = pred_label
                    max_idx = temp_idx        
        
                se = pd.Series([precision, recall, fpr, border, border_sec],
                               index=['precision', 'recall', 'fpr', 'border', 'border_sec'])
                data = data.append(se, ignore_index=True)
                                
        #Single model 
        else:
            pred_label = [0]*len(positive_proba)
            for i in range(0,len(positive_proba)):
                pred_label[i] = 1 if positive_proba[i]>=border else 0
                        
            res = calculate_performance(true_label, pred_label)
            precision = res['precision']
            recall = res['recall']
            fpr = 1-res['specificity']                                
             
            temp_idx = None
            if maximize_metrics == 'f_score': temp_idx = (2*res['precision']*res['recall'])/(res['precision']+res['recall']) #F-score
            elif maximize_metrics == 'youden_index': temp_idx = recall+(1-fpr)-1 #Youden Index
            else:
                print('Select from ["f_score", "youden_index"] and choose it as the "maximize_metrics" argument.')
                return None
                
            if max_idx < temp_idx or best_pred == None:
                best_pred = pred_label
                max_idx = temp_idx
                best_border = border
        
            se = pd.Series([precision, recall, fpr, border],
                           index=['precision', 'recall', 'fpr', 'border'])
            data = data.append(se, ignore_index=True)
     
    #print(f'Best border is {best_border}')               
    return best_pred, data