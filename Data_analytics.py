import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#Get ttest result
def get_ttests(df_a, df_b, tgt_clm, name_a='df_a', name_b='df_b', max_ylim=None, display_gragh=False, gragh_mode='boxplot'):
        
    #For stack result
    df_out = pd.DataFrame()   
    
    #Process by column 
    for clm in tgt_clm:
        #Delete NaN
        data_a = np.array(df_a[clm][~df_a[clm].isnull()])
        data_b = np.array(df_b[clm][~df_b[clm].isnull()])
        
        #Stack ttest result by column
        result_T = stats.ttest_ind(data_a, data_b, equal_var = False)
        se = pd.Series([result_T.statistic,result_T.pvalue], index=['statistic','pvalue'], name=clm)
        df_out = df_out.append(se)
        
        #Display swarm-gragh with ttests if 'display_gragh' is TRUE
        if display_gragh:
            fig, ax = plt.subplots()
            if gragh_mode=='boxplot': ax = sns.boxplot(data=[data_a, data_b])
            elif gragh_mode=='swarmplot': ax = sns.swarmplot(data=[data_a, data_b])
            ax.set_xticklabels([name_a, name_b])
            if max_ylim != None: ax.set_ylim(0, max_ylim)
            ax.text(0.5, 1.05, clm, size=20, transform=ax.transAxes, horizontalalignment = 'center')
            ax.text(1.0, 1.17, 'two-sided Welch\'s t test', size=10, transform=ax.transAxes, horizontalalignment = 'right')
            ax.text(1.0, 1.10, 'p='+str(round((result_T.pvalue),3)), size=10, transform=ax.transAxes, horizontalalignment = 'right')
            ax.text(1.0, 1.03, 't='+str(round((result_T.statistic),3)), size=10, transform=ax.transAxes, horizontalalignment = 'right')
            
    return df_out
