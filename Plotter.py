import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D   
from sklearn.metrics import auc

#Plot confusion matrix
def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    sns.set_style("white")   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=18,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    return
    
#Plot beeswarm
def plot_beeswarm(data, x_label=None, y_label=None, x_ticklabels=[]):
    sns.set_style("whitegrid")  

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)    
    ax = sns.swarmplot(data=data)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticklabels(x_ticklabels)    

    plt.show()
    
    return
    
#Plot scatters
def plot_2D_scatters(data, ticklabels=None, group=None, group_name=None, display_correlation=False):
    sns.set_style("whitegrid")  

    data = pd.DataFrame(data, columns=ticklabels)
    
    #Define gragh size
    vertical = 1
    horizonal = 1
    
    #Define gragh size like square if data number is upper 3
    if len(data.columns)>=3:
        divisor = sorted([i for i in range(1,len(data.columns)+1) if len(data.columns)%i == 0])
        vertical = divisor[int(np.median(range(0,len(divisor))))]
        horizonal = int(len(data.columns)/vertical)

    fig, ax = plt.subplots(vertical, horizonal, figsize=(5*horizonal, 4*vertical), squeeze=False)
    
    for i, pair in enumerate(itertools.combinations(range(0,len(data.columns)), 2)):    
        hor_idx = i%len(data.columns)
        ver_idx = int(i/len(data.columns))
        
        if not group is None:
            for i in sorted(group.unique()):
                sns.scatterplot(x=data.iloc[:,pair[0]].loc[(group==i)],
                                y=data.iloc[:,pair[1]].loc[(group==i)],
                                label=group_name[i],ax=ax[ver_idx][hor_idx])
                ax[ver_idx][hor_idx].legend()
        else:
            sns.scatterplot(x=data.iloc[:,pair[0]], y=data.iloc[:,pair[1]], ax=ax[ver_idx][hor_idx])
        
        if display_correlation:
            corr = stats.pearsonr(data.iloc[:,pair[0]], data.iloc[:,pair[1]])[0]
            ax[ver_idx][hor_idx].text(0.95, 0.90, f'R={round(corr,3)}', size=10, transform=ax[ver_idx][hor_idx].transAxes,
                                      horizontalalignment = 'right', bbox=dict(facecolor='white', edgecolor='black'))
     
    plt.show()
    
    return
    
  
#Plot 3D scatter
def plot_3D_scattaer(x,y,z, x_label=None,y_label=None,z_label=None, group=None,group_name=None):
    '''  
    Parameters
    ----------
    x : TYPE:like array
        First demention data.
    y : TYPE:like array
        Second demention data.
    z : TYPE:like array
        Third demention data.
    x_label : TYPE:str
        First demention name. The default is None.
    y_label : TYPE:str
        Seconde demention name. The default is None.
    z_label : TYPE:str
        Third demention name. The default is None.
    group : TYPE:like array
        Data group (like class). The default is None.
    group_name : TYPE:dict
        Corresponds to the group name. The default is None.

    Returns
    -------
    None.
    '''
    
    sns.set_style("whitegrid") 

    fig = plt.figure()
    ax = Axes3D(fig)

    if not group is None:
        for i in sorted(group.unique()):
            ax.plot(x[group==i],y[group==i],z[group==i],marker=".",linestyle='None',
                    label=group_name[i])
        ax.legend()
    else:
        ax.plot(x,y,z,marker="o",linestyle='None')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    ax.set_xlim(max(list((x))), min(list(x)))

    plt.show()
    
    return
 
#Plot scatter as line gragh
def plot_curve(axes, x, y, x_label=None, y_label=None, title=None, display=False):    
    axes.plot(x, y, label='AUC = %.2f'%auc(x,y))
    axes.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=8)
    axes.grid(color='gray')
    if x_label != None: axes.set_xlabel(x_label)
    if y_label != None: axes.set_ylabel(y_label)
    if title != None: axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    
    if display: plt.show()

    return
    
