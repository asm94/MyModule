import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

#Plot confusion matrix
def plot_confusion_matrix(matrix, lebales, title=None):
    sns.set_style("white")

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(range(len(lebales)), lebales)
    plt.yticks(range(len(lebales)), lebales)

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], horizontalalignment="center", fontsize=18,
                 color="white" if matrix[i, j] > matrix.max()/2 else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    return
    
#Plot beeswarm
def plot_beeswarm(data, x_label=None, y_label=None, x_ticklabels=[], save_filename=None):
    sns.set_style("whitegrid")  

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 1, 1)    
    ax = sns.swarmplot(data=np.array(data))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticklabels(x_ticklabels)    

    if save_filename!=None: plt.savefig(save_filename, dpi=600, facecolor='white')
    plt.show()
    
    return
    
#Plot scatters
def plot_2D_scatters(data, ticklabels=None, group=None, display_correlation=False):
    sns.set_style("whitegrid")  

    data = pd.DataFrame(np.array(data), columns=ticklabels)
    combination_list = list(itertools.combinations(range(0,len(data.columns)), 2))
        
    #Define gragh size
    vertical = 1
    horizonal = 1
    
    #Define gragh size like square if number of the data is upper 3
    if len(combination_list)>=2:
        divisor = sorted([i for i in range(1,len(combination_list)+1) if len(combination_list)%i == 0])  
        vertical = divisor[int(np.median(range(0,len(divisor))))]
        horizonal = int(len(combination_list)/vertical)

    fig, ax = plt.subplots(vertical, horizonal, figsize=(4*horizonal, 4*vertical), squeeze=False)
    
    for i, pair in enumerate(combination_list): 
        hor_idx = i%horizonal
        ver_idx = i%vertical
        
        if not group is None:
            group = group.reset_index(drop=True)
            for name in sorted(group.unique()):
                sns.scatterplot(x=data.iloc[:,pair[0]].loc[(group==name)],
                                y=data.iloc[:,pair[1]].loc[(group==name)],
                                label=name, ax=ax[ver_idx][hor_idx],
                                alpha=0.5)
                ax[ver_idx][hor_idx].legend()
        else:
            sns.scatterplot(x=data.iloc[:,pair[0]], y=data.iloc[:,pair[1]], ax=ax[ver_idx][hor_idx])
            
        #ax[ver_idx][hor_idx].set_xlim(0.0,2.0)
        #ax[ver_idx][hor_idx].set_ylim(0.0,0.7)
        
        if display_correlation:
            corr = stats.pearsonr(data.iloc[:,pair[0]], data.iloc[:,pair[1]])[0]
            ax[ver_idx][hor_idx].text(0.98, 1.03, f'R={round(corr,3)}', size=9, transform=ax[ver_idx][hor_idx].transAxes,
                                      horizontalalignment = 'right', bbox=dict(facecolor='white', edgecolor='black'))
         
    plt.rcParams["font.size"] = 10
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
        for name in sorted(group.unique()):
            ax.plot(x[group==name],y[group==name],z[group==name],marker=".",linestyle='None',
                    label=name)
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
    
    xy = np.stack([np.array(x),np.array(y)])
    data = pd.DataFrame(xy.T, columns=['x','y']).sort_values(['x','y'])
    x, y = np.array(data['x']), np.array(data['y'])
    
    axes.plot(x, y, label='AUC = %.3f'%np.trapz(y,x))
    axes.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=8)
    axes.grid(color='gray')
    if x_label != None: axes.set_xlabel(x_label)
    if y_label != None: axes.set_ylabel(y_label)
    if title != None: axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    
    if display: plt.show()

    return
