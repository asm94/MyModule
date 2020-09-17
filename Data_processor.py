import umap
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from FileReader import read_pickle

#Compress dimention
def dimensional_compressor(df, ignore_column=[], dimention=1, random_seed=None):
    '''
    Parameters
    ----------
    df : TYPE:pd.DataFrame
        data.
    ignore_column : TYPE:like array
        Not compress columns. The default is [].
    demention : TYPE:integer
        Dimention after compression. The default is 1.
    random_seed : TYPE:integer
        Random seed. The default is None.

    Returns
    -------
    TYPE:pd.DataFrame
        Concatenation 'ignore_column' and compressed data.
    '''
    
    #Divide the 'df' into target data and ignore data
    if len(ignore_column)>=1:
        target_df = df.drop(ignore_column, axis=1).reset_index(drop=True)
        non_target_df = df[ignore_column].reset_index(drop=True)
    
    #Change 'dimention' to 1 if dimention of target data is greater than 'dimention'
    if len(target_df.columns) < dimention:        
        dimention = 1
        print('WARNING : Compress dimention is upper than columns. -> Processed as dimention=1 .')
    
    compressed_np = umap.UMAP(random_state=random_seed, n_components=dimention).fit_transform(target_df)
    compressed_df = pd.DataFrame(compressed_np)
    
    return pd.concat([non_target_df, compressed_df], axis=1)

#Padding ndarray
def padding_ndarray(data, target_dim, target_length, value=np.nan):
    
    data = np.array(data)
    
    padding_length = target_length - data.shape[data.ndim-target_dim]
    padding_shape = list(data.shape)
    padding_shape[data.ndim-target_dim] = padding_length
    
    padding_data = np.full(padding_shape, np.nan)
    
    return np.concatenate([data,padding_data], axis=data.ndim-target_dim)

#Standardize data
def data_to_std(data, ex_clm=[]):
    scaler = StandardScaler()
    data_temp = data.loc[:,ex_clm]
    data = data.drop(ex_clm, axis=1)
    data_std = pd.DataFrame(scaler.fit_transform(data), index=data.index.values, columns=data.columns.values)
    data = pd.concat([data_temp, data_std], axis=1)
        
    return data

#Generate data by CTGAN
from ctgan import CTGANSynthesizer
def generate_data_ctgan(raw_data, generate_sample=0, ex_column=[], epoch=50,
                        save_path=None, model_path=None, option_word=''):
    
    #Exclude specified column
    train_data = raw_data.drop(ex_column, axis=1)
    
    #Training CTGAN model
    if model_path==None:
        #Define CTGAN model architecture
        ctgan = CTGANSynthesizer(embedding_dim=128, gen_dim=(256,256), dis_dim=(256,256))
       
        #Training model
        ctgan.fit(train_data, train_data.columns, epochs=epoch)
    
        #Save model if "save_path" is specified
        if save_path != None:
            with open(save_path+'\\'+'CTGAN'+option_word+r'.pickle', mode='wb') as fp:
                pickle.dump(ctgan, fp, protocol=2)
    
    #Read trained model if "model_path" is specified
    else:
        ctgan = read_pickle(model_path+'\CTGANmodel', sub_check=True, target_name=option_word)

    #Generate data
    if generate_sample < 1: generate_sample = len(train_data) 
    samples = ctgan.sample(int(generate_sample))
    
    return samples