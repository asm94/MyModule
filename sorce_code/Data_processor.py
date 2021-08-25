import umap
import pandas as pd
import numpy as np
import pickle
from concurrent import futures
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from ctgan import CTGANSynthesizer

from .FileReader import read_pickle   

#Compress dimention
def dimensional_compressor(df, ignore_column=[], dimention=1, random_seed=None):
        
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


#Adjust the number per period based on the specified attributes.
def adjust_number(data, target_column, attribute, sub_attribute=None, period=10, return_exdata=False):
    
    #Returns without processing if the target dataset is empty or has only one attribute.
    if len(data)==0 or len(set(data[attribute]))==1: return data if not return_exdata else (data, pd.DataFrame())
        
    #Add "idx" column to get excluded data
    if return_exdata: data['idx'] = list(range(len(data)))
    
    #Set the upper and lower limits of the interval and the maximum value of the target column
    lower = 0 if data[target_column].min() >= 0 else data[target_column].min()
    upper = lower+period
    maximum = data[target_column].max()
        
    #For storing adjusted data
    data_adjusted = pd.DataFrame(columns=data.columns)
    
    #Adjustment of the number of data in the interval of each period (undersampling)
    while lower <= maximum:
        #Extraction of target section data
        data_in_range = data[(lower<=data.loc[:,target_column]) & (data.loc[:,target_column]<upper)]
        
        #If the data of the target section does not exist,
        #or if the data of the target section does not contain all the attributes,
        #then the next section
        if len(data_in_range) == 0 or set(data[attribute]) != set(data_in_range[attribute]):
            lower += period
            upper += period           
            continue
        
        #Get the number of data per attribute
        counts = data_in_range[attribute].value_counts()
        
        #Undersampling per attribute
        for att in counts.index:
                        
            sample = pd.DataFrame()
            #If there is no sub-attribute or the target attribute does not require undersampling
            if sub_attribute == None or counts[att]==counts.min():
                sample = data_in_range[data_in_range[attribute]==att].sample(n=counts.min(), random_state=42)
                
            #If there are sub-attributes, sample them as evenly as possible for each sub-attribute
            elif sub_attribute != None and counts[att]!=counts.min():
                data_att = data_in_range[data_in_range[attribute]==att]                
                sub_counts = data_att[sub_attribute].value_counts()
                
                #Obtains the remaining number of samples
                #when "the number of samples per sub-attribute divided equally by the number of sub-attribute types of the target attribute" is
                #subtracted from the number of data per sub-attribute.
                remaining = sub_counts - int(counts.min()/len(sub_counts))
                                
                #Calculates the number of remnants at the time of equal division
                surplus = counts.min() % len(sub_counts)
                
                #Calculates shortage of sub-attributes that cannot be sampled due to the number of data
                surplus += -1*remaining[remaining<0].sum()
                                                                
                #Adjust the distribution of the number of samples
                #until the total number of samples per sub-attribute reaches
                #the number of target attribute samples
                while surplus > 0:         
                    
                    #Set the number of sub-attributes that cannot be sampled anymore due to the number of data, to 0
                    remaining[remaining<0] = 0
                                        
                    #If the number of sub-attributes with more than 1 data remaining is
                    #less than the remaining required number of samples
                    if surplus > len(remaining[remaining>=1]):
                        
                        #Subtract the remaining required number of samples equally 
                        #from a sub-attribute with more than 1 data remaining               
                        remaining[remaining>=1] -= int(surplus/len(remaining[remaining>=1]))
                
                        #Calculate the remainder of the equitable distribution of the required number of samples
                        surplus += surplus % len(remaining[remaining>=1])
                        
                        #Subtract the remainder that has been distributed
                        surplus -= int(surplus/len(remaining[remaining>=1]))
                                         
                    else:
                        if len(data_adjusted)>0:
                            temp_counts = data_adjusted[sub_attribute].value_counts()
                            tgt_index = temp_counts[remaining.index].sort_values()[:surplus].index
                            remaining[tgt_index] -= 1
                        else:
                            remaining[:surplus] -= 1
                        break
                    
                        
                    #Calculates shortage of sub-attributes that cannot be sampled due to the number of data
                    surplus += -1*remaining[remaining<0].sum()       
                                     
                #Sampling a specified number of sub-attributes
                sub_sample = sub_counts - remaining
                for s_att in sub_sample.index:
                    sample = pd.concat([sample, data_att[data_att[sub_attribute]==s_att].sample(n=sub_sample[s_att], random_state=42)],
                                       axis=0, ignore_index=True)
                             
            #Concatenate adjusted data of a single attribute
            data_adjusted = pd.concat([data_adjusted, sample],axis=0, ignore_index=True)
                       
        #Section Update
        lower += period
        upper += period
     
    #Get excluded data and return
    if return_exdata:
        exdata = data[~data['idx'].isin(data_adjusted['idx'])]
        data_adjusted = data_adjusted.drop('idx', axis=1)
        exdata = exdata.drop('idx', axis=1)
        
        return data_adjusted, exdata
    
    return data_adjusted


#Complete nan values with KNN
def NaN_complete(df_target, df_train=pd.DataFrame(), ex_column=[], n_neighbors=3, batch_size=0): 
    
    #Define complementer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    #Completion
    #Completion source data is not data for completion
    if len(df_train)==0: df_train = df_target
        
    imputer.fit(df_train.drop(ex_column, axis=1))
        
    df_out = pd.DataFrame()
    if batch_size==0 or batch_size >= len(df_target): batch_size = len(df_target)
    
    '''for i in range((len(df_target)//batch_size)+1):
        st = batch_size*i
        if st >= len(df_target): break
        en = batch_size*(i+1)
        if en >= len(df_target): en = len(df_target)'''
        
    def one_process(d):  
        np_value = imputer.transform(d.drop(ex_column, axis=1)) 
        
        #Data forming
        df_comp = pd.DataFrame(np_value, columns=df_target.drop(ex_column, axis=1).columns,
                               index=d.index)
        df_comp = pd.concat([d.loc[:,ex_column], df_comp], axis=1)   
        
        return df_comp #df_out = df_out.append(df_comp)
        
    result_list = []
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i in range((len(df_target)//batch_size)+1):
            st = batch_size*i
            if st >= len(df_target): break
            en = batch_size*(i+1)
            if en >= len(df_target): en = len(df_target)
            
            future = executor.submit(one_process, d=df_target.iloc[st:en])
            result_list.append(future.result())
            print('\r{0} done / {1} processes.'.format(en, len(df_target), end=''))
        print('completed.')
    #result_list = [f.result() for f in future_list]
    
    df_out = pd.concat(result_list, axis=0)
                                    
    return df_out.loc[:,df_target.columns]