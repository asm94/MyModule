import umap
import pandas as pd

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