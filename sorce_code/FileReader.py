import glob
import cv2
import numpy as np
import pandas as pd
import pickle
import sys

#Get image in a folder
def read_img(path, sub_check=False, name_return=False):
        
    #Get full-path of target images
    target_files = glob.glob(path+'\**\*.[png|PNG|jpg|JPG|jpeg|JPEG|tiff|TIFF]*', recursive=True) if sub_check else glob.glob(path+'\*.[png|PNG|jpg|JPG|jpeg|JPEG|tiff|TIFF]*')
     
    #For return
    img_list = []
    if name_return: name_list = []

    #Explore target images
    for filename in target_files: 
        img = imread(filename)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR→RGB
        img_list.append(img)

        #Get file name
        if name_return:
            name = filename.split('\\')[-1]
            name_list.append(name)
     
    return img_list if not name_return else img_list, name_list


#For path name include japanese
#https://qiita.com/SKYS/items/cbde3775e2143cad7455
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


#Get csv in a folder as pandas.DataFrame
def read_csv(path, encode='cp932', sub_check=False, target_name=None):
    #Get all csv file-path in a folder 
    target_files = glob.glob(path+r'\**\*.csv', recursive=True) if sub_check else glob.glob(path+r'\*.csv')

    #For return
    merged_file = pd.DataFrame()

    #All csv combine
    for filepath in target_files:
        
        #Exclude file not including target name
        filename = filepath.split('\\')[-1]
        if target_name!=None and target_name not in filename: continue        
    
        #Read a csv as pandas.DataFrame
        input_file = pd.read_csv(filepath, encoding=encode, sep=",", engine='python')
        
        #New csv conbine
        merged_file = pd.concat([merged_file, input_file], axis=0)

    #Index reset
    merged_file = merged_file.reset_index(drop=True)
    
    return merged_file

#Get excel in a folder as pandas.DataFrame
def read_xls(path, sub_check=False, target_name=None):
    #Get all excel file-path in a folder 
    target_files = glob.glob(path+r'\**\*.xls*', recursive=True) if sub_check else glob.glob(path+r'\*.xls*')

    #For return
    merged_file = pd.DataFrame()

    #All excel combine
    for filepath in target_files:
        
        #Exclude file not including target name
        filename = filepath.split('\\')[-1]
        if target_name!=None and target_name not in filename: continue        
    
        #Read a excel as pandas.DataFrame
        input_file = pd.read_excel(filepath)
        
        #New excel conbine
        merged_file = pd.concat([merged_file, input_file], axis=0)

    #Index reset
    merged_file = merged_file.reset_index(drop=True)
    
    return merged_file


#Get pickle
def read_pickle(path, sub_check=False, target_name=None):
    #Get all csv file-path in a folder   
    target_files = glob.glob(path+'\**\*.pickle', recursive=True) if sub_check else glob.glob(path+'\*.pickle')

    #For return
    target_pickle = None

    #Explore all pickle
    for filepath in sorted(target_files):
        
        #Not specify target name
        if target_name == None:
            if target_pickle==None:
                with open(filepath, mode='rb') as fp:
                    target_pickle = pickle.load(fp)
            
            #Upper 2 pickle must not exist        
            else:
                sys.exit('@read_pickle function: Specify "target_name" when pickle exists upper 2.')
            
        #Specify target name
        else:
            #Extract filename
            filename = filepath.split('\\')[-1]
            
            #Get file including target name
            if target_pickle==None and target_name in filename:
                with open(filepath, mode='rb') as fp:
                    target_pickle = pickle.load(fp)
                    print(target_name)
                break
        
    return target_pickle