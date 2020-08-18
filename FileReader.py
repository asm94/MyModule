#Get image in a folder
import glob
import cv2
def read_img(path, sub_check=False, name_return=False):
    '''
    #Setting parameter(required)
    path => Detail:Path of target image folder.
            Type  :string 

    #Setting parameter(not absolutely necessary) 
    sub_check => Detail:Whether or not explore sub folder.
                 Type  :bool
    name_return => Detail:Whether or not get image name.
                   Type  :bool
    '''
    
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
import numpy as np
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None