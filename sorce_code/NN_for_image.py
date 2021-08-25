import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation, MaxPool2D, DepthwiseConv2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16, vgg19, efficientnet, InceptionV3, ResNet50V2, DenseNet201, NASNetLarge

import torch
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
import random

from ctgan import CTGANSynthesizer
from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer

def get_fitted_model(x_train,y_train,x_test=None,y_true=None,loss=None,optimizer=keras.optimizers.Adam(),
                     training_epoch=10,batch_size=8,class_weight=None,metrics=['accuracy'],callbacks=None,
                     display_training=False,plot_history=False,path_save_history=None,save_index=''):
    
    ##check keras backend image_data_format
    if keras.backend.image_data_format()=='channels_first':
        x_train = x_train.transpose((0,3,1,2))
        if x_test!=None: x_test = x_test.transpose((0,3,1,2))

    ##Define model
    #Make new model
    model = get_model(x_train.shape, np.unique(np.argmax(y_train,axis=1)).size)
    print(model.summary())

    #Define 'loss' depending on class number(when 'loss' is not defined)
    if loss==None: loss=keras.losses.BinaryCrossentropy() if np.unique(y_train).size==2 else keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if type(x_test)!=type(None) and type(y_true)!=type(None):
        model_history = model.fit(x_train, y_train, epochs=training_epoch, batch_size=batch_size, class_weight=class_weight,
                                  validation_data=(x_test,y_true), callbacks=callbacks,
                                  verbose=1
                                 )
    else:
        model_history = model.fit(x_train, y_train, epochs=training_epoch, batch_size=batch_size, class_weight=class_weight,
                                  validation_split=0.1, callbacks=callbacks,
                                  verbose=1
                                 )

    if plot_history:
        plot_training_history(model_history, path_save_history=path_save_history, save_index=save_index)

    return model
   
 
##Define model architecture
def get_model(shape, class_num):    
    shape = list(shape)
    shape.pop(0)
    shape = tuple(shape)    

    #model = multitask_cnn(shape, class_num)#, activation=FReLU)
    model = efficientNet(shape, class_num)

    return model

#VGG16
def vgg16(shape, class_num):
    base_model = vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = True
    
    #for train part of model
    layer_names = [l.name for l in base_model.layers]
    idx = layer_names.index('block5_conv1')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    
    return Model(inputs=base_model.input, outputs=output)

#VGG19
def vgg19(shape, class_num):
    base_model = vgg19.VGG19(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = True
    
    #for train part of model
    layer_names = [l.name for l in base_model.layers]
    idx = layer_names.index('block5_conv1')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    
    return Model(inputs=base_model.input, outputs=output)

#EfficientNet
def efficientNet(shape, class_num):
    base_model = efficientnet.EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    #nw = Dropout(.5)(nw)
    #nw = Dense(512, activation='relu')(nw)
    #nw = Dropout(.5)(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = False
        
    '''#for train part of model
    layer_names = [l.name for l in base_model.layers]   
    idx = layer_names.index('block6a_expand_conv')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    '''
    return Model(inputs=base_model.input, outputs=output)

#InceptionV3(googlenet)
def inceptionv3(shape, class_num):
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = False
        
    '''#for train part of model
    layer_names = [l.name for l in base_model.layers]   
    idx = layer_names.index('block7a_expand_conv')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    '''   
    return Model(inputs=base_model.input, outputs=output)

#ResNet
def resnet(shape, class_num):
    base_model = ResNet50V2(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = False
        
    '''#for train part of model
    layer_names = [l.name for l in base_model.layers]   
    idx = layer_names.index('block7a_expand_conv')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    '''   
    return Model(inputs=base_model.input, outputs=output)

#DenseNet
def densenet(shape, class_num):
    base_model = DenseNet201(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = False
        
    '''#for train part of model
    layer_names = [l.name for l in base_model.layers]   
    idx = layer_names.index('block7a_expand_conv')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    '''   
    return Model(inputs=base_model.input, outputs=output)


#NasNet
def nasnet(shape, class_num):
    base_model = NASNetLarge(include_top=False, weights='imagenet', pooling='avg')#, input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = False
        
    '''#for train part of model
    layer_names = [l.name for l in base_model.layers]   
    idx = layer_names.index('block7a_expand_conv')
    for layer in base_model.layers[:idx]:
        layer.trainable = False
    '''   
    return Model(inputs=base_model.input, outputs=output)


#https://github.com/MaciejMazurowski/thyroid-us
def multitask_cnn(data_shape, class_num, activation=Activation('relu')):
    # n^2x1
    input_tensor = Input(shape=data_shape, name="thyroid_input")
    # n^2x8
    x = Conv2D(8, (3, 3), padding="same")(input_tensor)
    x = activation(x)
    # ((n/2)^2)x8
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # ((n/2)^2)x12
    x = Conv2D(12, (3, 3), padding="same")(x)
    x = activation(x)
    # ((n/4)^2)x12
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # ((n/4)^2)x16
    x = Conv2D(16, (3, 3), padding="same")(x)
    x = activation(x)
    # ((n/8)^2)x16
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # ((n/8)^2)x24
    x = Conv2D(24, (3, 3), padding="same")(x)
    x = activation(x)
    # ((n/16)^2)x24
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # ((n/16)^2)x32
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = activation(x)
    # ((n/32)^2)x32
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # ((n/32)^2)x48
    x = Conv2D(48, (3, 3), padding="same")(x)
    x = activation(x)
    # ((n/32)^2)x48
    x = Dropout(0.5)(x)

    # ((n/160)^2)×1
    y_cancer = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)
    y_cancer = Flatten()(y_cancer)
    y_cancer = Dense(class_num, activation='sigmoid', name='output')(y_cancer)
    #y_cancer = Activation("sigmoid", name="out_cancer")(y_cancer)

    return Model(
        inputs=[input_tensor],
        outputs=[y_cancer],
    )

#FReLU
#https://qiita.com/rabbitcaptain/items/26304b5a5e401db5bae2
def FReLU(inputs, kernel_size = 3):
    
    #T(x)
    x = DepthwiseConv2D(kernel_size, strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    
    #max(x, T(x))
    x = tf.maximum(inputs, x)
    
    return x


##Plot and save trainig history
def plot_training_history(model_history, path_save_history=None, save_index=''):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(range(1, len(model_history.history['accuracy'])+1), model_history.history['accuracy'], label="training")
    ax[0].plot(range(1, len(model_history.history['accuracy'])+1), model_history.history['val_accuracy'], label="validation")
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'], label="training")
    ax[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['val_loss'], label="validation")
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

    if path_save_history!=None:
        fig.savefig(path_save_history+r'\history_data{0}.png'.format(save_index),dpi=100)
    