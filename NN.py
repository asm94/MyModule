from tensorflow import keras
import numpy as np
def get_fitted_model(x_train,y_train,x_test=None,y_true=None,loss=None,optimizer=keras.optimizers.Adam(),
                     training_epoch=10,batch_size=8,class_weight=None,metrics=['accuracy'],callbacks=None,
                     display_training=False,plot_history=False,path_save_history=None,save_index=''):
    '''
    #Data parameter(required)
    x_train => Detail:Image data for train (only resized).
               Type:numpy.ndarray(data_number,image_width,image_height,channel)
    y_train => Detail:Target label for train.
               Type:numpy.ndarray(data_number,onehot_label)
    
    #Data parameter(not absolutely necessary)
    x_test => Detail:Image data for test (only resized).
              Type  :numpy.ndarray(data_number,image_width,image_height,channel)
    y_true => Detail:Target label for test.
              Type  :numpy.ndarray(data_number,onehot_label)

    #Setting parameter(not absolutely necessary) 
    loss => Detail:Loss function.
            Type  :str or keras.losses
    optimizer => Detail:Optimizer.
                 Type  :str or keras.optimizers
    metrics => Detail:Metrics for model training.
               Type  :list[function]
    callbacks => Detail:Callback function for model training.
                 Type  :list[function] 
    display_training => Detail:Whether or not display training status.
                        Type  :bool
    training_epoch => Detail:Training epoch.
                      Type  :integer
    batch_size => Detail:Image number is used by 1 epoch.
                  Type  :integer
    class_weight => Detail:Weight is considered by model training.
                    Type  :dict
    plot_history => Detail:Whether or not plot training history.
                    Type  :bool
    path_save_history => Detail:Path of training history as image.
                         Type  :string   
    save_index => Detail:Whether or not plot training history.
                  Type  :numeric or string
    '''

    ##check keras backend image_data_format
    if keras.backend.image_data_format()=='channels_first':
        x_train = x_train.transpose((0,3,1,2))
        if x_test!=None: x_test = x_test.transpose((0,3,1,2))

    ##Define model
    #Make new model
    model = get_model(x_train.shape[1], x_train.shape[2], x_train.shape[3], np.unique(np.argmax(y_train,axis=1)).size)
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
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import efficientnet  
def get_model(input_height, input_width, input_channel, class_num):
    '''
    #Data parameter(required)
    input_height => Detail:Image height. Type:integer
    input_width => Detail:Image width. Type:integer
    input_channel => Detail:Image channel. Type:integer
    class_num => Detail:Output class number. Type:integer
    
    #Setting parameter
    part_trainable => Detail:Whether or not tunig part of model. Type:bool
    '''
    
    inputs = Input(shape=(input_height, input_width, input_channel))
    in_net = Lambda(vgg16.preprocess_input, name='preprocess')(inputs)

    base_model = vgg16.VGG16(include_top=False, weights=None, input_tensor=in_net, pooling='avg')
    nw = base_model.output
       
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    output = Dense(class_num, activation='softmax', name='output')(nw)        

    model = Model(inputs=base_model.input, outputs=output)

    base_model.trainable = True
    
    #for train part of model
    layer_names = [l.name for l in base_model.layers]
    idx = layer_names.index('block5_conv1') #For VGG16        
    #idx = layer_names.index('block7a_expand_conv') #For EfficientNetB7
    for layer in base_model.layers[:idx]:
        layer.trainable = False        

    return model

##Plot and save trainig history
import matplotlib.pyplot as plt
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
