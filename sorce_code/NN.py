import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda, BatchNormalization, Activation, MaxPool2D, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import efficientnet

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

class uCTGANSynthesizer(CTGANSynthesizer):
    def __init__(self, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256), l2scale=1e-6, batch_size=500):
        super().__init__(embedding_dim=embedding_dim, gen_dim=gen_dim, dis_dim=dis_dim,
                                                l2scale=l2scale, batch_size=batch_size)
    def to(self, device):
        self.device = device
        return self
            
    def fit(self, train_data, discrete_columns=tuple(), epochs=300, log_frequency=True, verbose=1):
        
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data,
            self.transformer.output_info,
            log_frequency
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim
        ).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim
        ).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self.device)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            
            if verbose>0:
                if i==0 or (i+1)%verbose==0:
                    print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                          (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                          flush=True)
                    
        torch.cuda.empty_cache()
            
            
    def discriminate(self, train_data, discrete_columns=tuple(), log_frequency=True):
        
        real = torch.from_numpy(np.array(train_data).astype('float32')).to(self.device)
        return np.squeeze(self.discriminator(real).to('cpu').detach().numpy().copy())
        
        assert self.batch_size % 2 == 0
        
        index_train = random.sample(list(range(len(train_data))), len(train_data))
        batch_idx_train = [index_train[i:i+self.batch_size] for i in range(0, len(index_train), self.batch_size)]
             
        proba_list = []
        for idx in batch_idx_train: 
            
            condvec = self.cond_generator.sample(len(idx))
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                real = train_data[idx]
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self.device)

                perm = np.arange(len(idx))
                np.random.shuffle(perm)
                real = train_data[idx]
                c2 = c1[perm]
             
            real = torch.from_numpy(real.astype('float32')).to(self.device)
            
            if c1 is not None: real_cat = torch.cat([real, c2], dim=1)
            else: real_cat = real
            y_real = self.discriminator(real_cat)
            
            proba_list.append(y_real.to('cpu').detach().numpy().copy())
         
        return np.squeeze(np.concatenate(proba_list)) 
        

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
    model = VGG16(shape, class_num)

    return model

#VGG16
def VGG16(shape, class_num):
    base_model = vgg16.VGG16(include_top=False, weights=None, pooling='avg', input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = True
        
    return Model(inputs=base_model.input, outputs=output)

#EfficientNet
def EfficientNet(shape, class_num):
    base_model = efficientnet.EfficientNetB0(include_top=False, weights=None, pooling='avg', input_tensor=Input(shape=shape)) 
    nw = base_model.output
    
    nw = Dense(512, activation='relu')(nw)
    nw = Dropout(.4)(nw)
    nw = Dense(512, activation='relu')(nw)
    
    if class_num<=2:
        output = Dense(class_num, activation='sigmoid', name='output')(nw)     
    else:
        output = Dense(class_num, activation='softmax', name='output')(nw)  
            
    base_model.trainable = True
        
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
        
