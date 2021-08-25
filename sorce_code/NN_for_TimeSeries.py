import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import TimeDistributed
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import torch
import torch.nn as nn
from qhoptim.pyt import QHAdam

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
import random
import os


class RecurrentNN(nn.Module):
    def __init__(self, num_feature, num_class, n_hidden=100, inter_num=0, inter_nodes=10, drop_rate=0.5):
        super(RecurrentNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size=num_feature, hidden_size=n_hidden, batch_first=True)
        
        inter_layers = []
        for i in range(inter_num):            
            inter_layers.append(nn.Linear(in_features=(n_hidden if i==0 else inter_nodes), out_features=inter_nodes))
            inter_layers.append(nn.ReLU())
            inter_layers.append(nn.Dropout(p=drop_rate))
        self.inter_layer = nn.Sequential(*inter_layers)
        
        out_nodes = inter_nodes if inter_num>0 else n_hidden
        self.output_layer = nn.Sequential(nn.Linear(in_features=out_nodes, out_features=num_class),
                                          nn.Softmax(dim=-1) if num_class>2 else nn.Sigmoid())
    
    def forward(self, x):
        x = x.to(torch.float32)
        
        rnn_out, _ = self.rnn(x)
        inter = self.inter_layer(rnn_out)                
        output = self.output_layer(inter)
        
        return output.to(torch.float64)
    

def get_LSTM(timesteps, num_feature, num_class, n_hidden=100, inter_num=0, inter_nodes=10, drop_rate=0.5,
             mask_value=None, optimizer='Adam', learning_rate=1e-2, loss=None, final_activation=None):
    
    opt = None
    if optimizer=='SGD': opt = SGD(lr=learning_rate)
    if optimizer=='RMSprop': opt = RMSprop(lr=learning_rate)
    if optimizer=='Adadelta': opt = Adadelta(lr=learning_rate)
    if optimizer=='Adam': opt = Adam(lr=learning_rate)
    
    if loss==None: loss = ('binary_crossentropy' if num_class<=2 else 'categorical_crossentropy')
    if final_activation==None: final_activation = ('sigmoid' if num_class<=2 else 'softmax')
    
    inputs = Input(shape=(timesteps, num_feature))
    x_nn = inputs
    if mask_value!=None: x_nn = Masking(mask_value=mask_value)(x_nn)
    lstm_out = LSTM(n_hidden, return_sequences=True)(x_nn)
    inter = lstm_out
    for i in range(inter_num):
        inter = TimeDistributed(Dense(inter_nodes, activation='relu'))(inter)
        inter = TimeDistributed(Dropout(drop_rate))(inter)
    
    outputs = TimeDistributed(Dense(num_class, activation=final_activation))(inter)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

def fit_model_keras(model, X_train, y_train, batch_size=None, validation_data=tuple(), epochs=50, verbose=1):
       
    num_unique = np.unique(np.argmax(y_train[:,-1,:], axis=1), return_counts=True)
    num_sample = num_unique[1]
    weight_list = 1/(num_sample/num_sample.max())
    sample_weight = np.array([weight_list[list(num_unique[0]).index(label)] for label in np.argmax(y_train[:,-1,:], axis=1)])
               
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=int(epochs*0.2))
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              sample_weight = sample_weight,
              validation_data=validation_data if len(validation_data)>0 else None,
              callbacks=[early_stopping],
              verbose=verbose              
             )
    
    return model

def fit_model(model, X_train, y_train, batch_size=None, validation_data=tuple(), epochs=50, 
              optimizer='Adam', learning_rate=1e-2, device=None, verbose=1):
    
    
    #For early stopping
    patience = int(epochs*0.2)
    min_val_loss = float('inf')
    stop_count = 0
    
    #For saving tentative best model
    best_model_param = None
        
    #Use GPU if it is available when "device" is not specified
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    #Set up parallel processing when multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    opt = None
    if optimizer=='SGD': opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer=='RMSprop': opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    if optimizer=='Adadelta': opt = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    if optimizer=='Adam': opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer=='QHAdam': opt = QHAdam(model.parameters(), lr=learning_rate, nus=(0.7, 1.0), betas=(0.95, 0.998))
    
    num_unique = np.unique(np.argmax(y_train[:,-1,:], axis=1), return_counts=True)
    num_sample = num_unique[1]
    weight_list = 1/(num_sample/num_sample.max())
    class_weight = torch.as_tensor(weight_list, device=device)
    #sample_weight = np.array([weight_list[list(num_unique[0]).index(label)] for label in np.argmax(y_train[:,-1,:], axis=1)])
       
    critertion = nn.CrossEntropyLoss(weight=class_weight, reduction='none') if class_weight.shape[0]>2\
                                     else nn.BCELoss(weight=class_weight, reduction='none')
              
    path = r'.\pytorch_check_point'
    os.makedirs(path, exist_ok=True)
        
    for i in range(epochs):
        
        #train        
        model.train()
        
        shuffled_idx = random.sample(list(range(len(y_train))), len(y_train))          
        train_loss = 0
        for st in list(range(0, len(y_train), (batch_size if batch_size!=None else len(y_train)))):
            en = st + (batch_size if batch_size!=None else len(y_train))
            if en>len(y_train): en = len(y_train)            
            target_idx = shuffled_idx[st:en]
            
            inputs = torch.as_tensor(X_train[target_idx], device=device)
            inputs.requires_grad = True
            target = torch.as_tensor(np.argmax(y_train[target_idx], axis=2), device=device)
            
            #Predict the train data
            out = model(inputs)
            
            #Calculate average loss of all timesteps per sample
            loss = 0
            for j in range(out.shape[0]):
                loss += critertion(torch.log(out[j]), target[j]).mean() #Softmax is part of the model architecture
            train_loss += loss.item()
            
            #Get the avrage loss of one batch
            loss = loss.mean()
            
            #Initialize gradient
            opt.zero_grad()
            
            #Calculate gradient
            loss.backward()
            
            #Update the parameter
            opt.step()            
            
            del loss, inputs, target, out

        #Get the average train loss of all samples
        train_loss /= len(y_train)                     
                
            
            #train_loss += model.train_on_batch(X_train[target_idx], y_train[target_idx],
            #                                  sample_weight=sample_weight[target_idx])            
                    
        #test
        if len(validation_data)>0:      
            X_test = validation_data[0]
            y_true = validation_data[1]
            
            val_loss = test_model(model, X_test, y_true, batch_size=batch_size, critertion=critertion, device=device)
                                            
                #val_loss += model.evaluate(X_test[target_idx], y_true[target_idx], verbose=0)
                
            if verbose==1:
                print('epoch{0}:\t train_loss = {1}\t val_loss = {2}'.format(i+1, train_loss, val_loss))
                
            #early stop
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model_param = model.state_dict() #Save tentative best model
                stop_count = 0
                
            else:
                stop_count += 1
                if stop_count > patience: break
                
        else:
            if verbose==1:
                print('epoch{0}:\t train_loss = {1}'.format(i+1, train_loss))
           
    del class_weight
    
    if best_model_param!=None:
        model.load_state_dict(best_model_param)
        min_val_loss = val_loss
    
    return model


def test_model(model, X_test, y_true, batch_size=None, final_timestep=False, critertion=None, device=None, return_proba=False):

    model.eval()    

    #Set the loss function according to the number of classes when "critertion" is not specified
    if critertion==None:
        num_unique = np.unique(np.argmax(y_true[:,-1,:], axis=1), return_counts=True)   
        critertion = nn.CrossEntropyLoss(reduction='none') if num_unique[1].shape[0]>2 else nn.BCELoss(reduction='none') 

    #Use GPU if it is available when "device" is not specified
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    #Process per batch
    shuffled_idx = random.sample(list(range(len(y_true))), len(y_true))  
    val_loss = 0
    probas = []
    for st in list(range(0, len(y_true), (batch_size if batch_size!=None else len(y_true)))):
        
        #Get Index of one batch
        en = st + (batch_size if batch_size!=None else len(y_true))
        if en>len(y_true): en = len(y_true)            
        target_idx = shuffled_idx[st:en]
         
        #Predict
        with torch.no_grad():
            inputs = torch.as_tensor(X_test[target_idx], device=device)
            target = torch.as_tensor(np.argmax(y_true[target_idx], axis=-1), device=device)
            
            #Predict the test data
            out = model(inputs)
            
            #Calculate loss (per sample)
            for j in range(out.shape[0]):
                
                #Loss of one batch is only final timestep or average of all timesteps
                loss = critertion(torch.log(out[j]), target[j]) #Softmax is part of the model architecture
                val_loss += loss[-1].item() if final_timestep else loss.mean().item()
                del loss
                
            probas.append(out.to(torch.device('cpu')).clone().numpy())      
              
            del inputs, target, out
                    
    #Get average of val_loss (per sample)
    val_loss /= len(y_true)
    
    return (val_loss, {'proba':np.concatenate(probas, axis=0), 'label':y_true[shuffled_idx]}) if return_proba else val_loss

def validate_multidata(dataset_list, param_dict=dict(), return_proba=False, return_model=False, 
                       use_CPU=False, verbose=1):
    
    #Use GPU if it is available
    device = 'cpu' if use_CPU else 'cuda' if torch.cuda.is_available() else 'cpu'
        
    score_list = []
    proba_list = []
    model_list = []
    for i, data in enumerate(dataset_list):
        if verbose==1: print('Dataset {0}/{1}:'.format(i+1, len(dataset_list)))
        
        X_train = data['train']['X']
        y_train = data['train']['y']
        X_test = data['test']['X']
        y_true = data['test']['y']
                   
        timesteps = X_train.shape[1]
        num_feature = X_train.shape[2]
        num_class = y_train.shape[2]
        
        model = get_LSTM(timesteps, num_feature, num_class, **param_dict)
        model = fit_model_keras(model, X_train, y_train, batch_size=None, validation_data=(X_test, y_true),
                                epochs=50, verbose=verbose)
        one_score = model.evaluate(X_test, y_true, verbose=verbose)
        probas = {'proba':model.predict(X_test), 'label':y_true}
        
        '''
        model = RecurrentNN(num_feature, num_class, n_hidden=param_dict['n_hidden'], inter_num=param_dict['inter_num'],
                            inter_nodes=param_dict['inter_nodes'], drop_rate=param_dict['drop_rate']).to(device)        
        model = fit_model(model, X_train, y_train, batch_size=5000, validation_data=(X_test, y_true),
                          optimizer=param_dict['optimizer'], learning_rate=param_dict['learning_rate'],
                          device=device, verbose=verbose)        
        one_score, probas = test_model(model, X_test, y_true, batch_size=5000, final_timestep=True, device=device, return_proba=True)        
        '''
        
        score_list.append(one_score)
        proba_list.append(probas)
        model_list.append(model)#.state_dict())
        
        #score += model.evaluate(X_test, y_true, verbose=0)
                
        # Memory release
        del model
        #torch.cuda.empty_cache()
        clear_session()
                
    return score_list if not (return_proba and return_model) else (score_list, proba_list, model_list)


def tuning_model(dataset_list, max_iter=50, random_seed=99, fitting_verbose=0):
    
    optimizer_list = ['SGD', 'RMSprop', 'Adadelta', 'Adam']#, 'QHAdam']
    hyperopt_parameters = {'n_hidden':hp.randint('n_hidden', 1024)+1,
                           'inter_num':hp.randint('inter_num', 10+1),
                           'inter_nodes':hp.randint('inter_nodes', 1024)+1,
                           'drop_rate':hp.uniform('drop_rate', 0.0, 1.0),
                           'optimizer':hp.choice('optimizer',optimizer_list),
                           'learning_rate':hp.loguniform('learning_rate', -8, 0),
                          }
    
    def objective(args):     
    
        score_list = validate_multidata(dataset_list, param_dict=args, verbose=fitting_verbose)
        result = sum(score_list)/len(score_list)
    
        return {'loss': result,
                'status': STATUS_OK,
                'estimate_time': time.time(),
                'vals': args,
               }
    
    trials = Trials()  
        
    best = fmin(
            objective,
            hyperopt_parameters,
            algo=tpe.suggest,
            max_evals=max_iter,
            trials=trials,
            rstate = np.random.RandomState(random_seed),
            verbose=1
        )
    
    best['optimizer'] = optimizer_list[best['optimizer']]
    
    return best

    
