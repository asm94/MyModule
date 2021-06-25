import numpy as np
from sklearn.model_selection import StratifiedKFold
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
from hyperopt.fmin import generate_trials_to_calculate
from functools import partial
import math
import time
import random
import os
import sys


class LSTMclassifier(nn.Module):
    def __init__(self, num_feature, num_class, n_hidden=100, num_layer=1, lstm_drop_rate=0.5,
                 inter_num=0, inter_nodes=10, inter_drop_rate=0.5):
        super(LSTMclassifier, self).__init__()
        
        if num_layer<=1: lstm_drop_rate = 0.0            
        self.rnn = nn.LSTM(input_size=num_feature, hidden_size=n_hidden, num_layers=int(num_layer),
                           dropout=lstm_drop_rate, batch_first=True)
        
        inter_layers = []
        for i in range(inter_num):            
            inter_layers.append(nn.Linear(in_features=(n_hidden if i==0 else inter_nodes), out_features=inter_nodes))
            inter_layers.append(nn.ReLU())
            inter_layers.append(nn.Dropout(p=inter_drop_rate))
        self.inter_layer = nn.Sequential(*inter_layers)
        
        out_nodes = inter_nodes if inter_num>0 else n_hidden
        self.output_layer = nn.Sequential(nn.Linear(in_features=out_nodes, out_features=num_class),
                                          nn.Softmax(dim=-1) if num_class>2 else nn.Sigmoid())
    
    def forward(self, x, last_only=False):
        
        rnn_out, _ = self.rnn(x)
        
        inter = self.inter_layer(rnn_out[:,-1,:] if last_only else rnn_out)                        
        output = self.output_layer(inter)
        
        return output
    

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


def generate_batch(index, label=[], batch_size=None):    
    if batch_size==None or len(index)<=batch_size:
        yield index
        
    else:
        if len(label)==0:
            shuffled_idx = random.sample(list(index), len(index))
            for st in range(0, len(shuffled_idx), batch_size):
                en = st + batch_size
                if en>len(shuffled_idx): en = len(shuffled_idx)
                yield shuffled_idx[st:en]
                
        else:
            if len(index)!=len(label): sys.exit('ERROR: Length of "index" is not equal to length of "label".')
            
            index = np.array(index)
            label = np.array(label)
            
            uni, cnt = np.unique(label, return_counts=True)
            
            idx_dic = {}
            for l in uni:
                idx_dic[l] = np.random.choice(index[np.where(label==l)], size=len(index[np.where(label==l)]), replace=False)
            
            amount = np.zeros(uni.shape, dtype='int')
            for i, st in enumerate(range(0, len(index), batch_size)):
                en = st + batch_size
                if en>len(index): en = len(index)
                    
                cnt_rate = (cnt-amount)/(cnt-amount).sum()
                batch_by_labels = np.round(cnt_rate*(en-st)).astype('int')
                
                sampling_gap = batch_by_labels.sum()-(en-st)                
                if sampling_gap>0:
                    rank_desc = np.argsort(np.argsort((-1)*batch_by_labels))
                    batch_by_labels[np.where(rank_desc<sampling_gap)] -= 1
                elif sampling_gap<0:
                    rank_asc = np.argsort(np.argsort(batch_by_labels))
                    batch_by_labels[np.where(rank_asc<abs(sampling_gap))] += 1                
                
                yield np.concatenate([idx_dic[l][amount[l]:amount[l]+batch_by_labels[l]] for l in uni])
                            
                amount += batch_by_labels
                

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
              optimizer='Adam', learning_rate=1e-2, lr_schedule=False, weight_alpha=1.0, device=None, verbose=1):
        
    #For early stopping
    patience = 50 if epochs>1000 else 20
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
        
    #Define optimizer
    opt = None
    if optimizer=='SGD': opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer=='RMSprop': opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    if optimizer=='Adadelta': opt = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    if optimizer=='Adam': opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer=='QHAdam': opt = QHAdam(model.parameters(), lr=learning_rate)
        
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda step: 0.1 if step>=patience else 1)
            
    #Caluculate the weight for each class
    num_unique = np.unique(np.argmax(y_train[:,-1,:], axis=1), return_counts=True)
    num_sample = num_unique[1]
    weight_list = np.power(1/(num_sample/num_sample.max()), weight_alpha)
     
    #Apply the weight for the class at the final time-step to all time-steps of the same sample
    sample_weight = np.array([[weight_list[list(num_unique[0]).index(np.argmax(sample[-1], axis=-1))]]*y_train.shape[1] for sample in y_train])
        
    #Define loss function
    criterion = nn.CrossEntropyLoss(reduction='none') if len(num_unique[0])>2 else nn.BCELoss(reduction='none')                                    
              
    path = r'.\pytorch_check_point'
    os.makedirs(path, exist_ok=True)
        
    #Training and testing
    for i in range(epochs):        
        #train 
        model.train()
             
        train_loss = []
        batch_generator = generate_batch(index=list(range(len(y_train))),
                                         label=np.argmax(y_train[:,-1,:], axis=-1),
                                         batch_size=batch_size)
        for target_idx in batch_generator:
                   
            inputs = torch.as_tensor(X_train[target_idx], device=device, dtype=torch.float32)
            inputs.requires_grad = True
            target = torch.as_tensor(np.argmax(y_train[target_idx], axis=-1), device=device, dtype=torch.int64)
            weight = torch.as_tensor(sample_weight[target_idx], device=device, dtype=torch.float32)
            
            #Predict the train data
            out = model(inputs)
            
            #Calculate average loss of all samples (sample*timestep)
            loss = None
            if len(out.shape)==3:
                loss = criterion(torch.log(out.reshape(-1, out.size(-1))+1e-40), target.reshape(-1))
                loss = loss*weight.reshape(-1)
            else:
                loss = criterion(torch.log(out+1e-40), target[:,-1])
                loss = loss*weight[:,-1]            
            
            loss = loss.mean()
            train_loss.append(loss.item())
                        
            #Initialize gradient
            opt.zero_grad()
            
            #Calculate gradient
            loss.backward()
            
            #Update the parameter
            opt.step()     
            
            del loss, inputs, target, weight, out

        #Get the average train loss of all samples
        train_loss = sum(train_loss)/len(train_loss)             
                    
        #test
        if len(validation_data)>0:      
            X_test = validation_data[0]
            y_true = validation_data[1]
            
            val_loss = test_model(model, X_test, y_true, batch_size=batch_size, criterion=None, device=device)
                                                     
            if verbose==1:
                print('epoch{0}:\t train_loss = {1}\t val_loss = {2}'.format(i+1, train_loss, val_loss))
                              
            #early stop
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model_param = model.state_dict() #Save tentative best model
                stop_count = 0
                
            elif lr_schedule:
                scheduler.step()
                if scheduler.get_lr()[0]<=(learning_rate*0.1): stop_count += 1
                if stop_count > patience: break
                    
            else:
                stop_count += 1
                if stop_count > patience: break
                
        else:
            if verbose==1:
                print('epoch{0}:\t train_loss = {1}'.format(i+1, train_loss))
                        
    del sample_weight
    
    if best_model_param!=None:
        model.load_state_dict(best_model_param)
        min_val_loss = val_loss
    
    return model


def test_model(model, X_test, y_true, batch_size=None, final_timestep=False, criterion=None, device=None, return_proba=False):

    model.eval()

    #Set the loss function according to the number of classes when "critertion" is not specified
    if criterion==None:
        num_unique = np.unique(np.argmax(y_true[:,-1,:], axis=1), return_counts=True)   
        criterion = nn.CrossEntropyLoss(reduction='none') if num_unique[1].shape[0]>2 else nn.BCELoss(reduction='none') 

    #Use GPU if it is available when "device" is not specified
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    #Process per batch 
    val_loss = 0
    probas = []
    labels = []
    
    #Get batch generator considered class balance
    batch_generator = generate_batch(index=list(range(len(y_true))),
                                     label=np.argmax(y_true[:,-1,:], axis=-1),
                                     batch_size=batch_size)
    for target_idx in batch_generator:
             
        #Predict
        with torch.no_grad():
            inputs = torch.as_tensor(X_test[target_idx], device=device, dtype=torch.float32)
            target = torch.as_tensor(np.argmax(y_true[target_idx], axis=-1), device=device, dtype=torch.int64)
                        
            #Predict the test data
            out = model(inputs)
        
            #Calculate loss (average of all samples (sample*timesteps))
            if len(out.shape)==3:
                if final_timestep: val_loss += criterion(torch.log(out[:,-1,:].reshape(-1, out.size(-1))+1e-40),
                                                         target[:,-1].reshape(-1)).mean().item()
                else:
                    val_loss += criterion(torch.log(out.reshape(-1, out.size(-1))+1e-40), target.reshape(-1)).mean().item()
                    
            else:
                val_loss += criterion(torch.log(out+1e-40), target[:,-1]).mean().item()
                        
            probas.append(out.to(torch.device('cpu')).clone().numpy())  
            labels.append(y_true[target_idx] if len(out.shape)==3 else y_true[target_idx][:,-1,:])
              
            del inputs, target, out
                    
    #Get average of val_loss (per sample)
    val_loss /= len(probas)
    
    return (val_loss, {'proba':np.concatenate(probas, axis=0), 'label':np.concatenate(labels, axis=0)}) if return_proba else val_loss

def validate_multidata(dataset_list, param_dict=dict(), return_proba=False, return_model=False, 
                       use_CPU=False, interrupt_max_border=float('inf'), lr_schedule=False, epoch=5000, random_seed=99, verbose=1):
    
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
        '''
        model = get_LSTM(timesteps, num_feature, num_class, **param_dict)
        model = fit_model_keras(model, X_train, y_train, batch_size=None, validation_data=(X_test, y_true),
                                epochs=50, verbose=verbose)
        one_score = model.evaluate(X_test, y_true, verbose=verbose)
        probas = {'proba':model.predict(X_test), 'label':y_true}
        '''
        
        torch.manual_seed(random_seed)
        model = LSTMclassifier(num_feature,
                               num_class,
                               n_hidden=param_dict['n_hidden'] if 'n_hidden' in param_dict.keys() else num_feature,
                               num_layer=param_dict['num_lstm_layer'] if 'num_lstm_layer' in param_dict.keys() else 1,
                               lstm_drop_rate=param_dict['lstm_drop_rate'] if 'lstm_drop_rate' in param_dict.keys() else 0.0,                           
                               inter_num=param_dict['inter_num'] if 'inter_num' in param_dict.keys() else 1,
                               inter_nodes=param_dict['inter_nodes'] if 'inter_nodes' in param_dict.keys() else num_feature,
                               inter_drop_rate=param_dict['inter_drop_rate'] if 'inter_drop_rate' in param_dict.keys() else 0.0,
                              ).to(device)
        
        model = fit_model(model, X_train, y_train, batch_size=1000, validation_data=(X_test, y_true), epochs=epoch,
                          optimizer=param_dict['optimizer'] if 'optimizer' in param_dict.keys() else 'Adam',
                          learning_rate=param_dict['learning_rate'] if 'learning_rate' in param_dict.keys() else 0.01,
                          weight_alpha=param_dict['weight_alpha'] if 'weight_alpha' in param_dict.keys() else 1.0,
                          lr_schedule=lr_schedule, device=device, verbose=verbose)
        one_score, probas = test_model(model, X_test, y_true, batch_size=2000, final_timestep=True, device=device, return_proba=True)        
               
        score_list.append(one_score)
        proba_list.append(probas)
        model_list.append(model.state_dict())
                                
        # Memory release
        del model
        torch.cuda.empty_cache()
        clear_session()
        
        
        # Interrupt if score more than "interrupt_border"
        if interrupt_max_border<sum(score_list) and (i+1)<len(dataset_list):
            print(score_list)
            break
            
                
    return score_list if not (return_proba and return_model) else (score_list, proba_list, model_list)


def tuning_model(dataset_list, max_iter=50, interrupt_value=1.0, init_vals=None, random_seed=99, fitting_verbose=0):
      
    def objective(args, options=None, interrupt_val=float('inf')):
        if options!=None: args = {**args, **options}
    
        print('param:{0}'.format(args))
        
        score = validate_multidata(dataset_list, param_dict=args, interrupt_max_border=len(dataset_list)*interrupt_val,
                                   epoch=5000, random_seed=random_seed, verbose=fitting_verbose)
        result = sum(score)/len(score) if type(score)==type([0]) else score
        if result!=result: result = float('inf')
        
        print('->loss:{0}\n'.format(result))
    
        return {'loss': result,
                'status': STATUS_OK,
                'estimate_time': time.time(),
                'vals': args,
               }
        
    optimizer_list = ['SGD', 'RMSprop', 'Adadelta', 'Adam', 'QHAdam']        
    architect_parameters = {
        'n_hidden':          hp.randint('n_hidden', 512)+1,
        'num_lstm_layer':    hp.randint('num_lstm_layer', 8)+1,
        'inter_num':         hp.randint('inter_num', 10+1),
        'inter_nodes':       hp.randint('inter_nodes', 512)+1,
        'optimizer':         hp.choice('optimizer',optimizer_list),
    }
    other_parameters = {
        'lstm_drop_rate':    hp.uniform('lstm_drop_rate', 0.0, 0.8),
        'inter_drop_rate':   hp.uniform('inter_drop_rate', 0.0, 0.8),
        'learning_rate':     hp.loguniform('learning_rate', math.log(1e-3), math.log(1e-1)),
        'weight_alpha':      hp.uniform('weight_alpha', 0.0, 2.0),
    }

    init_architect = {}
    init_rate = {}
    if init_vals!=None:
        init_architect = dict([item for item in init_vals.items() if item[0] in architect_parameters.keys()])
        init_rate = dict([item for item in init_vals.items() if item[0] in other_parameters.keys()])
    
        init_architect['n_hidden'] = init_vals['n_hidden']-1
        init_architect['num_lstm_layer'] = init_vals['num_lstm_layer']-1
        init_architect['inter_nodes'] = init_vals['inter_nodes']-1
        init_architect['optimizer'] = optimizer_list.index(init_vals['optimizer'])
                
    trials = generate_trials_to_calculate([init_architect]) if init_vals!=None else Trials()
        
    best_first = fmin(
        partial(objective, options=init_rate),#, interrupt_val=interrupt_value),
        architect_parameters,
        algo=tpe.suggest,
        max_evals=max_iter,
        trials=trials,
        #rstate = np.random.RandomState(random_seed),
        verbose=1
    )
    out_trials = [trials]
    
    #best_loss = min(trials.losses())
    best_first['n_hidden'] += 1
    best_first['num_lstm_layer'] += 1
    best_first['inter_nodes'] += 1
    best_first['optimizer'] = optimizer_list[best_first['optimizer']]    
            
    trials = generate_trials_to_calculate([init_rate]) if init_vals!=None else Trials()
        
    best_sec = fmin(
        partial(objective, options=best_first, interrupt_val=interrupt_value),
        other_parameters,
        algo=tpe.suggest,
        max_evals=max_iter,
        trials=trials,
        #rstate = np.random.RandomState(random_seed),
        verbose=1
    )
    out_trials.append(trials)
            
    best = {**best_first, **best_sec}
    
    return best, out_trials
   
