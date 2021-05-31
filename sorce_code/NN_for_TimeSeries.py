import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import TimeDistributed
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam 
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time

def get_LSTM(timesteps, num_feature, num_class, n_hidden=100, inter_num=0, inter_nodes=10, drop_rate=0.5,
             mask_value=None, optimizer=Adam(lr=0.001), loss=None, final_activation=None):
    
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


def fit_model(model, X_train, y_train, validation_data=tuple(), epochs=50, verbose=1):
    
    num_unique = np.unique(np.argmax(y_train[:,-1,:], axis=1), return_counts=True)
    num_sample = num_unique[1]
    weight_list = 1/(num_sample/num_sample.max())
    sample_weight = np.array([weight_list[list(num_unique[0]).index(label)] for label in np.argmax(y_train[:,-1,:], axis=1)])
        
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=int(epochs*0.2))
    model.fit(X_train, y_train,
              epochs=epochs,
              sample_weight = sample_weight,
              validation_data=validation_data if len(validation_data)>0 else None,
              callbacks=[early_stopping],
              verbose=verbose              
             )
    
    return model


def validate_multidata(dataset_list, param_dict=dict(), return_proba=False, return_model=False, verbose=1):
    
    score = 0
    proba_list = []
    model_list = []
    for data in dataset_list:
        X_train = data['train']['X']
        y_train = data['train']['y']
        X_test = data['test']['X']
        y_true = data['test']['y']
                   
        timesteps = X_train.shape[1]
        num_feature = X_train.shape[2]
        num_class = y_train.shape[2]
        
        model = get_LSTM(timesteps, num_feature, num_class, **param_dict)
        model = fit_model(model, X_train, y_train, validation_data=(X_test, y_true), verbose=verbose)
        score += model.evaluate(X_test, y_true, verbose=0)
        
        if return_proba: proba_list.append(model.predict(X_test))
        if return_model: model_list.append(model)
        
    return (score / len(dataset_list)) if not (return_proba and return_model) else ((score / len(dataset_list)), proba_list, model_list)


def tuning_model(dataset_list, max_iter=50, random_seed=99, fitting_verbose=0):
    
    optimizer_list = [SGD(lr=0.01), RMSprop(lr=0.001), Adadelta(lr=1.0), Adam(lr=0.001)]
    hyperopt_parameters = {'n_hidden':hp.randint('n_hidden', 1024)+1,
                           'optimizer':hp.choice('optimizer',optimizer_list),
                           'inter_num':hp.randint('inter_num', 10+1),
                           'inter_nodes':hp.randint('inter_nodes', 1024)+1,
                           'drop_rate':hp.uniform('drop_rate', 0.0, 1.0),
                          }
    
    def objective(args):       
    
        result = validate_multidata(dataset_list, param_dict=args, verbose=fitting_verbose)
    
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
    
