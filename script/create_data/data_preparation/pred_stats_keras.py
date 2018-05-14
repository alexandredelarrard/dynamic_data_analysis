# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:33:29 2018

@author: User
"""

import pandas as pd
import os
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def extract_games_number(x):
    try:
        x = re.sub(r'\([^)]*\)', '', x)
        x = x.replace(" ",",").replace("-",",").split(",")
        return sum([int(a) for a in x if a !=""])
    
    except Exception:
        print(x)
        
        

def set_extract(x, taille):
    if len(x)>=taille:    
        return re.sub(r'\([^)]*\)', '', x[taille-1])
    else:
        return np.nan
     
        
def count_sets(x):
    x = re.sub(r'\([^)]*\)', '', x)
    return x.count("-")
    

def games_extract(x, w_l):
    
    try:
        if x != "RET" and x != "W/O" and x != "W/O " and x != "DEF" and pd.isnull(x) == False and x != "":
            return x.split("-")[w_l]
        else:
            return np.nan
    except Exception:
        print(x)
        
def data_prep(data):
    
    cols = [ 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', "indoor_flag", # 'minutes',
            'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'score',
            'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'best_of', 'winner_hand', 'winner_ht','round', 'masters', 'Currency',
            'winner_age', 'winner_rank', 'winner_rank_points', 'loser_id', 'loser_hand', 'loser_ht', 'match_num',"prize",'missing_stats',
            'loser_age', 'loser_rank', 'loser_rank_points', 'elo1', 'elo2', 'surface', 'draw_size', 'tourney_level', "total_games"]
    
    data["loser_hand"] = np.where(data["loser_hand"] == "Right-Handed", 1, 0).astype(int)
    data["winner_hand"] = np.where(data["winner_hand"] == "Right-Handed", 1, 0).astype(int)
    data["indoor_flag"] = np.where(data["indoor_flag"] == "Outdoor", 0,1).astype(int)
    
    data = data[cols]
    data["target"] = 1
    
    ### take care of round
    dico = {"R32": 32, "R16": 16, "R64": 64, "R128":128, "QF": 8, "SF": 4, "F":2, "RR": 4}
    data['round'] = data['round'].map(dico).astype(int)  
    
    for col in ['tourney_level', 'masters', 'Currency', "surface"]:
        a = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, a], axis=1)
        del data[col]
        
    data2 = data.copy()
    for col in [x for x in data2.columns if "w_" in x and "draw" not in x]:
        data2.rename(columns={col : col.replace("w_","l_"), col.replace("w_","l_") : col}, inplace = True)
        
    for col in [x for x in data2.columns if "_w" in x]:
        data2.rename(columns={col : col.replace("_w","_l"), col.replace("_w","_l") : col}, inplace = True)
    
    for col in [x for x in data2.columns if "winner_" in x]:
        data2.rename(columns={col : col.replace("winner_","loser_"), col.replace("winner_","loser_") : col}, inplace = True)
    
    data2.rename(columns={"elo1" : "elo2", "elo2" : "elo1"}, inplace = True)
    data2["target"] = 0
    
    total_data = pd.concat([data,data2],axis=0)
    
    total_data['N_set']  = total_data['score'].apply(lambda x : count_sets(x))
    total_data['score'] = total_data['score'].str.split(" ")
    total_data["S1"] = total_data['score'].apply(lambda x : set_extract(x, 1))
    total_data["S2"] = total_data['score'].apply(lambda x : set_extract(x, 2))
    total_data["S3"] = total_data['score'].apply(lambda x :set_extract(x, 3))
    total_data["S4"] = total_data['score'].apply(lambda x :set_extract(x,4))
    total_data["S5"] = total_data['score'].apply(lambda x :set_extract(x, 5))
    
    for i in range(1,6):
        for j, w_l in enumerate(["w", "l"]):
            total_data[w_l + "_S%i"%i] = total_data["S%i"%i].apply(lambda x : games_extract(x, j)).fillna(-99).astype(int)
    
    total_data = total_data.drop(["S1","S2","S3","S4","S5", "score"], axis= 1)        
    train, test = total_data.loc[data['missing_stats'] !=1], total_data.loc[data['missing_stats'] ==1]
    
    return train, test

def model_def(inputs, outputs):
    
    model = Sequential()
#    model.add(BatchNormalization(axis=1))
    model.add(Dense(32, input_shape=(inputs,), kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(16, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(outputs))
    
    adam = optimizers.Adam(lr=0.00003)
    model.compile(loss='mean_squared_error', 
                  optimizer=adam,
                  metrics=["mse", "mae"])
    model.summary()
    
    return model


def callback(file_path, patience):
    checkpoint = ModelCheckpoint(file_path,
                                 monitor='val_loss', verbose=0, 
                                 save_best_only=True, mode='min')
    
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    callbacks_list = [checkpoint, early]
    return callbacks_list


def modelling(train, test):
    
    cols_to_predict = ['w_svpt', 'l_svpt', 'w_1stIn', 'l_1stIn'] # , 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 
#                    'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    
    n_splits = 10
    batch_size = 128
    
    train = train.reset_index(drop=True)
    train = train.drop(['w_ace', 'w_df', 'l_ace', 'l_df', 'l_1stWon', 'l_2ndWon', 
                        'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'w_1stWon', 'w_2ndWon', 'w_SvGms',  # 'minutes', 
                        'w_bpSaved', 'w_bpFaced'],axis= 1)
    
    cols_to_train = list(set(train.columns) - set(cols_to_predict))
    
    X_train, output_train = train[cols_to_train], train[cols_to_predict]
    X_test = np.array(test[cols_to_train])
    
    inputs = X_train.shape[1]
    outputs = len(cols_to_predict)
    
    file_path = os.environ["DATA_PATH"] + "/models/missing_values_prediction/weights/model_weights.hdf5"
    
    total_acc = []
    total_loss = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=36750) 
    i = 0

    for train_index, test_index in kfold.split(X_train, output_train):
        
        x_train, y_train = np.array(X_train.loc[train_index]), np.array(output_train.loc[train_index])
        x_valid , y_valid  = np.array(X_train.loc[test_index]), np.array(output_train.loc[test_index])
        
        scaler = StandardScaler().fit(x_train)
        scaler.transform(x_train)  
        scaler.transform(x_valid) 
        scaler.transform(X_test) 
        
        file_path = r"D:\projects\jigsaw\scripts\improve_activ_lstm\model_weights_%i.hdf5"%i
        callbacks_list = callback(file_path, patience = 25)
        model = model_def(inputs, outputs)
    
        history = model.fit(x_train, y_train, batch_size= batch_size, epochs= 250, 
                            validation_data = (x_valid , y_valid), callbacks=callbacks_list)
        
        model.load_weights(file_path)
        score = model.evaluate(x_valid , y_valid, verbose=0)
        
        print('Kfold : {0} : mse: {1}, mae : {2}'.format(i, score[0], score[1]))
        total_loss.append(score[0])
        total_acc.append(score[1])
        
        
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('model square error')
        plt.ylabel('mean_squared_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for loss
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('model mean_absolute_error')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        if i ==0:
            overall_pred = model.predict(X_test,batch_size=1024)
        else:
            overall_pred += model.predict(X_test, batch_size=1024)
            
        y_pred  = model.predict(x_valid, batch_size=1024)
        i +=1
        
        for i in range(y_valid.shape[1]):
            print("MAE for {0} = {1}".format(cols_to_predict[i], np.mean((abs(y_valid[:,i] - y_pred[:,i])/y_valid[:,i]))))

    overall_pred = overall_pred/float(n_splits)
    y_pred = y_pred/float(n_splits)
    
    print("Finish mse tot = %f +/- %f, mae tot = %f +/- %f"%(np.mean(total_acc), np.std(total_acc), np.mean(total_loss), np.std(total_loss)))
   
    return overall_pred, y_pred, y_train
    
    
if __name__ == "__main__":
    data = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/historical/matches_elo_V1.csv") # 3576
    
    data["total_games"] = data["score"].apply(lambda x : extract_games_number(x))
    
    ### add l svpt
    data.loc[data["l_svpt"] == 0, 'missing_stats'] = 1 # 3604 (+28)
    data.loc[data["w_svpt"] < data["total_games"]*4/2, 'missing_stats'] = 1 #3714 (+110)
    
    train, test = data_prep(data)
    overall_pred, y_pred, y_train = modelling(train, test)
    