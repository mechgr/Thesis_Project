
import os, psutil
p = psutil.Process(os.getpid())

import tensorflow as tf
import random
import numpy as np
import time
from tensorflow import keras
#import Data_DH_1
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from tensorflow.keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer

import numpy as np
import pandas as pd
import gc

def data_read(s, r):
    Label = pd.read_csv('Label.csv')

    vibr_list = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3', 'Fault-4', 'Fault-5']
    vibr_all_normal, vibr_all_fault1, vibr_all_fault2, vibr_all_fault3, vibr_all_fault4, vibr_all_fault5 = [], [], [], [], [], []

    for i in range(1, 7):
        vibr = pd.read_csv('{}.csv'.format(vibr_list[i-1]), nrows=r).drop('Index', axis=1).reset_index(drop=True).values.reshape([-1, 4000])
        vibr1 = np.array(vibr)

        if i == 1:
            for j in range(1, r+1):
                vibr_all_normal.append(vibr1[j-1])
        elif i == 2:
            for j in range(1, r+1):
                vibr_all_fault1.append(vibr1[j-1])
        elif i == 3:
            for j in range(1, r+1):
                vibr_all_fault2.append(vibr1[j-1])
        elif i == 4:
            for j in range(1, r+1):
                vibr_all_fault3.append(vibr1[j-1])
        elif i == 5:
            for j in range(1, r+1):
                vibr_all_fault4.append(vibr1[j-1])
        else:
            for j in range(1, r+1):
                vibr_all_fault5.append(vibr1[j-1])

        del vibr, vibr1
        gc.collect()

    vibr_all_normal = pd.DataFrame(vibr_all_normal).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault1 = pd.DataFrame(vibr_all_fault1).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault2 = pd.DataFrame(vibr_all_fault2).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault3 = pd.DataFrame(vibr_all_fault3).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault4 = pd.DataFrame(vibr_all_fault4).reset_index(drop=True).values.reshape((-1, 1))
    vibr_all_fault5 = pd.DataFrame(vibr_all_fault5).reset_index(drop=True).values.reshape((-1, 1))

    X_dataset = pd.concat([pd.DataFrame(vibr_all_normal),\
                              pd.DataFrame(vibr_all_fault1),\
                              pd.DataFrame(vibr_all_fault2),\
                              pd.DataFrame(vibr_all_fault3),\
                              pd.DataFrame(vibr_all_fault4),\
                              pd.DataFrame(vibr_all_fault5)], axis=0).reset_index(drop=True).values.reshape([-1, 1])

    num_samples = int((len(vibr_all_normal)/s))*6
    X_data = pd.concat([pd.DataFrame(X_dataset[0:num_samples * s])], axis=1).reset_index(drop=True).values.reshape((-1, s, 1))

    y_data = pd.concat([Label["L0"].loc[0:num_samples/6-1],\
                         Label["L1"].loc[0:num_samples/6-1],\
                         Label["L2"].loc[0:num_samples/6-1],\
                         Label["L3"].loc[0:num_samples/6-1],\
                         Label["L4"].loc[0:num_samples/6-1],\
                         Label["L5"].loc[0:num_samples/6-1]], axis=0).reset_index(drop=True).values.reshape((-1, 1))

    return X_data, y_data


seq_length = 1024
line_num = 1000

# Data
X_data, y_data = data_read(seq_length, line_num)

print("Total data volume: {}".format(len(X_data)))

# Shuffle
Data = list(zip(X_data, y_data))
random.shuffle(Data)
X_data, y_data = zip(*Data)
X_data, y_data = np.array(X_data), np.array(y_data)

# Data split
X_train, y_train = X_data[0:int(len(X_data)*0.7)-1], y_data[0:int(len(y_data)*0.7)-1]
X_valuate, y_valuate = X_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1], y_data[int(len(X_data)*0.7):int(len(X_data)*0.9)-1]
X_test, y_test = X_data[int(len(X_data)*0.9):len(X_data)-1], y_data[int(len(X_data)*0.9):len(y_data)-1]
print("Train data volume: {}".format(len(X_train)), "Valuate data volume: {}".format(len(X_valuate)), "Test data volume: {}".format(len(X_test)))


X_data= X_data.reshape(23436,1024)
X_train = X_train.reshape(len(X_train), 1024)
X_valuate = X_valuate.reshape(len(X_valuate),1024)
X_test = X_test.reshape(len(X_test),1024)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_valuate_prep = scaler.transform(X_valuate)
X_test_prep = scaler.transform(X_test)

X_tr = X_train_prep.reshape((len(X_train_prep),32,32,1))
X_val = X_valuate_prep.reshape((len(X_valuate_prep),32,32,1))
X_te = X_test_prep.reshape((len(X_test_prep),32,32,1))

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience = 5, restore_best_weights =True)

print("Memory Usage (before import):", p.memory_info().rss/1024/1024, "MB")

Mem_0 = p.memory_info().rss/1024/1024

from functools import partial
import time


def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))

    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
    att = tf.einsum('baik,baij->bakj',q, k)/np.sqrt(dv)
    
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)
    out = tf.einsum('bajk,baik->baji',att, v)
    
    out = Reshape([l, d])(out)
    
    out = Add()([out, q1])

    out = Dense(dout, activation = "relu")(out)

    return  Model(inputs=[q1,k1,v1], outputs=out)
    
  
nb_classes = 6

print("X_train original shape", X_tr.shape)
print("y_train original shape", y_train.shape)
#X_tr = X_tr.astype('float32')
#X_te = X_te.astype('float32')
#X_tr /= 255
#X_te /= 255
print("Training matrix shape", X_tr.shape)
print("Testing matrix shape", X_te.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valuate= np_utils.to_categorical(y_valuate, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience = 10, restore_best_weights =True)

print("Memory Usage (before import):", p.memory_info().rss/1024/1024, "MB")

Mem_0 = p.memory_info().rss/1024/1024

inp = Input(shape = (32,32,1))
x = Conv2D(64,kernel_size = 5,activation='relu', padding='same')(inp)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32,kernel_size = 3,activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.5)(x)


x = Reshape([8*8,32])(x)    
att = MultiHeadsAttModel(l=8*8, d=32 , dv=4, dout=32, nv = 8 )
x = att([x,x,x])
x = Reshape([8,8,32])(x)   
x = BatchNormalization()(x)
    
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = Dense(6, activation='softmax')(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001,decay=1e-4),metrics=['accuracy'])

print(model.summary())

history = model.fit(X_tr,Y_train,batch_size=64, epochs=200, validation_data = (X_val, Y_valuate))
print("Score on the test set is : ")

score = model.evaluate(X_te, Y_test)

print("Memory Usage (after  import):", p.memory_info().rss/1024/1024, "MB")

Mem_1 = p.memory_info().rss/1024/1024

print("Memory used in training : ", Mem_1-Mem_0 , "MB") 

import matplotlib.pyplot as plt


pd.DataFrame(history.history).plot(figsize=(10, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Loss Plot")
plt.show()


############# Evaluation in noisy data ##############

xtest = np.array(X_test) + np.random.normal(0,0.01, X_test.shape[1])

xtestprep = scaler.transform(xtest)

xte = xtestprep.reshape((len(X_test),32,32,1))

model.evaluate(xte,y_test)

from functools import partial
from sklearn.model_selection import StratifiedKFold

########### K-FOLD CV ##########
import time

start = time.time()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
for train, test in kfold.split(X_tr, y_train):
  # create model
    inp = Input(shape = (32,32,1))
    x = Conv2D(64,kernel_size = 5,activation='relu', padding='same')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32,kernel_size = 3,activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)


    x = Reshape([8*8,32])(x)    
    att = MultiHeadsAttModel(l=8*8, d=32 , dv=4, dout=32, nv = 8 )
    x = att([x,x,x])
    x = Reshape([8,8,32])(x)   
        
    x = BatchNormalization()(x)
    
    x = Flatten()(x) 
    
    x = Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(6, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)

    # Compile model and plot Loss-Plot
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-4),metrics=['accuracy'])
    # Fit the model
    history = model.fit(X_tr[train], y_train[train],batch_size=64, epochs = 200, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_tr[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

end = time.time()
print(end - start)
