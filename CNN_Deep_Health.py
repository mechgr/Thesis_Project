
import os, psutil
p = psutil.Process(os.getpid())

import tensorflow as tf
import random
import numpy as np
import time
from tensorflow import keras

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import gc


#import Data_DH_1

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
    ## MAIN DATASET
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

start = time.time()

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=5, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=5, input_shape=[32,32,1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=32, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=6, activation='softmax'),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0002, decay=1e-4), metrics=["accuracy"])
history = model.fit(X_tr, y_train, epochs=150,validation_data = (X_val, y_valuate),callbacks = early_stopping)

end = time.time()
print(end - start)

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

xtest = np.array(X_test) + np.random.normal(0,0.1, X_test.shape[1])

xtestprep = scaler.transform(xtest)

xte = xtestprep.reshape((len(X_test),32,32,1))

model.evaluate(xte,y_test)

from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
for train, test in kfold.split(X_tr, y_train):
  # create model
    
    model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=5, input_shape=[32,32,1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=32, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=6, activation='softmax'),
    ])
    
    # Compile model and print Loss-Plot
    model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(lr =0.0002, decay = 1e-4), metrics=["accuracy"])
    # Fit the model
    history = model.fit(X_tr[train], y_train[train], epochs=50, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_tr[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
