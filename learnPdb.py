#パラメータ
epochs=50
testcsv="if/VLCQAHx.csv"


import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import backend as K

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
import os
from tensorflow.keras.utils import multi_gpu_model

def tmodel():
    model = Sequential()
    model.add(Dense(500, activation='sigmoid', input_shape=(24,)))
    model.add(Dense(1500, activation='sigmoid'))
    model.add(Dense(800, activation='sigmoid'))
    model.add(Dense(120, activation='sigmoid'))
    return model

def train(fx,fy,we,gpu):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu,allow_growth=True))
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    K.set_session(sess)#メモリを使い尽くさないようにする設定

    x = np.loadtxt(fx, delimiter=",", skiprows=0,usecols=(range(24))) # 読み込みたい列番号
    x = ((x+180)/360) #正規化
    y = np.loadtxt(fy, delimiter=",", skiprows=0, usecols=(range(120))) # 読み込みたい列番号
    xx = pd.read_csv(testcsv, header=None) #dont end ,
    xx = ((xx+180)/360)
    in_train = x.astype('float32')
    out_train = y#.astype('float32')
    in_test = xx.astype('float32')
    model = tmodel()

    wname = "weight/" + we + ".h5"

    if os.path.exists(wname):
        model.load_weights(wname)
        print(wname)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(in_train, out_train, epochs=epochs, batch_size=int(in_train.shape[0]/10.0)+1 ,shuffle=True)#batchはレコード総数の10%
    model.save_weights(wname, save_format="hdf5")
#    model.load_weights(wname)

#    decoded_fix = model.predict(x[:9])#正規化する二次元配列

    resList = {
            1:"A",2:"R",3:"N",4:"D",5:"C",6:"Q",7:"E",8:"G",9:"H",10:"I",11:"L",
            12:"K",13:"M",14:"F",15:"P",16:"S",17:"T",18:"W",19:"Y",20:"V",0:"■",
            }

    decoded_fix = model.predict(xx)#正規化する二次元配列
    print(xx.shape)
    print(model.predict(xx).shape)
    with open("log/reslog.txt", mode='a') as log:
        print(fx +" | "+ fy +" | "+ we, file=log)
        for row in range(0,1):
            j=0
            for i in range(row,row+6):
                min = j*20
                max = (j+1)*20
                ans = decoded_fix[row,range(min,max)] / np.sum(decoded_fix[row,range(min,max)])
                ans = ans / np.sum(ans)
                valmax = np.argmax(ans)
                if valmax+1 in resList:
                    print(resList[valmax+1], end=" :", file=log)
                print(np.max(ans), file=log)
                decoded_fix[row,range(min,max)] = [0] * 20
                decoded_fix[row,min+valmax] = 1
                j+=1
            print("", file=log)
"""
def plot_history(history):# 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
#    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
#    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
"""
