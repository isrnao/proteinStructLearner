#Copyright © 2020 DoiNaoya All rights reserved.

#パラメータ
accnum=10#全データの何分の1で学習度確認をするか

from sys import argv
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import backend as K
import learnPdb as ptt

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def accPdb(w,x,y):
    print("----------acc-----------")
    xx = np.loadtxt("data/data/" +x+".csv", delimiter=",", skiprows=0,usecols=(range(24))) # 予測入力
    yy = np.loadtxt("data/data/" +y+".csv", delimiter=",", skiprows=0,usecols=(range(120))) # 答え入力

    xx = ((xx+180)/360) #正規化
    adam = ptt.tmodel()
    #adam = load_model('model/5nm.h5', compile=True)

    wt = 'weight/' + w + '.h5'

    adam.load_weights(wt)

    resList = {
            1:"A",2:"R",3:"N",4:"D",5:"C",6:"Q",7:"E",8:"G",9:"H",10:"I",11:"L",
            12:"K",13:"M",14:"F",15:"P",16:"S",17:"T",18:"W",19:"Y",20:"V",0:"■",
            }

    decoded_yy = np.array(yy)
    decoded_fix = adam.predict(xx)#正規化する二次元配列

    """
print(xx.shape)
print(adam.predict(xx).shape)
    print(yy.shape)
    print(decoded_yy.shape)
    print(yy)
    print(decoded_yy)
    """
#print(decoded_fix)

    right, wrong = 0, 0
    num = int(yy.shape[0]/accnum)
#    print(num)

    for row in range(0,num):
        j=0
        for i in range(row,row+6):
            min = j*20
            max = (j+1)*20

            ans = decoded_fix[row,range(min,max)] / np.sum(decoded_fix[row,range(min,max)])
            ans = ans / np.sum(ans)

            valmax = np.argmax(decoded_yy[row,range(min,max)])
            valmaxx = np.argmax(ans)
            if valmax+1 in resList:
                print(resList[valmax+1], end=" :")
                print(resList[valmaxx+1])
                if resList[valmax+1] == resList[valmaxx+1]:right+=1
                else:wrong+=1
            print(ans)
#            print(decoded_fix[row,range(min,max)])
            decoded_fix[row,range(min,max)] = [0] * 20
            decoded_fix[row,min+valmax] = 1
            j+=1
        print(row,"/",num-1)
#  print("")
#ans = np.reshape(decoded_yy[row,],decoded_fix[row,], (6, 20))
    with open("log/aclog.txt", mode='a') as aclog:
        print("right :" + str(right) + " | wrong :" + str(wrong))
        print("right :" + str(right) + " | wrong :" + str(wrong) + wt, file=aclog)
#        print("acc :" + str(right/600/num*10000) + "%")
#        print("acc :" + str(right/600/num*10000) + "%", file=aclog)
        print("acc :" + str(right/(right+wrong)*100) + "%")
        print("acc :" + str(right/(right+wrong)*100) + "%", file=aclog)
#    with open("like1aki.txt", mode='a') as ly:
#        print("","acc :" + str(right/(right+wrong)*100) + "%",wt, file=ly)






#if __name__ == '__main__':
#    num = 0
#    f = "{0:02d}".format(fnum)
#    accPdb()
