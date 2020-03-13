#パラメータ
dir="data/data/" #csv dir
gpu = argv[2] #0 or 1


import learnPdb as ptt
import learnAcc as acc
import os, subprocess
import glob
from sys import argv
fname = glob.glob("data/data/x*.csv")
num = 0
if not os.path.exists("weight"):
    os.makedirs("weight")
for fx in fname:
    x = "x" + str(num)
    y = "y" + str(num)
    w = argv[1] +  str(num)
    for _ in range(8):
        print(dir+x+".csv", dir+y+".csv", w)
        ptt.train(dir+x+".csv", dir+y+".csv", w, gpu)
        acc.accPdb(w,x,y)
    num+=1
