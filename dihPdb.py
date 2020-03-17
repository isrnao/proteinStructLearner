#Copyright © 2020 DoiNaoya All rights reserved.
#! /usr/bin/env python
from sys import argv
#進捗確認>watch -td cat log/tqdm.txt

#パラメータ
fnum = 1000#int(argv[1]) #1データセットサイズ（単位：PDB件数）
fsum = int(argv[1]) #データセット作成数（単位：ファイル数）
pdbdir = "data/pdb/"
datadir = "data/data/"


import prody
import string
from pyrosetta import *
import glob
from tqdm import tqdm
from numpy.random import *
from joblib import Parallel, delayed
num = 0
pdbnum = fnum*fsum #総PDB数（データセット量*作成数=総PDB数）

def normalization(n): #nは残基名
    nlzList = {
        "ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,
        "GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
        "LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,
        "SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,
    }
    nlz="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    if n == "ALA":
        return str("1," + nlz[2:])
    elif n == 'VAL':
        return str(nlz[:37] + ",1")
    else:
        return str(nlz[:int(nlzList.get(n,21)*2-2)] + "1," + nlz[int(nlzList.get(n,21)*2):])

def cutPdb(pdbname,num):
    with open("log/tqdm.txt", "w") as tq:
        print(str(num)+"/"+str(pdbnum), file=tq)
    num = num%fsum
 #   num = 0
    name = ""
    res = ""

    try:
        plen=prody.parsePDB(pdbname)
        pyrosetta.init()
        pdb = pose_from_pdb(pdbname)

        n, i = 1, 0

        c = plen.select('name CA')
        amino = len(c)
        print(amino)

        resn = 6

    except:
        return

    for _ in range(1,amino*resn):
        residue = str(plen['A',n])
        print(residue)
        if str(residue)=="None":
            name += "None "

            i += 1
            n += 1
        try:
            if pdb.residue(n).name()[:3] in {"GLY", "ALA"}:
                #name += pdb.residue(n).name()[:3]
                name += str(pdb.phi(n)) + "," + str(pdb.psi(n)) + "," + str(pdb.omega(n)) + ",0.00,"
                res += normalization(pdb.residue(n).name()[:3]) + ","

            else:
                #name += pdb.residue(n).name()[:3]
                name += str(pdb.phi(n)) + "," + str(pdb.psi(n)) + "," + str(pdb.omega(n)) + "," + str(pdb.chi(1,n)) + ","
                res += normalization(pdb.residue(n).name()[:3]) + ","

        except:
            continue
        n += 1
        i += 1

        if i%resn==0:
            n -= resn-1
            if ('None' in name)==True:
                name = ""
                res = ""
                continue
            elif ('0.0,' in name)==True:
                name = ""
                res = ""
                continue
            else:
                with open(str(datadir) + "x" + str(num) + ".csv", "a") as fx:
                    print(name, file=fx)
                with open(str(datadir) + "y" + str(num) + ".csv", "a") as fy:
                    print(res, file=fy)
#                        print(res,pdbname, file=fy)
                name = ""
                res = ""

#    print amino
def main_calc():
    if not os.path.exists("log"):
        os.makedirs("log")
    pdbn = glob.glob(pdbdir+"*.pdb")
    Parallel(n_jobs=14)([delayed(cutPdb)(pdbn[randint(0,len(pdbn)-1)],n) for n in range(0,pdbnum)])


if __name__ == '__main__':
#    f = "{0:02d}".format(fnum)
    main_calc()
    print("fin")
