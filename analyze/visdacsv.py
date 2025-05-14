#import pandas
import pandas as pd
import os,sys
from collections import defaultdict
import numpy as np

dm1 = ['Art', 'Clipart', 'Product', 'RealWorld']
dm2 = ['amazon', 'dslr', 'webcam']
dm3 = ['train', 'validation']
ttype = 'PDA'
ttype = 'OPDA'
#ttype = 'OSDA'

myd = defaultdict(list)
fd = '..'
print('===========', fd, '===========')
for f in os.listdir(fd):
    if f.endswith('.csv'):
        #print(f)
        ff = os.path.join(fd, f)
        res = pd.read_csv(ff, header=None).iloc[1].tolist()
        f2 = f.replace('Real_World', 'RealWorld')
        #print(f2)
        if (res[1] < 1) and (res[1] > 0):
            hos, acc_test, nmi, k_acc, uk_nmi = res[1:6]
            epoch = None
        else:
            hos, acc_test, nmi, k_acc, uk_nmi = res[2:7]
            epoch = int(res[1])
        if len(f2[:-4].split('_')) == 15:
            _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs = f2[:-4].split('_')
            #if domain == ttype:
                # myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs, max_k)].append(
                #     [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
            if src == 'train' and domain == 'OPDA':
                print(f)
                print(hos, acc_test, nmi)


