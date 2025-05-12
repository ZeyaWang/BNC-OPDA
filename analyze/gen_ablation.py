#import pandas
import pandas as pd
import os,sys
from collections import defaultdict
import numpy as np
import argparse

dm1 = ['Art', 'Clipart', 'Product', 'RealWorld']
dm2 = ['amazon', 'dslr', 'webcam']
dm3 = ['train', 'validation']

dmd = {
    'officehome': dm1,
    'office': dm2
}

parser = argparse.ArgumentParser(description='Process.')
parser.add_argument('--ttype', type=str, help='ttype', default='OPDA')
parser.add_argument('--fd', type=str, help='dir', default='..')
parser.add_argument('--abtype', type=str, help='ab', default='balance')

args = parser.parse_args()
ttype = args.ttype

fd = args.fd
abtype = args.abtype

print('===========', fd, '===========')

base =  {'officehome': ('0.01', '0.0001', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100'),
        'office': ('0.01', '0.01', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100')}

abs = {'KK': [5, 10, 20, 50], 'balance': [0.001, 0.01, 0.1, 0.0, 1.0], 'alpha': [0.0, 100.0, 200.0, 400.0]}
pos = {'KK': 5, 'balance': 0, 'alpha': 4}

ab = abs[abtype]
ps = pos[abtype]

abpool = {}

for ds in ['office', 'officehome']:
    abpool[ds] = []
    bs = base[ds]
    for v in ab:
        bst = list(bs)
        bst[ps] = str(v)
        bst = tuple(bst)
        abpool[ds].append(bst)

epochs = [0, 10]
myr = {}
for epoch in epochs:
    myd = defaultdict(list)
    for f in os.listdir(fd):
        if f.endswith('.csv'):
            ff = os.path.join(fd, f)
            res = pd.read_csv(ff, header=None).iloc[1+epoch].tolist()
            f2 = f.replace('Real_World', 'RealWorld')
            if (res[1] < 1) and (res[1] > 0):
                hos, acc_test, nmi, k_acc, uk_nmi = res[1:6]
                epoch = None
            else:
                hos, acc_test, nmi, k_acc, uk_nmi = res[2:7]
                epoch = int(res[1])
            hos = round(hos*100, 1)
            acc_test = round(acc_test*100, 1)
            nmi = round(nmi*100, 1)
            k_acc = round(k_acc*100, 1)
            uk_nmi = round(uk_nmi*100, 1)
            if len(f2[:-4].split('_')) == 15:
                _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs = f2[:-4].split('_')
                if domain == ttype:
                    myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs, max_k)].append(
                        [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
            elif len(f2[:-4].split('_')) == 16:
                _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs, dt = f2[:-4].split('_')
                if domain == ttype:
                    myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs, dt, max_k)].append(
                        [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
            elif len(f2[:-4].split('_')) == 14:
                _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_')
                if domain == ttype:
                    myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, max_k)].append(
                        [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
            elif len(f2[:-4].split('_')) == 13:
                _, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_')
                myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, max_k)].append(
                    [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
    for k in myd:
        myd[k].sort()
    myr[epoch] = myd

# print(abpool)
# print(myr)

np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=1000)
for epoch, myd in myr.items():
    for ds in ['office', 'officehome']:
        abp = abpool[ds]
        for k in myd:
            if k in abp:
                dd = pd.DataFrame(myd[k])
                ddd = dd.fillna('')
                #print(ddd)
                dm = dmd[ds]
                dd1 = ddd[ddd[0].isin(dm)]
                #print(dm, dd1)
                dd1['task'] = dd1[0].astype(str) + '_' +  dd1[1].astype(str)
                dd1 = dd1.drop(columns=[0, 1])
                mean1 = dd1.drop(columns=['task']).mean()
                mean1['task'] = 'Average'
                dd1.loc['mean'] = mean1
                #print(dd1)
                ddd = pd.concat([dd1], ignore_index=True)
                ddd = ddd[['task',2,3,4,5,6,7]]
                ddd = ddd.rename(columns={2: 'hos',3: 'acc',4: 'nmi',5: 'known acc',6: 'unknown nmi', 7: 'epoch'})
                ddd = ddd.round(1)
                # hos = ddd['hos'].astype(str).tolist()
                # acc = ddd['acc'].astype(str).tolist()
                # nmi = ddd['nmi'].astype(str).tolist()
                print(k)
                print(ddd)
                # if k == ('0.01', '0.0001', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100'):
                #     hos_b_h = '& '.join(hos[:13])
                #     acc_b_h = '& '.join(acc[:13])
                #     nmi_b_h = '& '.join(nmi[:13])
                # if k == ('0.01', '0.01', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100'):
                #     hos_b = '& '.join(hos[13:])
                #     acc_b = '& '.join(acc[13:])
                #     nmi_b = '& '.join(nmi[13:])

# ########################################################################################################################
# epoch = [10]
#
# for i in epoch:
#     print('========== results epoch {} ==========='.format(i))
#     myd = defaultdict(list)
#     for f in os.listdir(fd):
#         if f.endswith('.csv'):
#             # print(f)
#             ff = os.path.join(fd, f)
#             res = pd.read_csv(ff, header=None).iloc[i+1].tolist()
#             #print(res)
#             f2 = f.replace('Real_World', 'RealWorld')
#             # print(f2)
#             # _, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_') # old version
#             if (res[1] < 1) and (res[1] > 0):
#                 hos, acc_test, nmi, k_acc, uk_nmi = res[1:6]
#                 epoch = None
#             else:
#                 hos, acc_test, nmi, k_acc, uk_nmi = res[2:7]
#                 epoch = int(res[1])
#             # if lam not in ['0.1','1.0'] and interval != '10':
#             # myd[(lam, lam2, interval)].append([src, tar, hos, acc_test, nmi, k_acc, uk_nmi])
#             # if (balance == '0.01'):# and (KK != '5'):
#             hos = round(hos * 100, 1)
#             acc_test = round(acc_test * 100, 1)
#             nmi = round(nmi * 100, 1)
#             k_acc = round(k_acc * 100, 1)
#             uk_nmi = round(uk_nmi * 100, 1)
#             if len(f2[:-4].split('_')) == 15:
#                 _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs = f2[
#                                                                                                              :-4].split(
#                     '_')
#                 if domain == ttype:
#                     myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs, max_k)].append(
#                         [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#             elif len(f2[:-4].split('_')) == 16:
#                 _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs, dt = f2[
#                                                                                                                  :-4].split(
#                     '_')
#                 if domain == ttype:
#                     myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs, dt, max_k)].append(
#                         [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#             elif len(f2[:-4].split('_')) == 14:
#                 _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split(
#                     '_')
#                 if domain == ttype:
#                     myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, max_k)].append(
#                         [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#             elif len(f2[:-4].split('_')) == 13:
#                 _, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_')
#                 myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, max_k)].append(
#                     [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#
#     for k in myd:
#         myd[k].sort()
#     #print(myd)
#
#     np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=1000)
#     for k in myd:
#         if (k == ('0.01', '0.0001', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100')) or (
#                 k == ('0.01', '0.01', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100')):
#             dd = pd.DataFrame(myd[k])
#             # nc = dd.select_dtypes(include='number').mean()
#             # nc = pd.DataFrame(nc).T
#             # ddd = pd.concat([dd, nc], ignore_index=True)
#             ddd = dd.fillna('')
#             # fn = "sum" + "_".join(map(str, k)) + ".csv"
#             # print(k, ddd)
#             # fn = "sum" + "_".join(map(str, k)) + ".csv"
#             # print(k, ddd)
#             dd1 = ddd[ddd[0].isin(dm1)]
#             dd2 = ddd[ddd[0].isin(dm2)]
#             dd3 = ddd[ddd[0].isin(dm3)]
#             # print(dd1)
#             # print(dd2)
#             # print(dd3)
#             dd1['task'] = dd1[0].astype(str) + '_' + dd1[1].astype(str)
#             dd1 = dd1.drop(columns=[0, 1])
#             dd2['task'] = dd2[0].astype(str) + '_' + dd2[1].astype(str)
#             dd2 = dd2.drop(columns=[0, 1])
#             dd3['task'] = dd3[0].astype(str) + '_' + dd3[1].astype(str)
#             dd3 = dd3.drop(columns=[0, 1])
#             mean1 = dd1.drop(columns=['task']).mean()
#             mean1['task'] = 'Average'
#             dd1.loc['mean'] = mean1
#             mean2 = dd2.drop(columns=['task']).mean()
#             mean2['task'] = 'Average'
#             dd2.loc['mean'] = mean2
#             ddd = pd.concat([dd1, dd2, dd3], ignore_index=True)
#             ddd = ddd[['task', 2, 3, 4, 5, 6, 7]]
#             ddd = ddd.rename(columns={2: 'hos', 3: 'acc', 4: 'nmi', 5: 'known acc', 6: 'unknown nmi', 7: 'epoch'})
#             ddd = ddd.round(1)
#             hos = ddd['hos'].astype(str).tolist()
#             acc = ddd['acc'].astype(str).tolist()
#             nmi = ddd['nmi'].astype(str).tolist()
#             print(k)
#             print(ddd)
#             if k == ('0.01', '0.0001', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100'):
#                 hos_f_h = '& '.join(hos[:13])
#                 acc_f_h = '& '.join(acc[:13])
#                 nmi_f_h = '& '.join(nmi[:13])
#             if k == ('0.01', '0.01', '0.1', '1', '0.0', '5', '0.001', 'cos', 'True', '64', '100'):
#                 hos_f = '& '.join(hos[13:])
#                 acc_f = '& '.join(acc[13:])
#                 nmi_f = '& '.join(nmi[13:])
#             # hos = '& '.join(hos)
#             # acc = '& '.join(acc)
#             # nmi = '& '.join(nmi)
#             # print(hos)
#             # print(acc)
#             # print(nmi)
# print(hos_b_h)
# print(hos_f_h)
# print(hos_b)
# print(hos_f)
# print(acc_b_h)
# print(acc_f_h)
# print(acc_b)
# print(acc_f)
# print(nmi_b_h)
# print(nmi_f_h)
# print(nmi_b)
# print(nmi_f)

########################################################################################################################
# print('========== final results ===========')
# myd = defaultdict(list)
# for f in os.listdir(fd):
#     if f.endswith('.csv'):
#         # print(f)
#         ff = os.path.join(fd, f)
#         res = pd.read_csv(ff, header=None).iloc[10].tolist()
#         f2 = f.replace('Real_World', 'RealWorld')
#         # print(f2)
#         # _, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_') # old version
#         if (res[1] < 1) and (res[1] > 0):
#             hos, acc_test, nmi, k_acc, uk_nmi = res[1:6]
#             epoch = None
#         else:
#             hos, acc_test, nmi, k_acc, uk_nmi = res[2:7]
#             epoch = int(res[1])
#         # if lam not in ['0.1','1.0'] and interval != '10':
#         # myd[(lam, lam2, interval)].append([src, tar, hos, acc_test, nmi, k_acc, uk_nmi])
#         # if (balance == '0.01'):# and (KK != '5'):
#         if len(f2[:-4].split('_')) == 15:
#             _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf, bs = f2[:-4].split(
#                 '_')
#             if domain == ttype:
#                 myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf, bs)].append(
#                     [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#         elif len(f2[:-4].split('_')) == 14:
#             _, domain, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_')
#             if domain == ttype:
#                 myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf)].append(
#                     [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#         elif len(f2[:-4].split('_')) == 13:
#             _, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, clf = f2[:-4].split('_')
#             myd[(balance, lr, lr_scale, interval, lambdav, KK, cov, sc, clf)].append(
#                 [src, tar, hos, acc_test, nmi, k_acc, uk_nmi, epoch])
#
# for k in myd:
#     myd[k].sort()
# # print(myd)
#
# np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=1000)
# for k in myd:
#     dd = pd.DataFrame(myd[k])
#     nc = dd.select_dtypes(include='number').mean()
#     nc = pd.DataFrame(nc).T
#     ddd = pd.concat([dd, nc], ignore_index=True)
#     ddd = ddd.fillna('')
#     # fn = "sum" + "_".join(map(str, k)) + ".csv"
#     # print(k, ddd)
#     dd1 = ddd[ddd[0].isin(dm1)]
#     dd2 = ddd[ddd[0].isin(dm2)]
#     dd3 = ddd[ddd[0].isin(dm3)]
#     dd1['task'] = dd1[0].astype(str) + '_' + dd1[1].astype(str)
#     dd1 = dd1.drop(columns=[0, 1])
#     dd2['task'] = dd2[0].astype(str) + '_' + dd2[1].astype(str)
#     dd2 = dd2.drop(columns=[0, 1])
#     dd3['task'] = dd3[0].astype(str) + '_' + dd3[1].astype(str)
#     dd3 = dd3.drop(columns=[0, 1])
#     mean1 = dd1.drop(columns=['task']).mean()
#     mean1['task'] = 'Average'
#     dd1.loc['mean'] = mean1
#     mean2 = dd2.drop(columns=['task']).mean()
#     mean2['task'] = 'Average'
#     dd2.loc['mean'] = mean2
#     ddd = pd.concat([dd1, dd2, dd3], ignore_index=True)
#     ddd = ddd[['task', 2, 3, 4, 5, 6, 7]]
#     ddd = ddd.rename(columns={2: 'hos', 3: 'acc', 4: 'nmi', 5: 'known acc', 6: 'unknown nmi', 7: 'epoch'})
#     print(k)
#     print(ddd)

########################################################################################################################


