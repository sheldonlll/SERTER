# -*- coding = utf-8 -*-
# @Time : 2022/3/20 17:56
# @Author : yusun
# @File : bi_test_vc.py
# @Softwore : PyCharm

import numpy as np
import matplotlib.pyplot as plt
import math
SEED=123456

def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

DICT = {'01': 0, '07': 1, '04': 2, '05': 3}
label=[]
with open('D:/pycharm/CondaEnv/ERmain/IEMOCAP/IEMOCAP_bitest_{}.csv'.format(SEED),'r') as f:
    fr=f.readlines()
for line in fr:
    label.append(DICT[line.replace('\n', '').split('\t')[3]])
audio_results = np.load('D:/pycharm/CondaEnv/ERmain/test/CNN_mfcc_SERtest_{}.npy'.format(SEED),allow_pickle=True)
audio_results=audio_results.squeeze()
audio_label=np.argmax(audio_results,1)
#with open('/program/xumingke/bert/base/{}/test_results.tsv'.format(SEED),'r') as f:
#with open('D:/pycharm/CondaEnv/ERmain/output2/test_results.tsv','r') as f:
with open('D:/pycharm/CondaEnv/ERmain/tsv/bert_output_123456_exp.tsv','r') as f:
    fr=f.readlines()
trans_results=[]

for line in fr:
    num = line.replace('\n', '').split('\t')
    trans_results.append([float(num[0]), float(num[1]), float(num[2]), float(num[3])])
trans_results=np.array(trans_results)

label_a = []
for i in label:
    aka = [0,0,0,0]
    aka[i] = 1
    label_a.append(aka)
label_a = np.array(label_a)

weight1 = []
weight2 = []

for n , end in enumerate(label_a):
    A = audio_results[i]
    B = trans_results[i]
    var1 = np.var(end)
    cov1 = covariance(A,end)
    cov2 = covariance(B,end)
    R1 = (cov1+cov2)/var1
    R2 = cov1/var1
    R3 = cov2/var1
    B1 = R1-R2+R3/2
    B2 = R1-R3+R2/2

    C1 = math.exp(B1)
    C2 = math.exp(B2)
    weight1.append(C1/(C1+C2))
    weight2.append(C2/(C1+C2))
weight1 = np.array(weight1)
weight2 = np.array(weight2)


bi_result = []
for i , trans_wei in enumerate(weight1):
    bi_result.append(trans_wei*trans_results[i]+weight2[i]*audio_results[i])

bi_result=np.array(bi_result)

bi_corr = 0
bi_class = [0, 0, 0, 0]
bi_class_corr = [0, 0, 0, 0]
bi_label = np.argmax(bi_result, 1)
for i in range(len(label)):
    if (bi_label[i] == label[i]):
        bi_corr += 1
        bi_class_corr[label[i]] += 1
    bi_class[label[i]] += 1
UA = 0
for i in range(4):
    UA += bi_class_corr[i] / bi_class[i]
UA = UA / 4
print(str(bi_corr / len(label)) + ',' + str(UA))

'''
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel('text_weigh')
plt.ylabel('audio_weigh')
plt.xlim(xmax=1,xmin=0)
plt.ylim(ymax=1,ymin=0)

#colors1 = '#00CED1'
#area = np.pi * 4**2

#plt.scatter(tr_a_soft, au_a_soft, s=area, c=colors1, alpha=0.4, label='weigh')
plt.scatter(weight1, weight2, alpha=0.1, label='weigh')
plt.legend()
plt.show()
'''


print()







