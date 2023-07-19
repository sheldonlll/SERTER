# -*- coding = utf-8 -*-
# @Time : 2022/3/15 9:49
# @Author : yusun
# @File : bi_test_base.py
# @Softwore : PyCharm


import numpy as np
import matplotlib.pyplot as plt
import math
SEED=123456

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

audio_results_x = []
trans_results_x = []
for a in audio_results:
    a_x = max(a)
    audio_results_x.append(a_x)

for i,b in enumerate(trans_results):
    b_x = b[audio_label[i]]
    trans_results_x.append(b_x)

tr_a_soft = []
au_a_soft = []
for i , trans_inm in enumerate(trans_results_x):
    tr_a = math.exp(trans_inm)/(math.exp(trans_inm)+math.exp(audio_results_x[i]))
    tr_a_soft.append(tr_a)
    au_a = math.exp(audio_results_x[i])/(math.exp(trans_inm)+math.exp(audio_results_x[i]))
    au_a_soft.append(au_a)

tr_a_soft = np.array(tr_a_soft)
au_a_soft = np.array(au_a_soft)

bi_result = []
for i , trans_wei in enumerate(tr_a_soft):
    bi_result.append(trans_wei*trans_results[i]+au_a_soft[i]*audio_results[i])

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
plt.scatter(tr_a_soft, au_a_soft, alpha=0.4, label='weigh')
plt.legend()
plt.show()
'''



print()



