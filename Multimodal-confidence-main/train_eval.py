# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import librosa
import logging
from python_speech_features import sigproc, fbank, logfbank
from torch.utils.data import Dataset, DataLoader
import pickle


# 语音数据特征提取
class FeatureExtractor(object):
    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)
        if features_to_use in ('mfcc', 26):
            X_features = self.get_mfcc(X)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            X_features = self.get_spectrogram(X)
        if features_to_use in ('pase'):
            X_features = self.get_Pase(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.rate)
            mel = np.log10(mel + 1e-10)
            return mel

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

    def get_Pase(self, X):
        return X

# 特征矩阵封装处理
class DataSet(Dataset):
    def __init__(self, X, Y):
            self.X = X
            self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        # x = torch.from_numpy(x).unsqueeze(0)
        x = torch.from_numpy(x)
        x = x.float()
        y = self.Y[index]
        y = dict[y]
        y = y.long()
        return x, y

    def __len__(self):
        return len(self.X)

# 加载语音dataloader
def load_dataloader (data_Loader,i , test = False):
    for j, data in enumerate(data_Loader):
        if test == False:
            x, y = data
        else:
            x, y = data_Loader[data]['X'], data_Loader[data]['y']
        if j == i:
            break
    return x,y



# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, train_audio, dev_audio, test_audio ,train_y , dev_y):

    # 语音数据准备工作-------------------------------------------------------------------------------------------------
    featuresExist = False
    FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
    impro_or_script = 'impro'
    featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
    RATE = 16000
    toSaveFeatures = True
    #BATCH_SIZE = 32
    SEED = 123456

    if (featuresExist == True):
        with open(featuresFileName, 'rb')as f:
            features = pickle.load(f)
        train_X_features = features['train_X']
        train_y = features['train_y']
        valid_features_dict = features['val_dict']
    else:
        logging.info("creating meta dict...")
        #train_X, train_y, val_dict = process_data(WAV_PATH, t=2, train_overlap=1)
        #print(train_X.shape)
        #print(len(val_dict))

        print("getting features")
        logging.info('getting features')
        feature_extractor = FeatureExtractor(rate=RATE)
        # 提取特征-------------------------------------------------------------------------
        train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_audio)
        dev_X_features = feature_extractor.get_features(FEATURES_TO_USE, dev_audio)

        # -------------------------------------------------------------------------------

        valid_features_dict = {}
        for _, i in enumerate(test_audio):
            X1 = feature_extractor.get_features(FEATURES_TO_USE, test_audio[i]['X'])
            valid_features_dict[i] = {
                'X': X1,
                'y': test_audio[i]['y']
            }

        if (toSaveFeatures == True):
            features = {'train_X': train_X_features, 'train_y': train_y,
                        'val_dict': valid_features_dict}
            with open(featuresFileName, 'wb') as f:
                pickle.dump(features, f)

    dict = {
        'neutral': torch.Tensor([0]),
        'happy': torch.Tensor([1]),
        'sad': torch.Tensor([2]),
        'angry': torch.Tensor([3]),
        'calm': torch.Tensor([4]),
        'disgust': torch.Tensor([5]),
        'fearful': torch.Tensor([6]),
        'surprised': torch.Tensor([7]),
    }
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件

    log_name = 'test-result/seed-{}.log'.format(SEED)
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    train_data = DataSet(train_X_features, train_y)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    dev_data = DataSet(dev_X_features, dev_y)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True)

    #model = model.MACNN(attention_head, attention_hidden)
    #model = HeadFusion(attention_head, attention_hidden, 4)
    #if torch.cuda.is_available():
        #model = model.cuda()
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # 文本数据准备工作-------------------------------------------------------------------------------------------------

    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    #--------------------------------------------------------------------------------------------------------------

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # 音频特征读取-------------------------------------------------
            x, y = load_dataloader(train_loader , i)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # ----------------------------------------------------------
            outputs = model(trains , x.unsqueeze(1))  # 需要两个输入
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter ,dev_loader)  # 需要两个输入
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    test(config, model, test_iter , valid_features_dict)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion , list_all = evaluate(config, model, test_iter, test=True)  # 需要两个输入
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, data_loader, test = False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype = int)
    labels_all = np.array([], dtype = int)
    list_all = []
    with torch.no_grad():
        for i,(texts, labels) in enumerate(data_iter):
            x , y = load_dataloader(data_Loader=data_loader,i=i)
            if test == False:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
            else:
                x = torch.from_numpy(x).float()
                y = dict[y[0]].long()
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                if (x.size(0) == 1):
                    x = torch.cat((x, x), 0)
            outputs = model(texts , x.unsqueeze(1))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            # 测试代码-------------------------------------------------------
            for list_1 in outputs.data:
                list_end = str(float(list_1[0])) + '\t' + str(float(list_1[3])) + '\t' + str(float(list_1[1])) + '\t' + str(float(list_1[2])) + '\n'
                list_all.append(list_end)
            #list_1 = outputs.data
            #list_2 = list_1[0]
            #list_3 = str(float(list_2[0]))

            # --------------------------------------------------------------
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test == True:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion , list_all
    return acc, loss_total / len(data_iter)
