# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import librosa
import os


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    # audio数据集处理
    def process_data(path, overlap, type, t=2, RATE=16000, dataset='iemocap'):

        impro_or_script = 'impro'
        meta_dict = {}
        IEMOCAP_LABEL = {
            '01': 'neutral',
            # '02': 'frustration',
            # '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            # '06': 'fearful',
            '07': 'happy',  # excitement->happy
            # '08': 'surprised'
        }
        RAVDESS_LABEL = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

        # VLTP数据增强操作
        data_dir = 'data/IEMOCAP'
        data_files = []
        with open(path) as f:
            fr = f.readlines()
            for line in fr:
                data_files.append(data_dir + '/' + line.split('\t')[2])

        print("constructing meta dictionary for {}...".format(path))

        # 处理训练数据 account——train——num
        account_num = []
        if (type == 'test'):
            if (overlap >= t):
                overlap = t / 2
        for i, wav_file in enumerate(tqdm(data_files)):
            label = str(os.path.basename(wav_file).split('-')[2])
            if (dataset == 'iemocap'):
                if (label not in IEMOCAP_LABEL):
                    continue
                if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
                    continue
                label = IEMOCAP_LABEL[label]
            elif (dataset == 'ravdess'):
                if (label not in RAVDESS_LABEL):
                    continue
                label = RAVDESS_LABEL[label]
            wav_data, _ = librosa.load(wav_file,
                                       sr=RATE)  # librosa是python的一个音频处理的包，其中的load函数就是用来读取音频的。当然，读取之后，转化为了numpy的格式储存，而不再是音频的格式了。
            X1 = []
            y1 = []
            index = 0
            if (t * RATE >= len(wav_data)):
                continue

            num_account = 0  # 单个
            while (index + t * RATE < len(wav_data)):
                X1.append(wav_data[int(index):int(index + t * RATE)])
                y1.append(label)
                if (type != 'test'):
                    assert t - overlap > 0
                index += int((t - overlap) * RATE)
                # ------------------------------------------------------------
                num_account = num_account + 1
            account_num.append(num_account)
            # ------------------------------------------------------------
            X1 = np.array(X1)
            meta_dict[i] = {
                'X': X1,
                'y': y1,
                'path': wav_file
            }

        print("building X, y...")
        train_X = []
        train_y = []
        for k in meta_dict:
            train_X.append(meta_dict[k]['X'])
            train_y += meta_dict[k]['y']
        train_X = np.row_stack(train_X)
        train_y = np.array(train_y)
        # 检查数据长度一致
        assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
            train_X.shape, train_y.shape)

        # 根据type返回对应数据
        if (type == 'test'):
            return meta_dict, account_num
        else:
            return train_X, train_y, account_num

    # txt数据集处理
    def load_dataset(path, num, pad_size=32, type='train'):
        contents = []
        with open(path, 'r') as f:
            if (type == 'test'):
                for line in tqdm(f):
                    _1, content, _2, label = line.split('\t')
                    label = label.strip('\n')
                    if label == '01':
                        label = '0'
                    if label == '04':
                        label = '1'
                    if label == '05':
                        label = '2'
                    if label == '07':
                        label = '3'
                    content = content.strip()
                    token = config.tokenizer.tokenize(content)
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = config.tokenizer.convert_tokens_to_ids(token)

                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append((token_ids, int(label), seq_len, mask))
            else:
                for n in num:
                    for i in range(n):
                        for line in tqdm(f):
                            _1, content, _2, label = line.split('\t')
                            label = label.strip('\n')
                            if label == '01':
                                label = '0'
                            if label == '04':
                                label = '1'
                            if label == '05':
                                label = '2'
                            if label == '07':
                                label = '3'
                            content = content.strip()
                            token = config.tokenizer.tokenize(content)
                            token = [CLS] + token
                            seq_len = len(token)
                            mask = []
                            token_ids = config.tokenizer.convert_tokens_to_ids(token)

                            if pad_size:
                                if len(token) < pad_size:
                                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                                    token_ids += ([0] * (pad_size - len(token)))
                                else:
                                    mask = [1] * pad_size
                                    token_ids = token_ids[:pad_size]
                                    seq_len = pad_size
                            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train_audio, train_y, train_audio_num = process_data(path=config.train_path,overlap=1,type='train')
    train_txt = load_dataset(config.train_path, train_audio_num, config.pad_size, type='train')

    dev_audio, dev_y, dev_audio_num = process_data(path=config.dev_path,overlap=1,type='dev')
    dev_txt = load_dataset(config.dev_path, dev_audio_num, config.pad_size, type='dev')

    test_audio, test_audio_num = process_data(path=config.test_path,overlap=1.6,type='test')
    test_txt = load_dataset(config.test_path, test_audio_num, config.pad_size, type='test')
    return train_txt, dev_txt, test_txt, train_audio, dev_audio, test_audio, train_y, dev_y


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
